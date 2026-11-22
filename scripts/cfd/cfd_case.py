#!/usr/bin/env python3
"""Prepared pulmonary CFD runner. Standard library only; run on Linux or macOS."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
import zlib

TOTAL_STEPS = 8125
DT = 0.0002375
RESTART_EVERY = 401
CASE_NAME = "0084_0001"


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def case_info(case):
    case = Path(case)
    required = ["solver.inp", "geombc.dat.1", "restart.0.1", "numstart.dat", "bct.dat",
                "mesh-complete/mesh-complete.mesh.vtu"]
    for name in required:
        if not (case / name).is_file() or not (case / name).stat().st_size:
            raise ValueError(f"Missing or empty CFD input: {case / name}")
    with (case / required[-1]).open("rb") as stream:
        head = stream.read(4096).decode(errors="replace")
    nodes = re.search(r'NumberOfPoints="(\d+)"', head)
    cells = re.search(r'NumberOfCells="(\d+)"', head)
    if not nodes or not cells:
        raise ValueError("Cannot read mesh dimensions")
    outlets = re.search(r"^Number of Resistance Surfaces:\s*(\d+)", (case / "solver.inp").read_text(), re.M)
    return {"case": str(case), "nodes": int(nodes[1]), "elements": int(cells[1]),
            "resistance_outlets": int(outlets[1]) if outlets else None,
            "time_step_s": DT, "full_target_step": TOTAL_STEPS}


def prepare_archive(archive, destination, expected_sha=None):
    """Extract atomically once; never overwrite an existing simulation."""
    archive, destination = Path(archive), Path(destination)
    digest = sha256(archive)
    if expected_sha and digest != expected_sha:
        raise ValueError("Uploaded ZIP SHA-256 differs from the local archive")
    destination.mkdir(parents=True, exist_ok=True)
    case = destination / CASE_NAME
    if case.exists():
        manifest = case / "archive.json"
        if not manifest.exists() or json.loads(manifest.read_text())["sha256"] != digest:
            raise ValueError(f"Refusing to overwrite an existing case: {case}")
        return case_info(case)
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=destination) as tmp:
        staging = Path(tmp)
        with zipfile.ZipFile(archive) as z:
            if sum(i.file_size for i in z.infolist()) > 5 * 1024**3:
                raise ValueError("Archive exceeds the 5 GB extraction limit")
            for member in z.infolist():
                p = Path(member.filename)
                mode = member.external_attr >> 16
                if (p.is_absolute() or ".." in p.parts or not p.parts
                        or p.parts[0] != CASE_NAME or (mode & 0o170000) == 0o120000):
                    raise ValueError(f"Unsafe/unexpected ZIP member: {member.filename}")
            z.extractall(staging)
        case_info(staging / CASE_NAME)
        atomic_json(staging / CASE_NAME / "archive.json", {"sha256": digest})
        (staging / CASE_NAME).rename(case)
    return case_info(case)


def valid_restart(path, step):
    """Check gzip CRC, header step and complete field payloads."""
    try:
        data = Path(path).read_bytes()
        if data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
        def field(name, source):
            m = re.search(name + rb"\s*:\s*<\s*\d+\s*>\s*(\d+)\s+(\d+)\s+(\d+)\s*\n", source[:4096])
            if not m or int(m[3]) != step or int(m[1]) < 1 or int(m[2]) < 4:
                return None
            end = m.end() + int(m[1]) * int(m[2]) * 8
            return end if len(source) >= end else None
        end = field(rb"(?m)^solution", data)
        if end is None:
            return False
        return step == 0 or field(rb"time derivative of solution", data[end:]) is not None
    except (OSError, EOFError, ValueError, gzip.BadGzipFile, zlib.error):
        return False


def latest_checkpoint(directory, ranks):
    directory = Path(directory)
    steps = set()
    for path in directory.glob("restart.*.1"):
        m = re.fullmatch(r"restart\.(\d+)\.1", path.name)
        if m:
            steps.add(int(m[1]))
    for step in sorted(steps, reverse=True):
        if all(valid_restart(directory / f"restart.{step}.{rank}", step)
               for rank in range(1, ranks + 1)):
            return step
    return None


def configure(case, ranks, until, restart_every):
    case = Path(case)
    if ranks < 1 or until < 1 or restart_every < 1:
        raise ValueError("ranks, target step and restart interval must be positive")
    case_info(case)
    for other in case.glob("*-procs_case"):
        if other.name != f"{ranks}-procs_case":
            raise ValueError(f"Existing partition {other.name}: resume with its MPI ranks; use a separate case to change ranks")
    partition = case / f"{ranks}-procs_case"
    if partition.exists():
        for rank in range(1, ranks + 1):
            p = partition / f"geombc.dat.{rank}"
            if not p.is_file() or not p.stat().st_size:
                raise ValueError("Incomplete MPI partition. Use a new case; existing outputs were preserved")
        start = latest_checkpoint(partition, ranks)
        if start is None:
            raise ValueError("No complete checkpoint; existing outputs were preserved")
        (partition / "numstart.dat").write_text(f"{start}\n")
    else:
        start = 0
        if not valid_restart(case / "restart.0.1", 0):
            raise ValueError("Invalid initial restart")
        (case / "numstart.dat").write_text("0\n")
    if start >= until:
        return start, partition
    p = case / "solver.inp"
    backup = case / "solver.inp.original"
    if not backup.exists():
        shutil.copy2(p, backup)
    original = backup.read_text()
    # Legacy svSolver interprets this as ADDITIONAL steps, not the final index.
    settings = {"Number of Timesteps": until - start, "Time Step Size": DT,
                "Number of Timesteps between Restarts": restart_every}
    for key, value in settings.items():
        original, count = re.subn(rf"^{re.escape(key)}:[^\n]*", f"{key}: {value}", original, flags=re.M)
        if count != 1:
            raise ValueError(f"Expected exactly one solver setting: {key}")
    p.write_text(original)
    return start, partition


@contextlib.contextmanager
def case_lock(case):
    with (Path(case) / ".cfd.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A solver is already running in this case") from exc
        yield


def run_case(case, ranks, until=TOTAL_STEPS, restart_every=RESTART_EVERY):
    case = Path(case).resolve()
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    if available and ranks > available:
        raise ValueError(f"Requested {ranks} ranks but only {available} CPUs are available")
    mpi, solver = os.environ.get("MPIEXEC", "mpiexec"), os.environ.get("SVSOLVER", "svsolver")
    for exe in [mpi, solver]:
        if not shutil.which(exe):
            raise FileNotFoundError(f"Executable not found: {exe}; run setup_svsolver.sh first")
    with case_lock(case):
        start, partition = configure(case, ranks, until, restart_every)
        if start >= until:
            return {"status": "already_complete", "start_step": start, "end_step": start,
                    "ranks": ranks, "elapsed_s": 0, "steps_run": 0}
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            env.update(OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
        name = f"svsolver_{ranks}ranks_{start}_{until}_{time.time_ns()}.log"
        began = time.perf_counter()
        with (case / name).open("w") as log:
            process = subprocess.Popen([mpi, "-n", str(ranks), solver], cwd=case, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
            finally:
                process.stdout.close()
        elapsed = time.perf_counter() - began
        last = latest_checkpoint(partition, ranks) if partition.exists() else None
        report = {"status": "completed" if code == 0 and last == until else "failed",
                  "ranks": ranks, "start_step": start, "target_step": until, "end_step": last,
                  "steps_run": last - start if last is not None else 0, "elapsed_s": elapsed,
                  "exit_code": code, "log": name, "time_step_s": DT, "restart_every": restart_every}
        atomic_json(case / "last_run.json", report)
        if report["status"] != "completed":
            raise RuntimeError(f"Solver failed or did not reach step {until}: {report}")
        (partition / "numstart.dat").write_text(f"{until}\n")
        return report


def postprocess(case, ranks, start=4010, stop=8020, increment=RESTART_EVERY):
    case = Path(case).resolve()
    directory = case / f"{ranks}-procs_case"
    if start < 0 or stop < start or increment < 1:
        raise ValueError("Invalid postprocessing range")
    steps = list(range(start, stop + 1, increment))
    for step in steps:
        if not all(valid_restart(directory / f"restart.{step}.{rank}", step)
                   for rank in range(1, ranks + 1)):
            raise ValueError(f"Missing/incomplete checkpoint at timestep {step}")
    out = case / f"post_{start}_{stop}_{increment}"
    out.mkdir(exist_ok=True)
    command = [os.environ.get("SVPOST", "svpost"), "-indir", str(directory), "-outdir", str(out),
               "-start", str(start), "-stop", str(stop), "-incr", str(increment), "-vtu", "all_results"]
    with (out / "svpost.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    for step in steps:
        p = out / f"all_results_{step:05d}.vtu"
        if not p.is_file() or not p.stat().st_size:
            raise RuntimeError(f"svpost did not produce {p}; inspect svpost.log")
        with p.open("rb") as stream:
            if b"UnstructuredGrid" not in stream.read(4096):
                raise RuntimeError(f"Invalid VTU output: {p}")
    return {"output": str(out), "frames": len(steps), "steps": steps}


def benchmark(archive, root, ranks, hourly, warmup=50, until=401):
    if not 0 < warmup < until or hourly < 0:
        raise ValueError("Require 0 < warmup < until, and a nonnegative hourly price")
    namespace = Path(root) / f"benchmark_{ranks}"
    case = Path(prepare_archive(archive, namespace)["case"])
    report_path = namespace / "benchmark.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if (report["warmup"]["end_step"] != warmup or report["measured"]["end_step"] != until
                or report["hourly_compute_usd"] != hourly):
            raise ValueError("Benchmark parameters changed: use a new benchmark root")
        return report
    warm_path, measured_path = namespace / "warmup.json", namespace / "measured.json"
    partition = case / f"{ranks}-procs_case"
    latest = latest_checkpoint(partition, ranks) if partition.exists() else 0
    if latest is not None and latest > warmup and not measured_path.exists():
        raise ValueError("Interrupted measured phase: use a fresh benchmark root for trustworthy timing")
    if not warm_path.exists():
        warm = run_case(case, ranks, warmup)
        if warm["start_step"] != 0 or warm["end_step"] != warmup:
            raise ValueError("Warmup was interrupted: use a fresh benchmark root for trustworthy timing")
        atomic_json(warm_path, warm)
    if not measured_path.exists():
        measured = run_case(case, ranks, until)
        if measured["start_step"] != warmup or measured["end_step"] != until:
            raise ValueError("Measured phase did not cover the intended interval")
        atomic_json(measured_path, measured)
    warm, measured = json.loads(warm_path.read_text()), json.loads(measured_path.read_text())
    exported = postprocess(case, ranks, until, until, 1)
    seconds = measured["elapsed_s"] / measured["steps_run"]
    projected = seconds * TOTAL_STEPS
    report = {"status": "completed", **case_info(case), "ranks": ranks,
              "warmup": warm, "measured": measured, "postprocess": exported,
              "measured_seconds_per_step": seconds, "projected_full_solve_hours": projected / 3600,
              "hourly_compute_usd": hourly, "projected_full_compute_usd": projected / 3600 * hourly,
              "benchmark_compute_usd": (warm["elapsed_s"] + measured["elapsed_s"]) / 3600 * hourly,
              "limitations": ["Timing covers early-cycle flow only; later phases may differ",
                              "Measured phase includes MPI startup and checkpoint output",
                              "Projection excludes setup, upload, postprocessing, storage and tax",
                              "Not a mesh, residual, time-step or cycle-convergence validation"]}
    atomic_json(report_path, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("archive", type=Path)
    prepare.add_argument("destination", type=Path)
    prepare.add_argument("--sha256")
    for action in ["run", "postprocess"]:
        cmd = sub.add_parser(action)
        cmd.add_argument("case", type=Path)
        cmd.add_argument("--ranks", type=int, default=6)
        if action == "run":
            cmd.add_argument("--until", type=int, default=TOTAL_STEPS)
            cmd.add_argument("--restart-every", type=int, default=RESTART_EVERY)
        else:
            cmd.add_argument("--start", type=int, default=4010)
            cmd.add_argument("--stop", type=int, default=8020)
            cmd.add_argument("--increment", type=int, default=RESTART_EVERY)
    cmd = sub.add_parser("benchmark")
    cmd.add_argument("archive", type=Path)
    cmd.add_argument("root", type=Path)
    cmd.add_argument("--ranks", type=int, default=32)
    cmd.add_argument("--hourly", type=float, required=True)
    cmd.add_argument("--warmup", type=int, default=50)
    cmd.add_argument("--until", type=int, default=401)
    a = p.parse_args()
    if a.action == "prepare":
        result = prepare_archive(a.archive, a.destination, a.sha256)
    elif a.action == "run":
        result = run_case(a.case, a.ranks, a.until, a.restart_every)
    elif a.action == "postprocess":
        result = postprocess(a.case, a.ranks, a.start, a.stop, a.increment)
    else:
        result = benchmark(a.archive, a.root, a.ranks, a.hourly, a.warmup, a.until)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
