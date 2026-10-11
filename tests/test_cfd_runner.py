"""Workflow regression tests use a fake MPI process, never fabricate CFD measurements."""
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/cfd"))
from runpod_cfd import Remote

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/cfd"
spec = importlib.util.spec_from_file_location("cfd_case", SCRIPTS / "cfd_case.py")
cfd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfd)


def restart(step, compressed=False):
    data = f"# SimVascular Input File Version 2.0\nsolution : < 81 > 2 5 {step}\n".encode() + bytes(80) + b"\n"
    if step:
        data += f"time derivative of solution : < 81 > 2 5 {step}\n".encode() + bytes(80) + b"\n"
    return gzip.compress(data) if compressed else data


def make_case(path):
    path.mkdir(parents=True)
    (path / "solver.inp").write_text("Number of Timesteps: 8125\nTime Step Size: 0.0002375\n"
        "Number of Timesteps between Restarts: 1\nNumber of Resistance Surfaces: 82\nDensity: 1.06\n")
    (path / "geombc.dat.1").write_bytes(b"geometry")
    (path / "restart.0.1").write_bytes(restart(0, True))
    (path / "numstart.dat").write_text("0\n")
    (path / "bct.dat").write_text("waveform")
    (path / "mesh-complete").mkdir()
    (path / "mesh-complete/mesh-complete.mesh.vtu").write_text('<Piece NumberOfPoints="2" NumberOfCells="1">')
    return path


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.case = make_case(self.root / "case")

    def tearDown(self):
        self.tmp.cleanup()

    def partition(self, ranks=2, step=401):
        p = self.case / f"{ranks}-procs_case"
        p.mkdir()
        for rank in range(1, ranks + 1):
            (p / f"geombc.dat.{rank}").write_bytes(b"geometry")
            (p / f"restart.{step}.{rank}").write_bytes(restart(step, True))
        (p / "numstart.dat").write_text("0\n")
        return p

    def test_resumes_with_remaining_steps_and_partition_pointer(self):
        p = self.partition(step=401)
        start, _ = cfd.configure(self.case, 2, 8125, 401)
        self.assertEqual(start, 401)
        self.assertEqual((p / "numstart.dat").read_text(), "401\n")
        self.assertIn("Number of Timesteps: 7724\n", (self.case / "solver.inp").read_text())
        self.assertIn("Density: 1.06", (self.case / "solver.inp").read_text())
        self.assertIn("Number of Timesteps: 8125", (self.case / "solver.inp.original").read_text())

    def test_partial_newer_checkpoint_falls_back(self):
        p = self.partition()
        (p / "restart.802.1").write_bytes(restart(802))
        (p / "restart.802.2").write_bytes(restart(802)[:-10])
        self.assertEqual(cfd.latest_checkpoint(p, 2), 401)

    def test_bad_gzip_and_wrong_step_are_rejected(self):
        p = self.root / "restart"
        p.write_bytes(restart(401, True)[:-4])
        self.assertFalse(cfd.valid_restart(p, 401))
        p.write_bytes(restart(401))
        self.assertFalse(cfd.valid_restart(p, 402))
        broken = bytearray(restart(401, True))
        broken[10:13] = b"\xff\xff\xff"
        p.write_bytes(broken)
        self.assertFalse(cfd.valid_restart(p, 401))

    def test_final_checkpoint_wins_over_stale_numstart(self):
        p = self.partition(step=8125)
        start, _ = cfd.configure(self.case, 2, 8125, 401)
        self.assertEqual(start, 8125)
        self.assertEqual((p / "numstart.dat").read_text(), "8125\n")

    def test_rank_change_refused_without_overwriting(self):
        p = self.partition()
        before = (p / "numstart.dat").read_bytes()
        with self.assertRaisesRegex(ValueError, "Existing partition"):
            cfd.configure(self.case, 1, 8125, 401)
        self.assertEqual((p / "numstart.dat").read_bytes(), before)

    def test_missing_rank_geometry_and_concurrent_run_refused(self):
        p = self.partition()
        (p / "geombc.dat.2").unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete MPI"):
            cfd.configure(self.case, 2, 8125, 401)
        with cfd.case_lock(self.case):
            with self.assertRaisesRegex(RuntimeError, "already running"):
                with cfd.case_lock(self.case):
                    pass

    def executable(self, name, source):
        p = self.root / name
        p.write_text(f"#!{sys.executable}\n" + source)
        p.chmod(0o700)
        return str(p)

    def test_launcher_propagates_mpi_failure(self):
        mpi = self.executable("fail", "import sys\nsys.exit(7)\n")
        with mock.patch.dict(os.environ, {"MPIEXEC": mpi, "SVSOLVER": sys.executable}):
            r = subprocess.run(["bash", str(SCRIPTS / "run_0082_cfd.sh"), str(self.case), "1"], capture_output=True)
        self.assertNotEqual(r.returncode, 0)
        result = json.loads((self.case / "last_run.json").read_text())
        self.assertEqual(result["exit_code"], 7)
        self.assertEqual(result["status"], "failed")

    def test_zero_exit_without_checkpoint_is_not_success(self):
        mpi = self.executable("false-success", "pass\n")
        with mock.patch.dict(os.environ, {"MPIEXEC": mpi, "SVSOLVER": sys.executable}):
            with self.assertRaisesRegex(RuntimeError, "did not reach"):
                cfd.run_case(self.case, 1, 20)

    def archive(self):
        p = self.root / "case.zip"
        with zipfile.ZipFile(p, "w") as z:
            for f in self.case.rglob("*"):
                if f.is_file():
                    z.write(f, "0084_0001/" + str(f.relative_to(self.case)))
        return p

    def test_extraction_is_idempotent_and_does_not_reset_case(self):
        z = self.archive()
        first = cfd.prepare_archive(z, self.root / "prepared", cfd.sha256(z))
        p = Path(first["case"]) / "numstart.dat"
        p.write_text("401\n")
        cfd.prepare_archive(z, self.root / "prepared")
        self.assertEqual(p.read_text(), "401\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            cfd.prepare_archive(z, self.root / "elsewhere", "0" * 64)

    def test_unsafe_zip_rejected(self):
        z = self.root / "bad.zip"
        with zipfile.ZipFile(z, "w") as f:
            f.writestr("0084_0001/../../escaped", "bad")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            cfd.prepare_archive(z, self.root / "prepared")
        self.assertFalse((self.root / "escaped").exists())

    def test_benchmark_runs_warmup_resume_and_export(self):
        mpi = self.executable("mpi", "import os,sys\nos.environ['FAKE_RANKS']=sys.argv[2]\nos.execv(sys.argv[3],sys.argv[3:])\n")
        solver = self.executable("solver", '''import os, pathlib, re
p=pathlib.Path.cwd(); ranks=int(os.environ['FAKE_RANKS']); d=p/f'{ranks}-procs_case'
d.mkdir(exist_ok=True)
start=int((d/'numstart.dat').read_text()) if (d/'numstart.dat').exists() else 0
steps=int(re.search(r'Number of Timesteps: (\\d+)',(p/'solver.inp').read_text())[1])
end=start+steps
for r in range(1,ranks+1):
 (d/f'geombc.dat.{r}').write_bytes(b'geometry')
 data=f'solution : < 81 > 2 5 {end}\\n'.encode()+bytes(80)+b'\\n'
 data+=f'time derivative of solution : < 81 > 2 5 {end}\\n'.encode()+bytes(80)+b'\\n'
 (d/f'restart.{end}.{r}').write_bytes(data)
(d/'numstart.dat').write_text(str(end))
with (p/'calls.txt').open('a') as f:f.write(f'{start},{steps}\\n')
''')
        post = self.executable("post", '''import pathlib,sys
a=sys.argv; out=pathlib.Path(a[a.index('-outdir')+1]); step=int(a[a.index('-start')+1])
(out/f'all_results_{step:05d}.vtu').write_text('<VTKFile type="UnstructuredGrid">')
''')
        with mock.patch.dict(os.environ, {"MPIEXEC": mpi, "SVSOLVER": solver, "SVPOST": post}):
            report = cfd.benchmark(self.archive(), self.root / "runs", 1, 0.96)
            again = cfd.benchmark(self.root / "case.zip", self.root / "runs", 1, 0.96)
        self.assertEqual(report, again)
        self.assertEqual(report["measured"]["steps_run"], 351)
        self.assertEqual(report["postprocess"]["frames"], 1)
        self.assertEqual(Path(report["case"]).joinpath("calls.txt").read_text(), "0,50\n50,351\n")

    def test_postprocess_rejects_missing_checkpoint_before_invoking_solver(self):
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            cfd.postprocess(self.case, 2)

    def test_remote_command_failure_and_status_tail(self):
        # Execute the generated SSH stdin locally to validate shell/error behavior,
        # without contacting a server or claiming SSH authentication was tested.
        run = subprocess.run
        def local_ssh(args, **kwargs):
            self.assertEqual(args[0], "ssh")
            return run(["bash", "-s"], **kwargs)
        remote = Remote("example.invalid")
        job = self.root / "jobs/test"
        job.mkdir(parents=True)
        (job / "exit_code").write_text("7\n")
        (job / "job.log").write_text("x" * 10000 + "error tail")
        with mock.patch("runpod_cfd.subprocess.run", side_effect=local_ssh):
            with self.assertRaisesRegex(RuntimeError, "SSH command failed"):
                remote.run("false\necho should-not-be-reached")
            status = remote.status(str(self.root), "test")
        self.assertEqual(status["exit_code"], 7)
        self.assertTrue(status["tail"].endswith("error tail"))
        self.assertLessEqual(len(status["tail"]), 3000)


if __name__ == "__main__":
    unittest.main()
