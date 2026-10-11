"""Local notebook controller: Runpod provisioning, direct SSH, transfer and detached jobs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request

from cfd_case import atomic_json, sha256


class RunpodAPI:
    """Credentials stay in memory; never saved in notebook outputs or state files."""
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("Set RUNPOD_API_KEY or enter it with getpass")
        self.api_key = api_key
        self.base = "https://api.runpod.io/v2"

    def request(self, method, path, body=None):
        req = urllib.request.Request(self.base + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
            method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                raw = response.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            # Deliberately avoid printing request headers or API keys.
            raise RuntimeError(f"Runpod returned HTTP {exc.code}: {exc.read().decode()[:1500]}") from None

    def cpu_quote(self, vcpus=32):
        result = self.request("GET", "/catalog/cpus/cpu3c")
        result = result.get("cpu", result)
        return {"vcpus": vcpus, "ram_gb": result["ramGbPerVcpu"] * vcpus,
                "compute_usd_per_hour": result["price"]["securePerVcpu"] * vcpus}

    def create_cpu(self, public_key, vcpus=32, lifetime_seconds=21600, network_volume_id=None):
        if vcpus not in (2, 4, 8, 16, 32) or lifetime_seconds < 600:
            raise ValueError("Use 2/4/8/16/32 vCPUs and a lifetime of at least 600 seconds")
        if not re.fullmatch(r"(?:ssh-ed25519|ssh-rsa) [A-Za-z0-9+/=]+(?: [^\r\n]*)?", public_key.strip()):
            raise ValueError("Supply a single SSH public key line")
        # Install authentication in THIS pod only; never replace account SSH keys.
        boot = ("export DEBIAN_FRONTEND=noninteractive; "
                "apt-get update && apt-get install -y openssh-server python3 ca-certificates && "
                "mkdir -p /run/sshd /root/.ssh && chmod 700 /root/.ssh && "
                "printf '%s\\n' \"$CFD_SSH_PUBLIC_KEY\" > /root/.ssh/authorized_keys && "
                "chmod 600 /root/.ssh/authorized_keys && exec timeout --signal=TERM "
                + str(lifetime_seconds) + " /usr/sbin/sshd -D -e")
        body = {"name": "dynamag-0082-cfd", "cloud": "SECURE", "image": "ubuntu:22.04",
                "cpu": {"id": "cpu3c", "vcpuCount": vcpus}, "disk": 50,
                "ports": ["22/tcp"], "startSsh": False, "startJupyter": False,
                "env": {"CFD_SSH_PUBLIC_KEY": public_key.strip()},
                "entrypoint": ["/bin/bash", "-lc"], "cmd": [boot]}
        if network_volume_id:
            body["mounts"] = {"network": [{"path": "/workspace", "volumeId": network_volume_id}]}
        result = self.request("POST", "/pods", body)
        return result.get("pod", result)

    def pod(self, pod_id):
        result = self.request("GET", "/pods/" + pod_id)
        return result.get("pod", result)

    def terminate(self, pod_id):
        return self.request("DELETE", "/pods/" + pod_id)


class Remote:
    """Direct SSH/SCP also works with a Google Cloud or university Linux VM."""
    def __init__(self, host, port=22, user="root", key=None, known_hosts=None):
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", host) or not re.fullmatch(r"[A-Za-z0-9_.-]+", user):
            raise ValueError("Invalid SSH host or username")
        if not 0 < int(port) < 65536:
            raise ValueError("Invalid SSH port")
        self.host, self.port, self.user = host, int(port), user
        self.key = str(Path(key).expanduser().resolve()) if key else None
        self.options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                        "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30"]
        if known_hosts:
            path = Path(known_hosts).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.options += ["-o", "UserKnownHostsFile=" + str(path)]
        if self.key:
            if not Path(self.key).is_file():
                raise FileNotFoundError(self.key)
            self.options += ["-i", self.key, "-o", "IdentitiesOnly=yes"]

    @classmethod
    def from_pod(cls, pod, key, known_hosts=None):
        endpoint = (pod.get("ssh") or {}).get("direct")
        if not endpoint:
            raise RuntimeError("Pod has no direct SSH endpoint yet. Wait for provisioning, or enter host/port from its Connect tab")
        return cls(endpoint["host"], endpoint["port"], endpoint.get("user", "root"), key, known_hosts)

    def run(self, script, timeout=60):
        result = subprocess.run(["ssh", *self.options, "-p", str(self.port),
                                 f"{self.user}@{self.host}", "bash -s"],
                                input="set -euo pipefail\n" + script, text=True,
                                capture_output=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f"SSH command failed ({result.returncode}):\n{result.stdout[-1500:]}\n{result.stderr[-1500:]}")
        return result.stdout.strip()

    def upload(self, source, destination):
        target = f"{self.user}@{self.host}:{shlex.quote(str(destination))}"
        subprocess.run(["scp", "-O", *self.options, "-P", str(self.port), str(source), target], check=True)

    def download(self, source, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        target = f"{self.user}@{self.host}:{shlex.quote(str(source))}"
        subprocess.run(["scp", "-O", *self.options, "-P", str(self.port), target, str(destination)], check=True)

    def upload_workflow(self, scripts, archive, root):
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", root) or ".." in Path(root).parts:
            raise ValueError("Remote root must be an absolute path without spaces or traversal")
        scripts, archive = Path(scripts), Path(archive)
        self.run(f"mkdir -p {shlex.quote(root + '/scripts')} {shlex.quote(root + '/inputs')}")
        for name in ["cfd_case.py", "setup_svsolver.sh", "run_0082_cfd.sh", "postprocess_0082_cycle2.sh"]:
            self.upload(scripts / name, root + "/scripts/" + name)
        self.upload(archive, root + "/inputs/" + archive.name)
        digest = sha256(archive)
        verified = self.run("python3 - " + shlex.quote(root + "/inputs/" + archive.name) + " <<'PY'\n"
            "import hashlib, pathlib, sys\np=pathlib.Path(sys.argv[1]); h=hashlib.sha256()\n"
            "with p.open('rb') as s:\n for b in iter(lambda:s.read(1048576),b''): h.update(b)\n"
            "print(h.hexdigest())\nPY")
        if verified != digest:
            raise RuntimeError("Uploaded ZIP checksum mismatch")
        return digest

    def launch(self, root, name, command, timeout_seconds=7200):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or timeout_seconds < 60:
            raise ValueError("Invalid job name or timeout")
        job = root + "/jobs/" + name
        # A lock is held for the lifetime of the job, including setup and benchmark phases.
        payload = ("set +e; flock -n 9 || exit 75; "
                   + f"timeout --signal=TERM --kill-after=30s {int(timeout_seconds)} bash -lc "
                   + shlex.quote(command) + "; code=$?; printf '%s\\n' \"$code\" > "
                   + shlex.quote(job + "/exit_code") + "; exit \"$code\"")
        script = (f"mkdir -p {shlex.quote(job)}\n"
                  f"if [[ -f {shlex.quote(job + '/exit_code')} ]]; then echo completed; exit 0; fi\n"
                  f"flock -n {shlex.quote(job + '/lock')} -c true || {{ echo running; exit 0; }}\n"
                  f"setsid bash -c {shlex.quote(payload)} 9>{shlex.quote(job + '/lock')} "
                  f">{shlex.quote(job + '/job.log')} 2>&1 < /dev/null &\n"
                  f"echo $! > {shlex.quote(job + '/pid')}\necho started")
        return self.run(script)

    def status(self, root, name):
        job = root + "/jobs/" + name
        script = ("python3 - " + shlex.quote(job) + " <<'PY'\n"
                  "import json, pathlib\np=pathlib.Path(__import__('sys').argv[1]); done=p/'exit_code'\n"
                  "log=p/'job.log'; tail=''\n"
                  "if log.exists():\n with log.open('rb') as f:\n  f.seek(max(0,log.stat().st_size-3000)); tail=f.read().decode(errors='replace')\n"
                  "print(json.dumps({'exit_code':int(done.read_text()) if done.exists() else None,'tail':tail}))\nPY")
        return json.loads(self.run(script))

    def wait(self, root, name, seconds=7200):
        deadline = time.monotonic() + seconds
        last_tail = None
        while time.monotonic() < deadline:
            status = self.status(root, name)
            if status["tail"] != last_tail:
                print(status["tail"][-800:])
                last_tail = status["tail"]
            if status["exit_code"] is not None:
                if status["exit_code"]:
                    raise RuntimeError(f"Remote job exited {status['exit_code']}: {status['tail']}")
                return status
            time.sleep(10)
        raise TimeoutError("Local wait ended; the remote job remains bounded by its timeout. Inspect status or reconnect")

    def backup(self, root, local_directory):
        # Includes inputs, restart partitions and reports; excludes rebuildable toolchain files.
        self.run(f"tar -czf {shlex.quote(root + '/checkpoints.tar.gz')} -C {shlex.quote(root)} runs inputs jobs scripts", timeout=600)
        local = Path(local_directory) / "checkpoints.tar.gz"
        self.download(root + "/checkpoints.tar.gz", local)
        digest = self.run("sha256sum " + shlex.quote(root + "/checkpoints.tar.gz")).split()[0]
        if sha256(local) != digest:
            raise RuntimeError("Downloaded checkpoint backup checksum mismatch")
        atomic_json(Path(local_directory) / "backup.json", {"sha256": digest, "file": str(local)})
        return local

    def restore(self, archive, root):
        # Backups are restored only into a new namespace, never over existing checkpoints.
        self.run(f"test ! -e {shlex.quote(root + '/runs')}\nmkdir -p {shlex.quote(root)}")
        self.upload(archive, root + "/restore.tar.gz")
        script = "python3 - " + shlex.quote(root) + " <<'PY'\n" + """import pathlib, sys, tarfile
root=pathlib.Path(sys.argv[1])
with tarfile.open(root/'restore.tar.gz') as t:
 for m in t.getmembers():
  p=pathlib.PurePosixPath(m.name)
  if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] not in {'runs','inputs','jobs','scripts'} or not (m.isfile() or m.isdir()):
   raise ValueError('Unsafe checkpoint archive member: '+m.name)
 t.extractall(root)
print('Checkpoints restored; rebuild the toolchain before resuming')
PY"""
        return self.run(script, timeout=600)
