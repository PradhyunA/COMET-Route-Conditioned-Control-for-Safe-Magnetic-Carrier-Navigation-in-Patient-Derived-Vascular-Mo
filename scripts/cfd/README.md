# Pulmonary CFD — 0082 anatomy, 0084_0001 simulation

`run_0082_cfd.ipynb` is a Python notebook that runs locally and controls a Linux
x86_64 Ubuntu 20.04/22.04 machine over direct SSH. It supports a Runpod CPU pod,
a Google Cloud VM, or another Linux workstation. All runtime helpers use the
Python standard library; local SSH and SCP are required.

The supplied `0084_0001.zip` contains 338,438 mesh nodes, 1,676,978 volume elements,
82 resistance outlets, the inlet waveform and an initial restart. No remeshing
is performed. The native physical parameters and boundary conditions are retained.

## Notebook workflow

Open the notebook from the repository root or `scripts/cfd` with a Python kernel.
Edit its configuration cell:

- `ARCHIVE`: local ZIP path (defaults to `~/Downloads/0084_0001.zip`).
- `PROVIDER="ssh"`: set host, direct SSH port, username, key and remote workspace.
  On Google Cloud, use `/home/YOUR_LINUX_USERNAME/dynamag` for `REMOTE_ROOT`.
- `PROVIDER="runpod"`: set `POD_ID` to reuse a pod; API credentials come from
  `RUNPOD_API_KEY` or a private `getpass` prompt. Optional new provisioning requires
  `CREATE_RUNPOD=True` and a funded account. A live CPU quote is printed first.
- `VCPUS`: MPI rank count; use the same count when resuming a case.
- `RUN_BENCHMARK=True`: 50 warmup steps, followed by 351 measured steps through
  step 401, then a real VTU export. The benchmark has its own case directory.
- `RUN_FULL_SOLVE=False`: explicitly enable only after inspecting the benchmark.
- `POSTPROCESS_FULL=False`: enable after the full solve to export cycle 2.

Upload verifies the ZIP checksum. Software setup builds VTK 8.2.0 and svSolver
commit `47de08e103078587264ad56089a387435a46d3c0` with GCC 9 and OpenMPI in the same
Linux environment. Setup needs root/passwordless sudo and network access. The
installer verifies binary dependencies and a two-rank MPI launch. Source setup
is provided but has not yet been validated on a live VM for this repair.

Jobs run detached with logs, locks, exit statuses and bounded timeouts. Closing
Jupyter does not stop the remote computation. Reconnect and inspect `remote.status`
or call `remote.wait`; do not start a second solver in the same case. Failed jobs
keep their exit code: fix the cause and choose a new `JOB_TAG` to retry.

`outputs/cfd_runpod/` holds local connection state, setup metadata, benchmark
reports, SSH host fingerprints and downloaded checkpoint archives. API keys and
private keys are never saved there. Save a verified backup before deleting or
stopping an ephemeral-disk pod. Use a network volume for long-lived Runpod work.
A restored case retains its partition and rank count; rebuild the toolchain on
the new host, choose a fresh job tag and resume.

The default new-pod container lifetime is six hours, suitable for setup and a
benchmark. Increase it for a full solve. It ends the container, not storage
billing: termination remains necessary. The notebook's cleanup is opt-in and
requires a verified local backup; it only terminates the pod it created.

## Run without Jupyter

After transferring the scripts and ZIP to the Linux machine:

```bash
sudo bash scripts/cfd/setup_svsolver.sh /home/YOUR_LINUX_USERNAME/dynamag/toolchain
source /home/YOUR_LINUX_USERNAME/dynamag/toolchain/env.sh
python3 scripts/cfd/cfd_case.py prepare /path/to/0084_0001.zip /path/to/runs/production
python3 scripts/cfd/cfd_case.py benchmark /path/to/0084_0001.zip /path/to/runs/sample \
    --ranks 32 --hourly 0.96
bash scripts/cfd/run_0082_cfd.sh /path/to/runs/production/0084_0001 32
bash scripts/cfd/postprocess_0082_cycle2.sh /path/to/runs/production/0084_0001 32
```

Replace rank count and hourly resource price with the selected machine's actual
values. Trial credits cover resource charges; they do not alter the underlying
price used in the report.

## Resume and verification

The run defaults to a final timestep of 8,125 and a time step of 0.0002375 s.
Legacy svSolver's `Number of Timesteps` is an **additional** step count. The runner
subtracts the completed steps rather than solving another 8,125 after a resume.

Checkpoints live in `N-procs_case/`. The runner scans for the newest step with
complete solution and derivative payloads on every MPI rank (including gzip CRC
checks). It skips incomplete newer checkpoints and updates that partition's
`numstart.dat`; it never blindly resets an existing case to zero. Changing rank
count is refused. Concurrent runs in the same case are locked out. Solver errors
and a zero exit status without the requested checkpoint both fail the job.

Benchmark timing excludes the first 50 steps and their initial partitioning,
but includes MPI startup and checkpoint output for the measured phase. It samples
only early-cycle flow, not all waveform phases. An interrupted measured phase
requires a fresh benchmark root; partially timed work is not reported as a valid
measurement. Completed reports are reusable only with the same parameters.

The second-cycle checkpoints are 4,010 through 8,020 in increments of 401. The
postprocessor reads the MPI partition directory, checks all input checkpoints,
and verifies 11 expected files in `post_4010_8020_401/`:
`all_results_04010.vtu` through `all_results_08020.vtu`.

Neither a successful solve nor two cardiac cycles establish mesh independence,
time-step convergence, residual convergence or cycle-to-cycle periodicity.

## Free compute and a 24-hour target

As checked on 2026-10-10:

- Eligible new Google Cloud users receive $300 of trial credit for 90 days.
  Trial Compute Engine projects are limited to 8 CPUs and cannot request quota
  increases while remaining non-billable. Check the actual project quota before
  choosing a VM. A larger VM may require upgrading to paid billing and obtaining
  sufficient quota. Remaining trial credit still applies after upgrading, but
  charges beyond the credit become billable.
- An x86_64 `n2d-standard-32` VM has 32 vCPUs and 128 GiB RAM. The published Iowa
  on-demand compute rate is $1.351872/hour, or $32.44 for 24 hours, before disk,
  network and tax. Eligible credits can cover this; the VM itself is not part of
  the always-free tier. Try the 8-vCPU shape first if staying entirely within the
  unupgraded trial; its speed must be measured.
- Free Colab offers variable resources and at most 12-hour sessions; it cannot
  guarantee a continuous 24-hour run. Its GPU does not accelerate this legacy
  CPU/MPI solver. Do not use this SSH controller to turn a free Colab runtime into
  a remote compute server.
- The always-free Compute Engine VM is an `e2-micro`; it is not a suitable planned
  machine for this 1.68-million-element CFD workload and deadline.
- An existing university HPC allocation or sufficiently capable Linux workstation
  can avoid new cloud charges; allocation and wall-clock availability vary.

For 8,125 steps, an average of 10 seconds/step is 22.57 hours of solving. To fit
setup, transfer and export in 24 hours, the benchmark should be comfortably below
that rate; an early-cycle estimate still cannot guarantee later-cycle performance.
No live CFD runtime was measured during this repair: Runpod refused provisioning
because the account balance was too low, and the user chose to defer paid compute.

Sources:
- https://docs.cloud.google.com/free/docs/free-cloud-features
- https://docs.cloud.google.com/managed-spark/quotas
- https://cloud.google.com/products/compute/pricing/general-purpose
- https://research.google.com/colaboratory/faq.html
- https://github.com/SimVascular/svSolver

## Verification

```bash
python3 -m unittest discover -s tests -p test_cfd_runner.py -v
```

Regression tests use a fake MPI process to exercise failure propagation, actual
subprocess execution, restart completeness/fallback, preserved settings, locks,
archive safety, the warmup/resume sequence and export handling. They are workflow
tests, not CFD performance measurements. The supplied ZIP was also extracted and
its actual mesh header and initial restart payload checked successfully.

Do not commit large restart files, VTUs, credentials or CFD result directories.
