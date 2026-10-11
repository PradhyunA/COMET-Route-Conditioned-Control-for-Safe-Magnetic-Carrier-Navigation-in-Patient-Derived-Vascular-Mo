# CFD validation workflow

The Run 4 abdominal and pulmonary exports match the project's corrected SI
anatomies. No coordinate or velocity rescaling was applied. The abdominal mesh
spans 165.05 × 124.44 × 405.76 mm; pulmonary spans 147.83 × 118.20 × 186.30 mm.
Both have finite flow arrays, inward inlet vectors, positive computed cell
volumes, and excellent final mass balance. Source audits are in
`outputs/cfd_export_audit_run4_20261005/audit.json`.

## Environment

Use the separate `.venv-cfd` environment created for this workflow. The existing
Python 3.14 / NumPy 1.26 environment produced inconsistent numerical results
when converting large batches of faces and comparing candidate scales. The
conversion was independently checked with the bundled Python 3.12 / NumPy 2.3
runtime, then imported and tested using Python 3.13 / NumPy 2.5. The original
environment was preserved.

On another checkout, create the CFD environment from the repository root:

```bash
uv venv --python python3.13 .venv-cfd
uv pip install --python .venv-cfd/bin/python -r requirements-cfd.txt
```

## Import exports

Both fields are already imported locally. To regenerate them:

```bash
.venv-cfd/bin/python scripts/import_simscale_flow.py \
  --zip "$HOME/Downloads/AbdominalAortaCFD-Incompressible-Run_4.zip" \
  --vessel-file data/real_vessel_segments_cm_corrected.pkl \
  --output data/cfd/abdominal_aorta_steady.npz

.venv-cfd/bin/python scripts/import_simscale_flow.py \
  --zip "$HOME/Downloads/PulmonaryArteryCFD-Incompressible-Run_4.zip" \
  --vessel-file data/vessels/pulmonary_0082_H_PULM_H/real_vessel_segments_cm_corrected.pkl \
  --output data/cfd/pulmonary_steady.npz
```

The importer reads ASCII OpenFOAM data directly from the ZIP, preserving cell
ordering. It computes area-weighted face centroids and volume-weighted cell
centroids from the mesh connectivity, associates each centre with its exported
`U`, and includes the exported boundary velocities. It checks SI geometry
alignment, non-positive cell volumes, and final flux balance. The `phi` in these
exports is mass flux (kg/s), as declared by its dimensions. It is not volumetric
flux. The generated JSON records source and vessel hashes, units, mesh sizes,
flow statistics, and limitations. Generated NPZ arrays are Git-ignored and must
be regenerated on another machine; OpenFOAM and VTK are not required.

The flow sampler uses eight nearest samples with inverse-distance-squared
weights. Queries beyond twice the nearest sample's bounding radius fail with
an explicit error. This bounds local extrapolation but does not implement exact
polyhedral fluid-domain membership. There is no fallback to analytical flow.
The field is steady and adds no artificial pulsation.

## Run abdominal COMET at 1×: five cases

```bash
./run_comet_cfd_abdominal_1x.sh
```

This runs the five frozen abdominal routes using the supplied CFD velocities
at multiplier 1.0. It uses the benchmark settings described below and writes
results under `outputs/comet_cfd_abdominal_1x/<run_id>`. The script verifies
five distinct cases, CFD provenance, and a velocity multiplier of exactly 1.0.
For a short execution check, use
`STEPS=2 WORKERS=2 ./run_comet_cfd_abdominal_1x.sh`.

## Run both anatomies and four controllers at 1×: 40 cases

```bash
./run_cfd_abdominal_pulmonary_1x.sh
```

This runs five frozen abdominal and five frozen pulmonary routes with COMET,
tuned constrained CEM-MPC (`constrained_cem_mpc_tuned_recovery`), tuned ACBC,
and tuned MPC. It preserves the existing controller settings and the same
500 µm carrier, dt, 2,000-step budget, hardware, and target tolerance used below.
All actual particle motion and predictive flow sampling use each anatomy's
imported CFD field at multiplier 1.0. CEM's vectorized rollouts sample CFD
directly, preserving its bounded support check and steady vector directions.
Stagnation early stopping is disabled.

Results are written to `outputs/cfd_1x_four_controllers/<run_id>/abdominal` and
`pulmonary`, with a combined 40-row CSV and summary JSON at the parent level.
The script verifies case counts, frozen coordinates, finite trajectories, and
CFD provenance. Use `STEPS=2 WORKERS=2` for an execution smoke check. Supported
overrides are `PYTHON_BIN`, `WORKERS`, `STEPS`, `SEED`, `RUN_ID`, and `OUTPUT_ROOT`.

## Run abdominal COMET: 15 sensitivity cases

```bash
./run_comet_cfd_abdominal_15trials.sh
```

This runs five frozen routes from
`benchmark_manifests/icem_45case_500um/cm_corrected.csv` at velocity multipliers
1, 3, and 5. It preserves the 500 µm carrier, 0.02 s integration step, 2,000-step
budget, 0.5 mm target tolerance, 0.7 mm centre-to-wall clearance criterion,
moving adaptive gantry, seed 20260611, and COMET settings. Stagnation early
stopping is disabled. Normal termination on success or physical failure remains
active. Four trial processes are used by default; each numerical worker is
single-threaded. Source field arrays and spatial indices are reused within a
worker process.

Overrides:

```bash
WORKERS=6 ./run_comet_cfd_abdominal_15trials.sh
STEPS=2 WORKERS=2 ./run_comet_cfd_abdominal_15trials.sh  # plumbing smoke test
```

Other supported environment variables are `PYTHON_BIN`, `CFD_FILE`, `SEED`,
`RUN_ID`, and `OUTPUT_ROOT`. Each run gets a fresh timestamped directory under
`outputs/comet_cfd_abdominal`. Reusing an existing run ID is rejected to preserve
results. The script checks that all 15 distinct cases were recorded.

The publication runner also accepts `--cfd-file` directly. Its CFD mode supports
COMET, CEM-MPC, ACBC, and MPC variants. CEM's batched flow model samples the CFD
field when present, while retaining analytical behavior for ordinary runs.
`run_publication_benchmark_cfd.py` now calls
the actual publication runner and requires an explicit CFD file.

CFD midpoint vectors replace analytical segment flow in graph costs. Adverse
flow can exceed estimated COMET authority on the fixed routes. In CFD mode,
the route's authority filter is disabled so those cases are attempted and
failures remain in the benchmark. Clearance and connectivity requirements
remain active; controller authority costs and responses remain active. Negative
route reserves are recorded. Analytical benchmark behavior retains its original
authority filter. This avoids excluding difficult CFD cases before testing.

Each run config records the CFD source metadata and velocity-multiplier
semantics. Each trial JSON/NPZ records CFD source provenance and query statistics.

## Interpretation

- Only the 1× condition is the supplied CFD solution. Multiplying its velocities
  by 3 or 5 yields sensitivity fields, not independently solved CFD at those
  inlet conditions. Independent 3×/5× solves are needed for that stronger claim.
- Navigation, particle wall handling, and clearance metrics retain the existing
  centreline/tubular geometry. This validates response to imported flow within
  the project's model; it is not a complete CFD-surface particle solver.
- The exports are steady kOmegaSST solutions. Iteration 1000 is a steady solver
  iteration, not a physical cardiac time. No residual histories or mesh
  independence evidence were supplied, so final momentum convergence and mesh
  independence are not certified.
- The previously supplied VTU matches the 0050 cerebral anatomy and has unknown
  declared units and snapshot time. It cannot substitute for 40Left85Right in
  the current 45-case protocol. The matching aortic-root export remains needed.

## Verification performed

- Unit and regression checks cover geometry centroids/volumes, mixed large face
  batches, ASCII ordering, unit/alignment rejection, steady flow scaling,
  unsupported-query rejection, route authority direction, and COMET/benchmark
  regressions.
- A parallel two-step smoke run completed all 15 abdominal cases. This checks
  end-to-end execution, provenance, output creation, and case counts; it is not
  the full validation benchmark.
- A 100-step 5× inlet-to-outlet pilot completed without import/query errors.
  COMET stalled near a wall and did not reach the target in that short budget.
  This is a performance finding, not evidence of full-benchmark success. Its
  output is in `outputs/cfd_smoke_run4/pilot_flow5_100steps`.
