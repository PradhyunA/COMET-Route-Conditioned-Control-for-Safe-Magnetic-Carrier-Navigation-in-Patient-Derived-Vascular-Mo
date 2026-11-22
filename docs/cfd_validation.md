# CFD validation workflow

The Run 4 abdominal and pulmonary exports match the project's corrected SI
anatomies. No coordinate or velocity rescaling was applied. The abdominal mesh
spans 165.05 × 124.44 × 405.76 mm; pulmonary spans 147.83 × 118.20 × 186.30 mm.
Both have finite flow arrays, inward inlet vectors, positive computed cell
volumes, and excellent final mass balance. Source audits are in
the JSON files in `data/cfd/`.

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
  --vessel-file data/vessels/abdominal_0033_H_ABAO_AAA/real_vessel_segments.pkl \
  --output data/cfd/abdominal_aorta_steady.npz

.venv-cfd/bin/python scripts/import_simscale_flow.py \
  --zip "$HOME/Downloads/PulmonaryArteryCFD-Incompressible-Run_4.zip" \
  --vessel-file data/vessels/pulmonary_0082_H_PULM_H/real_vessel_segments.pkl \
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

## Run the paper's CFD study

The paper's steady-CFD study (abdominal and pulmonary models, five
controllers, flow multipliers 1, 3 and 5; 30 cases per controller) uses the
benchmark runner with `CFD=1`, which keeps only the anatomies that have an
imported field:

```bash
CFD=1 PYTHON_BIN=.venv-cfd/bin/python \
  CONTROLLERS=comet_v4,nmpc_tuned,nmpc_default,mpc_tuned,acbc_tuned \
  RUN_ID=cfd_steady ./run_physics_sensitivity_sweep.sh
```

The results reported in the paper are in `results/cfd_steady/`.

The publication runner also accepts `--cfd-file` directly. Its CFD mode supports
COMET, CEM-MPC, ACBC, and MPC variants. CEM's batched flow model samples the CFD
field when present, while retaining analytical behavior for ordinary runs.

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
  This is a performance finding, not evidence of full-benchmark success.
