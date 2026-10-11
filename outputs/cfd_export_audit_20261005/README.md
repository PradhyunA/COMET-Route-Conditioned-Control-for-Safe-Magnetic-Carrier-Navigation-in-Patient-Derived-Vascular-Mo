# CFD export audit — 5 October 2026

The supplied exports are not ready to replace the current 45-case benchmark flows. Simulation source files and benchmark code were left unchanged. The detailed numerical results are in `audit.json`.

| Export | Finding | Required action for the current benchmark |
| --- | --- | --- |
| AbdominalAortaCFD-Incompressible-Run_3.zip | Physical mesh spans 16.506 × 12.431 × 40.574 mm. Geometry aligns with the corrected abdominal anatomy when coordinates are multiplied by 10. | Scale the current SimScale CAD by 10 and rerun meshing and CFD. Expected surface spans: 165.062 × 124.309 × 405.739 mm. |
| PulmonaryArteryCFD-Incompressible-Run_3.zip | Physical mesh spans 14.783 × 11.820 × 18.630 mm. Geometry aligns with the corrected pulmonary anatomy when coordinates are multiplied by 10. | Scale the current SimScale CAD by 10 and rerun meshing and CFD. Expected surface spans: 147.828 × 118.196 × 186.303 mm. |
| all_results_2010.vtu_02010.vtu | Closely matches the 0050 cerebral anatomy using a 0.01 coordinate mapping to `data/0050_H_CERE_H/real_vessel_segments.pkl`; does not align with the 40Left85Right anatomy in the current benchmark. Contains instantaneous velocity, pressure, nonzero velocity derivatives, and scalar average speed. Does not declare physical units or actual snapshot time. | Supply CFD for 40Left85Right to preserve the existing benchmark, or explicitly change the third anatomy and trial routes. Supply source units/settings and additional snapshots if the 0050 transient data are intended. |

The scale conclusions are relative to the geometry used in this project, not an independent certification of patient dimensions. The relevant benchmark geometry files are:

- `data/real_vessel_segments_cm_corrected.pkl`
- `data/vessels/pulmonary_0082_H_PULM_H/real_vessel_segments_cm_corrected.pkl`
- `data/vessels/aortic_root_cerebral_40Left85Right/real_vessel_segments.pkl`

The new abdominal mesh is approximately 0.001 times the raw geometry in the first export. Matching this project's corrected anatomy required 0.01 times that original geometry. Thus the new run is ten times smaller than the intended benchmark anatomy. Multiplying the current mesh's coordinates by 10 after solving is not a corrected physical CFD solution: at fixed inlet velocity and viscosity, the geometry change changes Reynolds number by a factor of 10. Correct geometry before solving.

## Basic numerical checks

Both SimScale exports contain a final steady solution at iteration 1000 from `simscaleSimpleFoam`, with kOmegaSST RANS enabled, kinematic viscosity 3.302e-6 m²/s, and density 1060 kg/m³. Exported `U` has dimensions m/s, pressure has dimensions Pa, and `phi` is mass flux in kg/s; treating `phi` as volumetric flow would be incorrect.

| Check | Abdominal | Pulmonary |
| --- | ---: | ---: |
| Cells | 121,997 | 818,976 |
| Open boundary patches | 11 | 83 |
| Inlet speed | 0.040000 m/s | 0.039994 m/s |
| Maximum internal speed | 0.304188 m/s | 0.182413 m/s |
| Inflow | 1.95460476437e-4 kg/s | 2.43643651540e-4 kg/s |
| Outflow | 1.95460476434e-4 kg/s | 2.43643650857e-4 kg/s |
| Relative final boundary mass imbalance | 1.53e-11 | 2.81e-9 |
| Sum of absolute cell mass residuals / inlet mass flow | 3.73e-9 | 5.06e-6 |

All inspected internal velocity, pressure, k, omega, and eddy-viscosity values are finite. Inlet vectors point inward: abdominal face6 has velocity (0, 0, -0.04) m/s; pulmonary face92 has velocity (0, -0.02828, 0.02828) m/s. Every designated outlet has positive net outward flow. One abdominal outlet face has a small reverse flux; this does not by itself establish a problem. No inlet was found pointing outward.

Excellent final mass balance does not verify momentum convergence or mesh independence. Neither ZIP contains solver residual histories, convergence plots, or a mesh-refinement comparison. Also, iteration 1000 in this steady solver is not 1000 seconds of physical flow. See [SimScale's simulation-control documentation](https://www.simscale.com/docs/simulation-setup/simulation-control-fluid/).

## VTU identity and units

The VTU contains 153,082 nodes and 738,037 tetrahedra, all with positive signed volumes under the exported connectivity ordering. Its point fields and connectivity arrays decoded without errors. Raw geometry spans are 6.57434 × 7.90101 × 11.90859 coordinate units. The raw maximum instantaneous speed is 3.43664 velocity units, while maximum average speed is 6.63743. These values cannot be labelled m/s from this file alone.

To test identity, each candidate project's segment midpoint was compared with the closest VTU mesh vertex under uniform scales 1, 0.1, 0.01, and 0.001. At the best scale for each candidate, the 95th percentile nearest-vertex distance divided by segment radius was:

| Candidate anatomy | Best coordinate scale | 95th percentile distance / radius |
| --- | ---: | ---: |
| 0050 cerebral | 0.01 | 0.292 |
| Abdominal | 0.1 | 33.158 |
| 40Left85Right | 0.1 | 49.452 |
| Pulmonary | 0.1 | 62.239 |

This is strong geometric evidence for 0050, with no comparable match to 40Left85Right. The audit did not infer units from the filename. [SimVascular's solver documentation](https://github.com/SimVascular/simvascular.github.io-archive/blob/master/docsFlowSolver.html) states that the solver does not enforce physical units: users can choose CGS, MGS, or SI. If the originating simulation used CGS, coordinates and velocity would each need conversion by 0.01 to SI, and pressure by 0.1; those conversions must be confirmed from the source simulation rather than assumed.

Velocity derivatives are nonzero at 67.74% of nodes, providing evidence that the file contains an unsteady snapshot rather than a certified steady solution. A scalar `average_speed` field cannot supply a mean velocity vector or a time-dependent velocity field. A single snapshot can be used for a clearly labelled frozen-field experiment after units and anatomy are resolved; pulsatile validation needs the velocity time series and time/cycle settings.

## Benchmark status

CFD integration and COMET benchmark execution remain paused. No supplied flow field was rescaled or used to produce steering results. To resume the existing benchmark, provide abdominal and pulmonary solves at the project's corrected physical geometry, plus a matching 40Left85Right export. The existing protocol uses five frozen routes at three flow conditions for 15 cases per anatomy, 45 across three anatomies. Scaling one solved velocity field to 3× or 5× yields a sensitivity experiment; independently solved inlet conditions are needed if those cases are to be described as separate CFD solutions.
