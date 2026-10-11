# Run 4 export audit — 5 October 2026

Both exports match their corrected project anatomy without coordinate rescaling. Imported SI fields are ready for COMET flow-sensitivity benchmarking.

| Network | Surface spans (mm) | Cells | Final relative mass imbalance |
| --- | --- | ---: | ---: |
| Abdominal aorta | 165.05 × 124.44 × 405.76 | 649,072 | 2.59e-08 |
| Pulmonary artery | 147.83 × 118.20 × 186.30 | 1,324,006 | 7.51e-10 |

All inspected internal velocity, pressure, turbulent kinetic energy, specific dissipation rate, and eddy-viscosity values are finite. Inlet vectors point inward; every designated outlet has positive net outward flow. The importer additionally checked that all reconstructed mesh cell volumes are positive. No units correction was applied.

The abdominal inlet is (0, 0, -0.04) m/s. Pulmonary inlet is (0, -0.02828, 0.02828) m/s. Kinematic viscosity is 3.302e-6 m²/s and density 1060 kg/m³. Both runs use steady kOmegaSST. Final continuity checks are excellent; momentum convergence and mesh independence are unverified because histories and refinement studies were not supplied.

Abdominal has limited local outlet backflow while each outlet maintains net outflow. This is not by itself a rejection of the solution.

The current 45-case benchmark still requires matching 40Left85Right aortic-root CFD. The earlier VTU matches 0050 cerebral and has unresolved units/time metadata.

The imported fields are data/cfd/abdominal_aorta_steady.npz and data/cfd/pulmonary_steady.npz, with accompanying provenance JSON. The abdominal 15-case script is run_comet_cfd_abdominal_15trials.sh. See docs/cfd_validation.md for commands and interpretation limits.

Short validation: 19 targeted unit/regression checks passed; a parallel two-step run completed all 15 frozen abdominal cases. A 100-step 5x canonical pilot completed without mesh/query errors but stalled near a wall. These checks do not constitute the full 2000-step benchmark.

The source audit was recomputed in the separate Python 3.13 CFD environment after detecting inconsistent large-array calculations in the previous Python 3.14 / NumPy 1.26 environment. The original environment was preserved.
