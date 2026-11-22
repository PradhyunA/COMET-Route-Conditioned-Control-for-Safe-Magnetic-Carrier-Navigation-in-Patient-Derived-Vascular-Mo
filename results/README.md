# Results reported in the paper

Each run directory keeps the files written by the benchmark runner, without
per-trial trajectories: `run_config.json` (exact settings, software and git
state), `benchmark_summary.json`, `patient_positioning.json`, and
`paired_results.csv` (one row per controller and case) under
`<condition>/<anatomy>/`. Safe success is `reached_target` and
`clearance_target_met` and not `wall_exit`.

| Directory | Paper |
|---|---|
| `main_benchmark/five_anatomy_final_comet_v4_20261103/` | Main benchmark (Tables 3 and 4, Figures 4 and 5). `all_controllers_by_vessel_network.csv` combines COMET and all baselines; `ci_summary.csv` has bootstrap CIs and McNemar tests. |
| `main_benchmark/five_anatomy_final_0049_20261103/`, `..._20261103/`, `..._nmpc_20261103/`, `..._comet_v3_20261103/` | Source runs for the baselines and earlier COMET revisions in the combined table. |
| `replicates/` | Five jittered-start replicates (Table 5); `replicates_400_ci_20261105.csv`. |
| `actuator_sensitivity/` | 300 and 600 mT/m (Table 6, Figure 6); the `*_slow_*` runs add tuned MPC and ACBC. |
| `cfd_steady/` | Steady CFD study (Table 7); `cfd_steady_complete_paired_results_20261112.csv` merges the 150 trials. |
| `ablations/` | Mechanism ablations (Table 8). |
| `localization_noise/` | Localization noise (Section 4.7). |

Controller keys: `comet_v4` is COMET; `nmpc_tuned`, `nmpc_default`,
`mpc_tuned` (heuristic MPC) and `acbc_tuned` are the baselines.
