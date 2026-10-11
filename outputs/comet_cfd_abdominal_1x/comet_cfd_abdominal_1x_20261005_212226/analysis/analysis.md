# Abdominal COMET 1× CFD analysis

Run: `comet_cfd_abdominal_1x_20261005_212226`. All five frozen cases completed. Two reached the 0.5 mm
target tolerance (40%); three exhausted 2,000 steps / 40 s with stagnation
early stopping disabled. No CFD query exception or nonfinite saved trajectory
was found. CSV, NPZ, and per-step diagnostic counts agree, endpoints match the
frozen manifest, and every case records the expected source ZIP hash and
velocity multiplier 1.0. This is one deterministic five-route block with no
localization noise, not an estimate over a randomized trial population.

| Route | Outcome | Time (s) | Final error (mm) | Best error (mm) | Path (mm) | Min centre clearance (mm) | Boost/rescue time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Canonical 1: frozen inlet-to-outlet label | Reached | 7.36 | 0.433 | 0.433 | 562.7 | 0.720 | 4.6% |
| Canonical 2: random-to-publication target | Timeout | 40.00 | 164.925 | 158.356 | 734.7 | 0.714 | 83.4% |
| Canonical 3: publication start-to-random | Reached | 2.38 | 0.429 | 0.429 | 179.1 | 0.716 | 0.0% |
| Random 1 | Timeout | 40.00 | 151.235 | 137.980 | 1288.2 | 0.704 | 74.6% |
| Random 2 | Timeout | 40.00 | 56.982 | 54.894 | 722.9 | 0.785 | 81.8% |

## Why the three cases failed

Canonical 2 and Random 2 oscillate near their initial branch. Their net start-to-end
displacements are only 2.24 and 7.29 mm despite travelling 734.7 and 722.9 mm.
Their best target distances remain 158.36 and 54.89 mm. In the final ten seconds,
median local CFD velocity projected onto the original route is approximately
−116 mm/s for both. They spend 83.4% and 81.9% of control intervals in boost/rescue.
This supports adverse-flow stalls with recovery cycling; it does not prove that
the routes are unreachable under all possible control inputs.

Random 1 was already unsuccessful in the matching historical simplified-flow
run. It moves through more of the network, then loses route progress: recorded
route arc reaches 107.2 mm but ends at 14.0 mm. Replanning changes route arc's
reference, so this measure is a diagnostic, not cumulative distance. Its final
error is larger than its initial 138.88 mm error, and 74.6% of control intervals
are boost/rescue. The final ten-second median local flow projected on the
original route is approximately −88.8 mm/s.

## Boundary direction and meaning of 1×

The imported boundary metadata identifies `face6` as the inlet (negative outward
mass flux −0.0197339 kg/s), centred at approximately (−49.1, −54.1, −33.3) mm.
`face8` is an outlet (positive outward flux +0.000918872 kg/s), centred at
(70.6, 18.4, −434.7) mm. Frozen Canonical 1 starts near face8 and targets the
region near face6. Its historical “inlet-to-outlet” label therefore describes
the opposite of the actual CFD transport direction. This identifies a naming
mismatch; it does not establish that the SimScale boundary assignment is wrong.
The route should be reported as an upstream route with the actual boundary roles.

The current simplified model and CFD vectors agree in direction at the five
start points (cosines 0.966–0.990). CFD speed is roughly 5–20 times the simplified
speed there at t=0. In particular the two stalled branch starts increase from
approximately 5.7–6.1 mm/s to 113.9–116.7 mm/s. “1×” identifies each model's own
baseline; it does not establish matched speed or inlet conditions across models.
The CFD mesh and velocities remain in native SI units with no rescaling.

## Historical comparison and model limitations

The historical no-noise COMET rows in
`outputs/publication_benchmark_1mm/comet_localization_noise_45cases_noearlystop_proc/paired_results.csv` use identical endpoints, carrier,
dt, step budget, target tolerance, device parameters, and seed. They reached
4/5 targets (80%): Canonical 2 and Random 2 now fail; Random 1 fails in both.
Canonical 1 increases from 4.42 to 7.36 s; Canonical 3 from 1.74 to 2.38 s.
The historical run records Git commit 66a2083; the CFD run records 0531920
with local changes. This is a descriptive historical comparison, not a controlled
experiment isolating flow alone.

All five satisfy the existing 0.7 mm centre-to-wall criterion and report zero
wall exits/violations. The tightest centre clearance is 0.703910 mm (Random 1),
only 3.91 µm above the threshold. Minimum particle-surface clearance is
0.453910 mm after subtracting the 0.25 mm carrier radius. These clearances use
the retained centreline/tubular wall model; integration projects positions into
that model. They do not certify clearance from the actual CFD mesh surface.

COMET's branch-flow adaptation, high-flow rescue gate, and terminal authority
gate use the scalar `flow_scale >= 4.0`. They remain off for this CFD run at 1.0,
even where the measured local flow is much stronger than the old 1× model.
Wall riding and local CFD compensation remain active. The wall-riding side is
chosen geometrically; it does not search the CFD cross-section for the safest
low-flow corridor. Controller updates should use measured adverse flow relative
to achievable magnetic drift rather than infer difficulty from the multiplier.

Mean control computation is 61.0, 69.8, 14.9, 37.0, and 15.3 ms for Canonical
1/2/3 and Random 1/2. Three exceed the 20 ms control interval on this machine in
this parallel run. This affects real-time readiness, not the offline trajectories.

These results demonstrate successful integration and partial navigation under
the supplied steady CFD field. They do not yet establish reliable navigation
across the abdominal routes or full CFD-surface particle validation. Residual
histories and mesh refinement evidence remain unavailable.

## Recommended next experiment

Keep this result as the untuned baseline. Confirm the intended anatomical inlet
and correct route labels. Then test CFD-aware local flow/authority gating and a
cross-sectional corridor choice on the two shared branch stalls, preserving
the same endpoints, limits, tolerance, and 40 s budget. Run a fresh same-code
simplified-flow comparison if an isolated flow-effect claim is needed. Extending
the time budget alone is unlikely to resolve the observed repeated local cycles.

Sources: run_config.json, paired_results.csv, all five trial NPZs, comet_diagnostics.csv,
data/cfd/abdominal_aorta_steady.json, and the historical run cited above. Supporting
calculations are in metrics.json and route_flow_diagnostics.json.
