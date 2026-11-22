# SMC-DOB-inspired published-style baseline

This controller is an adapted published-style baseline based on Bhattacharjee
et al., *Robust Autonomous Control of a Magnetic Millirobot in In Vitro
Cardiac Flow* ([arXiv:2604.01523](https://arxiv.org/abs/2604.01523)). It is
labelled `smc_dob_inspired` and must not be described as an exact reproduction.

The source paper used a planar permanent-magnet millirobot, camera feedback, a
2-D in-vitro cardiac phantom, and a different electromagnetic coil system.
This project instead uses a 500-um magnetic carrier in a 3-D vessel network,
the shared simulated flow and electromagnetic array, and the same numerical
integration and hardware limits as the other manuscript controllers.

## Adapted controller

For local route reference position `x_d`, route-tangent reference velocity
`v_d`, realized velocity `v`, and position `x`, the controller uses

```text
e      = x_d - x
e_dot  = v_d - v
s      = gamma * LPF(e_dot) + lambda * e
d_hat  = LPF(v - v_mag,previous - v_flow,previous)
v_eq   = v_d + k_v * LPF(e_dot) + k_p * e
v_sw   = k_sw * tanh(s / phi)
v_mag  = v_eq + v_sw - g_flow * v_flow - g_dob * d_hat + v_wall
```

This preserves the paper's sliding surface, equivalent-control term, smooth
boundary-layer switching term, and disturbance-observer structure. Force-domain
terms are converted to magnetic-velocity terms because this simulator uses an
overdamped particle model, `v_mag = F_mag / gamma_eff`.

The desired magnetic velocity is mapped to signed coil currents by the existing
bounded SLSQP velocity allocator. The controller does not use COMET branch
probabilities, downstream value, terminal precision, commitment memory, or CEM
allocation.

## Benchmark protocol

`run_smc_dob_inspired_45trials.sh` runs unit tests and three short anatomy smoke
tests before launching the frozen 45-case benchmark:

- 3 anatomies
- 5 fixed Dijkstra routes per anatomy
- flow scales 1x, 3x, and 5x
- 500-um carrier profile
- `dt = 0.02 s`, 2,000 maximum steps, 0.5-mm target tolerance
- shared vessel, flow, particle, electromagnetic-array, current-limit, gantry,
  and integrator settings

The launcher writes a machine-readable aggregate report containing success
count, median and maximum final error, median steps, mean effective steps, and
the 5x-flow breakdown.
