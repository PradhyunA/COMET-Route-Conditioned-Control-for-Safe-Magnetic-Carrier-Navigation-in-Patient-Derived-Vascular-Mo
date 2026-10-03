"""Run the adapted SMC-DOB-inspired route-following baseline.

The simulator delegates shared physics, geometry, moving-array placement,
integration, and the bounded SLSQP current allocator to the route-controller
simulation harness.  Only the high-level control law changes.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynamag_simulation.core.constants import DEFAULT_TARGET_TOLERANCE
from dynamag_simulation.core.real_vessel_network import RealVesselNetwork
from dynamag_simulation.core.reporting import print_summary, save_results_csv

from simulation_3d_pid import (
    DEFAULT_VESSEL_FILE,
    save_smc_dob_diagnostics,
    simulate_trial_pid,
)


SMC_DOB_CONFIG_KEYS = {
    "sliding_gamma",
    "sliding_lambda",
    "equivalent_velocity_gain",
    "equivalent_position_gain",
    "switching_gain",
    "boundary_layer",
    "disturbance_time_constant",
    "disturbance_gain",
    "max_disturbance_norm",
    "route_speed",
    "terminal_distance",
    "terminal_position_gain",
    "max_step_fraction",
    "flow_compensation_gain",
    "wall_repulsion_gain",
    "max_control_norm",
    "current_effort_weight",
    "current_smoothness_weight",
    "wall_penalty_weight",
    "optimizer_maxiter",
}


def simulate_trial_smc_dob(start, target, network, **kwargs):
    """Simulate one SMC-DOB-inspired trial with the shared benchmark model."""
    kwargs = dict(kwargs)
    smc_dob_kwargs = {
        key: kwargs.pop(key)
        for key in tuple(kwargs)
        if key in SMC_DOB_CONFIG_KEYS
    }
    return simulate_trial_pid(
        start=start,
        target=target,
        network=network,
        controller_mode="smc_dob_inspired",
        smc_dob_kwargs=smc_dob_kwargs,
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run the adapted SMC-DOB-inspired route baseline."
    )
    parser.add_argument("--vessel-file", default=DEFAULT_VESSEL_FILE)
    parser.add_argument("--output-dir", default="outputs/simulation_outputs_3d_smc_dob")
    parser.add_argument("--flow-scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TARGET_TOLERANCE)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    network = RealVesselNetwork(args.vessel_file, flow_scale=args.flow_scale)
    result = simulate_trial_smc_dob(
        start=network.inlet_start(),
        target=network.outlet_end(),
        network=network,
        trial_name="smc_dob_inspired_inlet_to_outlet",
        dt=args.dt,
        steps=args.steps,
        tolerance=args.tolerance,
        debug=args.debug,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    print_summary(result)
    save_results_csv([result], args.output_dir)
    save_smc_dob_diagnostics([result], args.output_dir)


if __name__ == "__main__":
    main()
