#!/usr/bin/env python3
"""Validate DynaMag iCEM sampling against the pinned authors' repositories."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dynamag_simulation.core.cem_mpc_controller import (
    ICEMConfig,
    _powerlaw_psd_gaussian,
    icem_population_schedule,
)


PINNED_ICEM_COMMIT = "98c1c1fe2bfc94a87e4658b6c793f4e81bc30203"
PINNED_COLOREDNOISE_COMMIT = "6f0185969092186b08d5ea81dfca7023ed25399a"


def _head(path: str) -> str:
    return subprocess.check_output(
        ["git", "-C", path, "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _require_source_mechanics(source: str) -> None:
    expected_fragments = {
        "colored temporal noise and H/D transpose": (
            "colorednoise.powerlaw_psd_gaussian",
            ".transpose(\n                [0, 2, 1])",
        ),
        "hard action clipping": ("samples = np.clip(samples * self.std + self.mean",),
        "final-iteration mean injection": ("sampled_from_distribution[0] = self.mean",),
        "population decay": ("int(num_sim_traj / self.factor_decrease_num)",),
        "elite shift across MPC steps": ("reused_actions = actions[:, 1:]",),
        "elite retention across iterations": ("simulated_paths.extend(self.elite_samples",),
        "best sampled action execution": (
            "executed_action = simulated_paths[best_traj_idx][\"actions\"][0]",
        ),
        "smoothed mean/std update": (
            "self.mean = (1 - self.alpha) * new_mean + self.alpha * self.mean",
            "self.std = (1 - self.alpha) * new_std + self.alpha * self.std",
        ),
    }
    missing = [
        name
        for name, fragments in expected_fragments.items()
        if not all(fragment in source for fragment in fragments)
    ]
    if missing:
        raise AssertionError("upstream mechanics not recognized: " + ", ".join(missing))


def validate(icem_repo: str, colorednoise_repo: str) -> None:
    icem_head = _head(icem_repo)
    noise_head = _head(colorednoise_repo)
    if icem_head != PINNED_ICEM_COMMIT:
        raise AssertionError(f"iCEM HEAD {icem_head} != pinned {PINNED_ICEM_COMMIT}")
    if noise_head != PINNED_COLOREDNOISE_COMMIT:
        raise AssertionError(
            f"colorednoise HEAD {noise_head} != pinned {PINNED_COLOREDNOISE_COMMIT}"
        )

    settings_path = os.path.join(
        icem_repo,
        "icem",
        "settings",
        "defaults",
        "i-cem-blitz.json",
    )
    with open(settings_path, encoding="utf-8") as handle:
        upstream = json.load(handle)["controller_params"]
    sampler = upstream["action_sampler_params"]
    local = ICEMConfig()
    comparisons = {
        "horizon": (local.horizon, upstream["horizon"]),
        "population_size": (
            local.population_size,
            upstream["num_simulated_trajectories"],
        ),
        "population_decay_factor": (
            local.population_decay_factor,
            upstream["factor_decrease_num"],
        ),
        "alpha": (local.alpha, sampler["alpha"]),
        "num_elites": (local.num_elites, sampler["elites_size"]),
        "fraction_elites_reused": (
            local.fraction_elites_reused,
            sampler["fraction_elites_reused"],
        ),
        "init_std": (local.init_std, sampler["init_std"]),
        "keep_previous_elites": (
            local.keep_previous_elites,
            sampler["keep_previous_elites"],
        ),
        "noise_beta": (local.noise_beta, sampler["noise_beta"]),
        "max_iterations": (local.max_iterations, sampler["opt_iterations"]),
        "shift_elites_over_time": (
            local.shift_elites_over_time,
            sampler["shift_elites_over_time"],
        ),
        "use_mean_action": (local.use_mean_action, sampler["use_mean_actions"]),
    }
    mismatches = {
        key: values for key, values in comparisons.items() if values[0] != values[1]
    }
    if mismatches:
        raise AssertionError(f"configuration mismatches: {mismatches}")

    controller_path = os.path.join(icem_repo, "icem", "controllers", "icem.py")
    with open(controller_path, encoding="utf-8") as handle:
        _require_source_mechanics(handle.read())

    module_path = os.path.join(colorednoise_repo, "colorednoise.py")
    spec = importlib.util.spec_from_file_location("upstream_colorednoise", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    seed = 2021
    shape = (40, 9, 30)
    local_noise = _powerlaw_psd_gaussian(
        local.noise_beta,
        shape,
        rng=np.random.default_rng(seed),
    )
    upstream_noise = module.powerlaw_psd_gaussian(
        local.noise_beta,
        shape,
        random_state=np.random.default_rng(seed),
    )
    if not np.array_equal(local_noise, upstream_noise):
        difference = float(np.max(np.abs(local_noise - upstream_noise)))
        raise AssertionError(f"colored-noise samples differ; max abs error={difference}")

    schedule = icem_population_schedule(local)
    upstream_schedule = tuple(
        max(
            sampler["elites_size"] * 2,
            int(upstream["num_simulated_trajectories"] / upstream["factor_decrease_num"]**i),
        )
        for i in range(sampler["opt_iterations"])
    )
    if schedule != upstream_schedule:
        raise AssertionError(f"population schedule {schedule} != {upstream_schedule}")

    print(f"PASS iCEM reference commit: {icem_head}")
    print(f"PASS colorednoise reference commit: {noise_head}")
    print("PASS official i-cem-blitz configuration: 12/12 fields")
    print("PASS upstream sampling/update mechanics: 8/8 checks")
    print(f"PASS population schedule: {schedule} ({sum(schedule)} fresh trajectories)")
    print(f"PASS seeded colored-noise samples: exact array equality, shape={shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-repo", default="/tmp/dynamag-icem-upstream")
    parser.add_argument(
        "--colorednoise-repo",
        default="/tmp/dynamag-colorednoise-upstream",
    )
    args = parser.parse_args()
    validate(args.upstream_repo, args.colorednoise_repo)


if __name__ == "__main__":
    main()
