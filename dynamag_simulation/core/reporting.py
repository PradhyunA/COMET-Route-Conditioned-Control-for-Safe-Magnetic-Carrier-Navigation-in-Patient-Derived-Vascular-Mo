"""
Structured trial results and output (CSV, console summary).
"""

import os
import csv
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TrialResult:
    """Container for everything produced by a single simulation trial."""
    trial_name: str

    start: np.ndarray
    target: np.ndarray
    final_position: np.ndarray

    trajectory: np.ndarray
    distance_log: np.ndarray
    current_log: np.ndarray
    field_log: np.ndarray
    force_log: np.ndarray
    velocity_log: np.ndarray          # realized velocity magnitudes
    reynolds_log: np.ndarray
    cost_log: np.ndarray
    step_time_log: np.ndarray         # wall-clock seconds per control step

    reached: bool
    steps_taken: int
    final_error: float
    max_current_used: float

    # Physics validity
    mean_reynolds: float = 0.0
    max_reynolds: float = 0.0
    low_re_valid: bool = True
    stokes_regime: bool = True
    finite_re_drag_valid: bool = True
    drag_model: str = "schiller_naumann"
    drag_regime_label: str = "stokes"

    # Timing
    wall_time_s: float = 0.0
    avg_control_step_time_s: float = 0.0
    max_control_step_time_s: float = 0.0

    # Trial-level early stopping
    early_stopped: bool = False
    early_stop_reason: str = ""
    early_stop_window_steps: int = 0
    early_stop_improvement_threshold_m: float = np.nan
    early_stop_measured_improvement_m: float = np.nan

    # Metadata
    seed: int = 42
    profile_name: str = "microsphere_500um"
    dt: float = 0.02
    workspace_half: float = 0.0762

    magnets: Optional[np.ndarray] = None
    orientations: Optional[np.ndarray] = None
    params: dict = field(default_factory=dict)


def print_summary(result: TrialResult):
    """Print a formatted console summary for one trial."""
    print(f"\n{'=' * 60}")
    print(f"  {result.trial_name}")
    print(f"{'=' * 60}")
    print(f"  Profile         : {result.profile_name}")
    print(f"  Start           : {result.start}")
    print(f"  Target          : {result.target}")
    print(f"  Final position  : {result.final_position}")
    print(f"  Steps taken     : {result.steps_taken}")
    print(f"  Final error (m) : {result.final_error:.6f}")
    print(f"  Final error (mm): {result.final_error * 1000:.3f}")
    print(f"  Reached target  : {result.reached}")
    print(f"  Early stopped   : {result.early_stopped}")
    if result.early_stopped:
        print(f"  Stop reason     : {result.early_stop_reason}")
        print(f"  Stop window     : {result.early_stop_window_steps} steps")
        print(
            "  Stop threshold  : "
            f"{result.early_stop_improvement_threshold_m * 1000:.3f} mm"
        )
        print(
            "  Stop improvement: "
            f"{result.early_stop_measured_improvement_m * 1000:.3f} mm"
        )
    print(f"  Max current (A) : {result.max_current_used:.3f}")
    print(f"  Avg ctrl time(s): {result.avg_control_step_time_s:.4f}")
    print(f"  Wall time (s)   : {result.wall_time_s:.2f}")
    print(f"  Mean Re         : {result.mean_reynolds:.3e}")
    print(f"  Max  Re         : {result.max_reynolds:.3e}")
    print(f"  Stokes regime   : {result.stokes_regime}")
    print(f"  Drag model      : {result.drag_model}")
    print(f"  Drag regime     : {result.drag_regime_label}")
    print(f"  Finite-Re valid : {result.finite_re_drag_valid}")

    if len(result.force_log) > 0:
        print(f"  Peak |F|  (N)   : {np.max(result.force_log):.3e}")
        print(f"  Peak |v|  (m/s) : {np.max(result.velocity_log):.3e}")
        print(f"  Peak |B|  (T)   : {np.max(result.field_log):.3e}")


CSV_HEADER = [
    "trial_name",
    "start_x", "start_y", "start_z",
    "target_x", "target_y", "target_z",
    "final_x", "final_y", "final_z",
    "steps_taken", "sim_time_s",
    "final_error_m", "final_error_mm",
    "reached_target", "max_current_A",
    "early_stopped", "early_stop_reason",
    "early_stop_window_steps",
    "early_stop_improvement_threshold_m",
    "early_stop_improvement_threshold_mm",
    "early_stop_measured_improvement_m",
    "early_stop_measured_improvement_mm",
    "wall_time_s",
    "avg_control_step_time_s", "max_control_step_time_s",
    "mean_reynolds", "max_reynolds", "low_re_valid",
    "stokes_regime", "finite_re_drag_valid",
    "drag_model", "drag_regime_label",
    "seed", "profile_name",
    "peak_force_N", "peak_velocity_m_s", "peak_field_T",
]


def save_results_csv(results: list, output_dir: str,
                     filename: str = "trial_summary.csv"):
    """Write a summary CSV for a batch of TrialResult objects."""
    path = os.path.join(output_dir, filename)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)

        for r in results:
            w.writerow([
                r.trial_name,
                r.start[0], r.start[1], r.start[2],
                r.target[0], r.target[1], r.target[2],
                r.final_position[0], r.final_position[1], r.final_position[2],
                r.steps_taken,
                r.steps_taken * r.dt,
                r.final_error,
                r.final_error * 1000.0,
                r.reached,
                r.max_current_used,
                r.early_stopped,
                r.early_stop_reason,
                r.early_stop_window_steps,
                r.early_stop_improvement_threshold_m,
                r.early_stop_improvement_threshold_m * 1000.0,
                r.early_stop_measured_improvement_m,
                r.early_stop_measured_improvement_m * 1000.0,
                r.wall_time_s,
                r.avg_control_step_time_s,
                r.max_control_step_time_s,
                r.mean_reynolds,
                r.max_reynolds,
                r.low_re_valid,
                r.stokes_regime,
                r.finite_re_drag_valid,
                r.drag_model,
                r.drag_regime_label,
                r.seed,
                r.profile_name,
                float(np.max(r.force_log)) if len(r.force_log) > 0 else 0.0,
                float(np.max(r.velocity_log)) if len(r.velocity_log) > 0 else 0.0,
                float(np.max(r.field_log)) if len(r.field_log) > 0 else 0.0,
            ])
