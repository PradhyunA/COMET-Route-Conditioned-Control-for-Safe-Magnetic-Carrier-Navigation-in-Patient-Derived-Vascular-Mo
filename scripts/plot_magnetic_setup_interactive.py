"""
Interactive 3D visualization of the DynaMag magnetic scanner setup.

Creates:
    outputs/magnetic_setup_30cm_interactive.html

Shows:
    - vessel network centerlines
    - magnet/coil center positions
    - coil orientation arrows
    - scanner center
    - approximate scanner bore / placement radius
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

try:
    import plotly.graph_objects as go
except ImportError as exc:
    raise ImportError(
        "Plotly is required for this script. Install it with:\n"
        "    pip install plotly"
    ) from exc

from dynamag_simulation.core.magnets import ElectromagnetArray3D
from dynamag_simulation.core.real_vessel_network import RealVesselNetwork


DEFAULT_VESSEL_FILE = "real_vessel_segments_cm_corrected.pkl"
OUTPUT_DIR = "outputs"
OUTPUT_HTML = os.path.join(OUTPUT_DIR, "magnetic_setup_30cm_interactive.html")


# ---------------------------------------------------------------------
# Match your current body-scale scanner defaults
# ---------------------------------------------------------------------

PLACEMENT_RADIUS = 0.300  # m

# These should match the realistic scanner values you are using now.
# If magnets.py has different defaults, this script still explicitly shows
# the intended 30 cm body-scale scanner.
COIL_RADIUS = 0.070
COIL_LENGTH = 0.160
CORE_RADIUS = 0.030
TURNS = 600.0
MAX_CURRENT = 16.0
CORE_MU_R = 70.0
CORE_SAT_T = 1.8


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return v / n


def _make_ring(center, radius, axis="z", n=160):
    """
    Make a simple circle/ring for visualizing the scanner bore.
    """
    t = np.linspace(0.0, 2.0 * np.pi, n)

    center = np.asarray(center, dtype=float)

    if axis == "z":
        x = center[0] + radius * np.cos(t)
        y = center[1] + radius * np.sin(t)
        z = np.full_like(t, center[2])
    elif axis == "x":
        x = np.full_like(t, center[0])
        y = center[1] + radius * np.cos(t)
        z = center[2] + radius * np.sin(t)
    elif axis == "y":
        x = center[0] + radius * np.cos(t)
        y = np.full_like(t, center[1])
        z = center[2] + radius * np.sin(t)
    else:
        raise ValueError("axis must be x, y, or z")

    return x, y, z


def _segment_trace(network, max_segments=None):
    """
    Convert vessel segments to Plotly line traces.
    """
    traces = []

    segments = network.segments
    if max_segments is not None:
        segments = segments[:max_segments]

    xs = []
    ys = []
    zs = []

    for seg in segments:
        p0 = np.asarray(seg.start, dtype=float)
        p1 = np.asarray(seg.end, dtype=float)

        xs.extend([p0[0] * 1000.0, p1[0] * 1000.0, None])
        ys.extend([p0[1] * 1000.0, p1[1] * 1000.0, None])
        zs.extend([p0[2] * 1000.0, p1[2] * 1000.0, None])

    traces.append(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            name="Vessel centerlines",
            line=dict(width=2),
            opacity=0.65,
        )
    )

    return traces


def _magnet_position_trace(magnets):
    pos_mm = magnets.positions * 1000.0

    labels = [f"Magnet {i + 1}" for i in range(magnets.n_magnets)]

    return go.Scatter3d(
        x=pos_mm[:, 0],
        y=pos_mm[:, 1],
        z=pos_mm[:, 2],
        mode="markers+text",
        text=labels,
        textposition="top center",
        name="Electromagnets",
        marker=dict(
            size=7,
            symbol="circle",
        ),
    )


def _orientation_traces(magnets, arrow_len_m=0.070):
    """
    Plot orientation arrows as line segments from each magnet center.
    """
    traces = []

    xs = []
    ys = []
    zs = []

    for p, u in zip(magnets.positions, magnets.orientations):
        p = np.asarray(p, dtype=float)
        u = _unit(u)

        p0 = p
        p1 = p + arrow_len_m * u

        xs.extend([p0[0] * 1000.0, p1[0] * 1000.0, None])
        ys.extend([p0[1] * 1000.0, p1[1] * 1000.0, None])
        zs.extend([p0[2] * 1000.0, p1[2] * 1000.0, None])

    traces.append(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            name="Coil orientation arrows",
            line=dict(width=6),
        )
    )

    return traces


def _scanner_ring_traces(center, placement_radius):
    """
    Add three orthogonal rings to show scanner/body-scale envelope.
    """
    traces = []

    for axis, name in [
        ("z", "Scanner ring XY"),
        ("x", "Scanner ring YZ"),
        ("y", "Scanner ring XZ"),
    ]:
        x, y, z = _make_ring(center, placement_radius, axis=axis)

        traces.append(
            go.Scatter3d(
                x=x * 1000.0,
                y=y * 1000.0,
                z=z * 1000.0,
                mode="lines",
                name=name,
                line=dict(width=3, dash="dash"),
                opacity=0.55,
            )
        )

    return traces


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    network = RealVesselNetwork(DEFAULT_VESSEL_FILE, flow_scale=1.0)
    print(network.segment_summary())

    bounds_min, bounds_max = network.bounds(include_radius=True)
    anatomical_center = 0.5 * (bounds_min + bounds_max)

    magnets = ElectromagnetArray3D(
        coil_radius=COIL_RADIUS,
        coil_length=COIL_LENGTH,
        turns=TURNS,
        max_current=MAX_CURRENT,
        placement_radius=PLACEMENT_RADIUS,
        center=anatomical_center,
        core_radius=CORE_RADIUS,
        core_relative_permeability=CORE_MU_R,
        core_saturation_tesla=CORE_SAT_T,
    )

    fig = go.Figure()

    for trace in _segment_trace(network):
        fig.add_trace(trace)

    fig.add_trace(_magnet_position_trace(magnets))

    for trace in _orientation_traces(magnets):
        fig.add_trace(trace)

    for trace in _scanner_ring_traces(anatomical_center, PLACEMENT_RADIUS):
        fig.add_trace(trace)

    center_mm = anatomical_center * 1000.0
    fig.add_trace(
        go.Scatter3d(
            x=[center_mm[0]],
            y=[center_mm[1]],
            z=[center_mm[2]],
            mode="markers+text",
            text=["Scanner center"],
            textposition="bottom center",
            name="Scanner center",
            marker=dict(size=6, symbol="diamond"),
        )
    )

    title = (
        "DynaMag 30 cm Body-Scale Magnetic Scanner Setup<br>"
        f"placement_radius={PLACEMENT_RADIUS * 1000:.0f} mm, "
        f"coil_radius={COIL_RADIUS * 1000:.0f} mm, "
        f"turns={TURNS:.0f}, max_current={MAX_CURRENT:.1f} A"
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x [mm]",
            yaxis_title="y [mm]",
            zaxis_title="z [mm]",
            aspectmode="data",
        ),
        legend=dict(
            x=0.02,
            y=0.98,
        ),
        margin=dict(l=0, r=0, t=70, b=0),
    )

    fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn")

    print(f"Saved interactive setup visualization to:")
    print(f"  {OUTPUT_HTML}")
    print()
    print("Open it with:")
    print(f"  open {OUTPUT_HTML}")


if __name__ == "__main__":
    main()