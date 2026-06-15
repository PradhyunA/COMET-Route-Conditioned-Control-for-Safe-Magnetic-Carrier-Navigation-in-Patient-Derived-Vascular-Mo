import pickle
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

# ============================================================
# Interactive Plotly/WebGL cerebral vessel + magnet visualizer
# ============================================================

vessel_file = Path("data/vessels/aortic_root_cerebral_40Left85Right/real_vessel_segments.pkl")
out_file = Path("outputs/interactive_40Left85Right_with_magnets.html")
out_file.parent.mkdir(parents=True, exist_ok=True)

# Display units
# Stored geometry is in meters. Convert to centimeters for plotting.
UNIT = 100.0

# Device visualization settings
placement_radius_m = 0.300
coil_length_m = 0.160
coil_radius_m = 0.070
front_back_depth_m = 0.180

# Hard cerebral canonical route from your benchmark, before cm correction:
# old start/target were in mm-scale meters, so after cm correction they are 10x larger.
hard_start_m = np.array([-104.88859564, 39.14727271, -376.76280737]) / 1000.0
hard_target_m = np.array([-76.45376772, 123.07464331, -166.75929725]) / 1000.0

# ============================================================
# Load vessel
# ============================================================

with open(vessel_file, "rb") as f:
    segments = pickle.load(f)

starts = []
ends = []
radii = []

for s in segments:
    if isinstance(s, dict):
        starts.append(np.asarray(s["start"], dtype=float))
        ends.append(np.asarray(s["end"], dtype=float))
        radii.append(float(s.get("radius", 0.0)))
    else:
        starts.append(np.asarray(s.start, dtype=float))
        ends.append(np.asarray(s.end, dtype=float))
        radii.append(float(getattr(s, "radius", 0.0)))

starts = np.asarray(starts)
ends = np.asarray(ends)
radii = np.asarray(radii)

all_pts_m = np.vstack([starts, ends])
all_pts_cm = all_pts_m * UNIT

bbox_min = all_pts_m.min(axis=0)
bbox_max = all_pts_m.max(axis=0)
bbox_center = (bbox_min + bbox_max) / 2.0
bbox_size = bbox_max - bbox_min

print("Loaded:", vessel_file)
print("Segments:", len(starts))
print("Bounding box size cm:", bbox_size * UNIT)
print("Radius range cm:", radii.min() * UNIT, "to", radii.max() * UNIT)
print("Gantry center cm:", bbox_center * UNIT)

# ============================================================
# Vessel centerline as one WebGL trace with None separators
# ============================================================

x, y, z = [], [], []

for a, b in zip(starts * UNIT, ends * UNIT):
    x.extend([a[0], b[0], None])
    y.extend([a[1], b[1], None])
    z.extend([a[2], b[2], None])

vessel_trace = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode="lines",
    name="40Left85Right vessel centerlines",
    line=dict(width=3),
    hoverinfo="skip",
)

# ============================================================
# Radius-colored midpoint cloud for detail
# ============================================================

midpoints_cm = ((starts + ends) / 2.0) * UNIT
radii_cm = radii * UNIT

radius_trace = go.Scatter3d(
    x=midpoints_cm[:, 0],
    y=midpoints_cm[:, 1],
    z=midpoints_cm[:, 2],
    mode="markers",
    name="Segment radius map",
    marker=dict(
        size=2.2,
        color=radii_cm,
        colorscale="Viridis",
        opacity=0.75,
        colorbar=dict(title="Radius cm"),
    ),
    text=[f"radius={r:.3f} cm" for r in radii_cm],
    hovertemplate="x=%{x:.2f} cm<br>y=%{y:.2f} cm<br>z=%{z:.2f} cm<br>%{text}<extra></extra>",
)

# ============================================================
# Terminal endpoints
# ============================================================

counts = {}
points = {}

for p in all_pts_m:
    key = tuple(np.round(p, 6))
    counts[key] = counts.get(key, 0) + 1
    points[key] = p

terminal_m = np.array([points[k] for k, c in counts.items() if c == 1])
terminal_cm = terminal_m * UNIT if len(terminal_m) else np.empty((0, 3))

endpoint_trace = go.Scatter3d(
    x=terminal_cm[:, 0],
    y=terminal_cm[:, 1],
    z=terminal_cm[:, 2],
    mode="markers+text",
    name="Terminal endpoints",
    marker=dict(size=7, symbol="circle"),
    text=[f"E{i:02d}" for i in range(len(terminal_cm))],
    textposition="top center",
    hovertemplate="Endpoint %{text}<br>x=%{x:.2f} cm<br>y=%{y:.2f} cm<br>z=%{z:.2f} cm<extra></extra>",
)

# ============================================================
# Start and target markers
# ============================================================

start_cm = hard_start_m * UNIT
target_cm = hard_target_m * UNIT

route_trace = go.Scatter3d(
    x=[start_cm[0], target_cm[0]],
    y=[start_cm[1], target_cm[1]],
    z=[start_cm[2], target_cm[2]],
    mode="markers+text+lines",
    name="Hard route start/target",
    marker=dict(size=10, symbol="diamond"),
    line=dict(width=5, dash="dash"),
    text=["START", "TARGET"],
    textposition="bottom center",
    hovertemplate="%{text}<br>x=%{x:.2f} cm<br>y=%{y:.2f} cm<br>z=%{z:.2f} cm<extra></extra>",
)

# ============================================================
# Magnet layout
# ============================================================

center = bbox_center
R = placement_radius_m
depth = front_back_depth_m

angles_front = np.deg2rad([90, 210, 330])
angles_mid = np.deg2rad([30, 150, 270])

magnet_positions = []
magnet_labels = []

# Front triangle A/B/C
for label, a in zip(["A", "B", "C"], angles_front):
    magnet_positions.append(center + np.array([R * np.cos(a), R * np.sin(a), -depth / 2]))
    magnet_labels.append(label)

# Back triangle D/E/F
for label, a in zip(["D", "E", "F"], angles_front + np.deg2rad(60)):
    magnet_positions.append(center + np.array([R * np.cos(a), R * np.sin(a), depth / 2]))
    magnet_labels.append(label)

# Middle ring G/H/I
for label, a in zip(["G", "H", "I"], angles_mid):
    magnet_positions.append(center + np.array([R * np.cos(a), R * np.sin(a), 0.0]))
    magnet_labels.append(label)

magnet_positions = np.asarray(magnet_positions)
magnet_positions_cm = magnet_positions * UNIT

magnet_trace = go.Scatter3d(
    x=magnet_positions_cm[:, 0],
    y=magnet_positions_cm[:, 1],
    z=magnet_positions_cm[:, 2],
    mode="markers+text",
    name="9 magnets",
    marker=dict(size=11, symbol="diamond"),
    text=magnet_labels,
    textposition="top center",
    hovertemplate="Magnet %{text}<br>x=%{x:.2f} cm<br>y=%{y:.2f} cm<br>z=%{z:.2f} cm<extra></extra>",
)

# Inward magnet arrows
arrow_traces = []
for label, pos_m in zip(magnet_labels, magnet_positions):
    direction = center - pos_m
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        continue
    direction = direction / norm

    start = pos_m * UNIT
    end = (pos_m + direction * coil_length_m) * UNIT

    arrow_traces.append(
        go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode="lines",
            name=f"{label} inward direction",
            line=dict(width=6),
            showlegend=False,
            hovertemplate=f"Magnet {label} inward direction<extra></extra>",
        )
    )

# ============================================================
# Gantry guide rings
# ============================================================

theta = np.linspace(0, 2 * np.pi, 240)
center_cm = center * UNIT
ring_radius_cm = placement_radius_m * UNIT

ring_xy = go.Scatter3d(
    x=center_cm[0] + ring_radius_cm * np.cos(theta),
    y=center_cm[1] + ring_radius_cm * np.sin(theta),
    z=np.full_like(theta, center_cm[2]),
    mode="lines",
    name="30 cm XY gantry ring",
    line=dict(width=4, dash="dash"),
    hoverinfo="skip",
)

ring_xz = go.Scatter3d(
    x=center_cm[0] + ring_radius_cm * np.cos(theta),
    y=np.full_like(theta, center_cm[1]),
    z=center_cm[2] + ring_radius_cm * np.sin(theta),
    mode="lines",
    name="30 cm XZ gantry ring",
    line=dict(width=3, dash="dot"),
    hoverinfo="skip",
)

ring_yz = go.Scatter3d(
    x=np.full_like(theta, center_cm[0]),
    y=center_cm[1] + ring_radius_cm * np.cos(theta),
    z=center_cm[2] + ring_radius_cm * np.sin(theta),
    mode="lines",
    name="30 cm YZ gantry ring",
    line=dict(width=3, dash="dot"),
    hoverinfo="skip",
)

center_trace = go.Scatter3d(
    x=[center_cm[0]],
    y=[center_cm[1]],
    z=[center_cm[2]],
    mode="markers+text",
    name="Gantry center",
    marker=dict(size=9, symbol="x"),
    text=["CENTER"],
    textposition="top center",
    hovertemplate="Gantry center<br>x=%{x:.2f} cm<br>y=%{y:.2f} cm<br>z=%{z:.2f} cm<extra></extra>",
)

# ============================================================
# Camera and scene
# ============================================================

all_display = np.vstack([all_pts_cm, magnet_positions_cm, center_cm[None, :]])
mins = all_display.min(axis=0)
maxs = all_display.max(axis=0)
plot_center = (mins + maxs) / 2.0
span = float(np.max(maxs - mins))

fig = go.Figure(
    data=[
        vessel_trace,
        radius_trace,
        endpoint_trace,
        route_trace,
        center_trace,
        magnet_trace,
        ring_xy,
        ring_xz,
        ring_yz,
        *arrow_traces,
    ]
)

fig.update_layout(
    title=(
        "Interactive 40Left85Right vessel network with 9-magnet enclosing gantry"
        "<br><sup>Drag to rotate • Scroll to zoom • Shift-drag/right-drag to pan • Double-click to reset</sup>"
    ),
    width=1500,
    height=950,
    scene=dict(
        xaxis_title="X cm",
        yaxis_title="Y cm",
        zaxis_title="Z cm",
        aspectmode="data",
        camera=dict(
            eye=dict(x=1.8, y=1.8, z=1.1),
            center=dict(x=0, y=0, z=0),
        ),
    ),
    legend=dict(
        x=0.02,
        y=0.98,
        bgcolor="rgba(255,255,255,0.75)",
    ),
    margin=dict(l=0, r=0, t=80, b=0),
)

fig.write_html(
    out_file,
    include_plotlyjs="cdn",
    full_html=True,
    config={
        "scrollZoom": True,
        "displaylogo": False,
        "responsive": True,
        "toImageButtonOptions": {
            "format": "png",
            "filename": "cerebral_with_magnets",
            "height": 1200,
            "width": 1800,
            "scale": 2,
        },
    },
)

print("Saved interactive viewer to:", out_file)
print("Open this file in your browser.")
