import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ------------------------------------------------------------
# File to visualize
# ------------------------------------------------------------
vessel_file = Path("data/0050_H_CERE_H/real_vessel_segments_cm_corrected.pkl")

# Magnet/device display parameters
# These match the approximate body-scale moving adaptive enclosing gantry scale.
placement_radius_m = 0.300   # 30 cm from gantry center
coil_length_m = 0.160        # 16 cm
coil_radius_m = 0.070        # 7 cm

# Use centimeters for plotting
UNIT = 100.0

# ------------------------------------------------------------
# Load vessel segments
# ------------------------------------------------------------
with open(vessel_file, "rb") as f:
    segments = pickle.load(f)

starts, ends, radii = [], [], []

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

print("Vessel:", vessel_file)
print("Bounding box size cm:", bbox_size * UNIT)
print("Radius range cm:", radii.min() * UNIT, "to", radii.max() * UNIT)
print("Gantry center cm:", bbox_center * UNIT)
print("Placement radius cm:", placement_radius_m * UNIT)

# ------------------------------------------------------------
# Vessel line geometry
# ------------------------------------------------------------
lines_cm = np.stack([starts, ends], axis=1) * UNIT

# Downsample for smoother interactive rendering if needed
stride = max(1, len(lines_cm) // 12000)
plot_lines = lines_cm[::stride]

# ------------------------------------------------------------
# Terminal endpoint detection
# ------------------------------------------------------------
counts = {}
points = {}

for p in all_pts_m:
    key = tuple(np.round(p, 6))
    counts[key] = counts.get(key, 0) + 1
    points[key] = p

terminal_m = np.array([points[k] for k, c in counts.items() if c == 1])
terminal_cm = terminal_m * UNIT if len(terminal_m) else np.empty((0, 3))

# ------------------------------------------------------------
# 9-magnet enclosing gantry layout
# Approximate triangular-prism style:
# front triangle, back triangle, and middle ring.
# ------------------------------------------------------------
center = bbox_center
R = placement_radius_m
depth = 0.18  # front/back split in meters for visualization

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

# Middle magnets G/H/I
for label, a in zip(["G", "H", "I"], angles_mid):
    magnet_positions.append(center + np.array([R * np.cos(a), R * np.sin(a), 0.0]))
    magnet_labels.append(label)

magnet_positions = np.asarray(magnet_positions)
magnet_positions_cm = magnet_positions * UNIT

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")

# Vessel
lc = Line3DCollection(plot_lines, linewidths=0.55, alpha=0.75)
ax.add_collection3d(lc)

# Endpoints
if len(terminal_cm):
    ax.scatter(
        terminal_cm[:, 0],
        terminal_cm[:, 1],
        terminal_cm[:, 2],
        s=35,
        marker="o",
        label="terminal endpoints",
    )

# Gantry center
center_cm = center * UNIT
ax.scatter(
    [center_cm[0]],
    [center_cm[1]],
    [center_cm[2]],
    s=80,
    marker="x",
    label="gantry center",
)

# Magnets
ax.scatter(
    magnet_positions_cm[:, 0],
    magnet_positions_cm[:, 1],
    magnet_positions_cm[:, 2],
    s=120,
    marker="^",
    label="magnets",
)

# Magnet labels and inward direction arrows
for label, pos_m, pos_cm in zip(magnet_labels, magnet_positions, magnet_positions_cm):
    ax.text(pos_cm[0], pos_cm[1], pos_cm[2], f" {label}", fontsize=11)

    direction_m = center - pos_m
    norm = np.linalg.norm(direction_m)
    if norm > 1e-12:
        direction_m = direction_m / norm

    arrow_len_cm = coil_length_m * UNIT
    ax.quiver(
        pos_cm[0],
        pos_cm[1],
        pos_cm[2],
        direction_m[0] * arrow_len_cm,
        direction_m[1] * arrow_len_cm,
        direction_m[2] * arrow_len_cm,
        linewidth=1.2,
        arrow_length_ratio=0.25,
    )

# Draw a transparent-ish gantry radius guide in XY plane
theta = np.linspace(0, 2 * np.pi, 240)
ring_x = center_cm[0] + placement_radius_m * UNIT * np.cos(theta)
ring_y = center_cm[1] + placement_radius_m * UNIT * np.sin(theta)
ring_z = np.full_like(theta, center_cm[2])
ax.plot(ring_x, ring_y, ring_z, linestyle="--", linewidth=1.0, label="30 cm gantry radius")

# Equal axis scaling
all_display_pts = np.vstack([all_pts_cm, magnet_positions_cm, center_cm[None, :]])
mins = all_display_pts.min(axis=0)
maxs = all_display_pts.max(axis=0)
plot_center = (mins + maxs) / 2.0
span = float(np.max(maxs - mins)) / 2.0
span *= 1.10

ax.set_xlim(plot_center[0] - span, plot_center[0] + span)
ax.set_ylim(plot_center[1] - span, plot_center[1] + span)
ax.set_zlim(plot_center[2] - span, plot_center[2] + span)

ax.set_xlabel("X cm")
ax.set_ylabel("Y cm")
ax.set_zlabel("Z cm")
ax.set_title("Interactive 3D cerebral vessel with 9-magnet enclosing gantry")
ax.legend(loc="upper left")

print()
print("Interactive viewer opened.")
print("Drag to rotate. Scroll/pinch to zoom.")

plt.tight_layout()
plt.show()
