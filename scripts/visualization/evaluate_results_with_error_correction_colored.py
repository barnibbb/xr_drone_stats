import yaml
import matplotlib.pyplot as plt
import numpy as np
import re
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.collections import LineCollection

class HandlerColormapLine(HandlerBase):
    def __init__(self, cmap, num_stripes=50, **kwargs):
        super().__init__(**kwargs)
        self.cmap = cmap
        self.num_stripes = num_stripes

    def create_artists(
        self, legend, orig_handle,
        xdescent, ydescent, width, height,
        fontsize, trans
    ):
        x = np.linspace(xdescent, xdescent + width, self.num_stripes + 1)
        y = ydescent + height / 2

        segments = [
            [[x[i], y], [x[i + 1], y]]
            for i in range(self.num_stripes)
        ]

        collection = LineCollection(
            segments,
            cmap=self.cmap,
            linewidth=2
        )

        collection.set_array(
            np.linspace(0, 1, self.num_stripes)
        )

        collection.set_transform(trans)

        return [collection]

# -----------------------------
# Load offset
# -----------------------------
with open("offset.txt", "r") as f:
    offset_text = f.read()

match = re.search(r"pos_x=([-.\d]+), pos_y=([-.\d]+), pos_z=([-.\d]+)", offset_text)

if not match:
    raise ValueError("Offset file format is incorrect or values not found.")

offset_x = float(match.group(1))
offset_y = float(match.group(2))
offset_z = float(match.group(3))


# -----------------------------
# Load physical cf1 trajectory
# -----------------------------
def load_cf1_pose(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    positions = []
    x, y, z = None, None, None

    for line in lines:
        line = line.strip()

        if line.startswith("- x:"):
            x = float(line.split(":")[1])
        elif line.startswith("y:"):
            y = float(line.split(":")[1])
        elif line.startswith("z:"):
            z = float(line.split(":")[1])

            if x is not None and y is not None:
                positions.append([x, y, z])
                x, y, z = None, None, None

    return np.array(positions)


physical_trajectory = load_cf1_pose("cf1_pose.txt")


# -----------------------------
# Load measurement log
# -----------------------------
with open("measurement_log.yaml", "r") as f:
    data = yaml.safe_load(f)

task_duration = data["task_duration"]
obstacle_errors = data["obstacle_placement_errors"]
collisions = data["collisions"]
trajectory = data["trajectory"]
waypoints = data["waypoints"]


# -----------------------------
# Unity -> real-world transform
# -----------------------------
def transform_unity_position(pos):
    x_unity = pos["x"]
    y_unity = pos["y"]
    z_unity = pos["z"]

    x_real = x_unity - offset_x
    y_real = z_unity - offset_z
    z_real = y_unity - offset_y

    x_rot = y_real
    y_rot = -x_real
    z_rot = z_real

    return [x_rot, y_rot, z_rot]


measurement_trajectory = np.array([
    transform_unity_position(p["position"]) for p in trajectory
])

waypoint_points = np.array([
    transform_unity_position(p) for p in waypoints
])

collision_points = np.array([
    transform_unity_position(c["position"]) for c in collisions
])


# -----------------------------
# Check data
# -----------------------------
if len(measurement_trajectory) == 0:
    raise ValueError("Measurement trajectory is empty.")

if len(physical_trajectory) == 0:
    raise ValueError("Physical cf1 trajectory is empty.")


# -----------------------------
# Align physical trajectory
# -----------------------------
z_visual_offset = 0.06

first_measurement_point = measurement_trajectory[0]

first_physical_point = physical_trajectory[0].copy()
first_physical_point[2] -= z_visual_offset

distance_vector = first_physical_point - first_measurement_point
distance_length = np.linalg.norm(distance_vector)

aligned_physical_trajectory = physical_trajectory.copy()
aligned_physical_trajectory[:, 0] -= distance_vector[0]
aligned_physical_trajectory[:, 1] -= distance_vector[1]
aligned_physical_trajectory[:, 2] -= distance_vector[2]


# -----------------------------
# Print only useful summary
# -----------------------------
summary = {
    "Task Duration (s)": task_duration,
    "Average Obstacle Placement Error (m)": np.mean(obstacle_errors) if obstacle_errors else 0.0,
    "Number of Collisions": len(collisions),
    "Initial Alignment Distance (m)": distance_length
}

print(summary)


# -----------------------------
# Plot aligned trajectories
# -----------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


# Measurement trajectory
ax.plot(
    measurement_trajectory[:, 0],
    measurement_trajectory[:, 1],
    measurement_trajectory[:, 2],
    label="Virtual trajectory",
    linewidth=1,
    color="green"
)


# -----------------------------
# Physical trajectory colored by time
# -----------------------------
physical_plot = aligned_physical_trajectory.copy()
physical_plot[:, 2] -= z_visual_offset


# Downsample ONLY for plotting
# Example: keep approximately 1000 points
max_plot_points = 1000

if len(physical_plot) > max_plot_points:
    indices = np.linspace(
        0,
        len(physical_plot) - 1,
        max_plot_points,
        dtype=int
    )

    physical_plot_reduced = physical_plot[indices]
else:
    physical_plot_reduced = physical_plot


# Time scale for reduced plotting trajectory
# First point = 0
# Last point = task_duration
physical_times = np.linspace(
    0,
    task_duration,
    len(physical_plot_reduced)
)


# Create line segments
segments = np.stack(
    [
        physical_plot_reduced[:-1],
        physical_plot_reduced[1:]
    ],
    axis=1
)


# Full color scale
norm = Normalize(
    vmin=0,
    vmax=task_duration
)

cmap = plt.get_cmap("plasma")


physical_line = Line3DCollection(
    segments,
    cmap=cmap,
    norm=norm,
    linewidth=2
)

physical_line.set_array(physical_times[:-1])

ax.add_collection3d(physical_line)


# -----------------------------
# Waypoints
# -----------------------------
if waypoint_points.size > 0:
    ax.scatter(
        waypoint_points[:, 0],
        waypoint_points[:, 1],
        waypoint_points[:, 2],
        c="blue",
        label="Waypoints",
        marker="^",
        s=30
    )


# -----------------------------
# Collisions
# -----------------------------
if collision_points.size > 0:
    ax.scatter(
        collision_points[:, 0],
        collision_points[:, 1],
        collision_points[:, 2],
        c="red",
        label="Collisions",
        marker="x",
        s=40
    )


# -----------------------------
# Colorbar
# -----------------------------
cbar = fig.colorbar(
    physical_line,
    ax=ax,
    pad=0.1,
    shrink=0.75
)

cbar.set_label("Time")


# -----------------------------
# Legend entry for colored trajectory
# -----------------------------
physical_legend = Line2D(
    [],
    [],
    label="Aligned Physical Drone Trajectory"
)

handles, labels = ax.get_legend_handles_labels()
handles.append(physical_legend)
labels.append("Aligned Physical Drone Trajectory")


# -----------------------------
# Labels
# -----------------------------
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Aligned Drone Trajectories and Events")

ax.legend(
    handles,
    labels,
    handler_map={
        physical_legend: HandlerColormapLine(cmap)
    }
)

plt.tight_layout()
plt.show()