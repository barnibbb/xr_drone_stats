import re
import yaml

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D




# =========================================================================
# Input data functions
# =========================================================================

# Transformation to world coordinates (might be good for later visualization)
def transform_point(pt, offset):
    x_real = pt["x"] - offset[0]
    y_real = pt["z"] - offset[2]
    z_real = pt["y"] - offset[1]

    # 90° clockwise rotation around Z
    x_rot = y_real
    y_rot = -x_real
    z_rot = z_real

    return [x_rot, y_rot, z_rot]


# Load offset file
def load_offset(filename):
    with open(filename, 'r') as f:
        offset_text = f.read()

    match = re.search(r'pos_x=([-.\d]+), pos_y=([-.\d]+), pos_z=([-.\d]+)', offset_text)

    if not match:
        raise ValueError("Offset file format is incorrect.")

    offset_x = float(match.group(1))
    offset_y = float(match.group(2))
    offset_z = float(match.group(3))

    return [offset_x, offset_y, offset_z]


# Load measurement log --> trajectory and waypoints
def load_virtual(trajectory_file, offset_file):
    offset = load_offset(offset_file)

    with open(trajectory_file, 'r') as file:
        data = yaml.safe_load(file)

    trajectory = data['trajectory']
    waypoints = data['waypoints']

    timestamps = np.array([p['time'] for p in trajectory], dtype=np.float64)
    positions = np.array([transform_point(p['position'], offset) for p in trajectory], dtype=np.float64)
    waypoints = np.array([transform_point(p, offset) for p in waypoints], dtype=np.float64)

    return timestamps, positions, waypoints


# =========================================================================
# Path functions
# =========================================================================

def cumulative_path_length(points):
    distances = np.linalg.norm(
        np.diff(points, axis=0),
        axis=1
    )

    return np.concatenate([
        [0.0],
        np.cumsum(distances)
    ])


def interpolate_path_position(points, cumulative_distance, target_distance):
    index = np.searchsorted(
        cumulative_distance,
        target_distance
    )

    if index == 0:
        return points[0]

    if index >= len(points):
        return points[-1]

    d0 = cumulative_distance[index - 1]
    d1 = cumulative_distance[index]

    alpha = (target_distance - d0) / (d1 - d0)

    return (
        points[index - 1]
        + alpha * (points[index] - points[index - 1])
    )


# =========================================================================
# Create reference path and midpoints
# =========================================================================

def create_reference_path(positions, waypoints):
    start = positions[0]
    end = positions[-1]

    reference_points = np.vstack([
        start,
        waypoints,
        end
    ])

    return reference_points


def create_midpoints(reference_points):
    midpoints = []

    for i in range(len(reference_points) - 1):
        midpoint = (
            reference_points[i]
            + reference_points[i + 1]
        ) / 2.0

        midpoints.append(midpoint)

    return np.array(midpoints)


# =========================================================================
# Divide actual path
# =========================================================================

def divide_actual_path(positions, reference_points, midpoints):

    # ---------------------------------------------------------
    # Reference path length
    # ---------------------------------------------------------

    reference_cumulative = cumulative_path_length(
        reference_points
    )

    reference_total_distance = reference_cumulative[-1]

    # ---------------------------------------------------------
    # Find where each midpoint lies along the reference path
    # ---------------------------------------------------------

    midpoint_distances = []

    for i in range(len(midpoints)):

        segment_start = reference_cumulative[i]
        segment_end = reference_cumulative[i + 1]

        midpoint_distance = (
            segment_start
            + (segment_end - segment_start) / 2.0
        )

        midpoint_distances.append(midpoint_distance)

    midpoint_distances = np.array(midpoint_distances)

    # Relative positions along reference path
    midpoint_ratios = (
        midpoint_distances
        / reference_total_distance
    )

    # ---------------------------------------------------------
    # Actual path length
    # ---------------------------------------------------------

    actual_cumulative = cumulative_path_length(
        positions
    )

    actual_total_distance = actual_cumulative[-1]

    # Same relative positions along actual path
    actual_midpoint_distances = (
        midpoint_ratios
        * actual_total_distance
    )

    # Find actual trajectory positions corresponding
    # to the reference midpoints
    actual_boundaries = np.array([
        interpolate_path_position(
            positions,
            actual_cumulative,
            distance
        )
        for distance in actual_midpoint_distances
    ])

    return (
        actual_boundaries,
        actual_cumulative
    )


# =========================================================================
# Visualization
# =========================================================================

def plot_divided_paths(positions, waypoints):

    # ---------------------------------------------------------
    # Reference path
    # ---------------------------------------------------------

    reference_points = create_reference_path(
        positions,
        waypoints
    )

    # ---------------------------------------------------------
    # Midpoints
    #
    # S -> C1 -> W1 -> C2 -> W2 -> C3 -> W3 -> C4 -> E
    #
    # But the actual colored segments are:
    #
    # S  -> C1
    # C1 -> C2    (W1)
    # C2 -> C3    (W2)
    # C3 -> C4    (W3)
    # C4 -> E
    # ---------------------------------------------------------

    midpoints = create_midpoints(
        reference_points
    )

    # ---------------------------------------------------------
    # Divide actual trajectory using midpoint ratios
    # ---------------------------------------------------------

    actual_boundaries, actual_cumulative = divide_actual_path(
        positions,
        reference_points,
        midpoints
    )

    actual_total_distance = actual_cumulative[-1]

    # Find cumulative distances of actual boundaries
    actual_boundary_distances = []

    for boundary in actual_boundaries:

        distances = np.linalg.norm(
            positions - boundary,
            axis=1
        )

        index = np.argmin(distances)

        actual_boundary_distances.append(
            actual_cumulative[index]
        )

    actual_boundary_distances = np.array(
        actual_boundary_distances
    )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    n_segments = len(waypoints) + 1

    colors = plt.get_cmap("tab20")(
        np.linspace(0, 1, n_segments)
    )

    # =========================================================
    # Actual trajectory
    # =========================================================

    # Add start and end to the actual boundaries
    actual_boundaries_full = np.vstack([
        positions[0],
        actual_boundaries,
        positions[-1]
    ])

    boundary_distances = np.concatenate([
        [0.0],
        actual_boundary_distances,
        [actual_total_distance]
    ])

    for i in range(n_segments):

        start_distance = boundary_distances[i]
        end_distance = boundary_distances[i + 1]

        start_idx = np.searchsorted(
            actual_cumulative,
            start_distance,
            side="left"
        )

        end_idx = np.searchsorted(
            actual_cumulative,
            end_distance,
            side="right"
        )

        segment = positions[start_idx:end_idx]

        segment = np.vstack([
            actual_boundaries_full[i],
            segment,
            actual_boundaries_full[i + 1]
        ])

        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=colors[i],
            linewidth=2.5,
            label=f"Segment {i}"
        )

    # =========================================================
    # Reference path
    #
    # IMPORTANT:
    # Reference is also divided using the MIDPOINTS.
    # =========================================================

    reference_boundaries = np.vstack([
        reference_points[0],
        midpoints,
        reference_points[-1]
    ])

    for i in range(n_segments):

        segment = reference_boundaries[i:i + 2]

        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=colors[i],
            linestyle="--",
            linewidth=2
        )

    # =========================================================
    # Waypoints
    # =========================================================

    for i, waypoint in enumerate(waypoints):

        ax.scatter(
            waypoint[0],
            waypoint[1],
            waypoint[2],
            color="black",
            s=100,
            marker="o",
            edgecolor="white",
            linewidth=1.5
        )

        ax.text(
            waypoint[0],
            waypoint[1],
            waypoint[2],
            f"  W{i + 1}",
            fontsize=10
        )

    # =========================================================
    # Start
    # =========================================================

    start = positions[0]

    ax.scatter(
        start[0],
        start[1],
        start[2],
        color="black",
        s=140,
        marker="^",
        label="Start"
    )

    ax.text(
        start[0],
        start[1],
        start[2],
        "  S",
        fontsize=10
    )

    # =========================================================
    # End
    # =========================================================

    end = positions[-1]

    ax.scatter(
        end[0],
        end[1],
        end[2],
        color="black",
        s=140,
        marker="X",
        label="End"
    )

    ax.text(
        end[0],
        end[1],
        end[2],
        "  E",
        fontsize=10
    )

    # =========================================================
    # Formatting
    # =========================================================

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    ax.set_title(
        "Reference and Actual Trajectory Division"
    )

    ax.legend()

    plt.tight_layout()
    plt.show()

    return {
        "reference_points": reference_points,
        "midpoints": midpoints,
        "actual_boundaries": actual_boundaries,
    }


# =========================================================================
# Example
# =========================================================================

def main():
    DATA_DIR = "/home/appuser/data/samples/2_P40/pinch_T2"

    trajectory_file = DATA_DIR + "/measurement_log.yaml"
    offset_file = DATA_DIR + "/offset.txt"

    timestamps, positions, waypoints = load_virtual(
        trajectory_file,
        offset_file
    )

    result = plot_divided_paths(
        positions,
        waypoints
    )


if __name__ == "__main__":
    main()

