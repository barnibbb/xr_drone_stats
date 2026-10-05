import os
import re
import yaml

import numpy as np
from scipy.signal import savgol_filter

# =========================================================================
# Input data functions
# =========================================================================

# Transformation to world coordinates (might be good for later visualization)
def transform_point(pt, offset):
    x_real = pt['x'] - offset[0]
    y_real = pt['z'] - offset[2]
    z_real = pt['y'] - offset[1]

    # Apply 90° CW rotation around Z
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
def load_measurement(trajectory_file):
    with open(trajectory_file, 'r') as file:
        data = yaml.safe_load(file)

    trajectory = data['trajectory']
    waypoints = data['waypoints']

    print(f"Loaded {len(trajectory)} trajectory points and {len(waypoints)} waypoints.")

    return trajectory, waypoints


def waypoint_metrics(waypoints, offset):
    # Transform waypoints to world coordinates
    points = np.array(
        [transform_point(wp, offset) for wp in waypoints],
        dtype=float
    )

    # ---------------------------------------------------------
    # Track length
    # ---------------------------------------------------------
    segment_vectors = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)

    track_length = np.sum(segment_lengths)

    # ---------------------------------------------------------
    # Curvature
    # ---------------------------------------------------------
    curvatures = []

    for i in range(1, len(points) - 1):
        p1 = points[i - 1]
        p2 = points[i]
        p3 = points[i + 1]

        a = p2 - p1
        b = p3 - p2
        c = p3 - p1

        a_len = np.linalg.norm(a)
        b_len = np.linalg.norm(b)
        c_len = np.linalg.norm(c)

        # Degenerate case
        if a_len == 0 or b_len == 0 or c_len == 0:
            curvatures.append(np.nan)
            continue

        cross = np.linalg.norm(np.cross(a, b))

        curvature = 2 * cross / (a_len * b_len * c_len)

        curvatures.append(curvature)

    curvatures = np.array(curvatures)

    mean_curvature = np.nanmean(curvatures)

    return track_length, mean_curvature, curvatures


def main():
    DATA_DIR = "/home/appuser/data/samples/12_P17/joy_T2"

    trajectory_file = DATA_DIR + "/measurement_log.yaml"
    offset_file = DATA_DIR + "/offset.txt"

    offset = load_offset(offset_file)

    trajectory, waypoints = load_measurement(trajectory_file)

    track_length, mean_curvature, curvatures = waypoint_metrics(
        waypoints,
        offset
    )

    print(f"Track length: {track_length:.3f} m")
    print(f"Mean curvature: {mean_curvature:.6f} 1/m")


if __name__ == "__main__":
    main()
