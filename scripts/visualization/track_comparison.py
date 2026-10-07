import os
import re
import yaml

import pandas as pd
import numpy as np

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
    points = np.array([transform_point(wp, offset) for wp in waypoints], dtype=float)

    # Basics
    n_waypoints = len(points)

    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0)

    dimensions = max_xyz - min_xyz


    # Segment geometry
    segment_vectors = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)

    track_length = np.sum(segment_lengths)

    direct_distance = np.linalg.norm(points[-1] - points[0])

    segment_mean = np.mean(segment_lengths)
    segment_median = np.median(segment_lengths)
    segment_std = np.std(segment_lengths, ddof=1)
    segment_min = np.min(segment_lengths)
    segment_max = np.max(segment_lengths)

    # Turning angles
    turning_angles = []

    for i in range(len(segment_vectors) - 1):
        v1 = segment_vectors[i]
        v2 = segment_vectors[i + 1]

        # Normalize vectors
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        # Calculate angle in radians
        cosine = np.dot(v1, v2) / (norm1 * norm2)
        cosine = np.clip(cosine, -1.0, 1.0)  # Ensure within valid range
        angle_rad = np.arccos(cosine)

        # Convert to degrees
        angle_deg = np.degrees(angle_rad)

        turning_angles.append(angle_deg)

    turning_angles = np.array(turning_angles)

    valid_angles = turning_angles[~np.isnan(turning_angles)]

    if len(valid_angles) > 0:
        mean_turning_angle = np.mean(valid_angles)
        median_turning_angle = np.median(valid_angles)
        total_turning_angle = np.sum(valid_angles)



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

    valid_curvatures = curvatures[~np.isnan(curvatures)]

    if len(valid_curvatures) > 0:
        mean_curvature = np.mean(valid_curvatures)
        median_curvature = np.median(valid_curvatures)
        std_curvature = np.std(valid_curvatures, ddof=1)
        max_curvature = np.max(valid_curvatures)

    return {
        "n_waypoints": n_waypoints,
        "track_length": track_length,
        "direct_distance": direct_distance,
        "segment_mean": segment_mean,
        "segment_median": segment_median,
        "mean_turning_angle": mean_turning_angle,
        "median_turning_angle": median_turning_angle,
        "total_turning_angle": total_turning_angle,
        "mean_curvature": mean_curvature,
        "median_curvature": median_curvature
    }


def main():
    DATA_DIR_1 = "/home/appuser/data/samples/12_P17/joy_T2"
    DATA_DIR_2 = "/home/appuser/data/samples/12_P17/palm_T1"

    tracks = { "T1": DATA_DIR_1, "T2": DATA_DIR_2 }

    results = []

    for track_name, track_dir in tracks.items():
        offset_file = os.path.join(track_dir, "offset.txt")
        trajectory_file = os.path.join(track_dir, "measurement_log.yaml")

        offset = load_offset(offset_file)
        _, waypoints = load_measurement(trajectory_file)

        metrics = waypoint_metrics(waypoints, offset)
        metrics["track"] = track_name

        results.append(metrics)

    results_df = pd.DataFrame(results)
    columns = ["track"] + [col for col in results_df.columns if col != "track"]

    results_df = results_df[columns]
    results_df.to_csv("/home/appuser/data/track_metrics.csv", index=False)


if __name__ == "__main__":
    main()
