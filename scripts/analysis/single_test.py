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


def load_physical(filename):
    with open(filename, 'r') as f:
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


# =========================================================================
# Alignment with physical trajectory
# =========================================================================

def align_physical_trajectory(simulated_timestamps, physical_positions):
    physical_timestamps = np.linspace(simulated_timestamps[0], simulated_timestamps[-1], len(physical_positions))
    physical_aligned = np.column_stack([np.interp(simulated_timestamps, physical_timestamps, physical_positions[:, i]) for i in range(3)])

    return physical_aligned, physical_timestamps



# =========================================================================
# Savitzky-Golay smoothing
# =========================================================================

def smooth(positions, timestamps):
    smoothing_time = 0.3
    polyorder = 3

    dt = np.mean(np.diff(timestamps))

    window_length = int(round(smoothing_time / dt))

    # Savitzky-Golay requires an odd window length
    if window_length % 2 == 0:
        window_length += 1

    smoothed_positions = savgol_filter(positions, window_length, polyorder, deriv=0, delta=dt, axis=0)
    smoothed_velocity = savgol_filter(positions, window_length, polyorder, deriv=1, delta=dt, axis=0)
    smoothed_accleration = savgol_filter(positions, window_length, polyorder, deriv=2, delta=dt, axis=0)
    smoothed_jerk = savgol_filter(positions, window_length, polyorder, deriv=3, delta=dt, axis=0)

    return smoothed_positions, smoothed_velocity, smoothed_accleration, smoothed_jerk




# =========================================================================
# Reference path comparison
# =========================================================================

def cumulative_path_length(points):
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
    return cumulative_length


def interpolate_path_positions(points, cumulative_distance, target_distance):
    index = np.searchsorted(cumulative_distance, target_distance)

    if index == 0:
        return points[0]
    elif index >= len(points):
        return points[-1]

    d0 = cumulative_distance[index - 1]
    d1 = cumulative_distance[index]

    alpha = (target_distance - d0) / (d1 - d0)

    interpolated_position = points[index - 1] + alpha * (points[index] - points[index - 1])

    return interpolated_position


def create_midpoints(reference_points):
    midpoints = []
    for i in range(len(reference_points) - 1):
        midpoint = (reference_points[i] + reference_points[i + 1]) / 2
        midpoints.append(midpoint)
    return np.array(midpoints)


def divide_actual_path(positions, reference_points):
    reference_cumulative = cumulative_path_length(reference_points)
    reference_total_length = reference_cumulative[-1]

    midpoint_distances = []

    for i in range(len(reference_points) - 1):
        start = reference_cumulative[i]
        end = reference_cumulative[i + 1]

        midpoint_distance = (start + (end - start) / 2)
        midpoint_distances.append(midpoint_distance)

    midpoint_distances = np.array(midpoint_distances)
    midpoint_ratios = midpoint_distances / reference_total_length

    actual_cumulative = cumulative_path_length(positions)
    actual_total_length = actual_cumulative[-1]

    actual_midpoint_distances = midpoint_ratios * actual_total_length
    actual_boundaries = np.array([
        interpolate_path_positions(positions, actual_cumulative, d) for d in actual_midpoint_distances])

    return actual_boundaries, actual_cumulative







# =========================================================================
# Evaluation metrics
# =========================================================================

def smoothness_metrics(positions, timestamps):
    dt = np.diff(timestamps)
    velocity = np.diff(positions, axis=0) / dt[:, None]
    speed = np.linalg.norm(velocity, axis=1)
    speed_cv = np.std(speed) / np.mean(speed)

    velocity_dt = (dt[:-1] + dt[1:]) / 2.0
    acceleration = np.diff(velocity, axis=0) / velocity_dt[:, None]

    acceleration_dt = (velocity_dt[:-1] + velocity_dt[1:]) / 2.0
    jerk = np.diff(acceleration, axis=0) / acceleration_dt[:, None]
    jerk_magnitude = np.linalg.norm(jerk, axis=1)
    rms_jerk = np.sqrt(np.mean(jerk_magnitude**2))

    return speed_cv, rms_jerk



def waypoint_metrics_ordered(positions, waypoints):
    n_points = len(positions)
    n_waypoints = len(waypoints)

    # Matrix of distances between each position and each waypoint
    distances = np.linalg.norm(positions[:, None, :] - waypoints[None, :, :], axis=2)

    # Dynamic programming to find the best matching of waypoints to positions
    dp = np.full((n_points, n_waypoints), np.inf)

    # Initialize the previous index matrix to reconstruct the path
    previous = np.full((n_points, n_waypoints), -1, dtype=int)

    dp[:, 0] = distances[:, 0]

    # Fill the DP table
    for j in range(1, n_waypoints):
        for i in range(j, n_points):
            # All possible previous positions for the current waypoint
            previous_costs = dp[:i, j - 1]

            # Find the index of the minimum cost from previous positions
            best_previous = np.argmin(previous_costs)

            # Update the DP table with the minimum cost to reach the current waypoint
            dp[i, j] = (distances[i, j] + previous_costs[best_previous])

            previous[i, j] = best_previous

    last_index = np.argmin(dp[:, -1])

    matched_indices = np.zeros(n_waypoints, dtype=int)
    matched_indices[-1] = last_index

    # Backtrack to find the matched indices for all waypoints
    for j in range(n_waypoints - 1, 0, -1):
        matched_indices[j - 1] = previous[matched_indices[j], j]

    waypoint_errors = distances[matched_indices, np.arange(n_waypoints)]

    return np.array(waypoint_errors)


def waypoint_metrics_simple(smoothed_positions, waypoints_positions):
    waypoint_errors = []

    for waypoint in waypoints_positions:
        distances = np.linalg.norm(
            smoothed_positions - waypoint,
            axis=1
        )

        waypoint_errors.append(np.min(distances))

    waypoint_errors = np.array(waypoint_errors)

    return waypoint_errors


def waypoint_metrics_segment(positions, waypoints, reference_points):
    actual_boundaries, actual_cumulative = divide_actual_path(positions, reference_points)

    actual_total_distance = actual_cumulative[-1]

    actual_boundary_distances = []

    for boundary in actual_boundaries:
        distances = np.linalg.norm(positions - boundary, axis=1)
        index = np.argmin(distances)
        actual_boundary_distances.append(actual_cumulative[index])

    actual_boundary_distances = np.array(actual_boundary_distances)

    boundary_distances = np.concatenate([[0], actual_boundary_distances, [actual_total_distance]])

    errors = []

    for i, waypoint in enumerate(waypoints):
        start_distance = boundary_distances[i + 1]
        end_distance = boundary_distances[i + 2]

        start_idx = np.searchsorted(actual_cumulative, start_distance, side='left')
        end_idx = np.searchsorted(actual_cumulative, end_distance, side='right')

        segment = positions[start_idx:end_idx]

        distances = np.linalg.norm(segment - waypoint, axis=1)
        errors.append(np.min(distances))

    return np.array(errors)


def reference_path_error(positions, reference_points, n_samples=100):
    cumulative_distance = cumulative_path_length(reference_points)
    total_distance = cumulative_distance[-1]

    sample_distances = np.linspace(0, total_distance, n_samples)

    sampled_points = np.array([
        interpolate_path_positions(reference_points, cumulative_distance, d) for d in sample_distances])

    waypoint_errors = waypoint_metrics_ordered(positions, sampled_points)

    return np.median(waypoint_errors)





# Single trajectory evaluation
def evaluate_trajectory(trajectory_file, offset_file, physical_file):
    # Load input
    offset = load_offset(offset_file)
    trajectory, waypoints = load_measurement(trajectory_file)
    physical_positions = load_physical(physical_file)

    # Extract timestamps and positions from trajectory
    timestamps = np.array([p['time'] for p in trajectory], dtype=np.float64)
    positions = np.array([transform_point(p['position'], offset) for p in trajectory], dtype=np.float64)
    waypoints_positions = np.array([transform_point(wp, offset) for wp in waypoints], dtype=np.float64)


    # Smoothing
    smoothed_positions, smoothed_velocity, smoothed_accleration, smoothed_jerk = smooth(positions, timestamps)


    #1 Mission time
    mission_time = timestamps[-1] - timestamps[0]


    #2 Total path length
    position_diffs = np.diff(smoothed_positions, axis=0)
    segment_distances = np.linalg.norm(position_diffs, axis=1)
    total_distance = np.sum(segment_distances)


    #3 Curvature
    speed = np.linalg.norm(smoothed_velocity, axis=1)
    cross_product = np.cross(smoothed_velocity, smoothed_accleration)
    curvature = np.linalg.norm(cross_product, axis=1) / (speed**3 + 1e-8)  # Avoid division by zero 
    mean_curvature = np.mean(curvature)


    #4 Speed variability
    speed_cv = np.std(speed) / np.mean(speed)
    

    #5 RMS jerk
    jerk_magnitude = np.linalg.norm(smoothed_jerk, axis=1)
    rms_jerk = np.sqrt(np.mean(jerk_magnitude**2))


    #6 Path efficiency
    ref_points = np.vstack([smoothed_positions[0], waypoints_positions, smoothed_positions[-1]])
    ref_diffs = np.diff(ref_points, axis=0)
    ref_distances = np.linalg.norm(ref_diffs, axis=1)
    total_ref_distance = np.sum(ref_distances)
    path_efficiency = total_ref_distance / total_distance


    #7 Waypoint errors
    waypoint_errors_ordered = waypoint_metrics_ordered(smoothed_positions, waypoints_positions)
    median_waypoint_error_ordered = np.median(waypoint_errors_ordered)
    # max_waypoint_error_ordered = np.max(waypoint_errors_ordered)

    # waypoint_errors_simple = waypoint_metrics_simple(smoothed_positions, waypoints_positions)
    # median_waypoint_error_simple = np.median(waypoint_errors_simple)
    # max_waypoint_error_simple = np.max(waypoint_errors_simple)


    #8 Segment-wise error
    waypoint_errors_segment = waypoint_metrics_segment(smoothed_positions, waypoints_positions, ref_points)
    median_waypoint_error_segment = np.median(waypoint_errors_segment)
    # max_waypoint_error_segment = np.max(waypoint_errors_segment)


    #9 Reference path error
    # median_reference_path_error = reference_path_error(smoothed_positions, ref_points)


    return {
        "mission_time": mission_time,
        "total_distance": total_distance,
        "mean_curvature": mean_curvature,
        "speed_cv": speed_cv,
        "rms_jerk": rms_jerk,
        "path_efficiency": path_efficiency,
        "ordered_error": median_waypoint_error_ordered,
        "segment_error": median_waypoint_error_segment
    }


def main():
    trial_path = "/home/appuser/data/samples/39_P12/holo_joy_T2"

    trajectory_file = os.path.join(trial_path, "measurement_log.yaml")
    offset_file = os.path.join(trial_path, "offset.txt")
    physical_file = os.path.join(trial_path, "cf1_pose.txt")

    metrics = evaluate_trajectory(trajectory_file, offset_file, physical_file)
    # print(metrics)

    offset = load_offset(offset_file)
    trajectory, waypoints = load_measurement(trajectory_file)

    


if __name__ == "__main__":
    main()





# =========================================================================
# Reference path
# =========================================================================

# Generating refrence path
# def generate_reference_path(waypoints, num_samples=REFERENCE_SAMPLES):
#     waypoint_distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
#     cumulative_distances = np.concatenate([[0], np.cumsum(waypoint_distances)])

#     spline_x = CubicSpline(cumulative_distances, waypoints[:, 0])
#     spline_y = CubicSpline(cumulative_distances, waypoints[:, 1])
#     spline_z = CubicSpline(cumulative_distances, waypoints[:, 2])

#     s_fine = np.linspace(cumulative_distances[0], cumulative_distances[-1], num_samples)

#     reference_path = np.column_stack([spline_x(s_fine), spline_y(s_fine), spline_z(s_fine)])

#     return reference_path


# def generate_reference_path2(waypoints, num_samples=REFERENCE_SAMPLES):
#     waypoint_distances = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
#     total_length = np.sum(waypoint_distances)

#     samples_per_segment = np.maximum(2, np.round(num_samples * (waypoint_distances / total_length)).astype(int))

#     reference_points = []

#     for i in range(len(waypoints) - 1):
#         segment = np.linspace(waypoints[i], waypoints[i + 1], samples_per_segment[i], endpoint=False)
#         reference_points.append(segment)

#     reference_points.append(waypoints[-1][None, :])
#     reference_path = np.vstack(reference_points)

#     return reference_path







