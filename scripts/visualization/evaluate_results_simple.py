import re
import yaml

import matplotlib.pyplot as plt
import numpy as np


# --- Waypoints and collisions (assume same Unity space, convert too) ---
def transform_point(pt, offset):
    x_real = pt['x'] - offset[0]
    y_real = pt['z'] - offset[2]
    z_real = pt['y'] - offset[1]

    # Apply 90° CW rotation around Z
    x_rot = y_real
    y_rot = -x_real
    z_rot = z_real

    return [x_rot, y_rot, z_rot]


# --- Load offset from offset.txt ---
def load_offset(filename):
    with open(filename, 'r') as f:
        offset_text = f.read()

    match = re.search(r'pos_x=([-.\d]+), pos_y=([-.\d]+), pos_z=([-.\d]+)', offset_text)
    if not match:
        raise ValueError("Offset file format is incorrect or values not found.")

    offset_x = float(match.group(1))
    offset_y = float(match.group(2))
    offset_z = float(match.group(3))
    return [offset_x, offset_y, offset_z]



# --- Load physical trajectory (no transformation needed) ---
def load_cf1_pose(filename):
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



# --- Load measurement YAML file ---
def load_measurement_yaml(filename, offset):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)

    task_duration = data['task_duration']
    obstacle_errors = data['obstacle_placement_errors']
    collisions = data['collisions']
    trajectory = data['trajectory']
    waypoints = data['waypoints']

    # --- Transform Unity trajectory to world space ---
    traj_points = []
    for p in trajectory:
        x_unity = p['position']['x']
        y_unity = p['position']['y']
        z_unity = p['position']['z']
        
        # Unity → real-world + offset removal
        x_real = x_unity - offset[0]
        y_real = z_unity - offset[2]
        z_real = y_unity - offset[1]

        # Rotate 90 degrees CW around Z axis
        x_rot = y_real
        y_rot = -x_real
        z_rot = z_real

        traj_points.append([x_rot, y_rot, z_rot])

    traj_points = np.array(traj_points)

    timestamps = np.array([p['time'] for p in trajectory], dtype=np.float64)

    return traj_points, waypoints, collisions, obstacle_errors, task_duration, timestamps


def trajectory_length(points):
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    
    return np.sum(segment_lengths)



def align_physical_to_simulated(simulated_timestamps, physical_trajectory):
    # Use the simulated mission duration as the common time interval
    physical_timestamps = np.linspace(
        simulated_timestamps[0],
        simulated_timestamps[-1],
        len(physical_trajectory)
    )

    # Interpolate physical trajectory onto simulated timestamps
    physical_aligned = np.column_stack([
        np.interp(
            simulated_timestamps,
            physical_timestamps,
            physical_trajectory[:, axis]
        )
        for axis in range(3)
    ])

    return physical_aligned




def main():

    DATA_DIR = "/home/appuser/data/samples/12_P17/joy_T2"

    trajectory_file = DATA_DIR + "/measurement_log.yaml"
    offset_file = DATA_DIR + "/offset.txt"
    physical_file = DATA_DIR + "/cf1_pose.txt"

    offset = load_offset(offset_file)

    physical_trajectory = load_cf1_pose(physical_file)

    traj_points, waypoints, collisions, obstacle_errors, task_duration, timestamps = load_measurement_yaml(trajectory_file, offset)
    waypoint_points = np.array([transform_point(p, offset) for p in waypoints])
    collision_points = np.array([transform_point(c['position'], offset) for c in collisions])
    
    


    # --- Output summary ---
    summary = {
        "Task Duration (s)": task_duration,
        "Average Obstacle Placement Error (m)": np.mean(obstacle_errors) if obstacle_errors else 0.0,
        "Number of Collisions": len(collisions)
    }

    print(summary)

    # --- Plotting ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Simulated trajectory (after correction)
    ax.plot(traj_points[:, 0], traj_points[:, 1], traj_points[:, 2],
            label='Simulated Trajectory', linewidth=1, color='blue')

    # Physical drone trajectory (raw cf1_pose)
    ax.plot(physical_trajectory[:, 0], physical_trajectory[:, 1], physical_trajectory[:, 2], #-0.06,
            label='Physical Drone Trajectory', linewidth=1, color='orange')

    # Waypoints
    if waypoint_points.size > 0:
        ax.scatter(waypoint_points[:, 0], waypoint_points[:, 1], waypoint_points[:, 2],
                c='green', label='Waypoints', marker='^', s=30)

    # Collisions
    if collision_points.size > 0:
        ax.scatter(collision_points[:, 0], collision_points[:, 1], collision_points[:, 2],
                c='red', label='Collisions', marker='x', s=40)

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Drone Trajectories and Events')
    ax.legend()

    plt.tight_layout()
    plt.show()

    print(summary)

    print("\n========== TRAJECTORY COMPARISON ==========")

    print("\nSimulated:")
    print("  Number of points:", len(traj_points))
    print("  Start:", traj_points[0])
    print("  End:  ", traj_points[-1])

    print("\nPhysical:")
    print("  Number of points:", len(physical_trajectory))
    print("  Start:", physical_trajectory[0])
    print("  End:  ", physical_trajectory[-1])

    start_difference = physical_trajectory[0] - traj_points[0]

    print("\nStart position difference:")
    print("  X:", start_difference[0])
    print("  Y:", start_difference[1])
    print("  Z:", start_difference[2])
    print("  Distance:", np.linalg.norm(start_difference))


    simulated_length = trajectory_length(traj_points)
    physical_length = trajectory_length(physical_trajectory)

    print("\n========== TRAJECTORY LENGTH ==========")
    print(f"Simulated trajectory: {simulated_length:.3f} m")
    print(f"Physical trajectory:  {physical_length:.3f} m")
    print(f"Difference:          {physical_length - simulated_length:.3f} m")
    print(
        f"Ratio (physical / simulated): "
        f"{physical_length / simulated_length:.3f}"
    )

    print("\n========== TIMING ==========")

    print("Simulated duration:",
        timestamps[-1] - timestamps[0])

    print("Physical duration assuming 120 Hz:",
        (len(physical_trajectory) - 1) / 120.0)

    print("Simulated mean frequency:",
        1 / np.mean(np.diff(timestamps)))

    print("Simulated median frequency:",
        1 / np.median(np.diff(timestamps)))


    physical_aligned = align_physical_to_simulated(
        timestamps,
        physical_trajectory
    )

    difference = physical_aligned - traj_points

    error = np.linalg.norm(difference, axis=1)

    print("\n========== TRAJECTORY ALIGNMENT ==========")
    print(f"Mean error:   {np.mean(error):.4f} m")
    print(f"Median error: {np.median(error):.4f} m")
    print(f"RMS error:    {np.sqrt(np.mean(error**2)):.4f} m")
    print(f"Max error:    {np.max(error):.4f} m")

    print("\nAxis errors:")
    print(f"X mean abs error: {np.mean(np.abs(difference[:, 0])):.4f} m")
    print(f"Y mean abs error: {np.mean(np.abs(difference[:, 1])):.4f} m")
    print(f"Z mean abs error: {np.mean(np.abs(difference[:, 2])):.4f} m")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(
        traj_points[:, 0],
        traj_points[:, 1],
        traj_points[:, 2],
        label='Simulated',
        linewidth=1
    )

    ax.plot(
        physical_aligned[:, 0],
        physical_aligned[:, 1],
        physical_aligned[:, 2],
        label='Physical',
        linewidth=1
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    plt.tight_layout()
    plt.show()


    progress = np.linspace(0, 1, len(error))

    plt.figure(figsize=(12, 5))
    plt.plot(progress, error)

    plt.xlabel("Mission progress")
    plt.ylabel("Position error [m]")
    plt.title("Simulated vs Physical Position Error")
    plt.grid()
    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    main()
