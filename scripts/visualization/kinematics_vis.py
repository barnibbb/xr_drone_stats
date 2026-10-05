import os
import re
import yaml

import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import savgol_filter



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



def compute_kinematics(timestamps, positions):
    dt = np.diff(timestamps)

    velocity = np.diff(positions, axis=0) / dt[:, None]
    speed = np.linalg.norm(velocity, axis=1)

    velocity_dt = (dt[:-1] + dt[1:]) / 2.0
    acceleration = (np.diff(velocity, axis=0) / velocity_dt[:, None])
    acceleration_magnitude = np.linalg.norm(acceleration, axis=1)

    acceleration_dt = (velocity_dt[:-1] + velocity_dt[1:]) / 2.0
    jerk = (np.diff(acceleration, axis=0) / acceleration_dt[:, None])
    jerk_magnitude = np.linalg.norm(jerk, axis=1)

    return {
        "dt": dt,
        "velocity": velocity,
        "speed": speed,
        "acceleration": acceleration,
        "acceleration_magnitude": acceleration_magnitude,
        "jerk": jerk,
        "jerk_magnitude": jerk_magnitude
    }



def smooth(positions, timestamps):
    window_length = 21
    polyorder = 3

    dt = np.mean(np.diff(timestamps))

    velocity = savgol_filter(positions, window_length, polyorder, deriv=1, delta=dt, axis=0)
    acceleration = savgol_filter(positions, window_length, polyorder, deriv=2, delta=dt, axis=0)
    jerk = savgol_filter(positions, window_length, polyorder, deriv=3, delta=dt, axis=0)

    speed = np.linalg.norm(velocity, axis=1)
    acceleration_magnitude = np.linalg.norm(acceleration, axis=1)
    jerk_magnitude = np.linalg.norm(jerk, axis=1)


    return {
        "velocity": velocity,
        "speed": speed,
        "acceleration": acceleration,
        "acceleration_magnitude": acceleration_magnitude,
        "jerk": jerk,
        "jerk_magnitude": jerk_magnitude
    }


def add_mean_line(ax, values):
    median_value = np.median(values)

    ax.axhline(
        median_value,
        linestyle="--",
        label=f"Median = {median_value:.3f}"
    )

    ax.legend()



def main():
    # Load trajectory
    trial_path = "/home/appuser/data/samples/18_P08/palm_T2"
    # trial_path = "/home/appuser/data/samples/39_P12/holo_joy_T2"
    # trial_path = "/home/appuser/data/samples/22_P74/holo_joy_T2"
    # trial_path = "/home/appuser/data/samples/22_P74/palm_T2"

    virtual_file = os.path.join(trial_path, "measurement_log.yaml")
    physical_file = os.path.join(trial_path, "cf1_pose.txt")
    offset_file = os.path.join(trial_path, "offset.txt")

    virtual_timestamps, virtual_positions = load_virtual(virtual_file, offset_file)
    physical_positions = load_physical(physical_file)


    # Mission time
    mission_time = virtual_timestamps[-1] - virtual_timestamps[0]
    virtual_timestamps = (virtual_timestamps - virtual_timestamps[0])
    physical_timestamps = np.linspace(0, mission_time, len(physical_positions))


    # Sampling info
    virtual_dt = np.diff(virtual_timestamps)
    physical_dt = np.diff(physical_timestamps)



    # Kinematics
    virtual_raw = compute_kinematics(virtual_timestamps, virtual_positions)
    physical_raw = compute_kinematics(physical_timestamps, physical_positions)

    virtual_smooth = smooth(virtual_positions, virtual_timestamps)
    physical_smooth = smooth(physical_positions, physical_timestamps)



    print("\n========== TRAJECTORIES ==========")


    print("\nVirtual raw:")
    print(f"Samples:  {len(virtual_positions)}")
    print(f"Duration: {mission_time:.6f} s")
    print(f"Mean dt:  {np.mean(virtual_dt):.6f} s")
    print(f"Mean Hz:  {1 / np.mean(virtual_dt):.2f} Hz")

    print("\nPhysical raw:")
    print(f"Samples:  {len(physical_positions)}")
    print(f"Duration: {physical_timestamps[-1]:.6f} s")
    print(f"Mean dt:  {np.mean(physical_dt):.6f} s")
    print(f"Mean Hz:  {1 / np.mean(physical_dt):.2f} Hz")


    print("\n========== VELOCITY ==========")

    print("\nVirtual raw:")
    print(f"Mean speed:   {np.mean(virtual_raw['speed']):.3f} m/s")
    print(f"Median speed: {np.median(virtual_raw['speed']):.3f} m/s")
    print(f"Max speed:    {np.max(virtual_raw['speed']):.3f} m/s")

    print("\nVirtual smooth:")  
    print(f"Mean speed:   {np.mean(virtual_smooth['speed']):.3f} m/s")
    print(f"Median speed: {np.median(virtual_smooth['speed']):.3f} m/s")
    print(f"Max speed:    {np.max(virtual_smooth['speed']):.3f} m/s")

    print("\nPhysical raw:")
    print(f"Mean speed:   {np.mean(physical_raw['speed']):.3f} m/s")
    print(f"Median speed: {np.median(physical_raw['speed']):.3f} m/s")
    print(f"Max speed:    {np.max(physical_raw['speed']):.3f} m/s")

    print("\nPhysical smooth:")
    print(f"Mean speed:   {np.mean(physical_smooth['speed']):.3f} m/s")
    print(f"Median speed: {np.median(physical_smooth['speed']):.3f} m/s")
    print(f"Max speed:    {np.max(physical_smooth['speed']):.3f} m/s")


    print("\n========== ACCELERATION ==========")

    print("\nVirtual raw:")
    print(f"Mean:   {np.mean(virtual_raw['acceleration_magnitude']):.3f} m/s²")
    print(f"Median: {np.median(virtual_raw['acceleration_magnitude']):.3f} m/s²")
    print(f"Max:    {np.max(virtual_raw['acceleration_magnitude']):.3f} m/s²")

    print("\nVirtual smooth:")  
    print(f"Mean:   {np.mean(virtual_smooth['acceleration_magnitude']):.3f} m/s²")
    print(f"Median: {np.median(virtual_smooth['acceleration_magnitude']):.3f} m/s²")
    print(f"Max:    {np.max(virtual_smooth['acceleration_magnitude']):.3f} m/s²")

    print("\nPhysical raw:")
    print(f"Mean:   {np.mean(physical_raw['acceleration_magnitude']):.3f} m/s²")
    print(f"Median: {np.median(physical_raw['acceleration_magnitude']):.3f} m/s²")
    print(f"Max:    {np.max(physical_raw['acceleration_magnitude']):.3f} m/s²")

    print("\nPhysical smooth:")
    print(f"Mean:   {np.mean(physical_smooth['acceleration_magnitude']):.3f} m/s²")
    print(f"Median: {np.median(physical_smooth['acceleration_magnitude']):.3f} m/s²")
    print(f"Max:    {np.max(physical_smooth['acceleration_magnitude']):.3f} m/s²")


    print("\n========== JERK ==========")

    print("\nVirtual raw:")
    print(f"Mean:   {np.mean(virtual_raw['jerk_magnitude']):.3f} m/s³")
    print(f"Median: {np.median(virtual_raw['jerk_magnitude']):.3f} m/s³")
    print(f"Max:    {np.max(virtual_raw['jerk_magnitude']):.3f} m/s³")
    print(f"RMS:    {np.sqrt(np.mean(virtual_raw['jerk_magnitude'] ** 2)):.3f} m/s³")

    print("\nVirtual smooth:")  
    print(f"Mean:   {np.mean(virtual_smooth['jerk_magnitude']):.3f} m/s³")
    print(f"Median: {np.median(virtual_smooth['jerk_magnitude']):.3f} m/s³")
    print(f"Max:    {np.max(virtual_smooth['jerk_magnitude']):.3f} m/s³")
    print(f"RMS:    {np.sqrt(np.mean(virtual_smooth['jerk_magnitude'] ** 2)):.3f} m/s³")

    print("\nPhysical raw:")
    print(f"Mean:   {np.mean(physical_raw['jerk_magnitude']):.3f} m/s³")
    print(f"Median: {np.median(physical_raw['jerk_magnitude']):.3f} m/s³")
    print(f"Max:    {np.max(physical_raw['jerk_magnitude']):.3f} m/s³")
    print(f"RMS:    {np.sqrt(np.mean(physical_raw['jerk_magnitude'] ** 2)):.3f} m/s³")

    print("\nPhysical smooth:") 
    print(f"Mean:   {np.mean(physical_smooth['jerk_magnitude']):.3f} m/s³")
    print(f"Median: {np.median(physical_smooth['jerk_magnitude']):.3f} m/s³")
    print(f"Max:    {np.max(physical_smooth['jerk_magnitude']):.3f} m/s³")
    print(f"RMS:    {np.sqrt(np.mean(physical_smooth['jerk_magnitude'] ** 2)):.3f} m/s³")





    # ============================================================
    # PLOTTING
    # ============================================================

    fig, axes = plt.subplots(
        4,
        3,
        figsize=(18, 14)
    )


    # ============================================================
    # Row 1: Virtual - Raw
    # ============================================================

    axes[0, 0].plot(
        virtual_timestamps[1:],
        virtual_raw["speed"]
    )
    add_mean_line(axes[0, 0], virtual_raw["speed"])
    axes[0, 0].set_title("Virtual - Raw Speed")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Speed [m/s]")
    axes[0, 0].grid(True)


    axes[0, 1].plot(
        virtual_timestamps[2:],
        virtual_raw["acceleration_magnitude"]
    )
    add_mean_line(
        axes[0, 1],
        virtual_raw["acceleration_magnitude"]
    )
    axes[0, 1].set_title("Virtual - Raw Acceleration")
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("Acceleration [m/s²]")
    axes[0, 1].grid(True)


    axes[0, 2].plot(
        virtual_timestamps[3:],
        virtual_raw["jerk_magnitude"]
    )
    add_mean_line(
        axes[0, 2],
        virtual_raw["jerk_magnitude"]
    )
    axes[0, 2].set_title("Virtual - Raw Jerk")
    axes[0, 2].set_xlabel("Time [s]")
    axes[0, 2].set_ylabel("Jerk [m/s³]")
    axes[0, 2].grid(True)


    # ============================================================
    # Row 2: Virtual - Savitzky-Golay
    # ============================================================

    axes[1, 0].plot(
        virtual_timestamps,
        virtual_smooth["speed"]
    )
    add_mean_line(axes[1, 0], virtual_smooth["speed"])
    axes[1, 0].set_title("Virtual - Savitzky-Golay Speed")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("Speed [m/s]")
    axes[1, 0].grid(True)


    axes[1, 1].plot(
        virtual_timestamps,
        virtual_smooth["acceleration_magnitude"]
    )
    add_mean_line(
        axes[1, 1],
        virtual_smooth["acceleration_magnitude"]
    )
    axes[1, 1].set_title(
        "Virtual - Savitzky-Golay Acceleration"
    )
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("Acceleration [m/s²]")
    axes[1, 1].grid(True)


    axes[1, 2].plot(
        virtual_timestamps,
        virtual_smooth["jerk_magnitude"]
    )
    add_mean_line(
        axes[1, 2],
        virtual_smooth["jerk_magnitude"]
    )
    axes[1, 2].set_title("Virtual - Savitzky-Golay Jerk")
    axes[1, 2].set_xlabel("Time [s]")
    axes[1, 2].set_ylabel("Jerk [m/s³]")
    axes[1, 2].grid(True)


    # ============================================================
    # Row 3: Physical - Raw
    # ============================================================

    axes[2, 0].plot(
        physical_timestamps[1:],
        physical_raw["speed"]
    )
    add_mean_line(axes[2, 0], physical_raw["speed"])
    axes[2, 0].set_title("Physical - Raw Speed")
    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 0].set_ylabel("Speed [m/s]")
    axes[2, 0].grid(True)


    axes[2, 1].plot(
        physical_timestamps[2:],
        physical_raw["acceleration_magnitude"]
    )
    add_mean_line(axes[2, 1], physical_raw["acceleration_magnitude"])
    axes[2, 1].set_title("Physical - Raw Acceleration")
    axes[2, 1].set_xlabel("Time [s]")
    axes[2, 1].set_ylabel("Acceleration [m/s²]")
    axes[2, 1].grid(True)


    axes[2, 2].plot(
        physical_timestamps[3:],
        physical_raw["jerk_magnitude"]
    )
    add_mean_line(axes[2, 2], physical_raw["jerk_magnitude"])
    axes[2, 2].set_title("Physical - Raw Jerk")
    axes[2, 2].set_xlabel("Time [s]")
    axes[2, 2].set_ylabel("Jerk [m/s³]")
    axes[2, 2].grid(True)


    # ============================================================
    # Row 4: Physical - Savitzky-Golay
    # ============================================================

    axes[3, 0].plot(
        physical_timestamps,
        physical_smooth["speed"]
    )
    add_mean_line(axes[3, 0], physical_smooth["speed"])
    axes[3, 0].set_title("Physical - Savitzky-Golay Speed")
    axes[3, 0].set_xlabel("Time [s]")
    axes[3, 0].set_ylabel("Speed [m/s]")
    axes[3, 0].grid(True)


    axes[3, 1].plot(
        physical_timestamps,
        physical_smooth["acceleration_magnitude"]
    )
    add_mean_line(axes[3, 1], physical_smooth["acceleration_magnitude"])
    axes[3, 1].set_title(
        "Physical - Savitzky-Golay Acceleration"
    )
    axes[3, 1].set_xlabel("Time [s]")
    axes[3, 1].set_ylabel("Acceleration [m/s²]")
    axes[3, 1].grid(True)


    axes[3, 2].plot(
        physical_timestamps,
        physical_smooth["jerk_magnitude"]
    )
    add_mean_line(axes[3, 2], physical_smooth["jerk_magnitude"])
    axes[3, 2].set_title("Physical - Savitzky-Golay Jerk")
    axes[3, 2].set_xlabel("Time [s]")
    axes[3, 2].set_ylabel("Jerk [m/s³]")
    axes[3, 2].grid(True)


    # ============================================================
    # Final plot formatting
    # ============================================================

    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()
