import os
import re

import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from single_test import evaluate_trajectory



DATA_DIR = "/home/appuser/data/samples"
OUTPUT_FILE = "/home/appuser/data/metrics.csv"


# =========================================================================
# Main processing loop
# =========================================================================

# Parse participant folder
def parse_participant_folder(folder_name):
    match = re.match(r"(\d+)_(.+)$", folder_name)

    oder_id = match.group(1)
    participant_id = match.group(2)

    return oder_id, participant_id



# Parse data folder
def parse_trial_folder(folder_name):
    match = re.match(r"(.+)_(T\d+)$", folder_name)

    control_mode = match.group(1)
    track = match.group(2)

    return control_mode, track



def process_trial(args):
    participant_folder, trial_folder = args

    participant_path = os.path.join(DATA_DIR, participant_folder)
    trial_path = os.path.join(participant_path, trial_folder)

    try:
        order_id, participant_id = parse_participant_folder(participant_folder)
        control_mode, track = parse_trial_folder(trial_folder)
    except Exception as e:
        print(f"Skipping folder {trial_folder}: {e}")
        return None

    trajectory_file = os.path.join(trial_path, "measurement_log.yaml")
    offset_file = os.path.join(trial_path, "offset.txt")
    physical_file = os.path.join(trial_path, "cf1_pose.txt")

    if not os.path.exists(trajectory_file) or not os.path.exists(offset_file) or not os.path.exists(physical_file):
        print(f"Missing files in {trial_path}. Skipping.")
        return None

    metrics = evaluate_trajectory(trajectory_file, offset_file, physical_file)

    result = {
        "order_id": order_id,
        "participant_id": participant_id,
        "control_mode": control_mode,
        "track": track,
        **metrics
    }

    return result



# Main processing loop
def main():
    tasks = []
    participant_folders = sorted(os.listdir(DATA_DIR))

    for participant_folder in participant_folders:
        participant_path = os.path.join(DATA_DIR, participant_folder)
        if not os.path.isdir(participant_path):
            continue

        trial_folders = sorted(os.listdir(participant_path))

        for trial_folder in trial_folders:
            trial_path = os.path.join(participant_path, trial_folder)
            if not os.path.isdir(trial_path):
                continue

            tasks.append((participant_folder, trial_folder))

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_trial, tasks))

    results = [result for result in results if result is not None]

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
