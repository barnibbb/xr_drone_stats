import numpy as np
import pandas as pd

from scipy.optimize import milp, LinearConstraint, Bounds

METRICS = [
    "mission_time",
    "total_distance",
    "mean_curvature",
    "speed_cv",
    "rms_jerk",
    "path_efficiency",
    "ordered_error",
    "segment_error"
]


def add_outlier_scores(df):
    df = df.copy()

    scores = np.zeros(len(df))

    for metric in METRICS:
        x = df[metric].to_numpy(dtype=np.float64)

        median = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - median))

        robust_z = np.abs(x - median) / (1.4826 * mad)

        excess = np.maximum(robust_z - 2.5, 0)

        scores += excess

    df["outlier_score"] = scores

    return df



def select_balanced_subset(df, track_modes, target_per_mode):
    data = df[df["track"].isin(track_modes.keys()) & df["control_mode"].isin(
        set(mode for modes in track_modes.values() for mode in modes))].copy()

    data = data.reset_index(drop=True)

    rng = np.random.default_rng(seed=42)

    objective = (data["outlier_score"].to_numpy() + rng.uniform(0, 1e-8, len(data)))

    n = len(data)

    integrality = np.ones(n)

    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))

    constraints = []

    lower = []
    upper = []

    # Constraint 1: exactly target_per_mode samples for each track and control_mode combination
    for track, modes in track_modes.items():
        for mode in modes:
            row = np.zeros(n)
            mask = (data["track"] == track) & (data["control_mode"] == mode)

            row[mask.to_numpy()] = 1

            constraints.append(row)

            lower.append(target_per_mode)
            upper.append(target_per_mode)

    # Constraint 2: one observation per participant per track
    participants = data["participant_id"].unique()

    for participant in participants:
        for track in track_modes.keys():
            row = np.zeros(n)
            mask = (data["participant_id"] == participant) & (data["track"] == track)

            row[mask.to_numpy()] = 1

            constraints.append(row)

            lower.append(1)
            upper.append(1)


    A = np.vstack(constraints)
    constraint_matrix = LinearConstraint(A, np.array(lower), np.array(upper))

    result = milp(c=objective, integrality=integrality, bounds=bounds, constraints=constraint_matrix, options={"time_limit": 60})

    selected = data[result.x > 0.5].copy()

    return selected



def main():
    df = pd.read_csv("/home/appuser/data/metrics.csv")
    results_dir = "/home/appuser/data"

    df = add_outlier_scores(df)

    participant_scores = df.groupby(["participant_id", "control_mode"])["outlier_score"].sum().unstack("control_mode").reset_index()
    participant_scores.to_csv(f"{results_dir}/participant_outlier_scores.csv")


    t1 = select_balanced_subset(df, track_modes={"T1": ["joy", "palm", "pinch"]}, target_per_mode=12)
    t1.to_csv(f"{results_dir}/balanced_subset_T1.csv", index=False)

    t2 = select_balanced_subset(df, track_modes={"T2": ["holo_joy", "joy", "palm", "pinch"]}, target_per_mode=9)
    t2.to_csv(f"{results_dir}/balanced_subset_T2.csv", index=False)

    balanced_2x3 = select_balanced_subset(df, track_modes={"T1": ["joy", "palm", "pinch"], "T2": ["joy", "palm", "pinch"]}, target_per_mode=12)
    balanced_2x3.to_csv(f"{results_dir}/balanced_subset_2x3.csv", index=False)







if __name__ == "__main__":
    main()

