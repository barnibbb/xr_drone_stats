import os
from xml.parsers.expat import model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pingouin as pg


iv = ["track", "control_mode"]
# iv = "control_mode"
RESULTS_DIR = "/home/appuser/data/analysis_1"
ALPHA = 0.05



def normality_test1(df, metrics):
    
    # Create paired data
    participant_means = (df.groupby(["participant_id", iv])[metrics].mean().reset_index())

    paired_data = participant_means.pivot(index="participant_id", columns=iv, values=metrics)

    paired_data.columns = [f"{metric}_{item}" for metric, item in paired_data.columns]

    paired_data = paired_data.reset_index()

    for metric in metrics:
        paired_data[f"{metric}_difference"] = paired_data[f"{metric}_T2"] - paired_data[f"{metric}_T1"]

    paired_data.to_csv(f"{RESULTS_DIR}/paired_data.csv", index=False)

    # Perform Shapiro-Wilk test for normality
    shapiro_results = []

    for metric in metrics:
        difference = paired_data[f"{metric}_difference"].dropna()

        stat, p_value = stats.shapiro(difference)

        shapiro_results.append({
            "metric": metric,
            "statistic": stat,
            "p_value": p_value,
            "normality": "Normal" if p_value > ALPHA else "Not Normal"
        })

    shapiro_results = pd.DataFrame(shapiro_results)
    shapiro_results.to_csv(f"{RESULTS_DIR}/shapiro_wilk_results.csv", index=False)

    # Create Q-Q plots for each metric
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        difference = paired_data[f"{metric}_difference"].dropna()

        stats.probplot(difference, dist="norm", plot=ax)
        ax.set_title(f"Q-Q Plot: {metric}")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")

    plt.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/figures/qq_plots.png", dpi=300, bbox_inches='tight')

    plt.close(fig)



def normality_test2(df, metrics):
    shapiro_results = []
    qq_deviation_results = []

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        data = df[["participant_id", iv, metric]].dropna()

        # Fitting repeated measures model
        model = smf.ols(f"Q('{metric}') ~ C(participant_id) + C(control_mode)", data=data).fit()

        residuals = model.resid

        # Quantitative check for normality
        residuals_sorted = np.sort(residuals)
        n = len(residuals_sorted)

        observed_z = (residuals_sorted - np.mean(residuals_sorted)) / np.std(residuals_sorted, ddof=1)
        probabilities = (np.arange(1, n + 1) - 0.5) / n
        expected_z = stats.norm.ppf(probabilities)

        qq_deviation = observed_z - expected_z
        abs_deviation = np.abs(qq_deviation)

        thresholds = [0.25, 0.50, 1.0]

        # Store deviation results
        qq_deviation_results.append({
            "metric": metric,
            "n": n,
            "max_abs_deviation": np.max(abs_deviation),
            "mean_abs_deviation": np.mean(abs_deviation),
            "median_abs_deviation": np.median(abs_deviation),
            "n_abs_dev_gt_0.25": np.sum(abs_deviation > 0.25),
            "n_abs_dev_gt_0.50": np.sum(abs_deviation > 0.50),
            "n_abs_dev_gt_1.00": np.sum(abs_deviation > 1.00)
        })


        # Perform Shapiro-Wilk test for normality of residuals
        stat, p_value = stats.shapiro(residuals)

        shapiro_results.append({
            "metric": metric,
            "statistic": stat,
            "p_value": p_value,
            "normality": "Normal" if p_value > ALPHA else "Not Normal"
        })

        # Q-Q plot for residuals
        stats.probplot(residuals, dist="norm", plot=ax)

        ax.set_title(f"Q-Q Plot of Residuals: {metric}")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")

    plt.tight_layout()

    fig.savefig(f"{RESULTS_DIR}/figures/qq_plots.png", dpi=300, bbox_inches='tight')
    
    plt.close(fig)

    shapiro_results = pd.DataFrame(shapiro_results)
    shapiro_results.to_csv(f"{RESULTS_DIR}/shapiro_wilk_results.csv", index=False)

    # Save Q-Q deviation results separately
    qq_deviation_results = pd.DataFrame(qq_deviation_results)
    qq_deviation_results.to_csv(f"{RESULTS_DIR}/qq_deviation_results.csv", index=False)
 
    


def normality_test3(df, metrics):
    shapiro_results = []

    for metric in metrics:
        modes = sorted(df[iv].dropna().unique())
        fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 5))

        for ax, mode in zip(axes, modes):
            values = df.loc[df[iv] == mode, metric].dropna()

            stat, p_value = stats.shapiro(values)

            shapiro_results.append({
                "metric": metric,
                "control_mode": mode,
                "n": len(values),
                "statistic": stat,
                "p_value": p_value,
                "normality": "Normal" if p_value > ALPHA else "Not Normal"
            })

            stats.probplot(values, dist="norm", plot=ax)
            ax.set_title(f"Q-Q Plot: {metric} ({mode})")
            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel("Sample Quantiles")

        plt.tight_layout()
        fig.savefig(f"{RESULTS_DIR}/figures/{metric}_qq_plots.png", dpi=300, bbox_inches='tight')

        plt.close(fig)

    shapiro_results = pd.DataFrame(shapiro_results)
    shapiro_results.to_csv(f"{RESULTS_DIR}/shapiro_wilk_results_2.csv", index=False)


def normality_test4(df, metrics):
    shapiro_results = []
    qq_deviation_results = []
    qq_observation_results = []

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        print(f"Processing metric: {metric}")

        data = df[["participant_id", "control_mode", "track", metric]].dropna()
        model = smf.mixedlm(f"{metric} ~ C(track) + C(control_mode)", data=data, groups=data["participant_id"]).fit(reml=False)

        print(metric, "participant variance:", model.cov_re.iloc[0, 0])
        print(metric, "converged:", model.converged)

        residuals = model.resid

        # Quantitative check for normality
        residuals_sorted = residuals.sort_values()
        n = len(residuals_sorted)

        observed_z = (residuals_sorted - np.mean(residuals_sorted)) / np.std(residuals_sorted, ddof=1)
        probabilities = (np.arange(1, n + 1) - 0.5) / n
        expected_z = stats.norm.ppf(probabilities)

        qq_deviation = observed_z - expected_z
        abs_deviation = np.abs(qq_deviation)

        thresholds = [0.25, 0.50, 1.0]

        # Store deviation results
        qq_deviation_results.append({
            "metric": metric,
            "n": n,
            "max_abs_deviation": np.max(abs_deviation),
            "mean_abs_deviation": np.mean(abs_deviation),
            "median_abs_deviation": np.median(abs_deviation),
            "n_abs_dev_gt_0.25": np.sum(abs_deviation > 0.25),
            "n_abs_dev_gt_0.50": np.sum(abs_deviation > 0.50),
            "n_abs_dev_gt_1.00": np.sum(abs_deviation > 1.00)
        })

        qq_observations = pd.DataFrame({
            "row_index": residuals_sorted.index.to_numpy(),
            "deviation": abs_deviation
        })

        original_data = data.reset_index().rename(
            columns={"index": "row_index"}
        )

        qq_observations = pd.DataFrame({
            "participant_id": original_data.loc[
                residuals_sorted.index, "participant_id"
            ].values,
            "deviation": abs_deviation
        })

        # IMPORTANT
        qq_observation_results.append(qq_observations)


        # Shapiro-Wilk test for normality of residuals
        stat, p_value = stats.shapiro(residuals)

        shapiro_results.append({
            "metric": metric,
            "statistic": stat,
            "p_value": p_value,
            "normality": "Normal" if p_value > ALPHA else "Not Normal"
        })

        # Q-Q plot for residuals
        sm.qqplot(residuals, line="45", fit=True, ax=ax)
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.set_title(f"{metric} - Mixed Model Residuals")

    plt.tight_layout()    

    fig.savefig(f"{RESULTS_DIR}/figures/qq_plots.png", dpi=300, bbox_inches='tight')

    plt.close(fig)

    shapiro_results = pd.DataFrame(shapiro_results)
    shapiro_results.to_csv(f"{RESULTS_DIR}/shapiro_wilk_results.csv", index=False)

    # Save Q-Q deviation results separately
    qq_deviation_results = pd.DataFrame(qq_deviation_results)
    qq_deviation_results.to_csv(f"{RESULTS_DIR}/qq_deviation_results.csv", index=False)

    # Save Q-Q observation results separately
    qq_observation_results = pd.concat(qq_observation_results, ignore_index=True)
    participant_deviation = (
    qq_observation_results
        .groupby("participant_id", as_index=False)["deviation"]
        .sum()
        .rename(columns={"deviation": "total_deviation"})
    )

    participant_deviation = participant_deviation.sort_values(
        "total_deviation",
        ascending=False
    )

    participant_deviation.to_csv(
        f"{RESULTS_DIR}/participant_qq_deviation.csv",
        index=False
    )


def sphericity_test(df, metrics):
    sphericity_results = []

    for metric in metrics:
        data = df[["participant_id", iv, metric]].dropna()

        wide = data.pivot(index="participant_id", columns=iv, values=metric)

        # Perform Mauchly's test for sphericity
        spher, W, chi2, dof, p_value  = pg.sphericity(wide)

        sphericity_results.append({
            "metric": metric,
            "W": W,
            "chi2": chi2,
            "dof": dof,
            "p_value": p_value,
            "sphericity": "Sphericity" if p_value > ALPHA else "Not Sphericity"
        })

    sphericity_results = pd.DataFrame(sphericity_results)
    sphericity_results.to_csv(f"{RESULTS_DIR}/sphericity_results.csv", index=False)



def variance_test(df, metrics):
    levene_results = []

    modes = sorted(df[iv].dropna().unique())

    for metric in metrics:
        groups = [df.loc[df[iv] == mode, metric].dropna().to_numpy() for mode in modes]

        stat, p_value = stats.levene(*groups, center='median')

        levene_results.append({
            "metric": metric,
            "statistic": stat,
            "p_value": p_value,
            "equal_variance": "Equal Variance" if p_value > ALPHA else "Not Equal Variance"
        })

    levene_results = pd.DataFrame(levene_results)
    levene_results.to_csv(f"{RESULTS_DIR}/levene_results.csv", index=False)





def main():
    # Load the CSV file into a DataFrame
    # metrics_path = f"/home/appuser/data/metrics.csv"
    
    metrics_path = f"/home/appuser/data/metrics.csv"
    df = pd.read_csv(metrics_path)

    metrics = [
        "mission_time",
        "total_distance",
        "mean_curvature",
        "speed_cv",
        "rms_jerk",
        "path_efficiency",
        "ordered_error",
        "segment_error"
    ]

    # Track analysis
    # normality_test1(df, metrics)

    # Control analysis
    # normality_test2(df, metrics)
    # sphericity_test(df, metrics)

    # Per-track control analysis
    # normality_test3(df, metrics)
    # variance_test(df, metrics)

    # LMM
    normality_test4(df, metrics)


if __name__ == "__main__":
    main()
