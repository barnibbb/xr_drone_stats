import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


RESULTS_DIR = "/home/appuser/data/primary_analysis"
FIGURES_DIR = f"{RESULTS_DIR}/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

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

ALPHA = 0.05
iv = ["track", "control_mode"]

# Computing basic statistics
def compute_stats(df, metrics):
    stats = (df.groupby(iv)[metrics].agg(["count", "mean", "median", "std", "min", "max"]))

    q1 = df.groupby(iv)[metrics].quantile(0.25)
    q3 = df.groupby(iv)[metrics].quantile(0.75)

    iqr = q3 - q1

    for metric in metrics:
        stats.loc[:, f"{metric}_IQR"] = iqr[metric]

    with open(f"{RESULTS_DIR}/stats.csv", "w") as f:
        for metric in metrics:
            f.write(f"{metric}\n")
            f.write(f"{iv},count,mean,median,std,min,max,IQR\n")

            for item in stats.index:
                row = [
                    stats.loc[item, (metric, "count")],
                    stats.loc[item, (metric, "mean")],
                    stats.loc[item, (metric, "median")],
                    stats.loc[item, (metric, "std")],
                    stats.loc[item, (metric, "min")],
                    stats.loc[item, (metric, "max")],
                    stats.loc[item, f"{metric}_IQR"]
                ]

                f.write(f"{item},{','.join(str(value) for value in row)}\n")

            f.write("\n")


# Computing histograms
def compute_histograms(df, metrics):
    fig, axes = plt.subplots(2, 4, figsize=(18,9))
    axes = axes.flatten()

    # # LMM
    df = df.copy()
    df["track_mode"] = df["track"] + "_" + df["control_mode"]

    for ax, metric in zip(axes, metrics):
        sns.histplot(data=df, x=metric, hue="track_mode", bins=15, kde=True, element="step", stat="density", common_norm=False, ax=ax)

        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel("Density")

    fig.suptitle(f"Metrics Distribution by {iv}")

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/histograms.png", dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory


## Computing boxplots
def compute_boxplots(df, metrics):
    fig, axes = plt.subplots(2, 4, figsize=(18,9))
    axes = axes.flatten()

    # LMM
    df = df.copy()
    df["track_mode"] = df["track"] + "_" + df["control_mode"]

    for ax, metric in zip(axes, metrics):
        sns.boxplot(data=df, x="track_mode", y=metric, ax=ax)

        ax.tick_params(axis='x', rotation=45)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle(f"Metrics Boxplots by {iv}")

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/boxplots.png", dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory


def normality_test(df, metrics):
    shapiro_results = []

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        print(f"Processing metric: {metric}")

        data = df[["participant_id", "control_mode", "track", metric]].dropna()
        model = smf.mixedlm(f"{metric} ~ C(track) + C(control_mode)", data=data, groups=data["participant_id"]).fit(reml=False)

        print(metric, "participant variance:", model.cov_re.iloc[0, 0])
        print(metric, "converged:", model.converged)

        residuals = model.resid

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


def make_design_vector(beta_index, mode=None, track=None):
    x = np.zeros(len(beta_index))

    x[beta_index.get_loc("Intercept")] = 1

    if mode is not None and mode != "holo_joy":
        x[beta_index.get_loc(f"C(control_mode)[T.{mode}]")] = 1

    if track == "T2":
        x[beta_index.get_loc(f"C(track)[T.T2]")] = 1

    return x


def fit_mixed_effects_model(df, metric):
    data = df[["participant_id", "control_mode", "track", metric]].dropna().copy()

    model = smf.mixedlm(f"{metric} ~ C(control_mode) + C(track)", data=data, groups=data["participant_id"])

    result = model.fit(reml=False)

    omnibus = result.wald_test_terms()

    tests = [("Track", "C(track)"), ("Control Mode", "C(control_mode)")]

    results = []

    for test_name, term in tests:
        statistic = omnibus.table.loc[term, "statistic"]
        p_value = omnibus.table.loc[term, "pvalue"]

        results.append({
            "metric": metric,
            "test": test_name,
            "statistic": statistic,
            "p_value": p_value,
            "significant": "Yes" if p_value < ALPHA else "No"
        })

    omnibus_results = pd.DataFrame(results)

    track_significance = omnibus.table.loc["C(track)", "pvalue"] < ALPHA

    control_mode_significance = omnibus.table.loc["C(control_mode)", "pvalue"] < ALPHA

    return result, omnibus_results, track_significance, control_mode_significance


def posthoc_control_mode(result, metric):
    beta = result.fe_params
    cov_beta = result.cov_params().loc[beta.index, beta.index]

    print(beta.index.tolist())
    print(beta)

    modes = ["holo_joy", "joy", "palm", "pinch"]

    for mode in modes:
        x = make_design_vector(beta.index, mode=mode)
        print(f"{mode:10s}: {x}")
    
    comparisons = [
        ("holo_joy", "joy"),
        ("holo_joy", "palm"),
        ("holo_joy", "pinch"),
        ("joy", "palm"),
        ("joy", "pinch"),
        ("palm", "pinch")
    ]
    
    results = []

    for mode1, mode2 in comparisons:
        
        x1 = make_design_vector(beta.index, mode=mode1)
        x2 = make_design_vector(beta.index, mode=mode2)

        contrast = x1 - x2

        estimate = contrast @ beta
        variance = contrast @ cov_beta @ contrast
        se = variance ** 0.5
        
        statistic = estimate / se
        p_value = 2 * stats.norm.sf(abs(statistic))

        residual_sd = np.sqrt(result.scale)
        effect_size = estimate / residual_sd

        results.append({
            "metric": metric,
            "mode1": mode1,
            "mode2": mode2,
            "statistic": statistic,
            "p_value": p_value,
            "significant": "Yes" if p_value < ALPHA else "No",
            "effect_size": effect_size
        })

    results = pd.DataFrame(results)

    # Adjust p-values for multiple comparison
    reject, p_adjusted, _, _ = multipletests(results["p_value"], alpha=ALPHA, method="holm")
        
    results["p_adjusted"] = p_adjusted
    results["significant_adjusted"] = np.where(reject, "Yes", "No")

    return results




def posthoc_track(result, metric):
    beta = result.fe_params
    cov_beta = result.cov_params().loc[beta.index, beta.index]

    x1 = make_design_vector(beta.index, track="T1")
    x2 = make_design_vector(beta.index, track="T2")

    print(f"x1: {x1}")
    print(f"x2: {x2}")

    contrast = x1 - x2

    estimate = contrast @ beta
    variance = contrast @ cov_beta @ contrast
    se = variance ** 0.5
    
    statistic = estimate / se
    p_value = 2 * stats.norm.sf(abs(statistic))

    residual_sd = np.sqrt(result.scale)
    effect_size = estimate / residual_sd

    results = pd.DataFrame([{
        "metric": metric,
        "mode1": "T1",
        "mode2": "T2",
        "statistic": statistic,
        "p_value": p_value,
        "significant": "Yes" if p_value < ALPHA else "No",
        "effect_size": effect_size,
        "p_adjusted": p_value,
        "significant_adjusted": "Yes" if p_value < ALPHA else "No"
    }])

    results = pd.DataFrame(results)

    return results





def main():
    # Load the CSV file into a DataFrame
    metrics_path = f"/home/appuser/data/metrics.csv"
    df = pd.read_csv(metrics_path)

    compute_stats(df, metrics)
    compute_histograms(df, metrics)
    compute_boxplots(df, metrics)
    normality_test(df, metrics)
    
    all_omnibus_results = []
    all_posthoc_results = []

    for metric in metrics:
        result, omnibus_results, track_significance, control_mode_significance = fit_mixed_effects_model(df, metric)
        all_omnibus_results.append(omnibus_results)

        if control_mode_significance:
            posthoc_result = posthoc_control_mode(result, metric)
            all_posthoc_results.append(posthoc_result)

        if track_significance:
            posthoc_result = posthoc_track(result, metric)
            all_posthoc_results.append(posthoc_result)

    all_omnibus_results = pd.concat(all_omnibus_results, ignore_index=True)
    all_omnibus_results.to_csv(f"{RESULTS_DIR}/mixed_effects_model_results.csv", index=False)

    all_posthoc_results = pd.concat(all_posthoc_results, ignore_index=True)
    all_posthoc_results.to_csv(f"{RESULTS_DIR}/mixed_effects_model_posthoc_results.csv", index=False)
    



if __name__ == "__main__":
    main()

