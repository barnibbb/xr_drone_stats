import itertools
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.formula.api as smf
import seaborn as sns
import pingouin as pg
from statsmodels.stats.multitest import multipletests


RESULTS_DIR = "/home/appuser/data/secondary_analysis_lim"
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
iv = "control_mode"



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


def compute_histograms(df, metrics):
    fig, axes = plt.subplots(2, 4, figsize=(18,9))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        sns.histplot(data=df, x=metric, hue=iv, bins=15, kde=True, element="step", stat="density", common_norm=False, ax=ax)
        
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel("Density")

    fig.suptitle(f"Metrics Distribution by {iv}")

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/histograms.png", dpi=300, bbox_inches='tight')
    plt.close(fig)



def compute_boxplots(df, metrics):
    fig, axes = plt.subplots(2, 4, figsize=(18,9))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        sns.boxplot(data=df, x=iv, y=metric, ax=ax)

        ax.tick_params(axis='x', rotation=45)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle(f"Metrics Boxplots by {iv}")

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/boxplots.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def normality_test(df, metrics):
    shapiro_results = []
    qq_deviation_results = []

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        data = df[["participant_id", iv, metric]].dropna()

        # Fitting repeated measures model
        model = smf.ols(f"Q('{metric}') ~ C(participant_id) + C(control_mode)", data=data).fit()

        residuals = model.resid

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


def within_subjects_omnibus(df, metrics):
    results = []

    for metric in metrics:
        data = df[[ "participant_id", iv, metric]].dropna()

        # Friedman test
        wide = data.pivot(index="participant_id", columns=iv, values=metric).dropna()
        modes = list(wide.columns)
        samples = [wide[mode].values for mode in modes]

        statistic, p_value = stats.friedmanchisquare(*samples)

        results.append({
            "metric": metric,
            "test": "Friedman Test",
            "statistic": statistic,
            "p_value": p_value,
            "significant": "Yes" if p_value < ALPHA else "No"
        })
            
    results = pd.DataFrame(results)
    
    results.to_csv(f"{RESULTS_DIR}/within_subjects_results.csv", index=False)    

    return results


def post_hoc_within(df, main_test_results, metrics):
    results_post_hoc = []

    for metric in metrics:
        main_result = main_test_results.loc[main_test_results["metric"] == metric].iloc[0]

        if main_result["significant"] != "Yes":
            continue  # Skip post-hoc tests if the main test is not significant

        data = df[["participant_id", iv, metric]].dropna()
        data = data.groupby(["participant_id", iv], as_index=False)[metric].mean()

        wide = data.pivot(index="participant_id", columns=iv, values=metric).dropna()

        modes = list(wide.columns)

        pairs = itertools.combinations(modes, 2)

        metric_results = []

        for mode1, mode2 in pairs:
            x = wide[mode1]
            y = wide[mode2]

            differences = y - x

            stat, p_value = stats.wilcoxon(y, x)

            # Rank biserial correlation as effect size for Wilcoxon test
            nonzero = differences[differences != 0]

            if len(nonzero) == 0:
                rank_biserial = np.nan
            else:
                ranks = stats.rankdata(np.abs(nonzero))
                positive_ranks = ranks[nonzero > 0].sum()
                negative_ranks = ranks[nonzero < 0].sum()

                rank_biserial = (positive_ranks - negative_ranks) / (positive_ranks + negative_ranks)

            metric_results.append({
                "metric": metric,
                "mode1": mode1,
                "mode2": mode2,
                "statistic": stat,
                "p_value": p_value,
                "significant": "Yes" if p_value < ALPHA else "No",
                "effect_size": -rank_biserial
            })

        # Adjust p-values for multiple comparisons using Holm correction
        p_values = [result["p_value"] for result in metric_results]

        reject, pvals_corrected, _, _ = multipletests(p_values, alpha=ALPHA, method='holm')

        for result, adjusted_p, significant in zip(metric_results, pvals_corrected, reject):
            result["p_adjusted"] = adjusted_p
            result["corrected_significant"] = "Yes" if significant else "No"

        results_post_hoc.extend(metric_results)

    results_post_hoc = pd.DataFrame(results_post_hoc)
    results_post_hoc.to_csv(f"{RESULTS_DIR}/post_hoc_results.csv", index=False)





def main():
    # Load the CSV file into a DataFrame
    metrics_path = f"/home/appuser/data/metrics_lim.csv"
    df = pd.read_csv(metrics_path)

    
    compute_stats(df, metrics)
    compute_histograms(df, metrics)
    compute_boxplots(df, metrics)
    normality_test(df, metrics)
    sphericity_test(df, metrics)

    main_test_results = within_subjects_omnibus(df, metrics)
    post_hoc_within(df, main_test_results, metrics)


if __name__ == "__main__":
    main()

