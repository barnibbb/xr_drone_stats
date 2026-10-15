import os
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt
import seaborn as sns
import bambi as bmb
import arviz as az
import inspect

RESULTS_DIR = "/home/appuser/data/control_analysis"
FIGURES_DIR = f"{RESULTS_DIR}/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


iv = "control_mode"

def compute_collision_stats(df):
    stats_df = df.groupby(iv)["collision_count"].agg(
        n="count",
        sum="sum",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max"
    )

    q1 = df.groupby(iv)["collision_count"].quantile(0.25)
    q3 = df.groupby(iv)["collision_count"].quantile(0.75)

    stats_df["IQR"] = q3 - q1

    stats_df["zero_collision_pct"] = (
        df.assign(zero=df["collision_count"].eq(0))
        .groupby(iv)["zero"]
        .mean()
        .mul(100)
    )

    stats_df.to_csv(f"{RESULTS_DIR}/collision_stats.csv", index=True)


def compute_collision_plots(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.boxplot(x=iv, y="collision_count", data=df, ax=axes[0])
    # sns.stripplot(x=iv, y="collision_count", data=df, color="black", alpha=0.5, ax=axes[0])
    axes[0].set_title("Collision Count by Control Mode")
    axes[0].set_xlabel("Control Mode")
    axes[0].set_ylabel("Collision Count")
    axes[0].tick_params(axis='x', rotation=45)

    sns.barplot(data=df, x=iv, y="collision_count", errorbar=("ci", 95), ax=axes[1])

    axes[1].set_title("Mean Collision Count with 95% CI")
    axes[1].set_xlabel("Control Mode")
    axes[1].set_ylabel("Mean Collision Count")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/collision_plots.png", dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory



def friedman_test(df):
    wide = df.pivot(index="participant_id", columns=iv, values="collision_count")

    modes = ["holo_joy", "joy", "palm", "pinch"]

    wide = wide[modes].dropna()

    n = len(wide)
    k = len(modes)

    statistic, p_value = stats.friedmanchisquare(*[wide[mode] for mode in modes])

    # Kendall's W effect size
    kendalls_w = statistic / (n * (k - 1))

    print(f"Friedman test statistic: {statistic}")
    print(f"p-value: {p_value}")
    print(f"Significant: {p_value < 0.05}")
    print(f"Kendall's W: {kendalls_w}")



def posthoc_wilcoxon(df):
    wide = df.pivot(index="participant_id", columns=iv, values="collision_count")
    
    modes = ["holo_joy", "joy", "palm", "pinch"]
    
    wide = wide[modes].dropna()

    results = []

    for mode_a, mode_b in combinations(modes, 2):
        a = wide[mode_a]
        b = wide[mode_b]

        test = stats.wilcoxon(a, b, alternative='two-sided', zero_method='wilcox', method='auto')
        statistic, p_value = test.statistic, test.pvalue


        results.append({
            "mode_a": mode_a,
            "mode_b": mode_b,
            "statistic": statistic,
            "p_value": p_value,
            "significant": "Yes" if p_value < 0.05 else "No"
        })

    results_df = pd.DataFrame(results)

    # Holm multiple comparisons correction
    results_df["p_holm"] = multipletests(results_df["p_value"], method='holm')[1]
    results_df["significant_holm"] = results_df["p_holm"].apply(lambda p: "Yes" if p < 0.05 else "No")

    results_df.to_csv(f"{RESULTS_DIR}/posthoc_wilcoxon_results.csv", index=False)



def main():
    # Load the CSV file into a DataFrame
    metrics_path = f"/home/appuser/data/metrics.csv"
    df = pd.read_csv(metrics_path)

    # Compute collision statistics
    # compute_collision_stats(df)
    # compute_collision_plots(df)
    # friedman_test(df)
    # posthoc_wilcoxon(df)

    df["participant_id"] = df["participant_id"].astype("category")
    df["control_mode"] = df["control_mode"].astype("category")
    df["track"] = df["track"].astype("category")

    model = bmb.Model("collision_count ~ control_mode + track + (1|participant_id)", data=df, family="negativebinomial")

    results = model.fit(draws=2000, tune=1000, chains=4, cores=4, random_seed=42)

    summary = az.summary(
        results,
        var_names=["Intercept", "control_mode", "track"],
        ci_prob=0.95,
        round_to=4
    )

    summary["IRR"] = np.exp(summary["mean"])
    summary["IRR_CI_low"] = np.exp(summary["eti95_lb"])
    summary["IRR_CI_high"] = np.exp(summary["eti95_ub"])

    print(summary[["mean", "eti95_lb", "eti95_ub", "IRR", "IRR_CI_low", "IRR_CI_high"]])




if __name__ == "__main__":
    main()

