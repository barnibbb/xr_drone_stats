import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


RESULTS_DIR = "/home/appuser/data/lmm"
iv = ["track", "control_mode"]



def compute_correlation_matrix(df, metrics):
    correlation_matrix = df[metrics].corr(method='pearson')
    
    correlation_matrix.to_csv(f"{RESULTS_DIR}/correlation_matrix.csv", index=True)


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

    # LMM
    df = df.copy()
    df["track_mode"] = df["track"] + "_" + df["control_mode"]

    for ax, metric in zip(axes, metrics):
        # sns.histplot(data=df, x=metric, hue=iv, bins=15, kde=True, element="step", stat="density", common_norm=False, ax=ax)
        # LMM
        sns.histplot(data=df, x=metric, hue="track_mode", bins=15, kde=True, element="step", stat="density", common_norm=False, ax=ax)

        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel("Density")

    fig.suptitle(f"Metrics Distribution by {iv}")

    plt.tight_layout()

    fig.savefig(f"{RESULTS_DIR}/figures/histograms.png", dpi=300, bbox_inches='tight')

    # plt.show()



def compute_boxplots(df, metrics):
    fig, axes = plt.subplots(2, 4, figsize=(18,9))
    axes = axes.flatten()

    # LMM
    df = df.copy()
    df["track_mode"] = df["track"] + "_" + df["control_mode"]

    for ax, metric in zip(axes, metrics):
        # sns.boxplot(data=df, x=iv, y=metric, ax=ax)
        # LMM
        sns.boxplot(data=df, x="track_mode", y=metric, ax=ax)

        ax.tick_params(axis='x', rotation=45)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle(f"Metrics Boxplots by {iv}")

    plt.tight_layout()

    fig.savefig(f"{RESULTS_DIR}/figures/boxplots.png", dpi=300, bbox_inches='tight')

    # plt.show()




def main():
    # Load the CSV file into a DataFrame
    # metrics_path = f"/home/appuser/data/metrics.csv"
    # metrics_path = f"/home/appuser/data/balanced_T1/balanced_subset_T1.csv"
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


    # allocation = (
    #     df.groupby(["track", "control_mode"])
    #     .size()
    #     .unstack(fill_value=0)
    # )

    # print(allocation)

    compute_correlation_matrix(df, metrics)
    compute_stats(df, metrics)
    compute_histograms(df, metrics)
    compute_boxplots(df, metrics)
    


if __name__ == "__main__":
    main()

