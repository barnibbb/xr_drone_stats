from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


WORK_DIR = Path("/home/appuser/data/post_condition")
INPUT_FILE = WORK_DIR / "post_condition_answers.csv"

QUESTIONS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10"]

PARTICIPANT_COL = "Participant ID"
CONTROL_COL = "Control Mode"

CONTROL_ORDER = [
    "A: Pinch control",
    "B: Palm control",
    "C: Joy control",
    "D: Drone-in-the-middle control"
]

SCALE_MIN = 1
SCALE_MAX = 7

sns.set_theme(style="whitegrid", palette="pastel")

# Load data
def load_data():
    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip()
    df[CONTROL_COL] = df[CONTROL_COL].str.strip()

    for question in QUESTIONS:
        df[question] = pd.to_numeric(df[question])

    return df


def to_long_format(df):
    long_df = df.melt(
        id_vars=[PARTICIPANT_COL, CONTROL_COL],
        value_vars=QUESTIONS,
        var_name="Question",
        value_name="Rating"
    )

    long_df[CONTROL_COL] = pd.Categorical(
        long_df[CONTROL_COL],
        categories=CONTROL_ORDER,
        ordered=True
    )

    long_df["Question"] = pd.Categorical(
        long_df["Question"],
        categories=QUESTIONS,
        ordered=True
    )

    return long_df


def basic_stats(long_df):
    stats_df = long_df.groupby(["Question", CONTROL_COL], observed=True, sort=False)["Rating"].agg(
        n="count",
        mean="mean",
        std="std",
        median="median",
        q1=lambda x: np.percentile(x, 25),
        q3=lambda x: np.percentile(x, 75),
        min="min",
        max="max"
    ).reset_index()

    stats_df["IQR"] = stats_df["q3"] - stats_df["q1"]

    stats_df.to_csv(WORK_DIR / "post_condition_stats.csv", index=False)

    return stats_df


def plot_boxplots(long_df):

    g = sns.catplot(
        data=long_df,
        x=CONTROL_COL,
        y="Rating",
        col="Question",
        col_order=QUESTIONS,
        col_wrap=5,
        order=CONTROL_ORDER,
        kind="box",
        sharey=True,
        height=4.5,
        aspect=1.05,
        color="lightsteelblue",
        showfliers=True
    )

    g.set_axis_labels("", "Rating (1-7)")
    g.set_titles("Question {col_name}")

    for ax in g.axes.flat:
        ax.set_ylim(0.5, 7.5)
        ax.set_yticks(range(SCALE_MIN, SCALE_MAX + 1))
        ax.tick_params(axis="x", rotation=25)

    g.figure.suptitle("Post-condition ratings by control mode", y=1.05)
    g.figure.savefig(WORK_DIR / "post_condition_boxplots.png", bbox_inches="tight", dpi=300)

    plt.close(g.figure)


def response_distribution(long_df):
    counts = long_df.groupby(["Question", CONTROL_COL, "Rating"], observed=True).size().rename("count").reset_index()

    counts["percentage"] = counts["count"] / counts.groupby(["Question", CONTROL_COL], observed=True)["count"].transform("sum") * 100

    grid = pd.MultiIndex.from_product([QUESTIONS, CONTROL_ORDER, range(SCALE_MIN, SCALE_MAX + 1)], names=["Question", CONTROL_COL, "Rating"]).to_frame(index=False)

    counts = grid.merge(counts, on=["Question", CONTROL_COL, "Rating"], how="left")

    counts[["count", "percentage"]] = counts[["count", "percentage"]].fillna(0)

    counts.to_csv(WORK_DIR / "post_condition_response_distribution.csv", index=False)

    g = sns.relplot(
        data=counts,
        x="Rating",
        y="percentage",
        hue=CONTROL_COL,
        hue_order=CONTROL_ORDER,
        col="Question",
        col_order=QUESTIONS,
        col_wrap=5,
        kind="line",
        marker="o",
        dashes=False,
        height=4.5,
        aspect=1.05
    )

    g.set_axis_labels("Rating (1-7)", "Percentage of responses")
    g.set_titles("Question {col_name}")

    for ax in g.axes.flat:
        ax.set_xlim(SCALE_MIN, SCALE_MAX)
        ax.set_xticks(range(SCALE_MIN, SCALE_MAX + 1))
        ax.set_ylim(bottom=0)

    g.figure.suptitle("Post-condition response distribution by control mode", y=1.05)
    g.figure.savefig(WORK_DIR / "post_condition_response_distribution.png", bbox_inches="tight", dpi=300)

    plt.close(g.figure)


def heatmap(long_df):
    medians = long_df.pivot_table(index=CONTROL_COL, columns="Question", values="Rating", aggfunc="median", observed=True)

    medians = medians.reindex(CONTROL_ORDER, columns=QUESTIONS)

    medians.to_csv(WORK_DIR / "post_condition_medians.csv")

    fig, ax = plt.subplots(figsize=(max(6, len(QUESTIONS) * 1.2), 4.5))

    sns.heatmap(
        medians,
        annot=True,
        fmt=".1f",
        cmap="vlag",
        center=4,
        vmin=SCALE_MIN,
        vmax=SCALE_MAX,
        linewidths=0.5,
        cbar_kws={"label": "Median Rating"},
        ax=ax
    )

    ax.set_title("Median post-condition ratings by control mode")
    ax.set_xlabel("Question")
    ax.set_ylabel("Control Mode")

    fig.savefig(WORK_DIR / "post_condition_median_heatmap.png", bbox_inches="tight", dpi=300)

    plt.close(fig)



def main():
    df = load_data()
    long_df = to_long_format(df)
    stats_df = basic_stats(long_df)
    plot_boxplots(long_df)
    response_distribution(long_df)
    heatmap(long_df)


if __name__ == "__main__":
    main()


