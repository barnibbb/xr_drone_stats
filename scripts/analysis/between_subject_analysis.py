import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import rankdata, norm, studentized_range
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.oneway import anova_oneway

# Configuration
DATA_DIR = "/home/appuser/data"

METRICS_FILE = os.path.join(DATA_DIR, "metrics.csv")
ENCODING_FILE = os.path.join(DATA_DIR, "encoding.csv")

dependent = "waypoint_error_mean"
RESULTS_DIR = os.path.join(DATA_DIR, dependent)
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHA = 0.05


# Load data
def load_data():
    metrics = pd.read_csv(METRICS_FILE)
    encoding = pd.read_csv(ENCODING_FILE)

    metrics.columns = metrics.columns.str.strip()
    encoding.columns = encoding.columns.str.strip()

    metrics["participant_id"] = (metrics["participant_id"].astype(str).str.strip())
    encoding["ID"] = (encoding["ID"].astype(str).str.strip())

    return metrics, encoding


# Data selection
def select_data(metrics, encoding, track):
    if track == "T1":
        encoding_column = "Track 1"
    elif track == "T2":
        encoding_column = "Track 2"
    else:
        raise ValueError("Invalid track.")

    selected = encoding[[ "ID", encoding_column]].copy()
    selected = selected.rename(columns={"ID": "participant_id", encoding_column: "control_mode"})
    selected["track"] = track

    # Merge with metrics
    merged = pd.merge(selected, metrics, on=["participant_id", "control_mode", "track"], how="left")
    result = merged[[ "participant_id", "track", "control_mode", dependent]].copy()

    return result







# Basic statistics
def basic_statistics(data):
    rows = []

    for mode, group in data.groupby("control_mode"):
        values = group[dependent].dropna()

        n = len(values)
        mean = values.mean()
        std = values.std(ddof=1)
        median = values.median()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        sem = stats.sem(values)
        ci_low, ci_high = stats.t.interval(0.95, n-1, mean, sem)

        rows.append({
            "control_mode": mode,
            "n": n,
            "mean": mean,
            "std": std,
            "median": median,
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
            "min": values.min(),
            "max": values.max(),
            "ci_low": ci_low,
            "ci_high": ci_high
        })

    return pd.DataFrame(rows)







# Q-Q plot
def qq_plot(data, track):
    qq_dir = os.path.join(RESULTS_DIR, track, "qq_plots")
    os.makedirs(qq_dir, exist_ok=True)

    for mode, group in data.groupby("control_mode"):
        values = group[dependent].dropna()

        plt.figure(figsize=(6, 6))
        stats.probplot(values, dist="norm", plot=plt)
        plt.title(f"Q-Q Plot for {mode} - {track}")
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Sample Quantiles")

        plt.savefig(os.path.join(qq_dir, f"qq_plot_{mode}.png"))
        plt.close()


# Shapiro-Wilk test
def shapiro_wilk_test(data):
    rows = []

    for mode, group in data.groupby("control_mode"):
        values = group[dependent].dropna()
        W_stat, p_value = stats.shapiro(values)

        rows.append({
            "control_mode": mode,
            "statistic": W_stat,
            "p_value": p_value,
            "normality_reject": p_value < ALPHA
        })

    return pd.DataFrame(rows)


# Variance test
def levene_variance_test(data):
    groups = [group[dependent].dropna().values for _, group in data.groupby("control_mode")]

    stat, p_value = stats.levene(*groups, center='median')

    variance_results = {
        "statistic": stat,
        "p_value": p_value,
        "equal_variance_reject": p_value < ALPHA
    }

    return pd.DataFrame([variance_results])







# One-way ANOVA test (normal, equal variance)
def one_way_anova_test(data):
    groups = [group[dependent].dropna().values for _, group in data.groupby("control_mode")]

    stat, p_value = stats.f_oneway(*groups)

    anova_results = {
        "statistic": stat,
        "p_value": p_value,
        "reject_null": p_value < ALPHA
    }

    return pd.DataFrame([anova_results])


# Welch's ANOVA test (normal, unequal variance)
def welch_anova_test(data):
    groups = [group[dependent].dropna().values for _, group in data.groupby("control_mode")]

    result = anova_oneway(groups, use_var="unequal")

    welch_results = {
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "reject_null": result.pvalue < ALPHA
    }

    return pd.DataFrame([welch_results])


# Kruskal-Wallis test (not normal, not equal variance)
def kruskal_wallis_test(data):
    groups = [group[dependent].dropna().values for _, group in data.groupby("control_mode")]

    stat, p_value = stats.kruskal(*groups)

    kruskal_results = {
        "statistic": stat,
        "p_value": p_value,
        "reject_null": p_value < ALPHA
    }

    return pd.DataFrame([kruskal_results])







# Holm correction
def holm_correction(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    adjusted = np.empty(n, dtype=np.float64)

    for rank, index in enumerate(sorted_indices):
        adjusted[index] = (n - rank) * p_values[index]

    # Monotonicity
    for i in range(1, n):
        current = sorted_indices[i]
        previous = sorted_indices[i - 1]

        adjusted[current] = max(adjusted[current], adjusted[previous])

    adjusted = np.minimum(adjusted, 1.0)

    return adjusted







# Dunn's post-hoc test + Holm correction
def dunn_posthoc_test(data):
    values = data[dependent].to_numpy()
    groups = data["control_mode"].to_numpy()

    ranks = rankdata(values)

    ranked_data = pd.DataFrame({"group": groups, "rank": ranks})

    group_stats = (ranked_data.groupby("group").agg(n=("rank", "size"), mean_rank=("rank", "mean")))

    N = len(values)

    # Tie correction
    _, counts = np.unique(values, return_counts=True)
    tie_sum = np.sum(counts[counts > 1] ** 3 - counts[counts > 1])
    tie_correction = 1 - tie_sum / (N ** 3 - N)

    modes = sorted(group_stats.index)

    rows = []

    for i in range(len(modes)):
        for j in range(i + 1, len(modes)):
            mode_i = modes[i]
            mode_j = modes[j]

            n_i = group_stats.loc[mode_i, "n"]
            n_j = group_stats.loc[mode_j, "n"]

            mean_rank_i = group_stats.loc[mode_i, "mean_rank"]
            mean_rank_j = group_stats.loc[mode_j, "mean_rank"]

            standard_error = np.sqrt((N * (N + 1) / 12) * (1 / n_i + 1 / n_j) * tie_correction)
            z_stat = (mean_rank_i - mean_rank_j) / standard_error
            p_value = 2 * norm.sf(np.abs(z_stat))

            rows.append({
                "group1": mode_i,
                "group2": mode_j,
                "z_stat": z_stat,
                "p_value": p_value
            })

    result = pd.DataFrame(rows)
    result["p_holm"] = holm_correction(result["p_value"].values)
    result["significant"] = result["p_holm"] < ALPHA

    return result


# Tukey's HSD test
def tukey_hsd_test(data):
    groups = data[[dependent, "control_mode"]].dropna()

    result = pairwise_tukeyhsd(endog=groups[dependent], groups=groups["control_mode"], alpha=ALPHA)

    rows = []

    for row in result.summary().data[1:]:
        rows.append({
            "group1": row[0],
            "group2": row[1],
            "meandiff": row[2],
            "p_value": row[3],
            "lower": row[4],
            "upper": row[5],
            "reject_null": row[6]
        })

    return pd.DataFrame(rows)


# Games-Howell test
def games_howell_test(data):
    groups = data[[dependent, "control_mode"]].dropna()
    grouped = groups.groupby("control_mode")[dependent]
    group_names = sorted(grouped.groups.keys())
    
    rows = []

    k = len(group_names)

    for i in range(k):
        for j in range(i + 1, k):
            group_i = group_names[i]
            group_j = group_names[j]

            x = grouped.get_group(group_i).values
            y = grouped.get_group(group_j).values

            n1 = len(x)
            n2 = len(y)

            mean1 = np.mean(x)
            mean2 = np.mean(y)

            var1 = np.var(x, ddof=1)
            var2 = np.var(y, ddof=1)

            mean_diff = mean1 - mean2
            se_diff = np.sqrt(var1 / n1 + var2 / n2)

            df_num = (var1 / n1 + var2 / n2) ** 2
            df_den = ((var1 / n1) ** 2) / (n1 - 1) + ((var2 / n2) ** 2) / (n2 - 1)
            df = df_num / df_den

            q_stat = np.abs(mean_diff) / np.sqrt(0.5 * (var1 / n1 + var2 / n2))

            p_value = studentized_range.sf(q_stat * np.sqrt(2), k, df)

            q_critical = studentized_range.ppf(1 - ALPHA, k, df)

            margin_of_error = q_critical * np.sqrt(0.5 * (var1 / n1 + var2 / n2))

            rows.append({
                "group1": group_i,
                "group2": group_j,
                "mean_diff": mean_diff,
                "q_stat": q_stat,
                "df": df,
                "q_critical": q_critical,
                "margin_of_error": margin_of_error,
                "p_value": p_value,
                "significant": p_value < ALPHA
            })

    return pd.DataFrame(rows)





# Pairwise Cohen's d effect size
def cohen_d(data):
    modes = sorted(data["control_mode"].unique())
    rows = []

    for i in range(len(modes)):
        for j in range(i + 1, len(modes)):
            mode_i = modes[i]
            mode_j = modes[j]

            x = data.loc[data["control_mode"] == mode_i, dependent].dropna().to_numpy()
            y = data.loc[data["control_mode"] == mode_j, dependent].dropna().to_numpy()

            n_x = len(x)
            n_y = len(y)

            mean_x = np.mean(x)
            mean_y = np.mean(y)

            std_x = np.std(x, ddof=1)
            std_y = np.std(y, ddof=1)

            pooled_std = np.sqrt(((n_x - 1) * std_x ** 2 + (n_y - 1) * std_y ** 2) / (n_x + n_y - 2))
            d = (mean_x - mean_y) / pooled_std

            rows.append({
                "group1": mode_i,
                "group2": mode_j,
                "cohen_d": d
            })

    result = pd.DataFrame(rows)
    return result








def generate_report(track, basic_stats, shapiro_results, variance_results, kruskal_results, anova_results, welch_results, tukey_results, games_howell_results, dunn_results, cohen_d_results, track_dir):
    report_file = os.path.join(RESULTS_DIR, track, "report.txt")

    with open(report_file, "w") as f:
        f.write("DESCRIPTIVE STATISTICS\n") 
        f.write("-" * 60 + "\n")
        cols = [ "control_mode", "n", "mean", "std", "median", "q1", "q3", "iqr" ] 
        f.write( basic_stats[cols].to_string( index=False, float_format=lambda x: f"{x:.3f}" ) ) 
        f.write("\n\n\n")

        f.write("NORMALITY: SHAPIRO-WILK\n") 
        f.write("-" * 60 + "\n")
        f.write( shapiro_results.to_string( index=False, float_format=lambda x: f"{x:.3f}" ) )
        f.write("\n\n\n")

        f.write("VARIANCE: LEVENE\n")
        f.write("-" * 60 + "\n")
        variance_stat = variance_results.iloc[0]["statistic"]
        variance_p = variance_results.iloc[0]["p_value"]
        variance_reject = variance_results.iloc[0]["equal_variance_reject"]
        f.write(f"Levene's test statistic: {variance_stat:.3f}, p-value: {variance_p:.3f}\n")
        if variance_reject:
            f.write("Variance is not equal across groups.\n")
        else:
            f.write("Variance is equal across groups.\n")
        f.write("\n\n\n")

        f.write("KRUSKAL-WALLIS TEST\n")
        f.write("-" * 60 + "\n")
        kruskal_stat = kruskal_results.iloc[0]["statistic"]
        kruskal_p = kruskal_results.iloc[0]["p_value"]
        kruskal_reject = kruskal_results.iloc[0]["reject_null"]
        f.write(f"Kruskal-Wallis test statistic: {kruskal_stat:.3f}, p-value: {kruskal_p:.3f}\n")
        if kruskal_reject:
            f.write("There is a significant difference between the groups.\n")
        else:
            f.write("There is no significant difference between the groups.\n")
        f.write("\n\n\n")


        f.write("ONE-WAY ANOVA TEST\n")
        f.write("-" * 60 + "\n")
        anova_stat = anova_results.iloc[0]["statistic"]
        anova_p = anova_results.iloc[0]["p_value"]
        anova_reject = anova_results.iloc[0]["reject_null"]
        f.write(f"One-way ANOVA test statistic: {anova_stat:.3f}, p-value: {anova_p:.3f}\n")
        if anova_reject:
            f.write("There is a significant difference between the groups.\n")
        else:
            f.write("There is no significant difference between the groups.\n")
        f.write("\n\n\n")

        f.write("WELCH'S T-TEST\n")
        f.write("-" * 60 + "\n")
        welch_stat = welch_results.iloc[0]["statistic"]
        welch_p = welch_results.iloc[0]["p_value"]
        welch_reject = welch_results.iloc[0]["reject_null"]
        f.write(f"Welch's t-test statistic: {welch_stat:.3f}, p-value: {welch_p:.3f}\n")
        if welch_reject:
            f.write("There is a significant difference between the groups.\n")
        else:
            f.write("There is no significant difference between the groups.\n")
        f.write("\n\n\n")

        f.write("TUKEY'S POST-HOC TEST\n")
        f.write("-" * 60 + "\n")
        tukey_cols = [ "group1", "group2", "meandiff", "p_value", "lower", "upper", "reject_null" ]
        f.write( tukey_results[tukey_cols].to_string( index=False, float_format=lambda x: f"{x:.3f}" ) )
        f.write("\n\n\n")

        f.write("GAMES-HOWELL POST-HOC TEST\n")
        f.write("-" * 60 + "\n")
        games_howell_cols = [ "group1", "group2", "mean_diff", "q_stat", "df", "q_critical", "margin_of_error", "p_value", "significant" ]
        f.write( games_howell_results[games_howell_cols].to_string( index=False, float_format=lambda x: f"{x:.3f}" ) )
        f.write("\n\n\n")

        f.write("DUNN'S POST-HOC TEST (HOLM CORRECTION)\n")
        f.write("-" * 60 + "\n")
        dunn_cols = [ "group1", "group2", "z_stat", "p_value", "p_holm", "significant" ]
        f.write( dunn_results[dunn_cols].to_string( index=False, float_format=lambda x: f"{x:.3f}" ) )
        f.write("\n\n\n")

        f.write("PAIRWISE COHEN'S D EFFECT SIZE\n")
        f.write("-" * 60 + "\n")
        f.write( cohen_d_results.to_string( index=False, float_format=lambda x: f"{x:.3f}" ) )
        f.write("\n\n\n")




def main():
    # Load
    metrics, encoding = load_data()

    for track in ["T1", "T2"]:
        selected = select_data(metrics, encoding, track)

        track_dir = os.path.join(RESULTS_DIR, track)
        os.makedirs(track_dir, exist_ok=True)


        # Basic statistics
        basic_stats = basic_statistics(selected)
        basic_stats.to_csv(os.path.join(track_dir, "basic_stats.csv"), index=False)
        print(f"Basic statistics for track {track}:")
        print(basic_stats.to_string(index=False))


        # Q-Q plot
        qq_plot(selected, track)


        # Shapiro-Wilk test
        shapiro_results = shapiro_wilk_test(selected)
        shapiro_results.to_csv(os.path.join(track_dir, "shapiro_wilk_test.csv"), index=False)
        print(f"\nShapiro-Wilk test for track {track}:")
        print(shapiro_results.to_string(index=False))

        
        # Levene variance test
        variance_results = levene_variance_test(selected)
        variance_results.to_csv(os.path.join(track_dir, "levene_variance_test.csv"), index=False)
        print(f"\nVariance test for track {track}:")
        print(variance_results.to_string(index=False))


        # Kruskal-Wallis test
        kruskal_results = kruskal_wallis_test(selected)
        kruskal_results.to_csv(os.path.join(track_dir, "kruskal_wallis_test.csv"), index=False)
        print(f"\nKruskal-Wallis test for track {track}:")
        print(kruskal_results.to_string(index=False))


        # One-way ANOVA test
        anova_results = one_way_anova_test(selected)
        anova_results.to_csv(os.path.join(track_dir, "one_way_anova_test.csv"), index=False)
        print(f"\nOne-way ANOVA test for track {track}:")
        print(anova_results.to_string(index=False))

        # Welch's ANOVA test
        welch_results = welch_anova_test(selected)
        welch_results.to_csv(os.path.join(track_dir, "welch_anova_test.csv"), index=False)
        print(f"\nWelch's ANOVA test for track {track}:")
        print(welch_results.to_string(index=False))


        # Dunn's post-hoc test
        dunn_results = dunn_posthoc_test(selected)
        dunn_results.to_csv(os.path.join(track_dir, "dunn_posthoc_test.csv"), index=False)  
        print(f"\nDunn's post-hoc test for track {track}:")
        print(dunn_results.to_string(index=False))


        # Tukey's HSD test
        tukey_results = tukey_hsd_test(selected)
        tukey_results.to_csv(os.path.join(track_dir, "tukey_hsd_test.csv"), index=False)
        print(f"\nTukey's HSD test for track {track}:")
        print(tukey_results.to_string(index=False))


        # Games-Howell test
        games_howell_results = games_howell_test(selected)
        games_howell_results.to_csv(os.path.join(track_dir, "games_howell_test.csv"), index=False)
        print(f"\nGames-Howell test for track {track}:")
        print(games_howell_results.to_string(index=False))
        

        # Pairwise Cohen's d effect size
        cohen_d_results = cohen_d(selected)
        cohen_d_results.to_csv(os.path.join(track_dir, "cohen_d.csv"), index=False)
        print(f"\nPairwise Cohen's d effect size for track {track}:")
        print(cohen_d_results.to_string(index=False))


        # Generate compact statistical report 
        report_file = generate_report( 
            track=track, 
            basic_stats=basic_stats, 
            shapiro_results=shapiro_results, 
            variance_results=variance_results, 
            kruskal_results=kruskal_results,
            anova_results=anova_results,
            welch_results=welch_results, 
            dunn_results=dunn_results, 
            tukey_results=tukey_results,
            games_howell_results=games_howell_results,
            cohen_d_results=cohen_d_results, 
            track_dir=track_dir) 

        print(f"\nStatistical report saved to: {report_file}")



if __name__ == "__main__":
    main()
