import itertools

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata, norm, studentized_range
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

RESULTS_DIR = "/home/appuser/data/track_analysis_selected"

iv = "track"
ALPHA = 0.05
ALPHA2 = 0.045


def within_subjects_pairwise(paired_data, shapiro_results, metrics):
    results = []

    for metric in metrics:
        data = paired_data[[f"{metric}_T1", f"{metric}_T2"]].dropna()

        t1 = paired_data[f"{metric}_T1"]
        t2 = paired_data[f"{metric}_T2"]   


        # Extract normality result from Shapiro-Wilk test results
        normality = shapiro_results.loc[shapiro_results["metric"] == metric, "normality"].values[0]

        differences = t2 - t1

        # Perform paired t-test or Wilcoxon signed-rank test based on normality
        if normality == "Normal":
            stat, p_value = stats.ttest_rel(t2, t1)
            test_name = "Paired t-test"

            mean_diff = differences.mean()
            sd_diff = differences.std(ddof=1)
    
            # Effect size (Cohen's d)
            effect_size_name = "Cohen's d"

            effect_size = mean_diff / sd_diff if sd_diff != 0 else float('inf')
    
            # Confidence interval
            standard_error = sd_diff / (len(differences) ** 0.5)
            t_critical = stats.t.ppf(1 - ALPHA / 2, df=len(differences) - 1)
    
            ci_lower = mean_diff - t_critical * standard_error
            ci_upper = mean_diff + t_critical * standard_error
        else:
            stat, p_value = stats.wilcoxon(t2, t1)
            test_name = "Wilcoxon signed-rank test"

            effect_size_name = "Rank Biserial Correlation"

            nonzero = differences[differences != 0]
            
            if len(nonzero) == 0:
                effect_size = np.nan
            else:
                ranks = stats.rankdata(np.abs(nonzero))
                positive_ranks = ranks[nonzero > 0].sum()
                negative_ranks = ranks[nonzero < 0].sum()

                effect_size = (positive_ranks - negative_ranks) / (positive_ranks + negative_ranks)

        
        results.append({
            "metric": metric,
            "test": test_name,
            "statistic": stat,
            "p_value": p_value,
            "significant": "Yes" if p_value < ALPHA else "No",
            "effect_size_name": effect_size_name,
            "effect_size": effect_size
        })

    results = pd.DataFrame(results)

    results.to_csv(f"{RESULTS_DIR}/within_subjects_pairwise_results.csv", index=False)


def within_subjects_omnibus(df, metrics, shapiro_results, sphericity_results):
    results = []

    for metric in metrics:
        data = df[[ "participant_id", iv, metric]].dropna()
        wide = data.pivot(index="participant_id", columns=iv, values=metric).dropna()
        data = wide.reset_index().melt(id_vars="participant_id", var_name=iv, value_name=metric)

        # Extract normality result from Shapiro-Wilk test results
        normality = shapiro_results.loc[shapiro_results["metric"] == metric, "normality"].values[0]

        # Extract sphericity result from Mauchly's test results
        sphericity = sphericity_results.loc[sphericity_results["metric"] == metric, "sphericity"].values[0]

        # Main test
        if normality == "Normal" and sphericity == "Sphericity":
            # Repeated measures ANOVA
            model = AnovaRM(data, depvar=metric, subject="participant_id", within=[iv]).fit()

            anova_table = model.anova_table

            statistic = anova_table.loc[iv, "F Value"]
            p_value = anova_table.loc[iv, "Pr > F"]
            df1 = anova_table.loc[iv, "Num DF"]
            df2 = anova_table.loc[iv, "Den DF"]
            test_name = "Repeated Measures ANOVA"

            # Effect size (Partial Eta Squared)
            partial_eta_squared = (statistic * df1) / (statistic * df1 + df2)
            effect_size = partial_eta_squared
            effect_size_name = "Partial Eta Squared"

        else:
            # Friedman test
            modes = list(data.columns)
            samples = [data[mode].values for mode in modes]

            statistic, p_value = stats.friedmanchisquare(*samples)

            test_name = "Friedman Test"

            # Kendall's W as effect size for Friedman test
            n = len(data)
            k = len(modes)

            kendalls_w = statistic / (n * (k - 1))
            effect_size = kendalls_w
            effect_size_name = "Kendall's W"



        results.append({
            "metric": metric,
            "test": test_name,
            "statistic": statistic,
            "p_value": p_value,
            "significant": "Yes" if p_value < ALPHA else "No",
            "effect_size_name": effect_size_name,
            "effect_size": effect_size
        })
            
    results = pd.DataFrame(results)
    
    results.to_csv(f"{RESULTS_DIR}/within_subjects_results.csv", index=False)    

    return results


def between_subjects_omnibus(df, metrics, shapiro_results, levene_results):
    results = []

    modes = sorted(df[iv].dropna().unique())
    k = len(modes)

    for metric in metrics:
        # Normality
        shapiro_metric = shapiro_results[shapiro_results["metric"] == metric]

        min_shapiro_p = shapiro_metric["p_value"].min()

        normal = min_shapiro_p > ALPHA2

        # Homogeneity of variance
        levene_metric = levene_results[levene_results["metric"] == metric]

        levene_p = levene_metric["p_value"].iloc[0]

        equal_variance = levene_p > ALPHA

        # Extract groups
        groups = [df.loc[df[iv] == mode, metric].dropna().to_numpy() for mode in modes]

        N = sum(len(group) for group in groups)

        # Omnibus test
        if not normal:
            # Kruskal-Wallis test
            stat, p_value = stats.kruskal(*groups)
            test_name = "Kruskal-Wallis Test"

            effect_size = max(0, (stat - k + 1) / (N - k))

            effect_size_name = "Epsilon Squared"
            test = "Kruskal-Wallis Test"

        elif not equal_variance:
            # Welch ANOVA
            data = df[[iv, metric]].dropna()
            welch = anova_oneway(data[metric], groups=data[iv], use_var="unequal")

            stat = welch.statistic
            p_value = welch.pvalue
            test_name = "Welch ANOVA"
            df1, df2 = welch.df

            effect_size = max(0, (df1 * (stat - 1)) / (df1 * stat + df2 + 1))

            test = "Welch ANOVA"
            effect_size_name = "Omega Squared"

        else:
            # One-way ANOVA
            data = df[[iv, metric]].dropna()
            anova = anova_oneway(data[metric], groups=data[iv], use_var="equal")

            stat = anova.statistic
            p_value = anova.pvalue
            test_name = "One-way ANOVA"
            df1, df2 = anova.df

            grand_mean = data[metric].mean()
            ss_between = sum(len(group) * (group[metric].mean() - grand_mean) ** 2 for _, group in data.groupby(iv))
            ss_total = ((data[metric] - grand_mean) ** 2).sum()

            effect_size = (ss_between / ss_total) if ss_total > 0 else np.nan

            test = "One-way ANOVA"
            effect_size_name = "Eta Squared"

        results.append({
            "metric": metric,
            "test": test_name,
            "statistic": stat,
            "p_value": p_value,
            "significant": "Yes" if p_value < ALPHA else "No",
            "effect_size_name": effect_size_name,
            "effect_size": effect_size
        })

    results = pd.DataFrame(results)

    results.to_csv(f"{RESULTS_DIR}/between_subjects_results.csv", index=False)

    return results




def post_hoc_within(df, main_test_results, metrics):
    results_post_hoc = []

    for metric in metrics:
        main_result = main_test_results.loc[main_test_results["metric"] == metric].iloc[0]

        if main_result["significant"] != "Yes":
            continue  # Skip post-hoc tests if the main test is not significant

        test_name = main_result["test"]

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

            if test_name == "Repeated Measures ANOVA":
                stat, p_value = stats.ttest_rel(y, x)
                test_type = "Paired t-test"

                mean_diff = differences.mean()
                sd_diff = differences.std(ddof=1)
        
        
                # Effect size (Cohen's d)
                cohen_d = mean_diff / sd_diff if sd_diff != 0 else float('inf')
        
                # Confidence interval
                standard_error = sd_diff / (len(differences) ** 0.5)
                t_critical = stats.t.ppf(1 - ALPHA / 2, df=len(differences) - 1)
        
                ci_lower = mean_diff - t_critical * standard_error
                ci_upper = mean_diff + t_critical * standard_error

                metric_results.append({
                    "metric": metric,
                    "mode1": mode1,
                    "mode2": mode2,
                    "test": test_type,
                    "statistic": stat,
                    "p_value": p_value,
                    "significant": "Yes" if p_value < ALPHA else "No",
                    "effect_size_name": "Cohen's d",
                    "effect_size": cohen_d
                })


            else:
                stat, p_value = stats.wilcoxon(y, x)
                test_type = "Wilcoxon signed-rank test"

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
                    "test": test_type,
                    "statistic": stat,
                    "p_value": p_value,
                    "significant": "Yes" if p_value < ALPHA else "No",
                    "effect_size_name": "Rank Biserial Correlation",
                    "effect_size": rank_biserial
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




def post_hoc_between(df, main_test_results, metrics):
    results_post_hoc = []

    for metric in metrics:
        main_result = main_test_results.loc[main_test_results["metric"] == metric].iloc[0]

        if main_result["significant"] != "Yes":
            continue  # Skip post-hoc tests if the main test is not significant

        test_name = main_result["test"]

        data = df[[iv, metric]].dropna()
        modes = sorted(data[iv].unique())
        
        if test_name == "One-way ANOVA":
            tukey = pairwise_tukeyhsd(endog=data[metric], groups=data[iv], alpha=ALPHA)

            for row in tukey.summary().data[1:]:

                group1 = row[0]
                group2 = row[1]
                p_adjusted = row[5]

                x1 = data.loc[data[iv] == group1, metric].to_numpy()
                x2 = data.loc[data[iv] == group2, metric].to_numpy()

                n1 = len(x1)
                n2 = len(x2)

                s1 = np.std(x1, ddof=1)
                s2 = np.std(x2, ddof=1)

                pooled_sd = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))

                effect_size = (np.mean(x1) - np.mean(x2)) / pooled_sd if pooled_sd > 0 else np.nan


                results_post_hoc.append({
                    "metric": metric,
                    "mode1": row[0],
                    "mode2": row[1],
                    "test": "Tukey HSD",
                    "statistic": row[2],
                    "p_value": row[3],
                    "significant": "Yes" if p_value < ALPHA else "No",
                    "effect_size_name": "Cohen's d",
                    "effect_size": effect_size,
                    "p_adjusted": p_adjusted,
                    "corrected_significant": "Yes" if p_adjusted < ALPHA else " No"
                })

        elif test_name == "Welch ANOVA":

            for group1, group2 in itertools.combinations(modes, 2):

                x1 = data.loc[data[iv] == group1, metric].to_numpy()
                x2 = data.loc[data[iv] == group2, metric].to_numpy()

                n1 = len(x1)
                n2 = len(x2)

                mean1 = np.mean(x1)
                mean2 = np.mean(x2)

                var1 = np.var(x1, ddof=1)
                var2 = np.var(x2, ddof=1)

                se = np.sqrt(var1 / n1 + var2 / n2)

                df_pair = ((var1 / n1 + var2 / n2) ** 2) / ((var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1))

                # Games-Howell q statistic
                q = abs(mean1 - mean2) / np.sqrt(0.5 * (var1 / n1 + var2 / n2))

                # Studentized range p value
                p_value = studentized_range.sf(q, k=len(modes), df=df_pair)


                # Hedges' g effect size
                pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

                d = ((mean1 - mean2) / pooled_sd) if pooled_sd > 0 else np.nan

                correction = (1 - 3 / (4 * (n1 + n2) - 9)) 

                g = d * correction


                results_post_hoc.append({
                    "metric": metric,
                    "mode1": group1,
                    "mode2": group2,
                    "test": "Games-Howell",
                    "statistic": q,
                    "p_value": p_value,
                    "significant": "Yes" if p_value < ALPHA else "No",
                    "effect_size_name": "Hedges' g",
                    "effect_size": g,
                    "p_adjusted": p_value, # Games-Howell does not require adjustment
                    "corrected_significant": "Yes" if p_value < ALPHA else "No"
                })

        elif test_name == "Kruskal-Wallis Test":

            pairs = list(itertools.combinations(modes, 2))

            raw_p_values = []
            pair_results = []

            for group1, group2 in pairs:
                x1 = data.loc[data[iv] == group1, metric].to_numpy()
                x2 = data.loc[data[iv] == group2, metric].to_numpy()

                # Mann-Whitney U test
                U, _ = stats.mannwhitneyu(x1, x2, alternative='two-sided')

                # Dunn-style rank comparison
                combined = np.concatenate([x1, x2])
                ranks = stats.rankdata(combined)

                n1 = len(x1)
                n2 = len(x2)
                N = n1 + n2

                rank1 = ranks[:n1].sum()
                rank2 = ranks[n1:].sum()

                mean_rank1 = rank1 / n1
                mean_rank2 = rank2 / n2

                # Tie correction
                _, counts = np.unique(ranks, return_counts=True)
                tie_term = np.sum(counts ** 3 - counts)
                tie_correction = 1 - tie_term / (N ** 3 - N)

                se = np.sqrt(N * (N + 1) / 12 * (1 / n1 + 1 / n2) * tie_correction)

                z = (mean_rank1 - mean_rank2) / se

                p_value = 2 * stats.norm.sf(abs(z))

                raw_p_values.append(p_value)

                # Rank biserial correlation
                r_rb = (2 * U) / (n1 * n2) - 1


                pair_results.append({
                    "metric": metric,
                    "mode1": group1,
                    "mode2": group2,
                    "test": "Dunn",
                    "statistic": z,
                    "p_value": p_value,
                    "significant": "Yes" if p_value < ALPHA else "No",
                    "effect_size_name": "Rank Biserial Correlation",
                    "effect_size": r_rb
                })

            # Holm correction for multiple comparisons
            adjusted = multipletests(raw_p_values, alpha=ALPHA, method='holm')

            for result, p_adjusted, significant in zip(pair_results, adjusted[1], adjusted[0]):
                result["p_adjusted"] = p_adjusted
                result["corrected_significant"] = "Yes" if significant else "No"
                results_post_hoc.append(result)

    results_post_hoc = pd.DataFrame(results_post_hoc)

    results_post_hoc.to_csv(f"{RESULTS_DIR}/post_hoc_results.csv", index=False)





def main():
    # metrics_path = "/home/appuser/data/metrics.csv"
    metrics_path = "/home/appuser/data/track_analysis_selected/metrics_selected.csv"
    df = pd.read_csv(metrics_path)

    shapiro_results_path = f"{RESULTS_DIR}/shapiro_wilk_results.csv"
    shapiro_results = pd.read_csv(shapiro_results_path)

    

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
    paired_data_path = f"{RESULTS_DIR}/paired_data.csv"
    paired_data = pd.read_csv(paired_data_path)
    within_subjects_pairwise(paired_data, shapiro_results, metrics)

    # Control analysis
    # sphericity_results_path = f"{RESULTS_DIR}/sphericity_results.csv"
    # sphericity_results = pd.read_csv(sphericity_results_path)
    # results = within_subjects(df, metrics, shapiro_results, sphericity_results)
    # post_hoc_tests(df, results, metrics)

    # Per-track control analysis
    # levene_results_path = f"{RESULTS_DIR}/levene_results.csv"
    # levene_results = pd.read_csv(levene_results_path)
    # results = between_subjects_omnibus(df, metrics, shapiro_results, levene_results)
    # post_hoc_between(df, results, metrics)


if __name__ == "__main__":
    main()



