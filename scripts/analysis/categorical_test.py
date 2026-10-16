import pandas as pd
import numpy as np
from scipy.stats import chisquare
from statsmodels.stats.contingency_tables import cochrans_q
from itertools import combinations
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests


INPUT_FILE = "/home/appuser/data/final/final_answers.csv"
MODES = ["pinch", "palm", "joy", "holo_joy"]
QUESTIONS = ["Q1", "Q2", "Q3"]

def main():
    df = pd.read_csv(INPUT_FILE)
    counts = pd.DataFrame({q: df[q].value_counts().reindex(MODES, fill_value=0) for q in QUESTIONS}).T

    percentages = counts.div(counts.sum(axis=1), axis=0) * 100

    summary = pd.concat([counts.add_suffix(" (n)"), percentages.round(1).add_suffix(" (%)")], axis=1)

    for q in QUESTIONS:
        observed = df[q].value_counts().reindex(MODES, fill_value=0)
        stat, p = chisquare(observed)
        # print(f"{q}: chi2={stat:.3f}, p={p:.4f}")

    for mode in MODES:
        selected = pd.DataFrame({q: (df[q] == mode).astype(int) for q in QUESTIONS})
        result = cochrans_q(selected.to_numpy())

        # print(f"{mode}: Cochran's Q={result.statistic:.3f}, p={result.pvalue:.4f}")


    pairwise_results = []
    for q in QUESTIONS:
        comparisons = []

        for a, b in combinations(MODES, 2):
            count_a = (df[q] == a).sum()
            count_b = (df[q] == b).sum()
            n = len(df)

            stat, p = proportions_ztest(count=[count_a, count_b], nobs=[n, n], alternative='two-sided')

            comparisons.append({
                "Question": q,
                "Mode_A": a,
                "Mode_B": b,
                "Z_statistic": stat,
                "p_value": p,
                "significant": "Yes" if p < 0.05 else "No"
            })

        adjusted_p = multipletests([x["p_value"] for x in comparisons], method="holm")[1]

        for result, adj_p in zip(comparisons, adjusted_p):
            result["p_holm"] = adj_p
            result["holm_significant"] = "Yes" if adj_p < 0.05 else "No"
            pairwise_results.append(result)

    pairwise_df = pd.DataFrame(pairwise_results)
    pairwise_df.to_csv("/home/appuser/data/final/pairwise_ztest_results.csv", index=False)            

    



if __name__ == "__main__":
    main()
