import pandas as pd
from itertools import permutations, combinations
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, norm
from statsmodels.stats.multitest import multipletests


INPUT_FILE = "/home/appuser/data/final/final_answers.csv"
PARTICIPANT_COL = "Participant ID"
MODES = ["pinch", "palm", "joy", "holo_joy"]


def kemeny_young(df):
    preferences = { a: { b: 0 for b in MODES if b != a } for a in MODES }

    # Count pairwise preferences
    for _, row in df[MODES].iterrows():
        for a in MODES:
            for b in MODES:
                if a != b and row[a] < row[b]:
                    preferences[a][b] += 1

    # Evaluate rankings
    results = []
    for ranking in permutations(MODES):
        score = sum(preferences[ranking[i]][ranking[j]]
                    for i in range(len(MODES)) 
                    for j in range(i + 1, len(MODES)))
        results.append((ranking, score))

    best_score = max(score for _, score in results)
    best_rankings = [ranking for ranking, score in results if score == best_score]

    max_score = len(df) * (len(MODES) * (len(MODES) - 1)) // 2

    borda_scores = (len(MODES) + 1 - df[MODES]).sum().sort_values(ascending=False)

    pairwise = pd.DataFrame([
        {"Mode A": a, "Mode B": b, 
         "Prefer A": preferences[a][b],
         "Prefer B": preferences[b][a]}
         for i, a in enumerate(MODES) 
         for b in MODES[i + 1:]
    ])

    return {
        "ranking": best_rankings,
        "score": best_score,
        "agreement": best_score / max_score if max_score else float("nan"),
        "borda_scores": borda_scores,
        "borda_ranking": borda_scores.index.tolist(),
        "pairwise": pairwise,
    }



def omnibus_test(df):
    n = len(df)
    k = len(MODES)

    friedman_stat, friedman_p = friedmanchisquare(*(df[mode].to_numpy() for mode in MODES))

    kendall_w = friedman_stat / (n * (k - 1))

    return {
        "friedman_statistic": friedman_stat,
        "friedman_p_value": friedman_p,
        "kendall_w": kendall_w,
        "friedman_significant": "Yes" if friedman_p < 0.05 else "No"
    }


def pairwise_wilcoxon(df):
    results = []

    for mode_a, mode_b in combinations(MODES, 2):
        a = df[mode_a]
        b = df[mode_b]

        differences = a - b

        stat, p_value = wilcoxon(differences, alternative='two-sided', zero_method='wilcox')

        nonzero = differences[differences != 0]
        ranks = nonzero.abs().rank()

        positive_ranks = ranks[nonzero > 0].sum()
        negative_ranks = ranks[nonzero < 0].sum()

        denominator = positive_ranks + negative_ranks

        effect = (
            (negative_ranks - positive_ranks) / denominator
            if denominator > 0 else 0.0
        )

        results.append({
            "Mode_A": mode_a,
            "Mode_B": mode_b,
            "Wilcoxon_statistic": stat,
            "effect_size": effect,
            "p_value": p_value,
            "significant": "Yes" if p_value < 0.05 else "No"
        })

    results_df = pd.DataFrame(results)

    results_df["p_holm"] = multipletests(results_df["p_value"], method="holm")[1]
    results_df["holm_significant"] = results_df["p_holm"].apply(lambda p: "Yes" if p < 0.05 else "No")

    results_df.to_csv("/home/appuser/data/final/pairwise_wilcoxon_results.csv", index=False)

    return results_df



def main():
    df = pd.read_csv(INPUT_FILE)
    result = kemeny_young(df)
    sig_result = omnibus_test(df)
    print("Friedman omnibus test")
    print(f"Statistic:   {sig_result['friedman_statistic']:.4f}")
    print(f"p-value:     {sig_result['friedman_p_value']:.6g}")
    print(f"Significant: {sig_result['friedman_significant']}")

    pairwise_results = pairwise_wilcoxon(df)
    result = kemeny_young(df)

    print("Best ranking(s):", result["ranking"])
    print(f"Agreement: {result['agreement']:.2%}")
    print(result["pairwise"])

    print("\nBorda scores:")
    print(result["borda_scores"])

    print("\nBorda ranking:", result["borda_ranking"])

if __name__ == "__main__":
    main()
