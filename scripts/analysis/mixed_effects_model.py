import numpy as np
import pandas as pd

import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests


RESULTS_DIR = "/home/appuser/data/balanced_2x3"
ALPHA = 0.05

METRICS = [
    "mission_time",
    "total_distance",
    "mean_curvature",
    "speed_cv",
    "rms_jerk",
    "path_efficiency",
    "ordered_error",
    "segment_error"
]


def fit_mixed_effects_model(df, metric):
    # Fit the mixed-effects model
    model = smf.mixedlm(f"{metric} ~ C(track) * C(control_mode)", data=df, groups=df["participant_id"])
    result = model.fit()

    omnibus = result.wald_test_terms()

    tests = [
        ("Track", "C(track)"),
        ("Control Mode", "C(control_mode)"),
        ("Track x Control Mode", "C(track):C(control_mode)")
    ]

    results = []

    for test_name, term in tests:
        statistic = omnibus.table.loc[term, "statistic"]
        df_test = omnibus.table.loc[term, "df_constraint"]
        p_value = omnibus.table.loc[term, "pvalue"]

        effect_size = statistic / (statistic + df_test)

        results.append({
            "metric": metric,
            "test": test_name,
            "statistic": statistic,
            "p_value": p_value,
            "significant": "Yes" if p_value < ALPHA else "No",
            "effect_size": effect_size
        })

    omnibus_results = pd.DataFrame(results)

    # Significance of effects
    track_significance = omnibus.table.loc["C(track)", "pvalue"] < ALPHA
    control_mode_significance = omnibus.table.loc["C(control_mode)", "pvalue"] < ALPHA
    interaction_significance = omnibus.table.loc["C(track):C(control_mode)", "pvalue"] < ALPHA


    return result, omnibus_results, track_significance, control_mode_significance, interaction_significance


def posthoc_interaction(result, metric):
    
    beta = result.fe_params
    cov_beta = result.cov_params().loc[beta.index, beta.index]

    # Matrix for track x control-mode combinations
    design = {
        ("T1", "joy"):   [1, 0, 0, 0, 0, 0],
        ("T1", "palm"):  [1, 0, 1, 0, 0, 0],
        ("T1", "pinch"): [1, 0, 0, 1, 0, 0],
        ("T2", "joy"):   [1, 1, 0, 0, 0, 0],
        ("T2", "palm"):  [1, 1, 1, 0, 1, 0],  
        ("T2", "pinch"): [1, 1, 0, 1, 0, 1]  
    }

    comparisons = [
        ("T1", "joy", "palm"),
        ("T1", "joy", "pinch"),
        ("T1", "palm", "pinch"),
        ("T2", "joy", "palm"),
        ("T2", "joy", "pinch"),
        ("T2", "palm", "pinch")
    ]


    results = []

    for track, mode1, mode2 in comparisons:

        x1 = np.array(design[(track, mode1)])
        x2 = np.array(design[(track, mode2)])

        contrast = x1 - x2

        estimate = contrast @ beta
        variance = contrast @ cov_beta @ contrast
        se = variance ** 0.5
        
        statistic = estimate / se
        p_value = 2 * stats.norm.sf(abs(statistic))

        ci_lower = estimate - 1.96 * se
        ci_upper = estimate + 1.96 * se

        residual_sd = np.sqrt(result.scale)
        effect_size = estimate / residual_sd

        results.append({
            "metric": metric,
            "track": track,
            "mode1": mode1,
            "mode2": mode2,
            # "test": "LMM contrast",
            "statistic": statistic,
            "p_value": p_value,
            "significant": "Yes" if p_value < ALPHA else "No",
            # "effect_size_name": "Standardized Mean Difference (SMD)",
            "effect_size": effect_size
        })

    results = pd.DataFrame(results)


    # Adjust p-values for multiple comparison
    reject, p_adjusted, _, _ = multipletests(results["p_value"], alpha=ALPHA, method="holm")
      
    results["p_adjusted"] = p_adjusted
    results["significant_adjusted"] = np.where(reject, "Yes", "No")

    return results
        

def posthoc_control_mode(result, metric):
    beta = result.fe_params
    cov_beta = result.cov_params().loc[beta.index, beta.index]


    # Estimated marginal means for control modes averaged across tracks
    design = {
        "joy":   np.array([1, 0.5, 0, 0, 0,   0  ]),
        "palm":  np.array([1, 0.5, 1, 0, 0.5, 0  ]),
        "pinch": np.array([1, 0.5, 0, 1, 0,   0.5])
    }

    comparisons = [
        ("joy", "palm"),
        ("joy", "pinch"),
        ("palm", "pinch")
    ]

    results = []

    for mode1, mode2 in comparisons:
        
        x1 = design[mode1]
        x2 = design[mode2]

        contrast = x1 - x2

        estimate = contrast @ beta
        variance = contrast @ cov_beta @ contrast
        se = variance ** 0.5
        
        statistic = estimate / se
        p_value = 2 * stats.norm.sf(abs(statistic))

        ci_lower = estimate - 1.96 * se
        ci_upper = estimate + 1.96 * se

        residual_sd = np.sqrt(result.scale)
        effect_size = estimate / residual_sd

        results.append({
            "metric": metric,
            "track": "Average",
            "mode1": mode1,
            "mode2": mode2,
            # "test": "LMM contrast",
            "statistic": statistic,
            "p_value": p_value,
            "significant": "Yes" if p_value < ALPHA else "No",
            # "effect_size_name": "Standardized Mean Difference (SMD)",
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


    # Estimated marginal means for control modes averaged across tracks
    design = {
        "T1": np.array([1, 0, 1/3, 1/3, 0,   0  ]),
        "T2": np.array([1, 1, 1/3, 1/3, 1/3, 1/3])
    }

    x1 = design["T1"]
    x2 = design["T2"]

    contrast = x1 - x2

    estimate = contrast @ beta
    variance = contrast @ cov_beta @ contrast
    se = variance ** 0.5
    
    statistic = estimate / se
    p_value = 2 * stats.norm.sf(abs(statistic))

    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se

    residual_sd = np.sqrt(result.scale)
    effect_size = estimate / residual_sd

    results = pd.DataFrame([{
        "metric": metric,
        "track": "Overall",
        "mode1": "T1",
        "mode2": "T2",
        # "test": "LMM contrast",
        "statistic": statistic,
        "p_value": p_value,
        "significant": "Yes" if p_value < ALPHA else "No",
        # "effect_size_name": "Standardized Mean Difference (SMD)",
        "effect_size": effect_size,
        "p_adjusted": p_value,
        "significant_adjusted": "Yes" if p_value < ALPHA else "No"
    }])

    results = pd.DataFrame(results)

    return results


    

def main():
    metrics_path = f"{RESULTS_DIR}/balanced_subset_2x3.csv"
    df = pd.read_csv(metrics_path)

    all_omnibus_results = []
    all_posthoc_results = []

    for metric in METRICS:
        result, omnibus_results, track_significance, control_mode_significance, interaction_significance = fit_mixed_effects_model(df, metric)
        all_omnibus_results.append(omnibus_results)

        if interaction_significance:
            posthoc_result = posthoc_interaction(result, metric)
            all_posthoc_results.append(posthoc_result)

        if control_mode_significance:
            posthoc_result = posthoc_control_mode(result, metric)
            all_posthoc_results.append(posthoc_result)

        if track_significance:
            posthoc_result = posthoc_track(result, metric)
            all_posthoc_results.append(posthoc_result)


    all_omnibus_results = pd.concat(all_omnibus_results, ignore_index=True)
    all_omnibus_results.to_csv(f"{RESULTS_DIR}/mixed_effects_model_results.csv", index=False)

    all_posthoc_results = pd.concat(all_posthoc_results, ignore_index=True)
    all_posthoc_results.to_csv(f"{RESULTS_DIR}/posthoc_interaction_results.csv", index=False)



if __name__ == "__main__":
    main()

