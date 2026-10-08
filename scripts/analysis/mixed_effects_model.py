import os

import numpy as np
import pandas as pd

import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests


RESULTS_DIR = "/home/appuser/data/lmm_lim"
FIGURES_DIR = f"{RESULTS_DIR}/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
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


def make_design_vector(beta_index, track, control_mode):
    x = np.zeros(len(beta_index))

    # Intercept
    x[beta_index.get_loc("Intercept")] = 1

    # Track
    if track == "T2":
        x[beta_index.get_loc("C(track)[T.T2]")] = 1

    # Control mode
    if control_mode != "holo_joy":
        x[
            beta_index.get_loc(
                f"C(control_mode)[T.{control_mode}]"
            )
        ] = 1

    # Track × control-mode interaction
    if track == "T2" and control_mode != "holo_joy":
        x[
            beta_index.get_loc(
                f"C(track)[T.T2]:C(control_mode)[T.{control_mode}]"
            )
        ] = 1

    return x

def make_weighted_design_vector(beta_index, control_mode, t2_weight):
    t1_weight = 1 - t2_weight

    x_T1 = make_design_vector(beta_index, track="T1", control_mode=control_mode)
    x_T2 = make_design_vector(beta_index, track="T2", control_mode=control_mode)

    return t1_weight * x_T1 + t2_weight * x_T2




def fit_mixed_effects_model(df, metric):
    # Fit the mixed-effects model
    model = smf.mixedlm(f"{metric} ~ C(track) * C(control_mode)", data=df, groups=df["participant_id"])
    result = model.fit(reml=False)

    print(result.converged)

    omnibus = result.wald_test_terms()

    tests = [
        ("Track", "C(track)"),
        ("Control Mode", "C(control_mode)"),
        ("Track x Control Mode", "C(track):C(control_mode)")
    ]

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

    # Significance of effects
    track_significance = omnibus.table.loc["C(track)", "pvalue"] < ALPHA
    control_mode_significance = omnibus.table.loc["C(control_mode)", "pvalue"] < ALPHA
    interaction_significance = omnibus.table.loc["C(track):C(control_mode)", "pvalue"] < ALPHA


    return result, omnibus_results, track_significance, control_mode_significance, interaction_significance


def posthoc_interaction(result, metric):
    
    beta = result.fe_params
    cov_beta = result.cov_params().loc[beta.index, beta.index]

    print(beta.index.tolist())
    print(beta)
    

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
    print(result.fe_params)

    beta = result.fe_params
    cov_beta = result.cov_params().loc[beta.index, beta.index]


    # design = {
    #     "holo_joy": np.array([1, 0.5, 0, 0, 0, 0,   0,   0]),
    #     "joy":      np.array([1, 0.5, 1, 0, 0, 0.5, 0,   0]),
    #     "palm":     np.array([1, 0.5, 0, 1, 0, 0,   0.5, 0]),
    #     "pinch":    np.array([1, 0.5, 0, 0, 1, 0,   0,   0.5])
    # }
    
    mode_distribution = {
        "holo_joy": (1, 35),
        "joy":      (24, 12),
        "palm":     (24, 12),
        "pinch":    (23, 13)
    }

    design = {}

    for mode, (n_T1, n_T2) in mode_distribution.items():
        total = n_T1 + n_T2
        t2_weight = n_T2 / total

        design[mode] = make_weighted_design_vector(beta.index, control_mode=mode, t2_weight=t2_weight)

        # x_T1 = make_design_vector(beta.index, track="T1", control_mode=mode)
        # x_T2 = make_design_vector(beta.index, track="T2", control_mode=mode)        
        # design[mode] = (x_T1 + x_T2) / 2



    print(f"Design matrix for control modes averaged across tracks:")
    for mode, vec in design.items():
        print(f"  {mode}: {vec}")


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
        
        x1 = design[mode1]
        x2 = design[mode2]

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
            "track": "Average",
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


    residual_sd = np.sqrt(result.scale)
    effect_size = estimate / residual_sd

    results = pd.DataFrame([{
        "metric": metric,
        "track": "Overall",
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
    metrics_path = f"/home/appuser/data/metrics_lim.csv"
    df = pd.read_csv(metrics_path)

    all_omnibus_results = []
    all_posthoc_results = []

    for metric in METRICS:
        result, omnibus_results, track_significance, control_mode_significance, interaction_significance = fit_mixed_effects_model(df, metric)
        all_omnibus_results.append(omnibus_results)

        # if interaction_significance:
        #     posthoc_result = posthoc_interaction(result, metric)
        #     all_posthoc_results.append(posthoc_result)

        if control_mode_significance:
            posthoc_result = posthoc_control_mode(result, metric)
            all_posthoc_results.append(posthoc_result)

        # if track_significance:
        #     posthoc_result = posthoc_track(result, metric)
        #     all_posthoc_results.append(posthoc_result)


    all_omnibus_results = pd.concat(all_omnibus_results, ignore_index=True)
    all_omnibus_results.to_csv(f"{RESULTS_DIR}/mixed_effects_model_results.csv", index=False)

    all_posthoc_results = pd.concat(all_posthoc_results, ignore_index=True)
    all_posthoc_results.to_csv(f"{RESULTS_DIR}/posthoc_interaction_results.csv", index=False)



if __name__ == "__main__":
    main()

