import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

def calculate_wilcoxon_test(list_A, list_B, metric_name, dataset_name, method, layer="N/A"):
    """
    Computes the Wilcoxon Signed-Rank Test between two paired lists of metrics (e.g., Setup A vs Setup B).
    """
    # If the lists are exactly identical, wilcoxon returns warning or error about zero differences
    diffs = np.array(list_A) - np.array(list_B)
    if np.all(diffs == 0):
        stat, p_val = 0.0, 1.0 # No difference
    else:
        stat, p_val = wilcoxon(list_A, list_B)
        
    return {
        "Dataset": dataset_name,
        "Method": method,
        "Layer": layer,
        "Metric": metric_name,
        "Mean A": np.mean(list_A),
        "Mean B": np.mean(list_B),
        "W-Statistic": stat,
        "P-Value": p_val,
        "Significant (p<0.05)": p_val < 0.05
    }

def compare_results_imbalance(wb_metrics, bb_metrics):
    """
    MODIFIED for Class Imbalance Study:
    Parses and compares accuracy, auc, macro_f1, recall_true, and recall_false.
    """
    rows = []
    
    # Add BB Row
    rows.append({
        "Method": "Black Box (Logprobs)",
        "Layer": "N/A",
        "Accuracy": bb_metrics.get('accuracy', 0),
        "AUC": bb_metrics.get('auc', 0),
        "Macro F1": bb_metrics.get('macro_f1', 0),
        "Recall (True)": bb_metrics.get('recall_true', 0),
        "Recall (False)": bb_metrics.get('recall_false', 0),
    })
    
    # Add WB Rows
    for layer, met in wb_metrics.items():
        rows.append({
            "Method": "White Box (Probes)",
            "Layer": layer,
            "Accuracy": met.get('accuracy', 0),
            "AUC": met.get('auc', 0),
            "Macro F1": met.get('macro_f1', 0),
            "Recall (True)": met.get('recall_true', 0),
            "Recall (False)": met.get('recall_false', 0),
        })
        
    df = pd.DataFrame(rows)
    return df
