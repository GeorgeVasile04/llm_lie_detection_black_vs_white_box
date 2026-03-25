import pandas as pd

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
