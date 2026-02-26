import pandas as pd

def compare_results(wb_metrics, bb_metrics):
    """
    Prints a summary table comparing the best WB layer vs BB.
    """
    rows = []
    
    # Add BB Row
    rows.append({
        "Method": "Black Box (Logprobs)",
        "Layer": "N/A",
        "Accuracy": bb_metrics['accuracy'],
        "AUC": bb_metrics['auc']
    })
    
    # Add WB Rows
    for layer, met in wb_metrics.items():
        rows.append({
            "Method": "White Box (Probes)",
            "Layer": layer,
            "Accuracy": met['accuracy'],
            "AUC": met['auc']
        })
        
    df = pd.DataFrame(rows)
    print("\n=== Final Comparison Results ===")
    print(df.sort_values(by="AUC", ascending=False))
    return df
