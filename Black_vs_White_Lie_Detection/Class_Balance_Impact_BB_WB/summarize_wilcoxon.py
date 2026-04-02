import pandas as pd

def analyze_wilcoxon_results(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return

    metrics = ["AUC", "Macro F1"]
    
    print("=" * 80)
    print(f"{'📊 WILCOXON SIGNED-RANK TEST: SCENARIO A (Balanced) vs SCENARIO B (Imbalanced) 📊':^80}")
    print("=" * 80)
    
    for metric in metrics:
        # Filter data for the specific metric
        df_metric = df[df["Metric"] == metric].copy()
        
        total_cases = len(df_metric)
        if total_cases == 0:
            continue
            
        # Calculate the absolute difference between Scenario A and Scenario B
        df_metric["Diff_A_minus_B"] = df_metric["Mean A"] - df_metric["Mean B"]
        
        # Count cases where Setup A was statistically significantly BETTER than Setup B
        a_better_mask = (df_metric["Significant (p<0.05)"] == True) & (df_metric["Diff_A_minus_B"] > 0)
        a_better_count = a_better_mask.sum()
        a_better_pct = (a_better_count / total_cases) * 100
        
        # Count cases where Setup B was statistically significantly BETTER than Setup A
        b_better_mask = (df_metric["Significant (p<0.05)"] == True) & (df_metric["Diff_A_minus_B"] < 0)
        b_better_count = b_better_mask.sum()
        b_better_pct = (b_better_count / total_cases) * 100
        
        # Count ties / non-significant differences
        tie_count = total_cases - a_better_count - b_better_count
        tie_pct = (tie_count / total_cases) * 100
        
        # Calculate Effect Size (Average absolute difference across all setups)
        avg_diff = df_metric["Diff_A_minus_B"].mean()
        
        print(f"📌 {metric.upper()} SUMMARY:")
        print(f"    • Total Cases Tested                : {total_cases} (Algorithms x Layers x Datasets)")
        print(f"    • Scenario A Significantly Better   : {a_better_count} cases ({a_better_pct:.2f}%)")
        print(f"    • Scenario B Significantly Better   : {b_better_count} cases ({b_better_pct:.2f}%)")
        print(f"    • No Significant Difference (Ties)  : {tie_count} cases ({tie_pct:.2f}%)")
        print(f"    • Average Difference (Mean A - B)   : {avg_diff:.4f}")
        print("-" * 80)

if __name__ == "__main__":
    csv_file = r"c:\ULB\MA2\Master Thesis\Practical Part\llm_lie_detection_black_vs_white_box\Black_vs_White_Lie_Detection\Class_Balance_Impact_BB_WB\out\wilcoxon_signed_rank_results.csv"
    analyze_wilcoxon_results(csv_file)
