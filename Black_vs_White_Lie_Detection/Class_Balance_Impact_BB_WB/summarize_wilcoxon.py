import pandas as pd
import os

def analyze_wilcoxon_v2_results(base_out_dir):
    metrics = ["AUC", "MAP", "BRP_90"]
    
    print("=" * 80)
    print(f"{'📊 CLASS IMBALANCE DEGRADATION ANALYSIS (SCENARIO A vs B) 📊':^80}")
    print("=" * 80)
    
    for metric in metrics:
        csv_path = os.path.join(base_out_dir, f"wilcoxon_pairs_{metric}_V2.csv")
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
            continue
            
        total_pairs = len(df)
        if total_pairs == 0:
            continue
            
        diff_col = "Difference (A - B)"
        
        # Overall stats
        a_better = (df[diff_col] > 0).sum()
        b_better = (df[diff_col] < 0).sum()
        ties = (df[diff_col] == 0).sum()
        avg_diff = df[diff_col].mean()
        
        print(f"\n📌 {metric.upper()} SUMMARY:")
        print(f"    • Total Dataset/Algorithm Pairs : {total_pairs}")
        print(f"    • Degraded by Imbalance (A > B) : {a_better} cases ({a_better/total_pairs*100:.1f}%)")
        print(f"    • Improved by Imbalance (B > A) : {b_better} cases ({b_better/total_pairs*100:.1f}%)")
        print(f"    • Unaffected (Ties)             : {ties} cases ({ties/total_pairs*100:.1f}%)")
        print(f"    • Average Degradation           : {avg_diff:.4f}")
        
        # Algorithm robustness ranking
        print("\n    🧠 ALGORITHM ROBUSTNESS (Sorted by severity of degradation):")
        print("      (Higher Avg Drop = More vulnerable to class imbalance)")
        algo_stats = df.groupby("Algorithm")[diff_col].agg(['mean', 'max', 'min']).sort_values(by='mean', ascending=False)
        for algo, row in algo_stats.iterrows():
            print(f"      - {algo:>5}: Avg Drop = {row['mean']:>7.4f} | Worst Drop observed = {row['max']:>7.4f}")
            
        # Dataset sensitivity
        print("\n    📂 DATASET SENSITIVITY (Sorted by average degradation):")
        print("      (Higher Avg Drop = Dataset is harder for imbalanced scenarios)")
        ds_stats = df.groupby("Dataset")[diff_col].mean().sort_values(ascending=False)
        for ds, mean_drop in ds_stats.items():
            print(f"      - {ds:>15}: Avg Drop = {mean_drop:>7.4f}")
            
        print("-" * 80)

if __name__ == "__main__":
    out_dir = r"c:\ULB\MA2\Master Thesis\Practical Part\llm_lie_detection_black_vs_white_box\Black_vs_White_Lie_Detection\Class_Balance_Impact_BB_WB\out"
    analyze_wilcoxon_v2_results(out_dir)
