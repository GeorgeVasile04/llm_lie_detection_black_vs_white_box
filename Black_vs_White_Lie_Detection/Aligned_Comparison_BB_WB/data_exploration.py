"""
Data Exploration for Aligned Comparison.
Analyzes the structure and size of the loaded datasets to match original methodology.
"""

from load_data import load_dataset_aligned
import numpy as np

def explore_data():
    print("--- Aligned Comparison Data Analysis ---\n")
    
    # 1. Load Data
    print("Loading CommonSenseQA (aligned)...")
    try:
        data = load_dataset_aligned(n_samples=None) # Load full
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
    
    if not data:
        print("No data loaded.")
        return

    total = len(data)
    print(f"\nTotal Samples: {total}")
    
    # 2. Analyze Balance
    positives = [d for d in data if d['label'] == 1]
    negatives = [d for d in data if d['label'] == 0]
    
    print(f"Positive Samples (True Answers): {len(positives)}")
    print(f"Negative Samples (False Answers): {len(negatives)}")
    
    ratio = len(negatives) / len(positives) if len(positives) > 0 else 0
    print(f"Negative-to-Positive Ratio: {ratio:.2f} (Expected ~4.0 for CommonSenseQA)")
    
    # 3. Analyze Question Structure
    unique_ids = set(d['id'] for d in data)
    print(f"Unique Questions: {len(unique_ids)}")
    
    # Check if we have completeness
    # Ideally 5 choices per question
    print(f"Average samples per question: {total / len(unique_ids):.2f}")
    
    # 4. Preview
    print("\n--- Sample Positive ---")
    if positives:
        print(positives[0])
        
    print("\n--- Sample Negative ---")
    if negatives:
        print(negatives[0])
        
if __name__ == "__main__":
    explore_data()
