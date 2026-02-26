"""
Data Exploration for Aligned Comparison.
Analyzes the structure and size of the loaded datasets to match original methodology.
"""

from load_data import load_dataset_aligned
from datasets import load_dataset
import numpy as np

def explore_data():
    print("--- Aligned Comparison Data Analysis ---\n")
    
    # 0. Raw Data Preview
    print("Loading Raw CommonSenseQA Sample (First Item)...")
    raw_dataset = load_dataset("commonsense_qa", split="validation")
    raw_sample = raw_dataset[0]
    
    print("\n[RAW DATA SAMPLE - DICTIONARY DUMP]")
    # Print the raw dictionary directly, no formatting
    print(raw_sample)
    print("-" * 50)

    # 1. Load Data
    print("\nLoading Transformed Aligned Data...")
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
    print("\n--- Detailed Question Sample ---")
    
    # Group by ID to show one full question
    if data:
        first_id = data[0]['id']
        question_group = [d for d in data if d['id'] == first_id]
        
        print(f"Question ID: {first_id}")
        print(f"Question: {question_group[0]['question']}")
        print(f"Total options: {len(question_group)}")
        
        print("\nCorrect Answer:")
        for d in question_group:
            if d['label'] == 1:
                print(f"  - {d['answer']}")
                
        print("\nFalse Answers:")
        for d in question_group:
            if d['label'] == 0:
                print(f"  - {d['answer']}")

if __name__ == "__main__":
    explore_data()
