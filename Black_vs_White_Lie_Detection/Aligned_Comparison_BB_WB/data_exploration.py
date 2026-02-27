"""
Data Exploration for Aligned Comparison.
Analyzes the structure and size of the loaded datasets to match original methodology.
"""

from load_data import load_dataset_aligned
import sys
from pathlib import Path

# Fix import path for White_Box_Lie_Detection if running directly
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

DATASETS_TO_EXPLORE = [
    "commonsense_qa",
    "race",
    "arc_easy",
    "arc_challenge",
    "open_book_qa",
    "boolq",
    "copa",
    "rte",
    "piqa",
    "imdb",
    "amazon_polarity",
    "ag_news",
    "dbpedia_14",
    "got_cities",
    "got_sp_en_trans",
    "got_larger_than",
    "got_cities_cities_conj",
    "got_cities_cities_disj"
]

def explore_data(dataset_name="commonsense_qa"):
    print(f"\n{'='*60}")
    print(f"--- Exploring Dataset: {dataset_name} ---")
    print(f"{'='*60}\n")
    
    # 1. Load Data (Validation Split by default)
    # Note: Some datasets might map 'validation' to 'test' internally in repeng (e.g. some GLUE tasks)
    print("Loading Aligned Data (Split: validation)...")
    try:
        data = load_dataset_aligned(dataset_name, split="validation", n_samples=None) 
    except Exception as e:
        print(f"Failed to load data for {dataset_name}: {e}")
        return
    
    if not data:
        # Try 'train' if validation is empty (some datasets might differ)
        print("Validation empty/failed, trying 'train' split...")
        try:
             data = load_dataset_aligned(dataset_name, split="train", n_samples=100)
        except:
             pass
             
    if not data:
        print(f"No data loaded for {dataset_name}.")
        return

    total = len(data)
    print(f"\nTotal Samples Loaded: {total}")
    
    # 2. Analyze Balance
    positives = [d for d in data if d['label'] == 1]
    negatives = [d for d in data if d['label'] == 0]
    
    print(f"Positive Samples (True): {len(positives)}")
    print(f"Negative Samples (False): {len(negatives)}")
    
    if len(positives) > 0:
        ratio = len(negatives) / len(positives)
        print(f"Negative-to-Positive Ratio: {ratio:.2f}")
    
    # 3. Preview One Full Group (Question + All Answers)
    print("\n--- Sample Group Preview (Raw Item) ---")
    
    if data:
        # Find the first ID
        first_item = data[0]
        print(first_item)
        
        print("\n--- Sample Group Preview (Readable) ---")
        first_id = first_item['id']
        question_group = [d for d in data if d['id'] == first_id]
        
        print(f"Group ID: {first_id}")
        # Print common fields from the first item
        first_item = question_group[0]
        
        # Display derived fields
        if 'question' in first_item:
            print(f"Question: {first_item['question']}")
        if 'context' in first_item:
            print(f"Context/Article: {first_item['context'][:100]}...")
            
        print(f"\nTotal options in group: {len(question_group)}")
        
        print("\nCorrect Answer(s):")
        for d in question_group:
            if d['label'] == 1:
                ans = d.get('answer', d.get('text', 'N/A')[-50:]) # Fallback to end of text
                print(f"  - {ans}")
                print(f"    Full Prompt: {d['text']}")
                
        print("\nFalse Answer(s):")
        for d in question_group:
            if d['label'] == 0:
                ans = d.get('answer', d.get('text', 'N/A')[-50:])
                print(f"  - {ans}")

if __name__ == "__main__":
    # If arguments provided, use them. Else loop all.
    if len(sys.argv) > 1:
        explore_data(sys.argv[1])
    else:
        print("Exploring all datasets defined in DATASETS_TO_EXPLORE...")
        for ds in DATASETS_TO_EXPLORE:
            explore_data(ds)

