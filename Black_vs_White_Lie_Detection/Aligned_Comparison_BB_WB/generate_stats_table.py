import sys
import os
from pathlib import Path
import time

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from load_data import load_dataset_aligned

DATASETS = [
    "commonsense_qa", "race", "arc_easy", "arc_challenge", "open_book_qa",
    "boolq", "copa", "rte", "piqa", "imdb", "amazon_polarity", "ag_news",
    "dbpedia_14", "got_cities", "got_sp_en_trans", "got_larger_than",
    "got_cities_cities_conj", "got_cities_cities_disj"
]

def generate_table():
    stats = []
    print("Generating statistics for all datasets...")
    
    total_all = 0
    pos_all = 0
    neg_all = 0

    for ds_name in DATASETS:
        print(f"Processing {ds_name}...", end="", flush=True)
        try:
            # Try validation first, fallback to train if empty (like PIQA might fail or need train)
            data = load_dataset_aligned(ds_name, split="validation")
            if not data:
                 data = load_dataset_aligned(ds_name, split="train", n_samples=None)
            
            if not data:
                print(" Failed/Empty.")
                stats.append((ds_name, 0, 0, 0))
                continue

            count = len(data)
            pos = sum(1 for d in data if d['label'] == 1)
            neg = sum(1 for d in data if d['label'] == 0)
            
            stats.append((ds_name, count, pos, neg))
            
            total_all += count
            pos_all += pos
            neg_all += neg
            print(f" Done ({count} samples)")
            
        except Exception as e:
            print(f" Error: {e}")
            stats.append((ds_name, 0, 0, 0))

    # Add Summary Row
    stats.append(("**TOTAL**", total_all, pos_all, neg_all))

    # Generate Markdown Table
    print("\n\n### Dataset Statistics")
    print("| Dataset | Total Samples | Positive (True) | Negative (False) |")
    print("| :--- | :---: | :---: | :---: |")
    
    for row in stats:
        name, t, p, n = row
        print(f"| {name} | {t} | {p} | {n} |")

if __name__ == "__main__":
    generate_table()
