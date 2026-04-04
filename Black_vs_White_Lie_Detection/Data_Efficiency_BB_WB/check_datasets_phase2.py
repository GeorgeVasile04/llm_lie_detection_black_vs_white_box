import sys
import os
from pathlib import Path
import pandas as pd

# Add paths to sys.path to easily import the repeng datasets
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
repeng_path = project_root / "White_Box_Lie_Detection"
if str(project_root) not in sys.path: sys.path.append(str(project_root))
if str(repeng_path) not in sys.path: sys.path.append(str(repeng_path))

from White_Box_Lie_Detection.repeng.datasets.elk import (
    geometry_of_truth, dlk, race, arc, open_book_qa, common_sense_qa
)

datasets_to_check = [
    "commonsense_qa", "race", "arc_easy", "arc_challenge", "open_book_qa", 
    "boolq", "copa", "rte", "piqa", "imdb", "amazon_polarity", "ag_news", 
    "dbpedia_14", "got_cities", "got_sp_en_trans", "got_larger_than", 
    "got_cities_cities_conj", "got_cities_cities_disj"
]

def load_dataset_full(name):
    data_dict = {}
    try:
        if name == "commonsense_qa": data_dict = common_sense_qa.get_common_sense_qa("repe")
        elif name == "race": data_dict = race.get_race("repe")
        elif name == "arc_easy": data_dict = arc.get_arc("easy", "repe")
        elif name == "arc_challenge": data_dict = arc.get_arc("challenge", "repe")
        elif name == "open_book_qa": data_dict = open_book_qa.get_open_book_qa("repe")
        elif name == "got_cities": data_dict = geometry_of_truth.get_geometry_of_truth("cities")
        elif name == "got_sp_en_trans": data_dict = geometry_of_truth.get_geometry_of_truth("sp_en_trans")
        elif name == "got_larger_than": data_dict = geometry_of_truth.get_geometry_of_truth("larger_than")
        elif name == "got_cities_cities_conj": data_dict = geometry_of_truth.get_geometry_of_truth("cities_cities_conj")
        elif name == "got_cities_cities_disj": data_dict = geometry_of_truth.get_geometry_of_truth("cities_cities_disj")
        elif name in ["imdb", "amazon_polarity", "ag_news", "dbpedia_14", "rte", "copa", "boolq", "piqa"]:
            data_dict = dlk.get_dlk_dataset(name)
        else: return []
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return []

    processed_data = []
    for key, row in data_dict.items():
        # NOTE: We grab train and validation/test altogether 
        # to see the max possible pool we can sample from.
        item = {
            "id": row.group_id if row.group_id else key,
            "label": 1 if row.label else 0,
        }
        processed_data.append(item)
    return processed_data

print(f"{'Dataset':<25} | {'Total Samples':<15} | {'True (1)':<10} | {'False (0)':<10} | {'Unique Questions':<18} | {'Good for Phase 2? (>=3k True & False)'}")
print("-" * 115)

for ds_name in datasets_to_check:
    data = load_dataset_full(ds_name)
    if not data:
        print(f"{ds_name:<25} | {'ERROR':<15} | {'-':<10} | {'-':<10} | {'-':<18} | {'No'}")
        continue
        
    df = pd.DataFrame(data)
    total_samples = len(df)
    n_true = df['label'].sum()
    n_false = total_samples - n_true
    
    # Unique questions are determined by counting how many unique IDs we have
    # For every question, we want to ensure we can pair 1 True / 1 False
    # So the maximum balanced pairs we can form is min(True, False) for each unique question.
    
    # Group by question ID 
    # and count how many trues and falses it has
    grouped = df.groupby('id').agg(
        n_true=('label', 'sum'),
        n_total=('label', 'count')
    )
    grouped['n_false'] = grouped['n_total'] - grouped['n_true']
    
    # A question can only contribute to a Balanced setup if it has at least 1 True and 1 False
    valid_balanced_questions = grouped[(grouped['n_true'] > 0) & (grouped['n_false'] > 0)]
    n_unique_questions_balanced = len(valid_balanced_questions)
    
    # Total unique questions (no matter if balanceable)
    n_unique_total = len(grouped)
    
    is_good = "YES" if (n_true >= 3000 and n_false >= 3000 and n_unique_questions_balanced >= 3000) else "NO"
    
    print(f"{ds_name:<25} | {total_samples:<15} | {n_true:<10} | {n_false:<10} | {n_unique_total:<18} | {is_good} (has {n_unique_questions_balanced} valid balanced qs)")

