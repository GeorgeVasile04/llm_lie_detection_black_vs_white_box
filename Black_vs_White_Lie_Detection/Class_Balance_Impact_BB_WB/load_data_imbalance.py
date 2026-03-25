import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent

# Add original module directory to import other dependencies if needed later
original_aligned_path = current_dir.parent / "Aligned_Comparison_BB_WB"
if str(original_aligned_path) not in sys.path:
    sys.path.append(str(original_aligned_path))

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from White_Box_Lie_Detection.repeng.datasets.elk import (
    geometry_of_truth,
    dlk,
    race,
    arc,
    open_book_qa,
    common_sense_qa
)

def load_dataset_imbalance(dataset_name="commonsense_qa", split='validation', n_samples=1000, true_ratio=0.5, random_seed=42):
    """
    Loads dataset and specifically constructs a split with an EXACT ratio of True vs False labels.
    Differs from original load_dataset_aligned by forcing strictly balanced/imbalanced class ratios.
    
    Args:
        dataset_name (str): The name of the dataset to load.
        split (str): 'train' or 'validation'.
        n_samples (int): Total number of samples required.
        true_ratio (float): The fraction of labels that must be True (label=1). 
                            0.5 = 50% True, 0.2 = 20% True.
        random_seed (int): Seed for reproducibility of sampling.
        
    Returns:
        pd.DataFrame: A pandas DataFrame containing exactly the requested distribution.
    """
    data_dict = {}
    name = dataset_name.lower().strip()
    if name == "csqa": name = "commonsense_qa"
    if name == "openbookqa": name = "open_book_qa"

    print(f"Loading {name} (split={split}) to construct split of size N={n_samples} with {true_ratio*100}% True...")

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
        else:
            print(f"Warning: Dataset '{name}' logic not explicitly mapped.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return pd.DataFrame()

    processed_data = []
    for key, row in data_dict.items():
        if split and split != 'all':
            if row.split != split:
                continue
            
        item = {
            "id": row.group_id if row.group_id else key,
            "text": row.text,
            "label": 1 if row.label else 0,
            "dataset": row.dataset_id,
            "split": row.split
        }
        processed_data.append(item)

    df_full = pd.DataFrame(processed_data)
    if len(df_full) == 0:
        return df_full
        
    df_true = df_full[df_full['label'] == 1]
    df_false = df_full[df_full['label'] == 0]
    
    n_true_needed = int(n_samples * true_ratio)
    n_false_needed = n_samples - n_true_needed
    
    # Validation safety check: if a dataset does not have enough truths, we dynamically fallback 
    # to the maximum we can achieve given the class proportions, or warn heavily.
    if len(df_true) < n_true_needed or len(df_false) < n_false_needed:
        print(f"⚠️ Warning {name}: Cannot satisfy requirements. "
              f"Has: {len(df_true)}T/{len(df_false)}F. Needs: {n_true_needed}T/{n_false_needed}F.")
        return df_full.head(n_samples) # fallback

    sampled_true = df_true.sample(n=n_true_needed, random_state=random_seed)
    sampled_false = df_false.sample(n=n_false_needed, random_state=random_seed)
    
    # Concat and shuffle
    df_balanced = pd.concat([sampled_true, sampled_false]).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    return df_balanced
