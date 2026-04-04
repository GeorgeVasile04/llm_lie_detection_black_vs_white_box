import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
repeng_path = project_root / "White_Box_Lie_Detection"
if str(project_root) not in sys.path: sys.path.append(str(project_root))
if str(repeng_path) not in sys.path: sys.path.append(str(repeng_path))

from White_Box_Lie_Detection.repeng.datasets.elk import (
    dlk, race, arc, open_book_qa, common_sense_qa
)

def load_dataset_efficiency(dataset_name="commonsense_qa", split='train', random_seed=42):
    """
    Loads datast for the Efficiency phase, maintaining a strictly 1:1 balance.
    It returns ALL possible balanced pairs. The notebook will then slice 
    this into 5k (train) or 1k (test).
    """
    data_dict = {}
    name = dataset_name.lower().strip()
    if name == "csqa": name = "commonsense_qa"
    if name == "openbookqa": name = "open_book_qa"

    try:
        if name == "commonsense_qa": data_dict = common_sense_qa.get_common_sense_qa("repe")
        elif name == "race": data_dict = race.get_race("repe")
        elif name == "arc_easy": data_dict = arc.get_arc("easy", "repe")
        elif name == "arc_challenge": data_dict = arc.get_arc("challenge", "repe")
        elif name == "open_book_qa": data_dict = open_book_qa.get_open_book_qa("repe")
        elif name in ["imdb", "amazon_polarity", "ag_news", "dbpedia_14", "rte", "boolq"]:
            data_dict = dlk.get_dlk_dataset(name)
        else: return pd.DataFrame()
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return pd.DataFrame()

    processed_data = []
    
    # Process dictionary to extract data list
    for key, row in data_dict.items():
        if split and split != 'all' and row.split != split: continue
        item = {
            "id": row.group_id if row.group_id else key,
            "text": row.text,
            "label": 1 if row.label else 0,
            "dataset": row.dataset_id, "split": row.split
        }
        processed_data.append(item)
    
    df = pd.DataFrame(processed_data)
    if df.empty: return df

    # Enforce strictly 1 True and 1 False per question
    np.random.seed(random_seed)
    
    grouped = df.groupby('id')
    balanced_samples = []
    
    for group_id, group_df in grouped:
        true_samples = group_df[group_df['label'] == 1]
        false_samples = group_df[group_df['label'] == 0]
        
        if len(true_samples) > 0 and len(false_samples) > 0:
            chosen_true = true_samples.sample(n=1, random_state=random_seed)
            chosen_false = false_samples.sample(n=1, random_state=random_seed)
            balanced_samples.append(chosen_true)
            balanced_samples.append(chosen_false)
            
    if len(balanced_samples) == 0:
        return pd.DataFrame()
        
    df_balanced = pd.concat(balanced_samples).reset_index(drop=True)
    return df_balanced

