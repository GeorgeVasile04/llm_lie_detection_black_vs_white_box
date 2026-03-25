import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
original_aligned_path = current_dir.parent / "Aligned_Comparison_BB_WB"

if str(original_aligned_path) not in sys.path: sys.path.append(str(original_aligned_path))
if str(project_root) not in sys.path: sys.path.append(str(project_root))

from White_Box_Lie_Detection.repeng.datasets.elk import (
    geometry_of_truth, dlk, race, arc, open_book_qa, common_sense_qa
)

def load_dataset_imbalance(dataset_name="commonsense_qa", split='validation', n_samples=1000, scenario="A", random_seed=42):
    """
    Loads dataset logically grouped by original question.
    scenario="A": strictly Balanced (1 True + 1 False per question)
    scenario="B": strictly Imbalanced (1 True + ALL False per question)
    """
    data_dict = {}
    name = dataset_name.lower().strip()
    if name == "csqa": name = "commonsense_qa"
    if name == "openbookqa": name = "open_book_qa"

    print(f"Loading {name} (split={split}) for Scenario {scenario}, Target N={n_samples}...")

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

    df_full = pd.DataFrame(processed_data)
    if len(df_full) == 0: return df_full

    unique_ids = list(df_full['id'].unique())
    np.random.seed(random_seed)
    # Randomly shuffle questions so we get a random sample
    np.random.shuffle(unique_ids)

    selected_rows = []
    collected_samples = 0

    # Group logically by questions
    for qid in unique_ids:
        q_df = df_full[df_full['id'] == qid]
        true_rows = q_df[q_df['label'] == 1]
        false_rows = q_df[q_df['label'] == 0]

        if len(true_rows) == 0 or len(false_rows) == 0: continue

        if scenario == "A":
            t = true_rows.sample(n=1, random_state=random_seed)
            f = false_rows.sample(n=1, random_state=random_seed)
            selected_rows.extend([t, f])
            collected_samples += 2
        elif scenario == "B":
            t = true_rows.sample(n=1, random_state=random_seed)
            selected_rows.extend([t, false_rows])
            collected_samples += (1 + len(false_rows))
            
        if collected_samples >= n_samples: break

    if not selected_rows: return pd.DataFrame()

    df_final = pd.concat(selected_rows).reset_index(drop=True)
    if len(df_final) > n_samples:
       # Optionally truncate exactly to n_samples if strict size is needed.
       # We will leave as is for now as this guarantees the logical pairing constraint.
       df_final = df_final.head(n_samples) 
       
    print(f"   -> Scenario {scenario}: Generated {len(df_final)} samples (grouped logically).")
    return df_final
