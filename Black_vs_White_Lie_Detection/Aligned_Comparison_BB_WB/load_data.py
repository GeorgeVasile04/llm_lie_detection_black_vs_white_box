import sys
import os
from pathlib import Path
import pandas as pd
from datasets import load_dataset

# Add project root to path to allow imports from White_Box_Lie_Detection
# Assuming this file is at Black_vs_White_Lie_Detection/Aligned_Comparison_BB_WB/load_data.py
# We need to add the folder containing 'White_Box_Lie_Detection' to sys.path
# That folder is 'llm_lie_detection_black_vs_white_box'
current_dir = Path(__file__).resolve().parent
# current_dir = Aligned_Comparison_BB_WB
# .parent = Black_vs_White_Lie_Detection
# .parent.parent = llm_lie_detection_black_vs_white_box (This is what we want)
project_root = current_dir.parent.parent

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

def load_dataset_aligned(dataset_name="commonsense_qa", split='validation', n_samples=None):
    """
    Loads dataset using the original White Box (RepEng) methodology.
    
    Args:
        dataset_name (str): The name of the dataset to load.
        split (str): The split to filter by ('train', 'validation', 'test').
        n_samples (int): Optional limit.
        
    Returns:
        List[dict]: A list of aligned samples.
    """
    
    # Map dataset names to repeng function calls
    data_dict = {}
    
    # Normalize name
    name = dataset_name.lower().strip()
    
    # Map common aliases
    if name == "csqa": name = "commonsense_qa"
    if name == "openbookqa": name = "open_book_qa"

    print(f"Loading {name} (split={split}) using RepEng logic...")

    try:
        # Dispatch to the correct loader
        if name == "commonsense_qa":
            data_dict = common_sense_qa.get_common_sense_qa("repe")
        elif name == "race":
             data_dict = race.get_race("repe")
        elif name == "arc_easy":
             data_dict = arc.get_arc("easy", "repe")
        elif name == "arc_challenge":
             data_dict = arc.get_arc("challenge", "repe")
        elif name == "open_book_qa":
             data_dict = open_book_qa.get_open_book_qa("repe")
             
        # Geometry of Truth
        elif name == "got_cities":
            data_dict = geometry_of_truth.get_geometry_of_truth("cities")
        elif name == "got_sp_en_trans":
            data_dict = geometry_of_truth.get_geometry_of_truth("sp_en_trans")
        elif name == "got_larger_than":
            data_dict = geometry_of_truth.get_geometry_of_truth("larger_than")
        elif name == "got_cities_cities_conj":
            data_dict = geometry_of_truth.get_geometry_of_truth("cities_cities_conj")
        elif name == "got_cities_cities_disj":
            data_dict = geometry_of_truth.get_geometry_of_truth("cities_cities_disj")
            
        # DLK Datasets (standard NLP tasks)
        elif name in ["imdb", "amazon_polarity", "ag_news", "dbpedia_14", "rte", "copa", "boolq", "piqa"]:
            data_dict = dlk.get_dlk_dataset(name)
        else:
            print(f"Warning: Dataset '{name}' logic not explicitly mapped.")
            return []

    except Exception as e:
        print(f"Error loading {name}: {e}")
        return []

    # Process into list
    processed_data = []
    count = 0
    
    for key, row in data_dict.items():
        # Check split
        current_split = row.split
        
        # Mapping logic for splits if necessary
        # RepEng splits: 'train', 'validation', 'train-hparams'
        
        if split and split != 'all':
            if current_split != split:
                continue
            
        # Extract fields
        # text: The full prompt text
        # label: Bool
        
        item = {
            "id": row.group_id if row.group_id else key,
            "text": row.text,         # Use the formatted prompt text as the primary input
            "label": 1 if row.label else 0,
            "dataset": row.dataset_id,
            "split": current_split,
            "answer_type": row.answer_type if hasattr(row, 'answer_type') else None
        }
        
        # Populate metadata for improved readability
        if row.format_args:
            item.update(row.format_args)
            
            # Normalize common fields
            if 'question' not in item and 'question_stem' in item:
                item['question'] = item['question_stem']
            
            if 'answer' not in item:
                if 'choice' in item:
                    item['answer'] = item['choice']
                elif 'statement' in item: 
                    item['answer'] = item['statement']

        processed_data.append(item)
        
        count += 1
        if n_samples and count >= n_samples:
            break
            
    print(f"Loaded {len(processed_data)} items.")
    return processed_data

if __name__ == "__main__":
    try:
        d = load_dataset_aligned("commonsense_qa", split="validation", n_samples=2)
        print(f"Sample CSQA: {d[0] if d else 'Empty'}")
    except Exception as e:
        print(f"Error: {e}")
