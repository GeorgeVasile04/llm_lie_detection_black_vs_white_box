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

import gc
import torch
from tqdm.auto import tqdm
from Black_vs_White_Lie_Detection.Aligned_Comparison_BB_WB.wb_activations import get_activations_for_dataset

def robust_get_activations(df, model, tokenizer, device, initial_batch_size, desc="Extracting", target_size=None):
    results = []
    idx = 0
    total_to_extract = target_size if target_size is not None else len(df)
    pbar = tqdm(total=total_to_extract, desc=desc)
    current_bs = initial_batch_size
    
    # We iterate until we've processed the full df OR reached exactly our target limit
    while idx < len(df) and len(results) < total_to_extract:
        # Dynamically adjust the chunk size so we never overshoot our specific limit
        remaining = total_to_extract - len(results)
        bs_to_use = min(current_bs, remaining, len(df) - idx)
        
        chunk = df.iloc[idx:idx + bs_to_use]
        try:
            # We call the existing extraction function on this chunk
            chunk_results = get_activations_for_dataset(
                chunk, model, tokenizer, device=device, batch_size=bs_to_use, show_progress=False
            )
            results.extend(chunk_results)
            idx += bs_to_use
            pbar.update(bs_to_use) # Exactly bs_to_use samples succeeded!
            
            # Flush completely after each successful chunk
            gc.collect()
            torch.cuda.empty_cache()
            
            # --- NEW: Speed recovery! ---
            # If it succeeds, immediately recover back to initial speed for the NEXT chunk
            current_bs = initial_batch_size
            
        except RuntimeError as e:
            # Check for Out Of Memory Error
            is_oom = "out of memory" in str(e).lower() or "outofmemoryerror" in str(getattr(e, "__class__", "")).lower()
            
            if is_oom:
                # Crucial: Clear traceback frames to release GPU memory tied to local variables (inputs, outputs)
                import traceback
                traceback.clear_frames(e.__traceback__)
                del e
                
                # Double collect to ensure frames are entirely wiped from memory
                gc.collect()
                torch.cuda.empty_cache()
                
                if bs_to_use > 1:
                    current_bs = max(1, current_bs // 2)
                    # Loop retries without advancing idx!
                else:
                    print(f"⚠️ Fatal OOM even with batch size 1 at index {idx}. Skipping this single bad sample to survive.")
                    idx += 1
                    # Notice we explicitly DO NOT update `pbar` or `results`. 
                    # By dropping a bad text, we automatically force the loop to process 
                    # an extra item later so we still strictly reach the 50,000 threshold perfectly.
                    
                    # Recover batch size after skipping the fatal sample
                    current_bs = initial_batch_size
            else:
                raise e

    pbar.close()
    return results

