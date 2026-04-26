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
    "boolq", "copa", "rte", "imdb", "amazon_polarity", "ag_news", 
    "dbpedia_14", "got_cities", "got_sp_en_trans", "got_larger_than", 
    "got_cities_cities_conj", "got_cities_cities_disj"
]

def load_dataset_full(name, num_examples=10, sort_by_length=False):
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
        else: return [], [], {}
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return [], [], {}

    processed_data = []
    
    structure_info = {}
    if data_dict:
        test_key = list(data_dict.keys())[0]
        test_row = data_dict[test_key]
        
        structure_info['dict_class'] = type(data_dict).__name__
        structure_info['row_class'] = type(test_row).__name__
        
        if hasattr(test_row, '__dict__'):
            structure_info['attributes'] = list(vars(test_row).keys())
        elif hasattr(test_row, '__slots__'):
            structure_info['attributes'] = list(test_row.__slots__)
        else:
            structure_info['attributes'] = [a for a in dir(test_row) if not a.startswith('_')]
            
        if hasattr(test_row, 'format_args') and isinstance(test_row.format_args, dict):
            structure_info['format_args_keys'] = list(test_row.format_args.keys())

    example_groups = []
    current_groups = {}
    
    for key, row in data_dict.items():
        g_id = row.group_id if getattr(row, 'group_id', None) else key
        
        if sort_by_length:
            if g_id not in current_groups:
                current_groups[g_id] = []
            current_groups[g_id].append(row)
        else:
            if g_id not in current_groups:
                if len(current_groups) < num_examples:
                    current_groups[g_id] = []
            if g_id in current_groups:
                current_groups[g_id].append(row)
            
        # NOTE: We grab train and validation/test altogether 
        # to see the max possible pool we can sample from.
        item = {
            "id": g_id,
            "label": 1 if row.label else 0,
        }
        processed_data.append(item)
        
    example_groups = list(current_groups.values())
    if sort_by_length and example_groups:
        # Sort groups by the length of the string representation of their first row
        example_groups.sort(key=lambda g: len(str(getattr(g[0], 'text', ''))) if hasattr(g[0], 'text') else 0)
        example_groups = example_groups[:num_examples]
        
    return processed_data, example_groups, structure_info

# CONTROL WHICH QUESTION INDEX TO PRINT HERE (0 to 9, since we load 10 by default)
# Note: For race, boolq, imdb, and amazon_polarity we load top 3 shortest, so index must be 0, 1, or 2 for those.
PRINT_INDEX = 0

examples_dict = {}
structure_dict = {}
stats_list = []

for ds_name in datasets_to_check:
    shorten = ds_name in ['race', 'boolq', 'imdb', 'amazon_polarity']
    num_ex = 3 if shorten else 10
    
    data, example_groups, struct_info = load_dataset_full(ds_name, num_examples=num_ex, sort_by_length=shorten)
    if not data:
        continue
        
    examples_dict[ds_name] = example_groups
    structure_dict[ds_name] = struct_info
    
    # Compute Exact True/False counts per question
    df = pd.DataFrame(data)
    grouped = df.groupby('id')['label'].agg(['sum', 'count'])
    avg_true = grouped['sum'].mean()
    avg_false = (grouped['count'] - grouped['sum']).mean()
    
    stats_list.append({
        "Dataset": ds_name,
        "Total Questions": len(grouped),
        "Avg True/Q": round(avg_true, 2),
        "Avg False/Q": round(avg_false, 2)
    })

stats_df = pd.DataFrame(stats_list)
print("\n" + "="*80)
print("DATASET TRUE/FALSE RATIO SUMMARY")
print("="*80)
print(stats_df.to_string(index=False))
print("="*80 + "\n")


print("\n" + "="*115)
print("DATASET EXAMPLES")
print("="*115 + "\n")

for ds_name, example_groups in examples_dict.items():
    if not example_groups or PRINT_INDEX >= len(example_groups):
        print(f"Skipping {ds_name} - Not enough questions loaded for index {PRINT_INDEX}")
        continue
        
    example_rows = example_groups[PRINT_INDEX]
    struct_info = structure_dict[ds_name]
    
    print(f"[{ds_name.upper()}] Example Request/Answers (Question Box Index: {PRINT_INDEX}):")
    print("-" * 50)
    print("ORIGINAL STRUCTURE DETAILS:")
    print(f"  Container Type: {struct_info.get('dict_class')}")
    print(f"  Row Type:       {struct_info.get('row_class')}")
    print(f"  Row Attributes: {struct_info.get('attributes')}")
    if 'format_args_keys' in struct_info:
        print(f"  Format Args (Columns): {struct_info.get('format_args_keys')}")
    print("-" * 50)
    
    # Try to extract common 'question' or context from format_args
    # Some datasets just have texts since they are not templated the same way
    
    first_row = example_rows[0]
    
    # Let's try to extract parts from format_args
    if hasattr(first_row, 'format_args') and first_row.format_args:
        args = first_row.format_args
        
        # Look for article/context
        for key in ['article', 'context', 'premise', 'text', 'sentence1', 'passage', 'content']:
            if key in args:
                val = args[key]
                if len(val) > 800: val = val[:800] + " ... [TRUNCATED]"
                print(f"Context/Premise/Content:\n{val}\n")
                break
                
        # Look for question
        for key in ['question', 'question_stem', 'hypothesis', 'sentence2', 'goal']:
            if key in args:
                print(f"Question/Hypothesis/Goal:\n{args[key]}\n")
                break
                
        # For datasets like COPA or PIQA that have choice1/choice2 in the context
        choices_printed = False
        for key in ['choice1', 'choice2', 'sol1', 'sol2']:
            if key in args:
                print(f"{key.capitalize()}: {args[key]}")
                choices_printed = True
        if choices_printed:
            print("")
                
        print("Answers:")
        for i, row in enumerate(example_rows):
            # 'label' is used by dlk datasets (like IMDb, AG News, etc.) for the answer option
            ans = row.format_args.get('answer', row.format_args.get('choice', row.format_args.get('label', 'None specifically formatted')))
            label_str = "TRUE" if row.label else "FALSE"
            print(f"  {i+1}) [Label: {label_str:<5}] {ans}")
            
    else:
        # Fallback to just printing the text
        if hasattr(first_row, 'text'):
            for i, row in enumerate(example_rows):
                label_str = "TRUE" if row.label else "FALSE"
                print(f"  {i+1}) [Label: {label_str:<5}]\n{row.text.strip()}\n")
        else:
            print("  (Cannot properly display, missing standard attributes.)")
            
    print("\n" + "="*80 + "\n")

