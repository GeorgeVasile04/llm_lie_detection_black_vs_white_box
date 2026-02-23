import json
import os
import pandas as pd
import glob

def load_anthropic_format(filepath):
    """
    Loads JSON that is in column-oriented format (dict of dicts).
    e.g. {"question": {"0": "...", "1": "..."}, "answer": {...}}
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert dict of dicts to list of dicts using pandas
    df = pd.DataFrame(data)
    
    # Add source dataset name
    filename = os.path.basename(filepath)
    df["source_dataset"] = filename.replace(".json", "")
    
    return df.to_dict(orient="records")

def load_standard_list_format(filepath):
    """
    Loads JSON that is already a list of records.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Validation: Ensure it's a list
    if not isinstance(data, list):
        print(f"Warning: {filepath} is not a list of records. Attempting to parse as column-oriented.")
        return load_anthropic_format(filepath)
        
    # Enrich with source info
    filename = os.path.basename(filepath)
    for record in data:
        record["source_dataset"] = filename.replace(".json", "")
        # Ensure category exists, default to 'general' if missing
        if "category" not in record:
            record["category"] = "general"
            
    return data

def create_unified_raw_dataset():
    data_dir = os.path.join("Black_vs_White_Lie_Detection", "Data")
    
    # Define file mapping
    # 1. Anthropic Awareness (Column-oriented)
    anthropic_path = os.path.join(data_dir, "anthropic_awareness_ai.json")
    
    # 2. Synthetic Facts (Column-oriented based on inspection)
    synthetic_path = os.path.join(data_dir, "synthetic_facts.json")
    
    # 3. Standard List formats (Assumed for others based on typical usage, but we'll check)
    # math_problems.json, common_sens_qa_v2.json, questions_1000.json
    
    all_records = []
    
    # Process Anthropic
    if os.path.exists(anthropic_path):
        print(f"Processing {anthropic_path}...")
        records = load_anthropic_format(anthropic_path)
        all_records.extend(records)
        
    # Process Synthetic Facts
    if os.path.exists(synthetic_path):
        print(f"Processing {synthetic_path}...")
        records = load_anthropic_format(synthetic_path)
        all_records.extend(records)

    # Process others (Generic loader)
    other_files = [
        "common_sens_qa_v2.json",
        "math_problems.json",
        "questions_1000.json"
    ]
    
    for filename in other_files:
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            print(f"Processing {path}...")
            # We try standard load first, if it fails format check, we fallback (inside the function)
            # Actually, let's peek at the file content structure briefly to be robust?
            # No, let's rely on try/except or the helper.
            try:
                records = load_standard_list_format(path)
                all_records.extend(records)
            except Exception as e:
                print(f"Error loading {path}: {e}")

    # Add unique IDs
    print("Assigning unique IDs...")
    for idx, record in enumerate(all_records):
        record["id"] = idx
        
        # Standardize fields
        # Some datasets might have "label" (0/1) instead of "answer". 
        # For this raw dataset, we want "answer" (text). 
        # If "answer" is missing but label exists, we might need to fix it, but 
        # based on your files, "answer" seems present in most.
        
        # Clean up any 'false_statement' or 'statement' fields from synthetic data?
        # The user wants "raw" questions. 
        # Synthetic facts has "statement" (True) and "false_statement" (Lie).
        # We should probably map "statement" -> "answer" (The truth).
        
        if "statement" in record and ("answer" not in record or record["answer"] is None):
             # In synthetic_facts.json (based on read_file output), 
             # "answer" is the short answer (e.g. "Santiago"), "statement" is the full sentence truth.
             # We prefer short answer for checking correctness? Or full sentence?
             # Usually short answer is better for checking "contains".
             pass # "answer" field already exists in synthetic_facts.json

    # Save
    output_path = os.path.join(data_dir, "raw_dataset.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, indent=2)
        
    print(f"Successfully created {output_path} with {len(all_records)} records.")

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import json
    
    # 1. Load Anthropic
    anth_path = os.path.join("Black_vs_White_Lie_Detection", "Data", "anthropic_awareness_ai.json")
    print(f"Loading {anth_path}...")
    with open(anth_path, 'r') as f:
        anth_data = json.load(f)
    # Convert 'question': {'0':..., '1':...} to list of records
    df_anth = pd.DataFrame(anth_data)
    df_anth["source_dataset"] = "anthropic_awareness"
    records_anth = df_anth.to_dict(orient="records")
    
    # 2. Load Synthetic Facts
    syn_path = os.path.join("Black_vs_White_Lie_Detection", "Data", "synthetic_facts.json")
    print(f"Loading {syn_path}...")
    with open(syn_path, 'r') as f:
        syn_data = json.load(f)
    df_syn = pd.DataFrame(syn_data)
    df_syn["source_dataset"] = "synthetic_facts"
    records_syn = df_syn.to_dict(orient="records")
    
    # 3. Load Common Sense
    cs_path = os.path.join("Black_vs_White_Lie_Detection", "Data", "common_sens_qa_v2.json")
    print(f"Loading {cs_path}...")
    with open(cs_path, 'r') as f:
        cs_data = json.load(f)
    # Check format: list or dict?
    if isinstance(cs_data, dict):
        df_cs = pd.DataFrame(cs_data)
        records_cs = df_cs.to_dict(orient="records")
    else:
        records_cs = cs_data
    for r in records_cs: r["source_dataset"] = "common_sense"
        
    # 4. Load Math Problems
    math_path = os.path.join("Black_vs_White_Lie_Detection", "Data", "math_problems.json")
    print(f"Loading {math_path}...")
    with open(math_path, 'r') as f:
        math_data = json.load(f)
    if isinstance(math_data, dict):
        df_math = pd.DataFrame(math_data)
        records_math = df_math.to_dict(orient="records")
    else:
        records_math = math_data
    for r in records_math: r["source_dataset"] = "math_problems"

    # 5. Load Questions 1000
    q1000_path = os.path.join("Black_vs_White_Lie_Detection", "Data", "questions_1000.json")
    print(f"Loading {q1000_path}...")
    with open(q1000_path, 'r') as f:
        q1000_data = json.load(f)
    if isinstance(q1000_data, dict):
        df_q1000 = pd.DataFrame(q1000_data)
        records_q1000 = df_q1000.to_dict(orient="records")
    else:
        records_q1000 = q1000_data
    for r in records_q1000: r["source_dataset"] = "questions_1000"

    # Combine all
    all_raw_records = records_anth + records_syn + records_cs + records_math + records_q1000
    
    # Filter and Standardize to only keep desired fields
    clean_records = []
    
    for i, r in enumerate(all_raw_records):
        
        # Handle answer mapping (sometimes it might be named differently or need extraction)
        answer = r.get("answer")
        # Fallback if answer is missing but statement exists (for synthetic facts) and we want the truth
        if answer is None and "answer" not in r and "statement" in r: 
             # Wait, in synthetic facts "answer" IS present (e.g. "Santiago").
             # "statement" is "The city with... is Santiago".
             # User wants the answer.
             pass

        # Handle category mapping
        category = r.get("category")
        if not category:
             category = r.get("source_dataset")
             
        new_record = {
            "id": i,
            "question": r.get("question", ""),
            "answer": answer,
            "category": category,
            "source_dataset": r.get("source_dataset")
        }
        
        # Only add if we have at least a question and answer (and answer isn't empty)
        if new_record["question"] and new_record["answer"] is not None:
             clean_records.append(new_record)
        
    # Save
    out_path = os.path.join("..", "Data", "raw_dataset.json") # Correct path relative to script location in Black_Box
    # Or absolute:
    # out_path = os.path.join("Black_vs_White_Lie_Detection", "Data", "raw_dataset.json") # depends on where we run it from
    # Let's use the same path logic as before but be explicit
    out_path = os.path.join(os.path.dirname(__file__), "..", "Data", "raw_dataset.json")
    
    with open(out_path, 'w') as f:
        json.dump(clean_records, f, indent=2)
        
    print(f"Done! Saved {len(clean_records)} clean records to {out_path}")
