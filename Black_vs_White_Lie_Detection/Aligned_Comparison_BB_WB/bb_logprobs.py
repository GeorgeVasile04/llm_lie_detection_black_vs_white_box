import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os

def load_probes(csv_path="probes.csv", probe_indices=None):
    """
    Loads elicitation questions from the CSV file.
    
    Args:
        csv_path: Path to probes.csv
        probe_indices: Optional numpy array or list of indices to select specific probes.
                      If None, loads all probes. If provided, filters to only these indices.
    
    Returns:
        DataFrame with 'probe_type' and 'probe' columns (filtered if indices provided).
    """
    # If path is relative, try to find it relative to this script
    if not os.path.exists(csv_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "probes.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find probes.csv at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Filter by indices if provided
    if probe_indices is not None:
        df = df.iloc[probe_indices].reset_index(drop=True)
    
    return df

def get_bb_logprobs(model, tokenizer, row, probes_df=None, device="cuda"):
    """
    For a given context (which ends with the model's lie or truth),
    append each elicitation question and get the probability of 'Yes' vs 'No'.
    
    Args:
        model: The LLM model
        tokenizer: The tokenizer
        row: The pandas row containing the text and label.
        probes_df: DataFrame with 'probe' and 'probe_type' columns. If None, loads from CSV.
    
    Returns:
        results: List of dictionaries:
        [
            {
                'category': 'Confession',
                'question': 'Did you lie?',
                'score': float (log(P(Yes)) - log(P(No))),
                'probs': {'yes': float, 'no': float}
            },
            ...
        ]
    """
    if probes_df is None:
        probes_df = load_probes()

    results = []
    from prompt_utils import get_black_box_context
    
    # Tokenizer prep - do this once
    # We look for " Yes" and " No" (with leading space) which is common for Llama-2/Alpaca
    # Check if tokenizer adds space automatically or if we need to.
    
    # Let's get IDs for variants to be robust against tokenizer differences
    # E.g. Llama uses " Yes" (token 3869) vs "Yes" (token 21185)
    yes_variants = ["Yes", " Yes", "yes", " yes"]
    no_variants = ["No", " No", "no", " no"]
    
    yes_ids = []
    no_ids = []
    
    for v in yes_variants:
        # encode returns list of ids, we want single token if possible
        ids = tokenizer.encode(v, add_special_tokens=False)
        if len(ids) == 1:
            yes_ids.append(ids[0])
            
    for v in no_variants:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if len(ids) == 1:
            no_ids.append(ids[0])
    
    # Remove duplicates
    yes_ids = list(set(yes_ids))
    no_ids = list(set(no_ids))
    
    if not yes_ids or not no_ids:
        # Fallback if vocabulary is weird
        print("Warning: Could not find single token ID for Yes/No variants.")
        return []
        
    for idx, p_row in probes_df.iterrows():
        probe_text = p_row['probe']
        category = p_row['probe_type']
        
        # PROMPT STRUCTURE FOR BLACK BOX:
        # Context (Question + forcing Answer)
        # \n
        # Probe Question
        # (Optional: "Answer:") - helps guide the model
        
        # We append the probe question to the context
        # context_text usually ends with the answer to the original question.
        # We add a newline, then the probe question.
        full_prompt = get_black_box_context(row, probe_text)
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Get logits of the last token generated (prediction for next token)
            logits = outputs.logits[0, -1, :]
            
        # Log-Sum-Exp trick or just sum probs
        probs = torch.softmax(logits, dim=0)
        
        yes_prob = sum([probs[i].item() for i in yes_ids])
        no_prob = sum([probs[i].item() for i in no_ids])
        
        # Avoid log(0)
        epsilon = 1e-10
        yes_log = np.log(max(yes_prob, epsilon))
        no_log = np.log(max(no_prob, epsilon))
        
        # Log-odds difference: log(P(Yes)/P(No))
        score = yes_log - no_log
        
        results.append({
            "category": category,
            "question": probe_text,
            "score": score,
            "probs": {"yes": yes_prob, "no": no_prob}
        })
            
    return results

def compute_bb_features_for_dataset(df, model, tokenizer, device="cuda", batch_size=1, probe_indices=None):
    """
    Iterates over the aligned dataset and computes Black Box features.
    
    Args:
        df: DataFrame with 'question', 'answer', 'label'
        model: The LLM model
        tokenizer: The tokenizer
        device: 'cuda' or 'cpu'
        batch_size: Number of samples to process together (default 1)
        probe_indices: Optional numpy array or list of probe indices to use.
                      If None, uses all probes. If provided, uses only these probes.
                      Set to use exact probes from original Black Box paper.
        
    Returns:
        List of samples with appended 'bb_features' key
    """
    probes_df = load_probes(probe_indices=probe_indices)
    results = []
    
    # Note: Black Box requires individual forward passes per probe per sample,
    # so batch_size mainly reduces redundancy in probe loading and setup,
    # not model computation. For now, we process samples one at a time through probes.
    
    from prompt_utils import get_white_box_context
    
    if batch_size == 1:
        # Original single-sample mode
        for index, row in df.iterrows():
            label = row.get('label', None)
            bb_result = get_bb_logprobs(model, tokenizer, row, probes_df=probes_df, device=device)
            results.append({
                "bb_features": bb_result,
                "label": label,
                "context": get_white_box_context(row)
            })
    else:
        # Process multiple samples (still iterate through probes individually)
        from tqdm import tqdm
        
        rows = list(df.iterrows())
        total_batches = (len(rows) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(total_batches), desc=f"Computing BB features (batch_size={batch_size})"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(rows))
            batch_rows = rows[start_idx:end_idx]
            
            for _, row in batch_rows:
                label = row.get('label', None)
                bb_result = get_bb_logprobs(model, tokenizer, row, probes_df=probes_df, device=device)
                results.append({
                    "bb_features": bb_result,
                    "label": label,
                    "context": get_white_box_context(row)
                })
    
    return results

