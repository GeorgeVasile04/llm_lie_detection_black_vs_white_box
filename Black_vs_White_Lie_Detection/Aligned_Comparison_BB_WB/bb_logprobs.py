import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os

def load_probes(csv_path="probes.csv"):
    """
    Loads elicitation questions from the CSV file.
    Returns a dataframe with 'probe_type' and 'probe' columns.
    """
    # If path is relative, try to find it relative to this script
    if not os.path.exists(csv_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "probes.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find probes.csv at {csv_path}")
        
    df = pd.read_csv(csv_path)
    return df

def get_bb_logprobs(model, tokenizer, context_text, probes_df=None, device="cuda"):
    """
    For a given context (which ends with the model's lie or truth),
    append each elicitation question and get the probability of 'Yes' vs 'No'.
    
    Args:
        model: The LLM model
        tokenizer: The tokenizer
        context_text: The conversation history (Question + Answer)
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
        
    for idx, row in probes_df.iterrows():
        probe_text = row['probe']
        category = row['probe_type']
        
        # PROMPT STRUCTURE FOR BLACK BOX:
        # Context (Question + forcing Answer)
        # \n
        # Probe Question
        # (Optional: "Answer:") - helps guide the model
        
        # We append the probe question to the context
        # context_text usually ends with the answer to the original question.
        # We add a newline, then the probe question.
        full_prompt = f"{context_text}\n{probe_text}"
        
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

def compute_bb_features_for_dataset(df, model, tokenizer, device="cuda"):
    """
    Iterates over the aligned dataset and computes BBQ features.
    
    Args:
        df: DataFrame with 'question', 'answer', 'label'
        
    Returns:
        List of samples with appended 'bb_features'
    """
    probes_df = load_probes()
    results = []
    
    from .wb_activations import format_prompt_wb
    
    # We iterate over the dataframe
    # For every sample (Question+Answer), we compute the vector of elicitation scores
    
    for index, row in df.iterrows():
        q = row['question']
        a = row['answer']
        label = row['label']
        
        # 1. Format Context (Same as White Box)
        context_text = format_prompt_wb(q, a)
        
        # 2. Get Features
        bb_result = get_bb_logprobs(model, tokenizer, context_text, probes_df, device)
        
        # 3. Store in a structured way
        # e.g. a flat vector for classification or the full dict
        # For classifiers, we likely want a flat vector, but let's keep metadata
        
        results.append({
            "bb_features": bb_result,
            "label": label,
            "context": context_text
        })
        
    return results

