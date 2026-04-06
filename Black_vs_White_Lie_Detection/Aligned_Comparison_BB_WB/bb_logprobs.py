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

def get_bb_logprobs(model, tokenizer, row, probes_df=None, device="cuda", probe_batch_size=24):
    """Compute Black Box logprobs for a single example using **independent** probe prompts.

    This avoids cross-probe contamination by making each probe a separate prompt.

    Args:
        model: The LLM model
        tokenizer: The tokenizer
        row: The pandas row containing the text and label.
        probes_df: DataFrame with 'probe' and 'probe_type' columns. If None, loads from CSV.
        device: Device to run model on.
        probe_batch_size: Ensure we don't OOM with long sequences by limiting max forward pass prompts.

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

    # Build one prompt per probe so that probes do NOT influence each other.
    from prompt_utils import get_black_box_context

    probes = []
    categories = []
    for _, p_row in probes_df.iterrows():
        probes.append(p_row["probe"])
        categories.append(p_row["probe_type"])

    prompts = [get_black_box_context(row, probe_text) for probe_text in probes]

    # Find token ids for yes/no (single-token variants only)
    yes_variants = ["Yes", "yes"]
    no_variants = ["No", "no"]

    yes_ids = [tokenizer.encode(v, add_special_tokens=False)[0] for v in yes_variants if len(tokenizer.encode(v, add_special_tokens=False)) == 1]
    no_ids = [tokenizer.encode(v, add_special_tokens=False)[0] for v in no_variants if len(tokenizer.encode(v, add_special_tokens=False)) == 1]

    if not yes_ids or not no_ids:
        print("Warning: Could not find single token ID for Yes/No variants.")
        return []

    results = []
    
    # Chunk the prompts to avoid Out Of Memory errors for long texts (like IMDb)
    for i in range(0, len(prompts), probe_batch_size):
        chunk_prompts = prompts[i:i + probe_batch_size]
        chunk_probes = probes[i:i + probe_batch_size]
        chunk_categories = categories[i:i + probe_batch_size]

        # Tokenize as a batch so we do a single forward pass.
        inputs = tokenizer(
            chunk_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # Shape: [batch, seq_len, vocab]

        # For each item in the batch, extract the logits at the last non-padding position.
        attention_mask = inputs["attention_mask"]
        last_token_idx = attention_mask.sum(dim=1) - 1  # [batch]

        for batch_i, (probe_text, category) in enumerate(zip(chunk_probes, chunk_categories)):
            idx = last_token_idx[batch_i].item()
            probe_logits = logits[batch_i, idx]
            probs = torch.softmax(probe_logits, dim=-1)

            yes_prob = sum([probs[j].item() for j in yes_ids])
            no_prob = sum([probs[j].item() for j in no_ids])

            epsilon = 1e-10
            yes_log = np.log(max(yes_prob, epsilon))
            no_log = np.log(max(no_prob, epsilon))
            score = yes_log - no_log

            results.append({
                "category": category,
                "question": probe_text,
                "score": score,
                "probs": {"yes": yes_prob, "no": no_prob},
            })

        # Explicitly clear memory to prevent OOM
        del inputs
        del outputs
        del logits
        del attention_mask
        torch.cuda.empty_cache()

    return results


def compute_bb_features_for_dataset(
    df,
    model,
    tokenizer,
    device="cuda",
    batch_size=1,
    probe_indices=None,
    show_progress=False,
    probe_batch_size=24,
):
    """Iterate over a dataset and compute Black Box probe features.

    Args:
        df: DataFrame with 'question', 'answer', 'label'
        model: The LLM model
        tokenizer: The tokenizer
        device: 'cuda' or 'cpu'
        batch_size: Number of samples to process together (default 1)
        probe_indices: Optional list/array of probe indices to use. If None, uses all probes.
        show_progress: Whether to show a tqdm progress bar for this dataset
        probe_batch_size: Chunk size for evaluating probes to avoid GPU OOM.

    Returns:
        List of samples with appended 'bb_features' key.
    """
    probes_df = load_probes(probe_indices=probe_indices)
    results = []

    from prompt_utils import get_white_box_context

    if batch_size == 1:
        for _, row in df.iterrows():
            label = row.get("label", None)
            bb_result = get_bb_logprobs(model, tokenizer, row, probes_df=probes_df, device=device, probe_batch_size=probe_batch_size)
            results.append({
                "bb_features": bb_result,
                "label": label,
                "context": get_white_box_context(row),
            })
    else:
        from tqdm.auto import tqdm

        rows = list(df.iterrows())
        total_batches = (len(rows) + batch_size - 1) // batch_size

        for batch_idx in tqdm(
            range(total_batches), desc=f"Computing BB features (batch_size={batch_size})", leave=False, disable=not show_progress
        ):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(rows))
            batch_rows = rows[start_idx:end_idx]

            for _, row in batch_rows:
                label = row.get("label", None)
                bb_result = get_bb_logprobs(
                    model, tokenizer, row, probes_df=probes_df, device=device, probe_batch_size=probe_batch_size
                )
                results.append({
                    "bb_features": bb_result,
                    "label": label,
                    "context": get_white_box_context(row),
                })

    return results

