import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt_utils import get_white_box_context

def load_model(model_name, device="cuda"):
    """
    Loads the model and tokenizer.
    Uses bfloat16 precision (same as original RepEng paper) for numerical stability.
    """
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

    # Llama-family tokenizers often do not define a pad token by default.
    # Batched extraction uses padding=True, so we map PAD -> EOS for safe padding.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # keep right padding for attention_mask-based last-token indexing
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer

def get_wb_activations(model, tokenizer, context_text, layer_nums=None, device="cuda"):
    """
    Extracts activations from the LAST token of the context_text.
    
    Args:
        model: HF Model
        tokenizer: HF Tokenizer
        context_text: formatted prompt (Question + Answer)
        layer_nums: List of integers (layers to extract). If None, extracts all.
        device: 'cuda' or 'cpu'.
        
    Returns:
        activations: Dict {layer_num: numpy_array}
    """
    inputs = tokenizer(context_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
    hidden_states = outputs.hidden_states
    # hidden_states tuple: (layer_0 (embeddings), layer_1, ..., layer_N)
    
    activations = {}
    total_layers = len(hidden_states)
    
    # Default to all layers if not specified
    if layer_nums is None:
        layer_nums = list(range(total_layers))
        
    for layer_idx in layer_nums:
        if layer_idx < total_layers:
            # Extract last token: [batch=0, seq_len=-1, hidden_dim]
            # .float() ensures we return float32 even if model is fp16/bf16
            act = hidden_states[layer_idx][0, -1, :].float().cpu().numpy()
            activations[layer_idx] = act
            
    return activations

def get_activations_for_dataset(df, model, tokenizer, device="cuda", batch_size=1, show_progress=False):
    """
    Iterates over the "Aligned" dataset layout (from load_data.py).
    Each row has 'question', 'answer', 'label'.
    
    Args:
        df: DataFrame with 'question', 'answer', 'label' columns
        model: HF Model
        tokenizer: HF Tokenizer
        device: 'cuda' or 'cpu'
        batch_size: Number of samples per forward pass (default 1, use 16+ for speed)
        show_progress: Whether to show a tqdm progress bar for this dataset
    
    Returns:
        List of dicts: [{'activations': ..., 'label': int, 'text': ...}, ...]
    """
    results = []
    
    if batch_size == 1:
        # Original single-sample mode (simpler, easier to debug)
        for index, row in df.iterrows():
            context_text = get_white_box_context(row)
            label = row['label']
            group_id = row.get('id', index)
            activations = get_wb_activations(model, tokenizer, context_text, device=device)
            results.append({
                "activations": activations,
                "label": label,
                "text": context_text,
                "id": group_id
            })
    else:
        # Batched mode: process multiple samples per forward pass
        from tqdm.auto import tqdm
        
        rows = list(df.iterrows())
        total_batches = (len(rows) + batch_size - 1) // batch_size
        
        # We set disable=not show_progress to avoid overloading the console
        for batch_idx in tqdm(range(total_batches), desc=f"Extracting activations (batch_size={batch_size})", leave=False, disable=not show_progress):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(rows))
            batch_rows = rows[start_idx:end_idx]
            
            # Prepare batch texts and labels
            batch_texts = []
            batch_labels = []
            batch_ids = []
            batch_row_data = []
            
            for index, row in batch_rows:
                context_text = get_white_box_context(row)
                batch_texts.append(context_text)
                batch_labels.append(row['label'])
                batch_ids.append(row.get('id', index))
                batch_row_data.append(context_text)
            
            # Tokenize all texts in batch
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            
            # Forward pass for entire batch
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            hidden_states = outputs.hidden_states
            total_layers = len(hidden_states)
            
            # Extract last token for each sample in batch
            # attention_mask tells us where the actual tokens end (ignoring padding)
            attention_mask = inputs['attention_mask']
            
            for sample_idx, label in enumerate(batch_labels):
                # Find the last non-padded token position for this sample
                last_token_pos = attention_mask[sample_idx].sum().item() - 1
                
                activations = {}
                for layer_idx in range(total_layers):
                    # Extract activation at last token position for this sample
                    # Move immediately to CPU numpy array to save GPU memory
                    act = hidden_states[layer_idx][sample_idx, last_token_pos, :].float().cpu().numpy()
                    activations[layer_idx] = act
                
                results.append({
                    "activations": activations,
                    "label": label,
                    "text": batch_row_data[sample_idx],
                    "id": batch_ids[sample_idx]
                })
            
            # Explicitly free batch memory to prevent OOM
            del inputs
            del outputs
            
            # Since hidden_states is a tuple, we must make sure all layers inside the tuple are deleted
            for t in hidden_states:
                del t
            del hidden_states
            del attention_mask
            torch.cuda.empty_cache()
    
    return results
