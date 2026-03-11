import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt_utils import get_white_box_context

def load_model(model_name, device="cuda"):
    """
    Loads the model and tokenizer.
    """
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
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

def get_activations_for_dataset(df, model, tokenizer, device="cuda"):
    """
    Iterates over the "Aligned" dataset layout (from load_data.py).
    Each row has 'question', 'answer', 'label'.
    
    Returns:
        List of dicts: [{'activations': ..., 'label': int}, ...]
    """
    results = []
    
    for index, row in df.iterrows():
        # Get the unified conversational context
        context_text = get_white_box_context(row)
        label = row['label']
        
        # Extract activations
        activations = get_wb_activations(model, tokenizer, context_text, device=device)
        
        results.append({
            "activations": activations,
            "label": label,
            "text": context_text
        })
        
    return results
