import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

def load_model(model_name, device="cuda"):
    """
    Loads the model and tokenizer.
    """
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    return model, tokenizer

def format_prompt(question, answer, mode="chat"):
    """
    Formats the input prompt.
    For base models: "Question: {q} Answer: {a}"
    For chat models: Standard chat template.
    """
    # Simple template for now, can be expanded
    return f"Question: {question}\nAnswer: {answer}"

def get_last_token_activations(model, tokenizer, text, layer_nums=None, device="cuda"):
    """
    Runs the model on the text and extracts hidden states (activations) of the last token.
    
    Args:
        model: Loaded HF model.
        tokenizer: Loaded HF tokenizer.
        text: Input text (Question + Answer).
        layer_nums: List of layer indices to extract. If None, extracts all.
        device: 'cuda' or 'cpu'.
        
    Returns:
        A dictionary mapping layer_num -> activation_vector (numpy array).
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
    hidden_states = outputs.hidden_states
    
    # hidden_states is a tuple of (num_layers + 1) tensors
    # shape: (batch_size, seq_len, hidden_dim)
    
    activations = {}
    total_layers = len(hidden_states)
    
    # Default to all layers if not specified
    if layer_nums is None:
        layer_nums = list(range(total_layers))
        
    for layer_idx in layer_nums:
        if layer_idx < total_layers:
            # Get last token hidden state: [0, -1, :]
            layer_vec = hidden_states[layer_idx][0, -1, :].cpu().numpy()
            activations[layer_idx] = layer_vec
            
    return activations

def get_activations_for_dataset(df, model, tokenizer, device="cuda"):
    """
    Iterates over the dataset and extracts activations for both Truth and Lie scenarios.
    
    Returns:
        X: List of activation vectors (n_samples, hidden_dim) - typically specific layer.
        y: List of labels (1 for Truth, 0 for Lie).
    """
    # This is a high-level function, might need to be specific about which layer to use for X
    # For now, let's return a dictionary structure storing all layers
    
    results = []
    
    for index, row in df.iterrows():
        q = row['question']
        true_a = row['true_answer']
        false_a = row['false_answer']
        
        # Truth Scenario
        text_truth = format_prompt(q, true_a)
        acts_truth = get_last_token_activations(model, tokenizer, text_truth, device=device)
        results.append({
            "activations": acts_truth,
            "label": 1,
            "text": text_truth
        })
        
        # Lie Scenario
        text_lie = format_prompt(q, false_a)
        acts_lie = get_last_token_activations(model, tokenizer, text_lie, device=device)
        results.append({
            "activations": acts_lie,
            "label": 0,
            "text": text_lie
        })
        
    return results
