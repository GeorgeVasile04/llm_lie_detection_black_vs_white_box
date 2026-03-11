import pandas as pd

def get_base_prompt(row):
    """
    Constructs the shared 'Forced Persona' base prompt for both White Box and Black Box methods.
    
    This replaces the original RepEng templates. Instead of treating the model as a passive 
    evaluator, we force the question and the (true or false) answer into a conversational format.
    The model "believes" it has just generated the answer.
    
    Format (Llama-style chat):
    [INST] Question: {question} [/INST]
    {answer}
    
    Args:
        row: A row from the loaded pandas DataFrame or dictionary.
        
    Returns:
        str: The unified base context.
    """
    # Extract the raw question and answer from the generic loader
    question = row.get('question', '')
    answer = row.get('answer', '')
    
    # If the text field has a fallback because 'question' wasn't clean, 
    # we do our best to map it. Our load_data does map most fields cleanly.
    if not question:
        question = row.get('text', '') # fallback
        
    base_context = f"[INST] Question: {question} [/INST]\n{answer}"
    return base_context

def get_white_box_context(row):
    """
    For the White Box, we extract activations precisely at the end of the base context.
    The model has just "finished" outputting the true or false statement.
    """
    return get_base_prompt(row)

def get_black_box_context(row, probe_text):
    """
    For the Black Box, we take the exact same base context and append the elicitation probe
    as the next user turn in the conversation.
    
    Format:
    [INST] Question: {question} [/INST]
    {answer}
    [INST] {probe_text} [/INST]
    """
    base_context = get_base_prompt(row)
    
    # Append the probe as the next user interaction
    bb_context = f"{base_context}\n[INST] {probe_text} [/INST]"
    return bb_context
