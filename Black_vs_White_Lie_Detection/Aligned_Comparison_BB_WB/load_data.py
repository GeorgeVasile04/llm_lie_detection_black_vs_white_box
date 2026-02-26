import pandas as pd
import json
import os
from datasets import load_dataset
import random

def load_commonsense_qa(split='validation'):
    """
    Loads the CommonSenseQA dataset and formats it for the lie detection task.
    
    Returns a list of dictionaries, where each dictionary contains:
    - question: The question text.
    - true_answer: The text of the correct answer.
    - false_answer: The text of an incorrect answer (randomly chosen from distractors).
    """
    print(f"Loading CommonSenseQA ({split})...")
    dataset = load_dataset("commonsense_qa", split=split)
    
    processed_data = []
    
    for item in dataset:
        question = item['question']
        choices = item['choices']
        answer_key = item['answerKey']
        
        # logic to find text for answerKey
        labels = choices['label']
        texts = choices['text']
        
        try:
            correct_index = labels.index(answer_key)
            true_answer_text = texts[correct_index]
            
            # Get false answers (distractors)
            false_indices = [i for i, label in enumerate(labels) if label != answer_key]
            if not false_indices:
                continue
                
            # Pick one false answer randomly for the lie scenario
            # In a rigorous setup, we might want to iterate over all, but for simplicity we pick one per question here.
            # Or we could return multiple pairs per question. Let's pick one for now to keep dataset balanced.
            false_index = random.choice(false_indices)
            false_answer_text = texts[false_index]
            
            processed_data.append({
                "id": item['id'],
                "question": question,
                "true_answer": true_answer_text,
                "false_answer": false_answer_text
            })
            
        except ValueError:
            continue
            
    print(f"Loaded {len(processed_data)} items from CommonSenseQA.")
    return pd.DataFrame(processed_data)

def load_local_dataset(file_path):
    """
    Loads a local JSON dataset if available.
    Expects JSON structure with 'question', 'answer', 'false_statement' or similar.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)
