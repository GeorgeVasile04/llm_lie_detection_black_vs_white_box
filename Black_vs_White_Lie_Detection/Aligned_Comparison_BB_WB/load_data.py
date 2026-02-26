import pandas as pd
from datasets import load_dataset
# Set a seed for reproducibility if needed, though we are taking all items now.

def load_dataset_aligned(dataset_name="commonsense_qa", split='validation', n_samples=None):
    """
    Loads the CommonSenseQA dataset and aligns it with the White Box methodology.
    
    In the original White Box paper (repeng), for CommonSenseQA:
    - They iterate over ALL choices for each question.
    - If choice == correct -> Label TRUE
    - If choice != correct -> Label FALSE
    
    So for each question with 5 choices, we get 5 samples (1 Pos, 4 Neg).
    
    This function returns a list of dictionaries:
    [
        {
            "id": "q_id",
            "question": "...",
            "answer": "...",
            "label": 1 (True) or 0 (False)
        },
        ...
    ]
    """
    print(f"Loading {dataset_name} ({split})...")
    
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

    if n_samples:
        dataset = dataset.select(range(min(n_samples, len(dataset))))

    processed_data = []

    for item in dataset:
        question = item['question']
        choices = item['choices']
        answer_key = item['answerKey'] # e.g. "A"
        
        labels = choices['label'] # ["A", "B", "C", "D", "E"]
        texts = choices['text']   # ["Answer A", "Answer B", ...]
        
        # We process ALL choices to match original White Box data volume/balance
        for label, text in zip(labels, texts):
            is_correct = (label == answer_key)
            
            processed_data.append({
                "id": item['id'],
                "question": question,
                "answer": text,
                "label": 1 if is_correct else 0
            })
            
    print(f"Loaded {len(dataset)} questions.")
    print(f"Generated {len(processed_data)} aligned samples (question-answer pairs).")
    
    return processed_data

if __name__ == "__main__":
    # Test loading
    data = load_dataset_aligned(n_samples=100)
    print(f"Sample 0: {data[0]}")
    print(f"Sample 1: {data[1]}")
