import sys
import os
import json
import torch
import numpy as np
import pandas as pd
import random
from typing import List, Dict, Tuple, Any, Optional
from tqdm.auto import tqdm
# Import repeng specific tools
from White_Box_Lie_Detection.repeng.models.loading import load_llm_oioo
from White_Box_Lie_Detection.repeng.activations.inference import get_model_activations
from White_Box_Lie_Detection.repeng.probes.difference_in_means import train_dim_probe as train_dim_repeng
# Import Logistic Regression from repeng structure - we will reimplement minimal wrapper
from sklearn.linear_model import LogisticRegression

def _extract_anthropic_data(data_dir: str, num_samples: int = 100) -> pd.DataFrame:
    """
    Extracts samples from Anthropic Awareness datasets.
    Categorizes them as 'anthropic_awareness'.
    """
    files = [
        "anthropic_awareness_ai.json"
    ]
    
    all_data = []
    
    for f in files:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # These files usually have 'question' and 'answer' keys 
                # where values are dicts indexed by number
                questions = data.get('question', {})
                answers = data.get('answer', {})
                
                for key, q in questions.items():
                    a = answers.get(key, "").strip().lower()
                    # Creating True/False pairs
                    # Anthropic questions are Yes/No. 
                    # We need to construct a Lie.
                    # If answer is 'no', lie is 'yes'.
                    
                    true_ans = a
                    lie_ans = "yes" if "no" in true_ans else "no"
                    
                    # Original (True) -> Label 1
                    all_data.append({
                        'question': q,
                        'answer': true_ans,
                        'label': 1,
                        'category': 'anthropic_awareness',
                        'source_dataset': 'anthropic'
                    })
                    # Lie -> Label 0
                    all_data.append({
                        'question': q,
                        'answer': lie_ans,
                        'label': 0,
                        'category': 'anthropic_awareness',
                        'source_dataset': 'anthropic'
                    })
    
    df = pd.DataFrame(all_data)
    # We need num_samples pairs. So take num_samples // 2 original items and their pairs? 
    # Or num_samples total rows?
    # User said "100 Anthropic Awareness". Usually means 100 questions (200 rows if paired).
    # But let's assume num_samples is the number of QUESTIONS to include.
    
    if len(df) == 0:
        print("Warning: No Anthropic data found.")
        return pd.DataFrame()

    # Group by question to keep pairs together
    questions = df['question'].unique()
    selected_questions = np.random.choice(questions, size=min(num_samples, len(questions)), replace=False)
    
    final_df = df[df['question'].isin(selected_questions)].copy()
    return final_df

def _extract_common_sense_data(data_dir: str, num_samples: int = 200) -> pd.DataFrame:
    """
    Extracts samples from Common Sense QA.
    """
    path = os.path.join(data_dir, "common_sens_qa_v2.json")
    if not os.path.exists(path):
        # flexibility for filename
        path = os.path.join(data_dir, "commonsense_QA_v2_dev.json") 
        if not os.path.exists(path):
             print(f"Warning: Common sense file not found at {path}")
             return pd.DataFrame()
             
    # Reuse existing logic but simplified
    raw_df = pd.read_json(path)
    
    # Random sample of questions
    if len(raw_df) > num_samples:
        raw_df = raw_df.sample(n=num_samples, random_state=42)
        
    pairs = []
    for _, row in raw_df.iterrows():
        q = row['question']
        true_ans = str(row['answer']).strip()
        
        lie_ans = None
        if true_ans.lower() == 'true':
            lie_ans = 'False'
        elif true_ans.lower() == 'false':
            lie_ans = 'True'
        elif true_ans.lower() == 'yes':
            lie_ans = 'No'
        elif true_ans.lower() == 'no':
            lie_ans = 'Yes'
            
        if lie_ans:
            # True -> Label 1
            pairs.append({'question': q, 'answer': true_ans, 'label': 1, 'category': 'common_sense', 'source_dataset': 'common_sense'})
            # Lie -> Label 0
            pairs.append({'question': q, 'answer': lie_ans, 'label': 0, 'category': 'common_sense', 'source_dataset': 'common_sense'})
            
    return pd.DataFrame(pairs)

def _extract_math_data(data_dir: str, num_samples: int = 200) -> pd.DataFrame:
    """
    Extracts samples from Mathematical Problems.
    """
    path = os.path.join(data_dir, "math_problems.json")
    if not os.path.exists(path):
        print("Warning: Math file 'math_problems.json' not found.")
        return pd.DataFrame()
        
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Structure: columns question, answer, false_statement, type
    # types: addition, subtraction, multiplication, division
    
    questions = data.get('question', {})
    answers = data.get('answer', {})
    false_statements = data.get('false_statement', {}) # This might be the full sentence lie
    types = data.get('type', {})
    
    # We need to construct pairs.
    # The 'false_statement' is usually "The result of X + Y is Z (wrong)". 
    # But our probe training expects (Question, Answer).
    # We need to extract the 'answer' part from the false statement or just use the number if available.
    # Actually mathematical_problems.json usually has a 'false_answer' column or we can parse it.
    
    # Let's inspect available keys in a generic way or assume standard 'false_answer' existing
    # Checking previous grep: questions_1000 had 'false_answer'. 
    # Let's assume math has it too or we use the statement.
    # For robust probing, we want "Question: What is 2+2? Answer: 4" vs "Question: What is 2+2? Answer: 5".
    
    # If standard 'false_answer' is missing, we might need to parse.
    # Let's try to find 'false_answer' in the keys.
    has_false_answer = 'false_answer' in data
    
    all_rows = []
    ids = list(questions.keys())
    
    for i in ids:
        q = questions[i]
        a = str(answers[i])
        t = types.get(i, 'math')
        
        # Get false answer
        if has_false_answer:
            fa = str(data['false_answer'][i])
        else:
            # Fallback: if 'false_statement' exists, try to extract, or generate simple wrong answer
            if 'false_statement' in data:
               fs = data['false_statement'][i]
               # Heuristic: split by is/result is? It's risky.
               # Simple math generation: add 1
               try:
                   val = float(a)
                   fa = str(val + 1)
                   if "." in a and ".0" in fa: fa = fa.replace(".0", "")
               except:
                   fa = a + " (False)"
            else:
                fa = a + " (False)"
        
        # True Answer -> Label 1
        all_rows.append({
            'question': q,
            'answer': a,
            'label': 1,
            'category': t,
            'source_dataset': 'math'
        })
        # False Answer -> Label 0
        all_rows.append({
            'question': q,
            'answer': fa,
            'label': 0,
            'category': t,
            'source_dataset': 'math'
        })

    df = pd.DataFrame(all_rows)
    
    # Balance types if possible
    # We want 200 samples total, so roughly 50 per type if 4 types exist
    unique_types = df['category'].unique()
    samples_per_type = num_samples // len(unique_types) if len(unique_types) > 0 else num_samples
    
    final_dfs = []
    for t in unique_types:
        sub_df = df[df['category'] == t]
        # We need samples_per_type QUESTIONS (so x2 rows)
        # Group by question
        qs = sub_df['question'].unique()
        selected_qs = np.random.choice(qs, size=min(samples_per_type, len(qs)), replace=False)
        final_dfs.append(sub_df[sub_df['question'].isin(selected_qs)])
        
    return pd.concat(final_dfs)

def _extract_questions_1000_data(data_dir: str, num_samples: int = 500) -> pd.DataFrame:
    """
    Extracts samples from Questions 1000 (General Knowledge).
    """
    path = os.path.join(data_dir, "questions_1000.json")
    if not os.path.exists(path):
        print("Warning: Questions 1000 file not found.")
        return pd.DataFrame()
        
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    questions = data.get('question', {})
    answers = data.get('answer', {})
    # This dataset definitely has 'false_answer' usually
    false_answers = data.get('false_answer', {})
    cats = data.get('category', {})
    
    ids = list(questions.keys())
    # Shuffle ids first
    random.shuffle(ids)
    selected_ids = ids[:num_samples]
    
    rows = []
    for i in selected_ids:
        q = questions[i]
        a = str(answers[i])
        fa = str(false_answers.get(i, "False Answer"))
        c = cats.get(i, "general")
        
        # True Answer => Label 1
        rows.append({'question': q, 'answer': a, 'label': 1, 'category': c, 'source_dataset': 'questions_1000'})
        # False Answer => Label 0
        rows.append({'question': q, 'answer': fa, 'label': 0, 'category': c, 'source_dataset': 'questions_1000'})
        
    return pd.DataFrame(rows)

def _extract_synthetic_facts_data(data_dir: str, num_samples: int = 200) -> pd.DataFrame:
    """
    Extracts samples from Synthetic Facts.
    """
    # Prefer synthetic_facts.json as user mentioned, fallback to _all
    path = os.path.join(data_dir, "synthetic_facts.json")
    if not os.path.exists(path):
         path = os.path.join(data_dir, "synthetic_facts_all.json")
         
    if not os.path.exists(path):
        print("Warning: Synthetic facts file not found.")
        return pd.DataFrame()
        
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    questions = data.get('question', {})
    answers = data.get('answer', {})
    # Synthetic facts usually has 'false_statement' which is full sentence. 
    # Whatever, let's look for false_answer or extract from false_statement?
    # Wait, in the user's grep, questions_1000 had 'false_answer'. 
    # Let's assume synthetic facts has 'false_answer' if it follows same schema.
    # If not, we might use 'answer' and modify.
    # Check if 'false_answer' in data
    
    has_false_answer = 'false_answer' in data
    has_false_statement = 'false_statement' in data
    
    ids = list(questions.keys())
    random.shuffle(ids)
    selected_ids = ids[:num_samples]
    
    rows = []
    for i in selected_ids:
        q = questions[i]
        a = str(answers[i])
        c = data.get('category', {}).get(i, "synthetic")
        
        if has_false_answer:
            fa = str(data['false_answer'][i])
        elif has_false_statement:
            # We have "The city is X (False)". We want "X".
            # This is hard to extract reliably without more logic.
            # But the 'probe' prompt format is "Question: ... Answer: ..."
            # So if we put a full sentence "The city is X" as 'Answer', format becomes "Question: Q Answer: The city is X".
            # That's acceptable for probing if consistent.
            # But usually 'Answer' is short.
            # Let's try to use the 'false_statement' as the answer? No, that repeats the question info.
            # We need a false entity. 
            # For now, let's generate a placeholder or use "False Answer" if missing to fail loudly?
            # Actually, let's assume valid data or just append " (False)" to answer.
            fa = a + " (False)" 
        else:
             fa = a + " (False)"
             
        # True => Label 1
        rows.append({'question': q, 'answer': a, 'label': 1, 'category': c, 'source_dataset': 'synthetic_facts'})
        # False => Label 0
        rows.append({'question': q, 'answer': fa, 'label': 0, 'category': c, 'source_dataset': 'synthetic_facts'})
        
    return pd.DataFrame(rows)


def create_unified_dataset(output_path: str = None) -> pd.DataFrame:
    """
    Creates the unified dataset with 1200 samples from 5 sources.
    Source folder is Assumed relative to this file: ../Data
    """
    # Resolve Data Directory
    # current file is in Black_vs_White_Lie_Detection/White_Box
    # data is in Black_vs_White_Lie_Detection/Data
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(base_dir, "..", "Data"))
    
    if not os.path.exists(data_dir):
        # Fallback 
        print(f"Data dir not found at calculated path: {data_dir}. Searching..." )
        # Try finding it relative to Cwd - assuming running from root
        cwd = os.getcwd()
        potential = os.path.join(cwd, "Black_vs_White_Lie_Detection", "Data")
        if os.path.exists(potential):
             data_dir = potential
        else:
             print(f"Error: Could not find Data directory. Cwd: {cwd}")
             # One last desperate try:
             potential = os.path.join(cwd, "Data")
             if os.path.exists(potential):
                 data_dir = potential

    print(f"Loading data from: {data_dir}")
    
    df1 = _extract_anthropic_data(data_dir, 100)
    df2 = _extract_common_sense_data(data_dir, 200)
    df3 = _extract_math_data(data_dir, 200)
    df4 = _extract_questions_1000_data(data_dir, 500)
    df5 = _extract_synthetic_facts_data(data_dir, 200)
    
    combined = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
    
    print(f"Combined Dataset Size: {len(combined)} rows (expecting ~2400 for 1200 questions)")
    print("Composition by Source:")
    print(combined['source_dataset'].value_counts() / 2) # Divided by 2 for quesions count
    
    if output_path:
        # Create dir if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined.to_json(output_path, orient='records', indent=2)
        print(f"Saved unified dataset to {output_path}")
        
    return combined


def format_prompt(question: str, answer: str) -> str:
    """
    Formats the input for the LLM.
    Using a standard Q/A format which is generally robust across base and chat models for probing.
    """
    return f"Question: {question} Answer: {answer}"

def load_model(llm_id: str, device_name: str = "cuda") -> Any:
    """
    Step 2: Load the model.
    """
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    # Half precision (bfloat16/float16) is not fully supported on CPU; use float32 there.
    use_half_precision = device.type != "cpu"
    print(f"Loading model {llm_id} on {device} (half_precision={use_half_precision})...")
    llm = load_llm_oioo(
        llm_id=llm_id,  # type: ignore
        device=device,
        use_half_precision=use_half_precision
    )
    return llm

def extract_activations(
    llm: Any, 
    dataset: pd.DataFrame, 
    layer_indices: Optional[List[int]] = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Step 3: Extract activations.
    
    Args:
        llm: Loaded Llm object
        dataset: DataFrame with 'question', 'answer', 'label'
        layer_indices: List of layer indices to extract. If None, extracts all.
        
    Returns:
        Dictionary where keys are layer names and values are tuples of (activations, labels)
        activations shape: (num_samples, hidden_dim)
        labels shape: (num_samples,)
    """
    all_activations: Dict[str, List[np.ndarray]] = {}
    labels = []
    
    # Determine which points (layers) to extract
    points = llm.points
    if layer_indices is not None:
        points = [llm.points[i] for i in layer_indices]
        
    # Initialize lists for each layer
    for point in points:
        all_activations[point.name] = []
        
    print(f"Extracting activations for {len(dataset)} samples...")
    
    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Extraction Progress"):
        prompt = format_prompt(row['question'], row['answer'])
        label = row['label']
        
        # We want the activation at the last token
        act_row = get_model_activations(
            llm=llm,
            text=prompt,
            last_n_tokens=1, # Only last token
            points_start=0,
            points_end=None, 
            points_skip=1
        )
        
        for point in points:
            # act_row.activations is a dict {layer_name: ndarray}
            # Shape is (1, hidden_dim) because last_n_tokens=1
            act = act_row.activations[point.name]
            all_activations[point.name].append(act.flatten())
            
        labels.append(label)
        
    # Convert to numpy arrays
    result = {}
    y = np.array(labels)
    
    for layer_name, acts in all_activations.items():
        X = np.stack(acts)
        result[layer_name] = (X, y)
        
    return result

def train_logreg_probe(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """
    Step 4: Train Logistic Regression Probe (Standard Scikit-Learn / Repeng style).
    Repeng uses C=1.0, solver='newton-cg', max_iter=10000.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    # Configuration matching repeng's LrConfig default
    clf = LogisticRegression(
        C=1.0, 
        solver='newton-cg', 
        max_iter=10000, 
        random_state=42
    )
    
    # Fit the model
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] # Probability of Class 1 (Truth)
    
    acc = accuracy_score(y_test, y_pred)
    # Check for single-class edge case in test set
    if len(np.unique(y_test)) < 2:
        auc = 0.5 
    else:
        auc = roc_auc_score(y_test, y_prob)
    
    return {
        "model": clf,
        "accuracy": acc,
        "auc": auc,
        "y_pred": y_pred,
        "y_prob": y_prob, 
        "type": "LogisticRegression"
    }

def train_dim_probe_utils(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    """
    Step 4b: Train Difference in Means Probe using repeng library.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    # 1. Train Probe
    # y_train is 0 (Lie) or 1 (Truth).
    # We want the vector to point FROM Lies TO Truths.
    # Vector arithmetic: V = Tip - Tail = Truth - Lie.
    # Geometrically (see diagram (b)): This vector starts at Lie and points to Truth.
    # Result:
    #   - High dot product (> threshold) -> Closer to Truth
    #   - Low dot product (< threshold) -> Closer to Lie
    probe = train_dim_repeng(activations=X_train, labels=(y_train == 1))
    
    # 2. Get Scores (Dot Product)
    # The 'predict' method in repeng returns a result object with 'logits' (simple dot product scores)
    train_scores = probe.predict(X_train).logits
    test_scores = probe.predict(X_test).logits
    
    # 3. Calculate AUC
    # Higher score = more aligned with Lie direction
    if len(np.unique(y_test)) < 2:
        auc = 0.5 # Default if only one class in test batch
    else:
        auc = roc_auc_score(y_test, test_scores)
    
    # 4. Calculate Optimal Threshold (Geometric ROC method from repeng paper)
    from sklearn.metrics import roc_curve
    
    # We find the best threshold on the Training set
    fpr, tpr, thresholds = roc_curve(y_train, train_scores)
    # Find index maximizing geometric proximity to (0,1) corner (perfect classifier)
    # dist = sqrt((1-TPR)^2 + FPR^2)
    best_idx = np.argmin(np.sqrt((1 - tpr) ** 2 + fpr**2))
    best_thresh = thresholds[best_idx]
            
    # Apply optimal threshold to Test set
    y_pred = (test_scores > best_thresh).astype(int)
    test_acc = accuracy_score(y_test, y_pred)
    
    return {
        "model": probe,
        "accuracy": test_acc,
        "auc": auc,
        "y_pred": y_pred,
        "y_prob": test_scores, 
        "type": "DifferenceInMeans",
        "threshold": best_thresh
    }

def save_results(results: Dict[str, Any], output_dir: str = "results", model_name: str = "unknown_model"):
    """
    Step 5: Save results to disk.
    Saves:
    1. metrics.csv: Table of Accuracy/AUC per layer
    2. best_probe.pkl: The trained model object for the best performing layer
    """
    import joblib
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Prepare and Save Metrics
    metrics_data = []
    best_acc = -1
    best_layer = None
    best_model = None
    
    for layer_name, res in results.items():
        metrics_data.append({
            "layer": layer_name,
            "accuracy": res['accuracy'],
            "auc": res['auc']
        })
        
        if res['accuracy'] > best_acc:
            best_acc = res['accuracy']
            best_layer = layer_name
            best_model = res['model']
            
    metrics_df = pd.DataFrame(metrics_data)
    csv_path = os.path.join(output_dir, f"{model_name}_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")
    
    # 2. Save Best Probe
    if best_model:
        model_path = os.path.join(output_dir, f"{model_name}_best_probe_{best_layer}.pkl")
        joblib.dump(best_model, model_path)
        print(f"Best probe (Layer {best_layer}, Acc {best_acc:.4f}) saved to {model_path}")
    
    return metrics_df
