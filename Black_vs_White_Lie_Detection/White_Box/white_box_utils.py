import sys
import os
import json
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from tqdm.auto import tqdm
# Import repeng specific tools
from White_Box_Lie_Detection.repeng.models.loading import load_llm_oioo
from White_Box_Lie_Detection.repeng.activations.inference import get_model_activations
from White_Box_Lie_Detection.repeng.probes.difference_in_means import train_dim_probe as train_dim_repeng

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Step 1: Load the dataset.
    Handles both 'records' (list of dicts) and 'columns' (dict of lists/dicts) JSON formats.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
        
    if file_path.endswith('.json'):
        # using orient=None allows pandas to automatically detect the structure 
        # (e.g. 'columns' like {"col": {"idx": val}} or 'records' like [{"col": val}])
        df = pd.read_json(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Use .json or .csv")
    
    return df

def generate_pairs_from_raw_data(df: pd.DataFrame, num_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Helper to process raw QA data into Truth/Lie pairs.
    """
    processed_rows = []
    
    if num_samples:
        df = df.head(num_samples)
    
    for _, row in df.iterrows():
        q = row['question']
        true_ans = str(row['answer']).strip()
        
        # Determine lie answer (simple inversion for True/False)
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
            # Honest example (Label 0)
            processed_rows.append({
                'question': q,
                'answer': true_ans,
                'label': 0
            })
            # Lying example (Label 1)
            processed_rows.append({
                'question': q,
                'answer': lie_ans,
                'label': 1
            })
            
    df_pairs = pd.DataFrame(processed_rows)
    # Shuffle the dataset so we don't have alternating True/False patterns
    return df_pairs.sample(frac=1, random_state=42).reset_index(drop=True)

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
    Step 4: Train Logistic Regression Probe.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
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
    # y_train is 0/1. Convert to boolean for repeng (1=True for mask)
    # If 1 is Lie, mask selects Lies. Probe = Mean(Lies) - Mean(Truths)
    # So the "direction" V points towards Lies.
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
