import sys
import os
import numpy as np
from typing import Literal, Dict, Any, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score
from pathlib import Path

# Add project root to path to allow imports from Aligned Comparison
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
original_aligned_path = current_dir.parent / "Aligned_Comparison_BB_WB"

if str(original_aligned_path) not in sys.path:
    sys.path.append(str(original_aligned_path))
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from Black_vs_White_Lie_Detection.Aligned_Comparison_BB_WB.wb_probes import (
    train_dim_probe,
    train_lr_probe,
    train_lr_g_probe,
    train_lda_probe,
    train_pca_probe,
    train_pca_g_probe,
    train_lat_probe,
    train_ccs_probe,
    ProbeMethod
)

def train_wb_probes_imbalance(
    activation_data: List[Dict[str, Any]],
    eval_data: Optional[List[Dict[str, Any]]] = None,
    layer_list: List[int] = [10, 15, 20],
    method: ProbeMethod = "lr",
    test_size: float = 0.2,
    verbose: bool = True,
):
    """
    MODIFIED from Aligned_Comparison_BB_WB for Class Imbalance Study:
    Also evaluates Macro F1, Recall for Truth, and Recall for Lies.
    """
    probes = {}
    metrics = {}
    
    y_all = np.array([item['label'] for item in activation_data])
    g_all = np.array([item.get('id', i) for i, item in enumerate(activation_data)])
    
    if eval_data is not None:
        y_eval_all = np.array([item['label'] for item in eval_data])
        g_eval_all = np.array([item.get('id', i) for i, item in enumerate(eval_data)])
    else:
        y_eval_all = None
        g_eval_all = None
        
    for layer in layer_list:
        if verbose:
            print(f"Training WB Probe ({method}) for Layer {layer}...")
        try:
            X_layer = np.array([item['activations'][layer] for item in activation_data])
        except KeyError:
            continue
            
        if eval_data is None:
            indices = np.arange(len(X_layer))
            train_idx, test_idx, y_train, y_test = train_test_split(indices, y_all, test_size=test_size, random_state=42)
            X_train, X_test = X_layer[train_idx], X_layer[test_idx]
            g_train, g_test = g_all[train_idx], g_all[test_idx]
        else:
            try:
                X_test = np.array([item['activations'][layer] for item in eval_data])
            except KeyError:
                continue
            X_train = X_layer
            y_train = y_all
            g_train = g_all
            y_test = y_eval_all
            g_test = g_eval_all
            
        if method == "lr":
            probe = train_lr_probe(X_train, y_train)
        elif method == "lr-g":
            probe = train_lr_g_probe(X_train, y_train, g_train)
        elif method == "dim":
            try:
                probe = train_dim_probe(X_train, y_train)
            except ValueError as e:
                print(str(e))
                continue
        elif method == "pca":
            probe = train_pca_probe(X_train)
        elif method == "pca-g":
            probe = train_pca_g_probe(X_train, g_train)
        elif method == "lat":
            probe = train_lat_probe(X_train)
        elif method == "lda":
            probe = train_lda_probe(X_train, y_train)
        elif method == "ccs":
            try:
                probe = train_ccs_probe(X_train, y_train, g_train)
            except ValueError as e:
                print(str(e))
                continue
        else:
            raise ValueError(f"Unknown Method: {method}")
            
        result = probe.predict(X_test, groups=g_test)
        logits = result.logits
        
        if method in ["dim", "pca", "pca-g", "lat", "ccs"]:
            train_logits = probe.predict(X_train, groups=g_train).logits
            
            # 1. SCIENTIFIC FIX: Resolve the sign using TRAIN data only!
            if len(np.unique(y_train)) == 2:
                train_auc_normal = roc_auc_score(y_train, train_logits)
                train_auc_flipped = roc_auc_score(y_train, -train_logits)
            else:
                train_auc_normal, train_auc_flipped = 0.5, 0.5
                
            # If the flipped version is better on the training data, lock that decision in
            if train_auc_flipped > train_auc_normal:
                logits = -logits
                train_logits = -train_logits
            
            # 2. Evaluate the test set using the permanently locked sign
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, logits)
            else:
                auc = 0.5
                
            # 3. Calculate thresholds using the corrected train_logits
            if len(np.unique(y_train)) == 2:
                mean_pos = np.mean(train_logits[y_train == 1])
                mean_neg = np.mean(train_logits[y_train == 0])
                threshold = (mean_pos + mean_neg) / 2
            else:
                threshold = 0.0
        else:
            threshold = 0.0
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, logits)
            else:
                auc = 0.5
                
        preds = (np.array(logits) > threshold).astype(int)
        
        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")
        recall_true = recall_score(y_test, preds, pos_label=1, zero_division=0)
        recall_false = recall_score(y_test, preds, pos_label=0, zero_division=0)
        
        probes[layer] = probe
        metrics[layer] = {
            'accuracy': acc, 
            'auc': auc,
            'macro_f1': macro_f1,
            'recall_true': recall_true,
            'recall_false': recall_false,
            'logits': logits,
            'preds': preds
        }
        print(f"Layer {layer} ({method}) - Acc: {acc:.4f}, AUC: {auc:.4f}, F1: {macro_f1:.4f}, Rec(T/F): {recall_true:.2f}/{recall_false:.2f}")
        
    return probes, metrics
