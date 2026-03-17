import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from dataclasses import dataclass
from typing import Literal, Dict, Any, List, Optional
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore")

ProbeMethod = Literal["lr", "dim"]

@dataclass
class PredictResult:
    logits: np.ndarray

class BaseProbe:
    def predict(self, activations: np.ndarray) -> PredictResult:
        raise NotImplementedError

class DotProductProbe(BaseProbe):
    def __init__(self, probe_vector: np.ndarray):
        self.probe = probe_vector

    def predict(self, activations: np.ndarray) -> PredictResult:
        logits = activations @ self.probe
        return PredictResult(logits=logits)

class LogisticRegressionProbe(BaseProbe):
    def __init__(self, model: LogisticRegression):
        self.model = model

    def predict(self, activations: np.ndarray) -> PredictResult:
        # decision_function returns dist from hyperplane (signed)
        logits = self.model.decision_function(activations)
        return PredictResult(logits=logits)

class DifferenceInMeansProbe(DotProductProbe):
    pass

def train_dim_probe(activations: np.ndarray, labels: np.ndarray) -> DifferenceInMeansProbe:
    """
    Trains a Difference-in-Means probe.
    Methodology:
    1. Compute mean of activations for positive class.
    2. Compute mean of activations for negative class.
    3. The probe direction is mean_pos - mean_neg.
    """
    labels = np.array(labels, dtype=bool)
    
    # Check if we have both classes
    if len(np.unique(labels)) < 2:
        # If only one class is present, we cannot compute a difference. 
        # For robustness, handle this gracefully or raise error
        raise ValueError("Results must contain both True and False samples for DiM training.")
        
    pos_mean = activations[labels].mean(axis=0)
    neg_mean = activations[~labels].mean(axis=0)
    
    probe_vector = pos_mean - neg_mean
    return DifferenceInMeansProbe(probe_vector)

def train_lr_probe(activations: np.ndarray, labels: np.ndarray, C: float = 1.0, max_iter: int = 10000) -> LogisticRegressionProbe:
    """
    Trains a Logistic Regression probe using sklearn.
    Default config from repeng: C=1.0, solver='newton-cg', max_iter=10000.
    """
    # Using 'newton-cg' as specified in original RepEng paper (faster for small datasets)
    model = LogisticRegression(
        C=C,
        solver='newton-cg',
        max_iter=max_iter,
        fit_intercept=True
    )
    model.fit(activations, labels)
    return LogisticRegressionProbe(model)

def train_wb_probes(activation_data: List[Dict[str, Any]], layer_list: List[int] = [10, 15, 20], method: ProbeMethod = "lr", test_size: float = 0.2):
    """
    Trains probes for specified layers using the chosen method.
    
    Args:
        activation_data: List of dicts with 'activations' key (dict of layer->vec) and 'label' key.
        layer_list: List of layers to train on.
        method: "lr" (Logistic Regression) or "dim" (Difference in Means).
        test_size: Proportion of data to use for validation.
        
    Returns:
        probes: {layer: probe_object}
        metrics: {layer: {'accuracy': float, 'auc': float}}
    """
    probes = {}
    metrics = {}
    
    # Extract labels once
    y_all = np.array([item['label'] for item in activation_data])
    
    for layer in layer_list:
        print(f"Training WB Probe ({method}) for Layer {layer}...")
        
        # Extract activations for this layer
        # Ensure consistent ordering
        try:
            X_layer = np.array([item['activations'][layer] for item in activation_data])
        except KeyError:
            print(f"Layer {layer} not found in activations. Skipping.")
            continue
            
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(X_layer, y_all, test_size=test_size, random_state=42)
        
        if method == "lr":
            probe = train_lr_probe(X_train, y_train)
        elif method == "dim":
            try:
                probe = train_dim_probe(X_train, y_train)
            except ValueError as e:
                print(f"Skipping Layer {layer}: {e}")
                continue
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # Evaluation
        result = probe.predict(X_test)
        logits = result.logits
        
        # Metric Calculation with Probe Direction Flip Detection (RepEng paper methodology)
        if method == "dim":
            # For DiM, test both probe directions (+ and -) and pick the better one
            # This accounts for label encoding ambiguity (truth=1 vs truth=0)
            train_logits = probe.predict(X_train).logits
            
            # Compute AUC for both directions
            if len(np.unique(y_test)) == 2:
                auc_normal = roc_auc_score(y_test, logits)
                auc_flipped = roc_auc_score(y_test, -logits)
            else:
                auc_normal = 0.5
                auc_flipped = 0.5
            
            # Use the direction with better AUC
            if auc_flipped > auc_normal:
                logits = -logits
                auc = auc_flipped
                is_flipped = True
            else:
                auc = auc_normal
                is_flipped = False
            
            # Compute threshold on training data
            mean_pos = np.mean(train_logits[y_train])
            mean_neg = np.mean(train_logits[~y_train])
            threshold = (mean_pos + mean_neg) / 2
        else:
            # LR decision function is 0-centered
            threshold = 0.0
            is_flipped = False
            if len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, logits)
            else:
                auc = 0.5
            
        preds = (logits > threshold).astype(int)
        acc = accuracy_score(y_test, preds)
            
        probes[layer] = probe
        metrics[layer] = {'accuracy': acc, 'auc': auc}
        
        print(f"Layer {layer} Results ({method}) - Acc: {acc:.4f}, AUC: {auc:.4f}")
        
    return probes, metrics
