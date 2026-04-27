import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from dataclasses import dataclass
from typing import Literal, Dict, Any, List, Optional
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore")

def best_recall_at_precision(y_true, y_scores, target_precision=0.90):
    """Calculates best recall maintaining at least 'target_precision'"""
    if len(np.unique(y_true)) < 2:
        return 0.0
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    # Find all indices where precision >= target_precision
    valid_idx = np.where(precisions >= target_precision)[0]
    if len(valid_idx) == 0:
        return 0.0  # No threshold gives required precision
    return np.max(recalls[valid_idx])

ProbeMethod = Literal["lr", "dim", "lat", "ccs", "lda", "pca", "pca-g", "lr-g"]

@dataclass
class PredictResult:
    logits: np.ndarray

class BaseProbe:
    def predict(self, activations: np.ndarray, groups: Optional[np.ndarray] = None) -> PredictResult:
        raise NotImplementedError

class DotProductProbe(BaseProbe):
    def __init__(self, probe_vector: np.ndarray):
        self.probe = probe_vector

    def predict(self, activations: np.ndarray, groups: Optional[np.ndarray] = None) -> PredictResult:
        logits = activations @ self.probe
        return PredictResult(logits=logits)

class CentredDotProductProbe(DotProductProbe):
    def __init__(self, probe_vector: np.ndarray, center: np.ndarray):
        super().__init__(probe_vector)
        self.center = center
        
    def predict(self, activations: np.ndarray, groups: Optional[np.ndarray] = None) -> PredictResult:
        return super().predict(activations - self.center, groups)

class LogisticRegressionProbe(BaseProbe):
    def __init__(self, model: LogisticRegression):
        self.model = model

    def predict(self, activations: np.ndarray, groups: Optional[np.ndarray] = None) -> PredictResult:
        # decision_function returns dist from hyperplane (signed)
        logits = self.model.decision_function(activations)
        return PredictResult(logits=logits)

class LogisticRegressionGroupedProbe(LogisticRegressionProbe):
    def predict(self, activations: np.ndarray, groups: Optional[np.ndarray] = None) -> PredictResult:
        if groups is not None:
            activations_centered = _center_by_group(activations, groups)
            logits = self.model.decision_function(activations_centered)
        else:
            logits = self.model.decision_function(activations)
        return PredictResult(logits=logits)

class LdaProbe(BaseProbe):
    def __init__(self, model: LinearDiscriminantAnalysis):
        self.model = model

    def predict(self, activations: np.ndarray, groups: Optional[np.ndarray] = None) -> PredictResult:
        logits = self.model.decision_function(activations)
        return PredictResult(logits=logits)

class CcsNetwork(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        result = self.linear(activations)
        result = torch.sigmoid(result)
        return result.squeeze(-1)

class CcsProbe(BaseProbe):
    def __init__(self, model: CcsNetwork):
        self.model = model
    
    def predict(self, activations: np.ndarray, groups: Optional[np.ndarray] = None) -> PredictResult:
        with torch.no_grad():
            self.model.eval()
            probs = self.model(torch.tensor(activations, dtype=torch.float32)).cpu().numpy()
        return PredictResult(logits=probs)

class DifferenceInMeansProbe(DotProductProbe):
    pass

def _center_by_group(activations: np.ndarray, groups: np.ndarray) -> np.ndarray:
    result = activations.copy()
    for g in np.unique(groups):
        mask = (groups == g)
        result[mask] = result[mask] - result[mask].mean(axis=0)
    return result

def train_dim_probe(activations: np.ndarray, labels: np.ndarray) -> DifferenceInMeansProbe:
    labels = np.array(labels, dtype=bool)
    if len(np.unique(labels)) < 2:
        raise ValueError("Results must contain both True and False samples for DiM training.")
        
    pos_mean = activations[labels].mean(axis=0)
    neg_mean = activations[~labels].mean(axis=0)
    
    probe_vector = pos_mean - neg_mean
    return DifferenceInMeansProbe(probe_vector)

def train_lr_probe(activations: np.ndarray, labels: np.ndarray, C: float = 1.0, max_iter: int = 10000) -> LogisticRegressionProbe:
    activations = np.asarray(activations, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    
    model = LogisticRegression(C=C, solver='newton-cg', max_iter=max_iter, fit_intercept=True)
    model.fit(activations, labels)
    return LogisticRegressionProbe(model)

def train_lr_g_probe(activations: np.ndarray, labels: np.ndarray, groups: np.ndarray, C: float = 1.0, max_iter: int = 10000) -> LogisticRegressionGroupedProbe:
    activations_centered = _center_by_group(activations, groups)
    probe = train_lr_probe(activations_centered, labels, C=C, max_iter=max_iter)
    return LogisticRegressionGroupedProbe(probe.model)

def train_lda_probe(activations: np.ndarray, labels: np.ndarray) -> LdaProbe:
    activations = np.asarray(activations, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    lda = LinearDiscriminantAnalysis()
    lda.fit(activations, labels)
    return LdaProbe(lda)

def train_pca_probe(activations: np.ndarray) -> DotProductProbe:
    activations_centered = activations - activations.mean(axis=0)
    pca = PCA(n_components=1)
    pca.fit(activations_centered)
    probe_vector = pca.components_.squeeze(0)
    probe_vector = probe_vector / np.linalg.norm(probe_vector)
    return DotProductProbe(probe_vector=probe_vector)

def train_pca_g_probe(activations: np.ndarray, groups: np.ndarray) -> DotProductProbe:
    activations_centered = _center_by_group(activations, groups)
    probe = train_pca_probe(activations_centered)
    return DotProductProbe(probe_vector=probe.probe)

def train_lat_probe(activations: np.ndarray) -> CentredDotProductProbe:
    indices = list(range(len(activations)))
    random.shuffle(indices)
    indices = np.array(indices)[: len(indices) // 2 * 2]
    indices_1, indices_2 = indices.reshape(2, -1)

    activation_diffs = activations[indices_1] - activations[indices_2]
    activations_center = np.mean(activation_diffs, axis=0)
    activation_diffs_norm = activation_diffs - activations_center
    pca = PCA(n_components=1)
    pca.fit(activation_diffs_norm)
    probe_vector = pca.components_.squeeze(0)
    probe_vector = probe_vector / np.linalg.norm(probe_vector)
    return CentredDotProductProbe(probe_vector=probe_vector, center=activations_center)

def train_ccs_probe(activations: np.ndarray, labels: np.ndarray, groups: np.ndarray, attempts: int = 2) -> CcsProbe:
    activations_1 = []
    activations_2 = []
    
    unique_groups = np.unique(groups)
    
    for group in unique_groups:
        mask = (groups == group)
        group_activations = activations[mask]
        group_labels = labels[mask]
        
        if True not in group_labels or False not in group_labels:
            continue
            
        idx_true = group_labels.tolist().index(True)
        idx_false = group_labels.tolist().index(False)
        
        activations_1.append(group_activations[idx_true])
        activations_2.append(group_activations[idx_false])
        
    if len(activations_1) == 0:
        raise ValueError("No true/false pairs found for CCS training.")
        
    act_1 = torch.tensor(np.array(activations_1), dtype=torch.float32)
    act_2 = torch.tensor(np.array(activations_2), dtype=torch.float32)
    
    _, hidden_dim = act_1.shape
    network = CcsNetwork(hidden_dim=hidden_dim)
    optimizer = torch.optim.LBFGS(network.parameters(), lr=0.1, max_iter=100)
    
    def closure():
        optimizer.zero_grad()
        probs_1 = network(act_1)
        probs_2 = network(act_2)
        loss_consistency = (probs_1 - (1 - probs_2)).pow(2).mean()
        loss_confidence = torch.min(probs_1, probs_2).pow(2).mean()
        loss_l2 = sum(param.norm() ** 2 for param in network.parameters()) * 0.01
        loss = loss_consistency + loss_confidence + loss_l2
        loss.backward()
        return loss
        
    optimizer.step(closure=closure)
    
    loss_val = closure().item()
    if loss_val > 0.2:
        if attempts > 0:
            return train_ccs_probe(activations, labels, groups, attempts=attempts - 1)
        print(f"Warning: CCS probe did not converge cleanly (loss {loss_val:.4f}). Returning best effort.")
        
    network.eval()
    return CcsProbe(model=network)

def train_wb_probes(
    activation_data: List[Dict[str, Any]],
    eval_data: Optional[List[Dict[str, Any]]] = None,
    layer_list: List[int] = [10, 15, 20],
    method: ProbeMethod = "lr",
    test_size: float = 0.2,
):
    """
    Train and evaluate WB probes.
    Args:
        activation_data: List of dicts with 'activations' (dict), 'label' (bool), and 'id' (int)
        eval_data: Optional separate eval dict list. If None, split activation_data.
    """
    probes = {}
    metrics = {}
    
    # Extract labels and group chunks
    y_all = np.array([item['label'] for item in activation_data])
    g_all = np.array([item.get('id', i) for i, item in enumerate(activation_data)])
    
    if eval_data is not None:
        y_eval_all = np.array([item['label'] for item in eval_data])
        g_eval_all = np.array([item.get('id', i) for i, item in enumerate(eval_data)])
    else:
        y_eval_all = None
        g_eval_all = None
        
    for layer in layer_list:
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
            
        # Select method
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
            
        # Evaluation Logic
        result = probe.predict(X_test, groups=g_test)
        logits = result.logits
        
        # Unsupervised Directional Flipper Correction — sign decided on TRAIN data only (no leakage)
        if method in ["dim", "pca", "pca-g", "lat", "ccs"]:
            train_logits = probe.predict(X_train, groups=g_train).logits
            if len(np.unique(y_train)) == 2:
                train_auc_normal = roc_auc_score(y_train, train_logits)
                train_auc_flipped = roc_auc_score(y_train, -train_logits)
            else:
                train_auc_normal, train_auc_flipped = 0.5, 0.5

            if train_auc_flipped > train_auc_normal:
                logits = -logits
                train_logits = -train_logits

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
            map_score = average_precision_score(y_test, logits)
            brp_90 = best_recall_at_precision(y_test, logits, target_precision=0.90)
        else:
            auc, map_score, brp_90 = 0.5, 0.5, 0.0

        probes[layer] = probe
        metrics[layer] = {'AUC': auc, 'MAP': map_score, 'BRP_90': brp_90}
        print(f"Layer {layer} Results ({method}) - AUC: {auc:.4f}, MAP: {map_score:.4f}, BRP_90: {brp_90:.4f}")
        
    return probes, metrics

