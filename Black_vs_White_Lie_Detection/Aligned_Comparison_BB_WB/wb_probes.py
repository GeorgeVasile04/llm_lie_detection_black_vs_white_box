from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

def train_wb_probes(activation_data, layer_list=[10, 15, 20]):
    """
    Trains a Logistic Regression probe for each specified layer.
    
    Args:
        activation_data: List of dicts with 'activations' and 'label'.
        layer_list: List of layers to train probes on.
        
    Returns:
        probes: dictionary {layer_num: trained_model}
        metrics: dictionary {layer_num: {'acc': float, 'auc': float}}
    """
    
    probes = {}
    metrics = {}
    
    # Organize data by layer
    # X_by_layer = {layer: []}
    # y = []
    
    # Basic data extraction
    y = [item['label'] for item in activation_data]
    
    for layer in layer_list:
        print(f"Training WB Probe for Layer {layer}...")
        X_layer = np.array([item['activations'][layer] for item in activation_data])
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X_layer, y, test_size=0.2, random_state=42)
        
        # Train
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        
        # Eval
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        
        probes[layer] = clf
        metrics[layer] = {'accuracy': acc, 'auc': auc}
        
        print(f"Layer {layer} Results - Acc: {acc:.4f}, AUC: {auc:.4f}")
        
    return probes, metrics
