from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

def train_bb_classifier(X, y):
    """
    Trains the Black Box classifier on the logprob features.
    
    Args:
        X: (n_samples, n_probes)
        y: (n_samples,)
        
    Returns:
        model: Trained LogisticRegression model
        metrics: dict with accuracy and auc
    """
    print("Training Black Box Classifier...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    print(f"Black Box Results - Acc: {acc:.4f}, AUC: {auc:.4f}")
    
    return clf, {'accuracy': acc, 'auc': auc}
