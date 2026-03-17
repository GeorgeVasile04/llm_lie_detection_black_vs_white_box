from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

def train_bb_classifier(X_train, X_test, y_train, y_test):
    """
    Trains the Black Box classifier on scaled logprob features.
    Follows original Black Box paper methodology with feature scaling.
    
    Args:
        X_train: (n_train, n_probes) - Already split at question level in notebook
        X_test: (n_test, n_probes) - Already split at question level in notebook
        y_train: (n_train,)
        y_test: (n_test,)
        
    Returns:
        clf: Trained LogisticRegression model
        metrics: dict with accuracy and auc
    """
    print("Training Black Box Classifier...")
    
    # Feature scaling (CRITICAL for logprob features)
    # Fit scaler ONLY on training data to avoid test data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression (matches original paper config)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    preds = clf.predict(X_test_scaled)
    probs = clf.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    print(f"Black Box Results - Acc: {acc:.4f}, AUC: {auc:.4f}")
    
    return clf, {'accuracy': acc, 'auc': auc}
