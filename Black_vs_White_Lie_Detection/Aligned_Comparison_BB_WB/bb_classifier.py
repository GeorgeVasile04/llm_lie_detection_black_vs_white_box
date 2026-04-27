from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import numpy as np

def best_recall_at_precision(y_true, y_scores, target_precision=0.90):
    """Calculates best recall maintaining at least 'target_precision'"""
    if len(np.unique(y_true)) < 2:
        return 0.0
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    valid_idx = np.where(precisions >= target_precision)[0]
    if len(valid_idx) == 0:
        return 0.0  
    return np.max(recalls[valid_idx])

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
    probs = clf.predict_proba(X_test_scaled)[:, 1]
    
    auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5
    map_score = average_precision_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5
    brp_90 = best_recall_at_precision(y_test, probs, target_precision=0.90)

    print(f"Black Box Results - AUC: {auc:.4f}, MAP: {map_score:.4f}, BRP_90: {brp_90:.4f}")

    return clf, {'AUC': auc, 'MAP': map_score, 'BRP_90': brp_90}
