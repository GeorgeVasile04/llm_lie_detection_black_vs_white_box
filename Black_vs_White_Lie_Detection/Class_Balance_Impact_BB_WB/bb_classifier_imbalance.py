from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score
import numpy as np

def train_bb_classifier_imbalance(X_train, X_test, y_train, y_test):
    """
    Trains the Black Box classifier on scaled logprob features.
    MODIFIED for Class Imbalance Study: Also returns Macro F1 and Class-wise Recall.
    
    Args:
        X_train: (n_train, n_probes)
        X_test: (n_test, n_probes)
        y_train: (n_train,)
        y_test: (n_test,)
        
    Returns:
        clf: Trained LogisticRegression model
        metrics: dict with accuracy, auc, macro_f1, recall_true, recall_false
    """
    print("Training Black Box Classifier (Imbalance setup)...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    preds = clf.predict(X_test_scaled)
    probs = clf.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.5
    macro_f1 = f1_score(y_test, preds, average="macro")
    
    # Calculate Recall for True (1) and False (0)
    recall_true = recall_score(y_test, preds, pos_label=1, zero_division=0)
    recall_false = recall_score(y_test, preds, pos_label=0, zero_division=0)
    
    print(f"Black Box Results - Acc: {acc:.4f}, AUC: {auc:.4f}, Macro-F1: {macro_f1:.4f}, Recall(T/L): {recall_true:.2f}/{recall_false:.2f}")
    
    return clf, {
        'accuracy': acc, 
        'auc': auc,
        'macro_f1': macro_f1,
        'recall_true': recall_true,
        'recall_false': recall_false
    }
