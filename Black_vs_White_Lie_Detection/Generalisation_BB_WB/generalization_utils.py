"""
Experiment 4 — Cross-Dataset Generalisation Utilities
======================================================
Helper functions for loading saved classifiers/probes from E3,
evaluating them on new target datasets, computing recovery metrics,
and plotting heatmaps.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve


def best_recall_at_precision(y_true, y_scores, target_precision=0.90):
    """Highest recall achievable while maintaining at least target_precision."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    valid = np.where(precisions >= target_precision)[0]
    return float(np.max(recalls[valid])) if len(valid) > 0 else 0.0


# ─────────────────────────────────────────────
#  Metric helpers
# ─────────────────────────────────────────────

def _compute_metrics(y_true, scores):
    """Compute AUC, MAP, BRP_90 from raw scores (logits or probabilities)."""
    if len(np.unique(y_true)) < 2:
        return {"AUC": 0.5, "MAP": 0.5, "BRP_90": 0.0}
    return {
        "AUC":    roc_auc_score(y_true, scores),
        "MAP":    average_precision_score(y_true, scores),
        "BRP_90": best_recall_at_precision(y_true, scores),
    }


# ─────────────────────────────────────────────
#  Evaluation functions
# ─────────────────────────────────────────────

def evaluate_bb_on_target(bundle, X_target, y_target):
    """
    Evaluate a saved L2 BB classifier on a target dataset.

    The source StandardScaler is applied to the target features before
    prediction — this is intentional for the transfer experiment (we
    deliberately use the source distribution's normalisation parameters).

    Args:
        bundle:   dict {"clf": LogisticRegression, "scaler": StandardScaler}
        X_target: np.ndarray (n_target, 48) — logprob features
        y_target: np.ndarray (n_target,)

    Returns:
        dict with keys AUC, MAP, BRP_90
    """
    X_scaled = bundle["scaler"].transform(X_target)
    probs = bundle["clf"].predict_proba(X_scaled)[:, 1]
    return _compute_metrics(y_target, probs)


def evaluate_wb_on_target(probe, method, X_source_train, y_source_train, X_target, y_target):
    """
    Evaluate a saved L3 WB probe on a target dataset.

    For unsupervised probes (dim, pca, pca-g, lat, ccs) the sign of the
    probe direction is ambiguous. The sign is resolved using SOURCE training
    data only — identical to the procedure in E3 — so no target-label
    information leaks into the decision.

    Args:
        probe:           Saved probe object (any subclass of BaseProbe)
        method:          Short method key, e.g. "lr", "dim", "ccs"
        X_source_train:  np.ndarray (n_train, hidden_dim) — source train activations
        y_source_train:  np.ndarray (n_train,) — source train labels
        X_target:        np.ndarray (n_target, hidden_dim) — target activations
        y_target:        np.ndarray (n_target,)

    Returns:
        dict with keys AUC, MAP, BRP_90
    """
    UNSUPERVISED = {"dim", "pca", "pca-g", "lat", "ccs"}

    logits = probe.predict(X_target).logits

    if method in UNSUPERVISED:
        # Resolve sign from source training data (no leakage).
        train_logits = probe.predict(X_source_train).logits
        if len(np.unique(y_source_train)) == 2:
            auc_normal  = roc_auc_score(y_source_train, train_logits)
            auc_flipped = roc_auc_score(y_source_train, -train_logits)
            if auc_flipped > auc_normal:
                logits = -logits

    return _compute_metrics(y_target, logits)


# ─────────────────────────────────────────────
#  Recovery metric
# ─────────────────────────────────────────────

def compute_recovered_metrics(in_dist, out_dist):
    """
    Compute recovered metric ratios: out_dist / in_dist.

    A value of 1.0 means the out-of-distribution performance equals the
    in-distribution performance. Values > 1.0 are capped at 1.0 in the
    heatmap plots but preserved in the CSV.

    Args:
        in_dist:  dict {"AUC": float, "MAP": float, "BRP_90": float}
        out_dist: dict {"AUC": float, "MAP": float, "BRP_90": float}

    Returns:
        dict {"Recovered_AUC": float, "Recovered_MAP": float, "Recovered_BRP_90": float}
    """
    result = {}
    for key in ["AUC", "MAP", "BRP_90"]:
        in_val  = in_dist.get(key, 0.0)
        out_val = out_dist.get(key, 0.0)
        result[f"Recovered_{key}"] = (out_val / in_val) if in_val > 1e-6 else 0.0
    return result


# ─────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────

def plot_recovery_heatmap(matrix_df, metric_label, title, save_path):
    """
    Plot a square heatmap of recovered metric values (0–100%).

    Args:
        matrix_df:    pd.DataFrame with datasets as both index and columns.
                      Values are recovery ratios in [0, 1]. Diagonal = 1.0.
        metric_label: String label for the colour bar (e.g. "Recovered AUC").
        title:        Plot title.
        save_path:    File path for the saved PNG.
    """
    # Cap at 1.0 for display only
    display_df = matrix_df.clip(upper=1.0)

    fig, ax = plt.subplots(figsize=(15, 12))
    sns.heatmap(
        display_df,
        annot=True,
        fmt=".0%",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.4,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": metric_label, "shrink": 0.8},
    )
    ax.set_title(title, fontsize=14, pad=15, fontweight="bold")
    ax.set_xlabel("Eval Dataset", fontsize=12)
    ax.set_ylabel("Train Dataset", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {save_path}")
