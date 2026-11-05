"""
SentryICU Clinical Metrics Suite
Computes discrimination, calibration, and threshold-dependent clinical risk metrics.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_fscore_support
)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) across n_bins equal-width confidence intervals.
    
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    
    Args:
        y_true: Ground truth binary labels (0 or 1), shape (N,) or (N, 1).
        y_prob: Predicted probabilities in [0.0, 1.0], shape (N,) or (N, 1).
        n_bins: Number of equal-width bins (default: 10).
        
    Returns:
        float: Expected Calibration Error value in [0.0, 1.0].
    """
    y_true = np.asarray(y_true).flatten().astype(float)
    y_prob = np.clip(np.asarray(y_prob).flatten().astype(float), 0.0, 1.0)
    
    if len(y_true) == 0:
        return 0.0
        
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    
    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        else:
            in_bin = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            
        bin_count = in_bin.sum()
        if bin_count > 0:
            avg_confidence = float(y_prob[in_bin].mean())
            empirical_accuracy = float(y_true[in_bin].mean())
            bin_weight = bin_count / n_samples
            ece += bin_weight * np.abs(empirical_accuracy - avg_confidence)
            
    return float(ece)


def brier_score_loss(y_true, y_prob):
    """
    Computes Brier score loss: mean squared error between probabilities and binary labels.
    
    Brier = (1 / N) * sum_{i=1}^N (p_i - y_i)^2
    
    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probabilities in [0.0, 1.0].
        
    Returns:
        float: Brier score in [0.0, 1.0].
    """
    y_true = np.asarray(y_true).flatten().astype(float)
    y_prob = np.asarray(y_prob).flatten().astype(float)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean((y_prob - y_true) ** 2))


def compute_metrics(y_true, y_prob, threshold=0.5, n_bins=10):
    """
    Computes comprehensive clinical classification metrics:
    AUROC, PR-AUC, F1, Sensitivity, Specificity, Precision, Accuracy, ECE, Brier Score.
    
    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probabilities in [0.0, 1.0].
        threshold: Decision threshold for discrete classification (default: 0.5).
        n_bins: Number of bins for ECE calculation (default: 10).
        
    Returns:
        dict: Complete dictionary containing all clinical metrics and confusion matrix.
    """
    y_true = np.asarray(y_true).flatten().astype(int)
    y_prob = np.clip(np.asarray(y_prob).flatten().astype(float), 0.0, 1.0)
    y_pred = (y_prob >= threshold).astype(int)

    # Discrimination Metrics
    unique_classes = np.unique(y_true)
    if len(unique_classes) > 1:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
    else:
        auroc = 0.5
        auprc = float(y_true.mean()) if len(y_true) > 0 else 0.0

    # Calibration Metrics
    brier = brier_score_loss(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)

    # Threshold-based Classification Metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / max(tp + fn, 1))
    specificity = float(tn / max(tn + fp, 1))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    accuracy = float((tp + tn) / max(tp + tn + fp + fn, 1))

    return {
        'auroc': auroc,
        'roc_auc': auroc,
        'auprc': auprc,
        'pr_auc': auprc,
        'prauc': auprc,
        'sensitivity': sensitivity,
        'recall': float(recall),
        'specificity': specificity,
        'precision': float(precision),
        'f1': float(f1),
        'accuracy': accuracy,
        'ece': ece,
        'brier_score': brier,
        'brier': brier,
        'confusion_matrix': {
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }
    }


# Backwards compatibility alias
compute_clinical_metrics = compute_metrics
