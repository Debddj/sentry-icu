import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, precision_recall_fscore_support
)

def compute_metrics(y_true, y_prob, threshold=0.5):
    """
    Computes clinical classification metrics: AUROC, AUPRC, Sensitivity, Specificity, F1, Brier Score.
    """
    y_true = np.array(y_true).flatten()
    y_prob = np.array(y_prob).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    auprc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    brier = brier_score_loss(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

    return {
        'auroc': float(auroc),
        'auprc': float(auprc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'brier_score': float(brier),
        'confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)}
    }
