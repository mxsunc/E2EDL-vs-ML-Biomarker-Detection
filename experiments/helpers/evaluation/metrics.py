import numpy as np
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    auc
)
from typing import Tuple, Dict, Optional, List
import warnings


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_pred))


def calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    fraction_of_positives = np.zeros(n_bins)
    mean_predicted_value = np.zeros(n_bins)

    for i in range(n_bins):
        if i == n_bins - 1:
            # Last bin includes right edge
            mask = (y_pred >= bin_boundaries[i]) & (y_pred <= bin_boundaries[i + 1])
        else:
            mask = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i + 1])

        if np.sum(mask) > 0:
            fraction_of_positives[i] = np.mean(y_true[mask])
            mean_predicted_value[i] = np.mean(y_pred[mask])
        else:
            fraction_of_positives[i] = np.nan
            mean_predicted_value[i] = bin_centers[i]

    return fraction_of_positives, mean_predicted_value, bin_centers


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> float:

    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_pred >= bin_boundaries[i]) & (y_pred <= bin_boundaries[i + 1])
        else:
            mask = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i + 1])

        n_in_bin = np.sum(mask)
        if n_in_bin > 0:
            avg_confidence = np.mean(y_pred[mask])
            avg_accuracy = np.mean(y_true[mask])
            ece += (n_in_bin / n_samples) * np.abs(avg_accuracy - avg_confidence)

    return float(ece)


def youden_j_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, Dict]:

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)

    info = {
        'youden_j': float(youden_j[best_idx]),
        'sensitivity': float(tpr[best_idx]),
        'specificity': float(1 - fpr[best_idx]),
        'threshold_method': 'youden_j'
    }

    return float(thresholds[best_idx]), info


def sensitivity_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_sensitivity: float = 0.95
) -> Tuple[float, Dict]:

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Find thresholds where sensitivity >= target
    valid_mask = tpr >= target_sensitivity

    if np.any(valid_mask):
        # Among valid thresholds, pick the one with highest specificity (lowest FPR)
        valid_indices = np.where(valid_mask)[0]
        best_idx = valid_indices[np.argmin(fpr[valid_mask])]
        target_achieved = True
    else:
        # Target not achievable - use closest (maximum sensitivity)
        best_idx = np.argmax(tpr)
        target_achieved = False

    specificity = 1 - fpr[best_idx]

    info = {
        'target_sensitivity': target_sensitivity,
        'achieved_sensitivity': float(tpr[best_idx]),
        'specificity': float(specificity),
        'threshold_method': f'sensitivity_{int(target_sensitivity * 100)}',
        'target_achieved': target_achieved
    }

    return float(thresholds[best_idx]), info


def specificity_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_specificity: float = 0.95
) -> Tuple[float, Dict]:

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Find thresholds where specificity >= target (FPR <= 1 - target)
    target_fpr = 1 - target_specificity
    valid_mask = fpr <= target_fpr

    if np.any(valid_mask):
        # Among valid thresholds, pick the one with highest sensitivity (highest TPR)
        valid_indices = np.where(valid_mask)[0]
        best_idx = valid_indices[np.argmax(tpr[valid_mask])]
        target_achieved = True
    else:
        # Target not achievable - use closest (maximum specificity, minimum FPR)
        best_idx = np.argmin(fpr)
        target_achieved = False

    info = {
        'target_specificity': target_specificity,
        'achieved_specificity': float(1 - fpr[best_idx]),
        'sensitivity': float(tpr[best_idx]),
        'threshold_method': f'specificity_{int(target_specificity * 100)}',
        'target_achieved': target_achieved
    }

    return float(thresholds[best_idx]), info


def get_threshold_info(
    threshold: float,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold_name: str = 'custom'
) -> Dict:

    y_pred_binary = (y_pred_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    balanced_acc = balanced_accuracy_score(y_true, y_pred_binary)

    return {
        'threshold': float(threshold),
        'threshold_name': threshold_name,
        'TP': int(tp),
        'FP': int(fp),
        'TN': int(tn),
        'FN': int(fn),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'f1': float(f1),
        'balanced_accuracy': float(balanced_acc)
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None,
    n_calibration_bins: int = 10
) -> Dict:

    results = {
        'calibration': {},
        'thresholds': {},
        'metrics_by_threshold': {}
    }

    # Calibration metrics
    results['calibration']['brier_score'] = brier_score(y_true, y_pred_proba)
    results['calibration']['ece'] = expected_calibration_error(y_true, y_pred_proba, n_calibration_bins)

    # Compute calibration curve data
    frac_pos, mean_pred, bin_centers = calibration_curve(y_true, y_pred_proba, n_calibration_bins)
    results['calibration']['curve_fraction_positives'] = frac_pos.tolist()
    results['calibration']['curve_mean_predicted'] = mean_pred.tolist()
    results['calibration']['curve_bin_centers'] = bin_centers.tolist()

    # Compute thresholds if not provided
    if thresholds is None:
        thresholds = {}

    if 'youden' not in thresholds:
        thresholds['youden'], _ = youden_j_threshold(y_true, y_pred_proba)

    if 'sens_95' not in thresholds:
        thresholds['sens_95'], sens_info = sensitivity_threshold(y_true, y_pred_proba, 0.95)
        results['thresholds']['sensitivity_95'] = {
            'value': thresholds['sens_95'],
            'target_achieved': sens_info['target_achieved'],
            'achieved_sensitivity': sens_info['achieved_sensitivity']
        }

    if 'spec_95' not in thresholds:
        thresholds['spec_95'], spec_info = specificity_threshold(y_true, y_pred_proba, 0.95)
        results['thresholds']['specificity_95'] = {
            'value': thresholds['spec_95'],
            'target_achieved': spec_info['target_achieved'],
            'achieved_specificity': spec_info['achieved_specificity']
        }

    # Compute metrics for each threshold
    threshold_configs = [
        ('youden', thresholds.get('youden'), 'Youden J'),
        ('sens_95', thresholds.get('sens_95'), 'Sensitivity 95%'),
        ('spec_95', thresholds.get('spec_95'), 'Specificity 95%')
    ]

    for key, thresh, name in threshold_configs:
        if thresh is not None:
            info = get_threshold_info(thresh, y_true, y_pred_proba, name)
            results['metrics_by_threshold'][key] = info

    return results
