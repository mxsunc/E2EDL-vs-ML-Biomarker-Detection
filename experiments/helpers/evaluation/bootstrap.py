import numpy as np
from typing import Callable, Tuple, Optional, List, Dict
from sklearn.metrics import f1_score


def _percentile_ci(
    bootstrap_samples: np.ndarray,
    ci_level: float = 0.95
) -> Tuple[float, float, float]:

    alpha = 1 - ci_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower = float(np.percentile(bootstrap_samples, lower_percentile))
    median = float(np.percentile(bootstrap_samples, 50))
    upper = float(np.percentile(bootstrap_samples, upper_percentile))

    return lower, median, upper


def _stratified_sample_indices(
    y_true: np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:

    # Get indices for each class
    class_0_indices = np.where(y_true == 0)[0]
    class_1_indices = np.where(y_true == 1)[0]

    # Sample with replacement within each class (same size as original)
    sampled_0 = rng.choice(class_0_indices, size=len(class_0_indices), replace=True)
    sampled_1 = rng.choice(class_1_indices, size=len(class_1_indices), replace=True)

    # Combine and shuffle
    all_indices = np.concatenate([sampled_0, sampled_1])
    rng.shuffle(all_indices)

    return all_indices


def bootstrap_metric_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 2000,
    ci_level: float = 0.95,
    statistic_name: str = 'metric',
    stratified: bool = True,
    seed: Optional[int] = None
) -> Dict:

    rng = np.random.Generator(np.random.PCG64(seed))

    # Point estimate on full data
    point_estimate = metric_fn(y_true, y_pred)

    # Bootstrap sampling
    n_samples = len(y_true)
    bootstrap_values = np.empty(n_bootstraps)

    for b in range(n_bootstraps):
        if stratified:
            indices = _stratified_sample_indices(y_true, rng)
        else:
            indices = rng.integers(0, n_samples, size=n_samples)

        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        bootstrap_values[b] = metric_fn(y_true_boot, y_pred_boot)

    # Calculate CI
    ci_lower, median, ci_upper = _percentile_ci(bootstrap_values, ci_level)

    return {
        statistic_name: float(point_estimate),
        f'{statistic_name}_ci_lower': ci_lower,
        f'{statistic_name}_ci_upper': ci_upper,
        f'{statistic_name}_median': median,
        f'{statistic_name}_bootstrap_samples': bootstrap_values
    }


def bootstrap_diff_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
    n_bootstraps: int = 2000,
    ci_level: float = 0.95,
    statistic_name: str = 'diff',
    stratified: bool = True,
    seed: Optional[int] = None
) -> Dict:

    rng = np.random.Generator(np.random.PCG64(seed))

    # Point estimate on full data
    metric_1 = metric_fn(y_true, y_pred_1)
    metric_2 = metric_fn(y_true, y_pred_2)
    diff_estimate = metric_1 - metric_2

    # Bootstrap sampling
    n_samples = len(y_true)
    bootstrap_diffs = np.empty(n_bootstraps)

    for b in range(n_bootstraps):
        if stratified:
            indices = _stratified_sample_indices(y_true, rng)
        else:
            indices = rng.integers(0, n_samples, size=n_samples)

        y_true_boot = y_true[indices]
        y_pred_1_boot = y_pred_1[indices]
        y_pred_2_boot = y_pred_2[indices]

        metric_1_boot = metric_fn(y_true_boot, y_pred_1_boot)
        metric_2_boot = metric_fn(y_true_boot, y_pred_2_boot)
        bootstrap_diffs[b] = metric_1_boot - metric_2_boot

    # Calculate CI
    ci_lower, median, ci_upper = _percentile_ci(bootstrap_diffs, ci_level)

    return {
        f'{statistic_name}_diff': float(diff_estimate),
        f'{statistic_name}_diff_ci_lower': ci_lower,
        f'{statistic_name}_diff_ci_upper': ci_upper,
        f'{statistic_name}_diff_median': median,
        f'{statistic_name}_diff_bootstrap_samples': bootstrap_diffs
    }


def bootstrap_all_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold_fn: Callable[[np.ndarray, np.ndarray], Tuple[float, Dict]],
    threshold_name: str = 'threshold',
    n_bootstraps: int = 2000,
    ci_level: float = 0.95,
    seed: Optional[int] = None,
    fixed_threshold: Optional[float] = None
) -> Dict:

    from .metrics import get_threshold_info

    rng = np.random.Generator(np.random.PCG64(seed))

    # Get threshold on full data
    if fixed_threshold is not None:
        threshold = fixed_threshold
    else:
        threshold, _ = threshold_fn(y_true, y_pred_proba)

    # Point estimates on full data
    point_metrics = get_threshold_info(threshold, y_true, y_pred_proba, threshold_name)

    # Storage for bootstrap samples
    bootstrap_metrics = {
        'accuracy': np.empty(n_bootstraps),
        'f1': np.empty(n_bootstraps),
        'sensitivity': np.empty(n_bootstraps),
        'specificity': np.empty(n_bootstraps),
        'balanced_accuracy': np.empty(n_bootstraps),
        'TP': np.empty(n_bootstraps, dtype=int),
        'FP': np.empty(n_bootstraps, dtype=int),
        'TN': np.empty(n_bootstraps, dtype=int),
        'FN': np.empty(n_bootstraps, dtype=int)
    }

    for b in range(n_bootstraps):
        # Stratified bootstrap sample
        indices = _stratified_sample_indices(y_true, rng)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred_proba[indices]

        # Use fixed threshold if provided, otherwise compute on bootstrap sample
        if fixed_threshold is not None:
            thresh_boot = fixed_threshold
        else:
            thresh_boot, _ = threshold_fn(y_true_boot, y_pred_boot)

        # Compute metrics
        boot_info = get_threshold_info(thresh_boot, y_true_boot, y_pred_boot, threshold_name)

        for key in ['accuracy', 'f1', 'sensitivity', 'specificity', 'balanced_accuracy']:
            bootstrap_metrics[key][b] = boot_info[key]
        for key in ['TP', 'FP', 'TN', 'FN']:
            bootstrap_metrics[key][b] = boot_info[key]

    # Build result dictionary
    result = {
        f'{threshold_name}_threshold': threshold,
        'threshold_method': threshold_name
    }

    for key, values in bootstrap_metrics.items():
        ci_lower, median, ci_upper = _percentile_ci(values, ci_level)
        result[key] = point_metrics[key]
        result[f'{key}_ci_lower'] = ci_lower
        result[f'{key}_ci_upper'] = ci_upper
        result[f'{key}_median'] = median

    return result
