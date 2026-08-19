
"""
Clinical Evaluation Pipeline for Binary Classification Models

This script takes model predictions as input and computes comprehensive
clinical evaluation metrics including:
- Clinical thresholds: Youden's J, 95% sensitivity, 95% specificity
- Calibration metrics: Brier score, Expected Calibration Error (ECE)
- Performance metrics: accuracy, F1, sensitivity, specificity, confusion matrix
- Bootstrap confidence intervals (stratified, 2000 samples)

Input Format (CSV):
    - patient_id: unique patient identifier
    - cohort: cohort name (e.g., TCGA_test, CPTAC_full)
    - fold: cross-validation fold (0-4)
    - label: true binary label (0 or 1)
    - prediction_score: predicted probability
    - model_name: name of the model (e.g., DL_MSI, XGB_MSI)

Output Files:
    - evaluation_results.csv: Aggregated metrics per model/cohort/threshold
    - per_patient_predictions.csv: Patient-level predictions with classifications
    - calibration_data.csv: Data for plotting reliability diagrams

Usage:
    python run_evaluation_pipeline.py --input predictions_all_folds.csv
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.metrics import (
    brier_score,
    expected_calibration_error,
    calibration_curve,
    youden_j_threshold,
    sensitivity_threshold,
    specificity_threshold,
    get_threshold_info,
    compute_all_metrics,
    roc_auc_score, 
    auc, 
    precision_recall_curve, 
    roc_curve
)
from evaluation.bootstrap import (
    bootstrap_metric_ci,
    bootstrap_all_metrics
)
from evaluation.io import (
    load_predictions_csv,
    save_predictions_csv,
    create_patient_level_output
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Clinical Evaluation Pipeline for Binary Classification Models'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input predictions CSV file'
    )
    parser.add_argument(
        '--input-dir', '-d',
        type=str,
        help='Directory containing prediction CSV files'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./evaluation_output',
        help='Output directory for results (default: ./evaluation_output)'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        help='Specific model names to evaluate (default: all found)'
    )
    parser.add_argument(
        '--cohorts',
        type=str,
        nargs='+',
        help='Specific cohorts to evaluate (default: all found)'
    )
    parser.add_argument(
        '--n-bootstraps',
        type=int,
        default=2000,
        help='Number of bootstrap samples (default: 2000)'
    )
    parser.add_argument(
        '--ci-level',
        type=float,
        default=0.95,
        help='Confidence interval level (default: 0.95)'
    )
    parser.add_argument(
        '--target-sensitivity',
        type=float,
        default=0.95,
        help='Target sensitivity for clinical threshold (default: 0.95)'
    )
    parser.add_argument(
        '--target-specificity',
        type=float,
        default=0.95,
        help='Target specificity for clinical threshold (default: 0.95)'
    )
    parser.add_argument(
        '--reference-cohort',
        type=str,
        default='TCGA_test',
        help='Cohort used to establish thresholds (default: TCGA_test). '
             'Thresholds (Youden, Sens95%, Spec95%) are computed on this '
             'cohort and applied to all others for external validation.'
    )
    parser.add_argument(
        '--skip-bootstrap',
        action='store_true',
        help='Skip bootstrap CI calculation (faster)'
    )
    parser.add_argument(
        '--generate-plots',
        action='store_true',
        default=True,
        help='Generate ROC and calibration plots'
    )

    return parser.parse_args()


def evaluate_cohort(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    cohort_name: str,
    target_sensitivity: float = 0.95,
    target_specificity: float = 0.95,
    n_bootstraps: int = 2000,
    ci_level: float = 0.95,
    skip_bootstrap: bool = False,
    seed: int = 42,
    fixed_thresholds: dict = None
) -> dict:

    print(f"  Evaluating {model_name} on {cohort_name}...")

    if np.any(y_pred < 0) or np.any(y_pred > 1):
        print(f"    Applying sigmoid transformation (detected logits)")
        y_pred = 1 / (1 + np.exp(-y_pred))

    results = {
        'model_name': model_name,
        'cohort': cohort_name,
        'n_samples': len(y_true),
        'n_positive': int(np.sum(y_true)),
        'n_negative': int(np.sum(y_true == 0))
    }

    if fixed_thresholds is not None:
        youden_thresh = fixed_thresholds['youden']
        sens_thresh = fixed_thresholds['sens_95']
        spec_thresh = fixed_thresholds['spec_95']
        youden_info = {'target_achieved': True}
        sens_info = {'target_achieved': fixed_thresholds.get('sens_target_achieved', True)}
        spec_info = {'target_achieved': fixed_thresholds.get('spec_target_achieved', True)}
        print(f"    Using fixed thresholds from reference cohort")
    else:
        youden_thresh, youden_info = youden_j_threshold(y_true, y_pred)
        sens_thresh, sens_info = sensitivity_threshold(y_true, y_pred, target_sensitivity)
        spec_thresh, spec_info = specificity_threshold(y_true, y_pred, target_specificity)

    results['youden_threshold'] = youden_thresh
    results['sensitivity_threshold'] = sens_thresh
    results['specificity_threshold'] = spec_thresh

    results['youden_target_achieved'] = True
    results['sensitivity_target_achieved'] = sens_info['target_achieved']
    results['specificity_target_achieved'] = spec_info['target_achieved']

    for method, thresh, name in [
        ('youden', youden_thresh, 'Youden J'),
        ('sensitivity_95', sens_thresh, f'Sensitivity {int(target_sensitivity*100)}%'),
        ('specificity_95', spec_thresh, f'Specificity {int(target_specificity*100)}%')
    ]:
        info = get_threshold_info(thresh, y_true, y_pred, name)
        for metric, value in info.items():
            if metric != 'threshold_name':
                results[f'{method}_{metric}'] = value

    results['brier_score'] = brier_score(y_true, y_pred)
    results['ece'] = expected_calibration_error(y_true, y_pred, n_bins=10)

 
    results['roc_auc'] = roc_auc_score(y_true, y_pred)

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    results['pr_auc'] = auc(recall, precision)

    if not skip_bootstrap:
        print(f"    Computing bootstrap CIs ({n_bootstraps} samples)...")

        for method, thresh, name in [
            ('youden', youden_thresh, 'Youden J'),
            ('sensitivity_95', sens_thresh, f'Sensitivity {int(target_sensitivity*100)}%'),
            ('specificity_95', spec_thresh, f'Specificity {int(target_specificity*100)}%')
        ]:
            bootstrap_results = bootstrap_all_metrics(
                y_true, y_pred,
                threshold_fn=lambda yt, yp, t=thresh: (t, {'threshold': t}),
                threshold_name=name,
                n_bootstraps=n_bootstraps,
                ci_level=ci_level,
                seed=seed,
                fixed_threshold=thresh if fixed_thresholds is not None else None
            )

            for metric in ['accuracy', 'f1', 'sensitivity', 'specificity', 'balanced_accuracy']:
                results[f'{method}_{metric}_ci_lower'] = bootstrap_results[f'{metric}_ci_lower']
                results[f'{method}_{metric}_ci_upper'] = bootstrap_results[f'{metric}_ci_upper']

        brier_boot = bootstrap_metric_ci(
            brier_score, y_true, y_pred,
            n_bootstraps=n_bootstraps,
            ci_level=ci_level,
            statistic_name='brier',
            stratified=True,
            seed=seed
        )
        results['brier_score_ci_lower'] = brier_boot['brier_ci_lower']
        results['brier_score_ci_upper'] = brier_boot['brier_ci_upper']

        ece_boot = bootstrap_metric_ci(
            lambda yt, yp: expected_calibration_error(yt, yp, n_bins=10),
            y_true, y_pred,
            n_bootstraps=n_bootstraps,
            ci_level=ci_level,
            statistic_name='ece',
            stratified=True,
            seed=seed
        )
        results['ece_ci_lower'] = ece_boot['ece_ci_lower']
        results['ece_ci_upper'] = ece_boot['ece_ci_upper']

        # Bootstrap for AUC metrics
        roc_boot = bootstrap_metric_ci(
            roc_auc_score, y_true, y_pred,
            n_bootstraps=n_bootstraps,
            ci_level=ci_level,
            statistic_name='roc_auc',
            stratified=True,
            seed=seed
        )
        results['roc_auc_ci_lower'] = roc_boot['roc_auc_ci_lower']
        results['roc_auc_ci_upper'] = roc_boot['roc_auc_ci_upper']

        def pr_auc_fn(yt, yp):
            p, r, _ = precision_recall_curve(yt, yp)
            return auc(r, p)

        pr_boot = bootstrap_metric_ci(
            pr_auc_fn, y_true, y_pred,
            n_bootstraps=n_bootstraps,
            ci_level=ci_level,
            statistic_name='pr_auc',
            stratified=True,
            seed=seed
        )
        results['pr_auc_ci_lower'] = pr_boot['pr_auc_ci_lower']
        results['pr_auc_ci_upper'] = pr_boot['pr_auc_ci_upper']

    return results


def generate_plots(
    predictions_df: pd.DataFrame,
    results: list,
    output_dir: str
):

    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    fold_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    for cohort in predictions_df['cohort'].unique():
        cohort_data = predictions_df[predictions_df['cohort'] == cohort]

        for model in cohort_data['model_name'].unique():
            model_data = cohort_data[cohort_data['model_name'] == model]
            model_slug = model.replace(' ', '_')

            plt.figure(figsize=(6, 6))
            for fold in sorted(model_data['fold'].unique()):
                fold_data = model_data[model_data['fold'] == fold]
                y_true = fold_data['label'].values
                y_pred = fold_data['prediction_score'].values

                fpr, tpr, _ = roc_curve(y_true, y_pred)
                roc_auc = auc(fpr, tpr)

                plt.plot(fpr, tpr, '-', color=fold_colors[fold % len(fold_colors)],
                        linewidth=2, label=f'Fold {fold + 1} (AUC = {roc_auc:.3f})')

            plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC - {model} / {cohort} (5-fold CV)')
            plt.legend(loc='lower right', fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'roc_{model_slug}_{cohort}.svg'), format='svg')
            plt.close()

            plt.figure(figsize=(6, 6))
            for fold in sorted(model_data['fold'].unique()):
                fold_data = model_data[model_data['fold'] == fold]
                y_true = fold_data['label'].values
                y_pred = fold_data['prediction_score'].values

                precision, recall, _ = precision_recall_curve(y_true, y_pred)
                pr_auc = auc(recall, precision)

                plt.plot(recall, precision, '-', color=fold_colors[fold % len(fold_colors)],
                        linewidth=2, label=f'Fold {fold + 1} (PR AUC = {pr_auc:.3f})')

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'PR Curve - {model} / {cohort} (5-fold CV)')
            plt.legend(loc='lower left', fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'pr_{model_slug}_{cohort}.svg'), format='svg')
            plt.close()

            plt.figure(figsize=(6, 6))
            for fold in sorted(model_data['fold'].unique()):
                fold_data = model_data[model_data['fold'] == fold]
                y_true = fold_data['label'].values
                y_pred = fold_data['prediction_score'].values

                if np.any(y_pred < 0) or np.any(y_pred > 1):
                    y_pred = 1 / (1 + np.exp(-y_pred))

                n_bins = 10
                bin_boundaries = np.linspace(0, 1, n_bins + 1)

                mean_predicted_vals = []
                fraction_positive_vals = []

                for i in range(n_bins):
                    if i == n_bins - 1:
                        mask = (y_pred >= bin_boundaries[i]) & (y_pred <= bin_boundaries[i + 1])
                    else:
                        mask = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i + 1])

                    if np.sum(mask) > 0:
                        mean_predicted_vals.append(np.mean(y_pred[mask]))
                        fraction_positive_vals.append(np.mean(y_true[mask]))

                if len(mean_predicted_vals) > 0:
                    plt.plot(mean_predicted_vals, fraction_positive_vals, 'o-',
                            color=fold_colors[fold % len(fold_colors)],
                            linewidth=2, markersize=6, markerfacecolor='white',
                            markeredgewidth=2, label=f'Fold {fold + 1}')

            plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title(f'Calibration - {model} / {cohort} (5-fold CV)')
            plt.legend(loc='upper left', fontsize=9)
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05, 1.05)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'calibration_{model_slug}_{cohort}.svg'), format='svg')
            plt.close()

            print(f"  Saved plots for {model} / {cohort}")


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print("Clinical Evaluation Pipeline")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print()

    if args.input:
        print(f"Loading predictions from: {args.input}")
        predictions_df = pd.read_csv(args.input)
    elif args.input_dir:
        print(f"Loading predictions from directory: {args.input_dir}")
        predictions_df = load_predictions_csv(args.input_dir)
    else:
        print("Error: Must specify --input or --input-dir")
        sys.exit(1)

    print(f"Loaded {len(predictions_df)} predictions")
    print(f"Models: {predictions_df['model_name'].unique().tolist()}")
    print(f"Cohorts: {predictions_df['cohort'].unique().tolist()}")
    print()

    if args.models:
        predictions_df = predictions_df[predictions_df['model_name'].isin(args.models)]
    if args.cohorts:
        predictions_df = predictions_df[predictions_df['cohort'].isin(args.cohorts)]

    reference_thresholds = {}
    for model in predictions_df['model_name'].unique():
        ref_cohort = args.reference_cohort
        if ref_cohort not in predictions_df[predictions_df['model_name'] == model]['cohort'].values:
            ref_cohort = predictions_df[predictions_df['model_name'] == model]['cohort'].iloc[0]
            print(f"  Note: {model} has no {ref_cohort} cohort, using {ref_cohort} as reference for thresholds")

        ref_data = predictions_df[
            (predictions_df['model_name'] == model) &
            (predictions_df['cohort'] == ref_cohort)
        ]
        y_true_ref = ref_data['label'].values
        y_pred_ref = ref_data['prediction_score'].values
        if np.any(y_pred_ref < 0) or np.any(y_pred_ref > 1):
            y_pred_ref = 1 / (1 + np.exp(-y_pred_ref))

        youden_th, youden_info = youden_j_threshold(y_true_ref, y_pred_ref)
        sens_th, sens_info = sensitivity_threshold(y_true_ref, y_pred_ref, args.target_sensitivity)
        spec_th, spec_info = specificity_threshold(y_true_ref, y_pred_ref, args.target_specificity)

        reference_thresholds[model] = {
            'youden': youden_th,
            'sens_95': sens_th,
            'spec_95': spec_th,
            'sens_target_achieved': sens_info['target_achieved'],
            'spec_target_achieved': spec_info['target_achieved'],
            'reference_cohort': ref_cohort
        }
        print(f"\n  Reference thresholds for {model} (from {ref_cohort}):")
        print(f"    Youden J:   {youden_th:.4f}")
        print(f"    Sens 95%:   {sens_th:.4f} (achieved: {sens_info['target_achieved']})")
        print(f"    Spec 95%:   {spec_th:.4f} (achieved: {spec_info['target_achieved']})")

    all_results = []
    for model in predictions_df['model_name'].unique():
        ref_thresh = reference_thresholds[model]

        for cohort in predictions_df['cohort'].unique():
            if cohort not in predictions_df[predictions_df['model_name'] == model]['cohort'].values:
                continue

            cohort_data = predictions_df[
                (predictions_df['model_name'] == model) &
                (predictions_df['cohort'] == cohort)
            ]

            if len(cohort_data) == 0:
                continue

            y_true = cohort_data['label'].values
            y_pred = cohort_data['prediction_score'].values

            fthresh = ref_thresh

            results = evaluate_cohort(
                y_true, y_pred, model, cohort,
                args.target_sensitivity, args.target_specificity,
                args.n_bootstraps, args.ci_level, args.skip_bootstrap,
                fixed_thresholds=fthresh
            )

            all_results.append(results)

    results_df = pd.DataFrame(all_results)

    base_cols = ['model_name', 'cohort', 'n_samples', 'n_positive', 'n_negative']
    threshold_cols = ['youden_threshold', 'sensitivity_threshold', 'specificity_threshold']
    auc_cols = ['roc_auc', 'roc_auc_ci_lower', 'roc_auc_ci_upper', 'pr_auc', 'pr_auc_ci_lower', 'pr_auc_ci_upper']
    calibration_cols = ['brier_score', 'brier_score_ci_lower', 'brier_score_ci_upper',
                        'ece', 'ece_ci_lower', 'ece_ci_upper']

    metric_cols = []
    for method in ['youden', 'sensitivity_95', 'specificity_95']:
        for metric in ['accuracy', 'f1', 'sensitivity', 'specificity', 'balanced_accuracy',
                       'TP', 'FP', 'TN', 'FN']:
            metric_cols.append(f'{method}_{metric}')
            if not args.skip_bootstrap:
                metric_cols.append(f'{method}_{metric}_ci_lower')
                metric_cols.append(f'{method}_{metric}_ci_upper')

    flag_cols = ['youden_target_achieved', 'sensitivity_target_achieved', 'specificity_target_achieved']

    all_cols = base_cols + threshold_cols + flag_cols + auc_cols + calibration_cols + metric_cols
    available_cols = [c for c in all_cols if c in results_df.columns]

    results_df = results_df[available_cols]

    results_path = output_dir / f"evaluation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved evaluation results: {results_path}")

    thresholds_by_model = {}
    for model, ref_thresh in reference_thresholds.items():
        thresholds_by_model[model] = {
            'youden': ref_thresh['youden'],
            f'sens_{int(args.target_sensitivity*100)}': ref_thresh['sens_95'],
            f'spec_{int(args.target_specificity*100)}': ref_thresh['spec_95']
        }

    patient_output = predictions_df.copy()
    patient_output['prediction_probability'] = np.nan

    for model in predictions_df['model_name'].unique():
        model_mask = patient_output['model_name'] == model
        model_data = patient_output[model_mask]

        for fold in model_data['fold'].unique():
            fold_mask = model_data['fold'] == fold
            fold_indices = model_data[fold_mask].index
            y_pred_fold = model_data.loc[fold_indices, 'prediction_score'].values

            if np.any(y_pred_fold < 0) or np.any(y_pred_fold > 1):
                y_pred_proba = 1 / (1 + np.exp(-y_pred_fold))
            else:
                y_pred_proba = y_pred_fold

            patient_output.loc[fold_indices, 'prediction_probability'] = y_pred_proba

        for name, thresh in thresholds_by_model[model].items():
            patient_output.loc[model_mask, f'pred_binary_{name}'] = (
                patient_output.loc[model_mask, 'prediction_probability'] >= thresh
            ).astype(int)
            patient_output.loc[model_mask, f'distance_from_{name}'] = (
                patient_output.loc[model_mask, 'prediction_probability'] - thresh
            )

    patient_output_path = output_dir / f"per_patient_predictions.csv"
    patient_output.to_csv(patient_output_path, index=False)
    print(f"Saved per-patient predictions: {patient_output_path}")

    for model in predictions_df['model_name'].unique():
        model_output = patient_output[patient_output['model_name'] == model]
        model_slug = model.replace(' ', '_')
        model_path = output_dir / f"per_patient_{model_slug}.csv"
        model_output.to_csv(model_path, index=False)

    calibration_data = []
    for model in predictions_df['model_name'].unique():
        for cohort in predictions_df['cohort'].unique():
            cohort_data = predictions_df[
                (predictions_df['model_name'] == model) &
                (predictions_df['cohort'] == cohort)
            ]
            if len(cohort_data) == 0:
                continue

            y_true_all = []
            y_pred_proba_all = []

            for fold in cohort_data['fold'].unique():
                fold_data = cohort_data[cohort_data['fold'] == fold]
                y_true_fold = fold_data['label'].values
                y_pred_fold = fold_data['prediction_score'].values

                # Apply sigmoid to this fold's predictions
                if np.any(y_pred_fold < 0) or np.any(y_pred_fold > 1):
                    y_pred_fold = 1 / (1 + np.exp(-y_pred_fold))

                y_true_all.extend(y_true_fold)
                y_pred_proba_all.extend(y_pred_fold)

            y_true_all = np.array(y_true_all)
            y_pred_proba_all = np.array(y_pred_proba_all)

            frac_pos, mean_pred, bin_centers = calibration_curve(y_true_all, y_pred_proba_all, n_bins=10)

            for i in range(10):
                calibration_data.append({
                    'model_name': model,
                    'cohort': cohort,
                    'bin': i + 1,
                    'bin_center': bin_centers[i],
                    'mean_predicted': mean_pred[i],
                    'fraction_positive': frac_pos[i]
                })

    cal_df = pd.DataFrame(calibration_data)
    cal_path = output_dir / f"calibration_data.csv"
    cal_df.to_csv(cal_path, index=False)
    print(f"Saved calibration data: {cal_path}")

    if args.generate_plots:
        print("\nGenerating plots...")
        generate_plots(predictions_df, all_results, str(output_dir))

    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")

    for _, row in results_df.iterrows():
        print(f"\n{row['model_name']} - {row['cohort']} (n={row['n_samples']})")
        print(f"  ROC AUC: {row['roc_auc']:.3f} (95% CI: {row.get('roc_auc_ci_lower', 'N/A'):.3f}-{row.get('roc_auc_ci_upper', 'N/A'):.3f})")
        print(f"  Brier:   {row['brier_score']:.4f} (95% CI: {row.get('brier_score_ci_lower', 'N/A'):.4f}-{row.get('brier_score_ci_upper', 'N/A'):.4f})")
        print(f"  ECE:     {row['ece']:.4f} (95% CI: {row.get('ece_ci_lower', 'N/A'):.4f}-{row.get('ece_ci_upper', 'N/A'):.4f})")
        print(f"  Thresholds:")
        print(f"    Youden J:     {row['youden_threshold']:.4f}")
        print(f"    Sens 95%:     {row['sensitivity_threshold']:.4f} (achieved: {row.get('sensitivity_target_achieved', 'N/A')})")
        print(f"    Spec 95%:     {row['specificity_threshold']:.4f} (achieved: {row.get('specificity_target_achieved', 'N/A')})")

    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
