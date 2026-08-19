import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Union
from pathlib import Path
import warnings


def save_predictions_csv(
    predictions_df: pd.DataFrame,
    output_path: str,
    model_name: str,
    cohort_name: Optional[str] = None,
    include_threshold_columns: bool = True
) -> str:

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate required columns
    required_cols = ['patient_id', 'cohort', 'fold', 'label', 'prediction_score']
    missing_cols = [col for col in required_cols if col not in predictions_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure correct dtypes
    df = predictions_df.copy()
    df['label'] = df['label'].astype(int)
    df['fold'] = df['fold'].astype(int)
    df['prediction_score'] = df['prediction_score'].astype(float)

    if cohort_name:
        # Save separate file per cohort
        cohort_df = df[df['cohort'] == cohort_name]
        filename = f"predictions_{model_name}_{cohort_name}.csv"
        cohort_df.to_csv(output_dir / filename, index=False)
        return str(output_dir / filename)
    else:
        # Save all cohorts together
        filename = f"predictions_{model_name}.csv"
        df.to_csv(output_dir / filename, index=False)
        return str(output_dir / filename)


def load_predictions_csv(
    input_path: str,
    model_name: Optional[str] = None,
    cohort_name: Optional[str] = None
) -> pd.DataFrame:

    input_path = Path(input_path)

    if input_path.is_file():
        df = pd.read_csv(input_path)
    elif input_path.is_dir():
        # Find all prediction CSVs in directory
        if model_name:
            pattern = f"predictions_{model_name}*.csv"
        else:
            pattern = "predictions_*.csv"

        files = list(input_path.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No prediction files found matching {pattern}")

        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # Apply filters
    if model_name and 'model_name' in df.columns:
        df = df[df['model_name'] == model_name]
    elif model_name and 'model_name' not in df.columns:
        # Try to infer from filename - add model_name column
        df['model_name'] = model_name

    if cohort_name and 'cohort' in df.columns:
        df = df[df['cohort'] == cohort_name]

    return df


def aggregate_predictions_by_model(
    pred_dir: str,
    output_path: Optional[str] = None
) -> pd.DataFrame:

    pred_dir = Path(pred_dir)
    files = list(pred_dir.glob("predictions_*.csv"))

    if not files:
        raise FileNotFoundError(f"No prediction files found in {pred_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Extract model name from filename if not in DataFrame
        if 'model_name' not in df.columns:
            # Filename format: predictions_{model_name}_{cohort}.csv or predictions_{model_name}.csv
            parts = f.stem.replace('predictions_', '').split('_')
            model_name = parts[0] if parts else f.stem
            df['model_name'] = model_name
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined['label'] = combined['label'].astype(int)
    combined['fold'] = combined['fold'].astype(int)
    combined['prediction_score'] = combined['prediction_score'].astype(float)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)

    return combined


def merge_model_predictions(
    base_predictions: pd.DataFrame,
    additional_predictions: Union[pd.DataFrame, str],
    on_columns: Optional[List[str]] = None
) -> pd.DataFrame:

    if isinstance(additional_predictions, str):
        additional_predictions = pd.read_csv(additional_predictions)

    if on_columns is None:
        on_columns = ['patient_id', 'cohort', 'fold', 'label']

    # Identify prediction score columns
    base_score_cols = [c for c in base_predictions.columns if 'score' in c.lower() or 'pred' in c.lower()]
    add_score_cols = [c for c in additional_predictions.columns if 'score' in c.lower() or 'pred' in c.lower()]

    # Rename score columns to include model name
    base_df = base_predictions.copy()
    add_df = additional_predictions.copy()

    if 'model_name' in base_df.columns:
        model_names = base_df['model_name'].unique()
        if len(model_names) == 1:
            model_name = model_names[0]
            for col in base_score_cols:
                if model_name not in col:
                    base_df = base_df.rename(columns={col: f'{col}_{model_name}'})

    if 'model_name' in add_df.columns:
        model_names = add_df['model_name'].unique()
        if len(model_names) == 1:
            model_name = model_names[0]
            for col in add_score_cols:
                if model_name not in col:
                    add_df = add_df.rename(columns={col: f'{col}_{model_name}'})

    # Merge
    merged = pd.merge(base_df, add_df, on=on_columns, how='outer', suffixes=('_base', '_add'))

    return merged


def save_evaluation_results(
    results_df: pd.DataFrame,
    output_path: str,
    include_timestamp: bool = True
) -> str:

    from datetime import datetime

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = output_path.stem
        suffix = output_path.suffix
        output_path = output_path.parent / f"{stem}_{timestamp}{suffix}"

    results_df.to_csv(output_path, index=False)
    return str(output_path)


def create_patient_level_output(
    predictions_df: pd.DataFrame,
    thresholds: Dict[str, float],
    output_path: Optional[str] = None
) -> pd.DataFrame:
 
    df = predictions_df.copy()

    # Add binary predictions for each threshold
    for name, thresh in thresholds.items():
        df[f'pred_binary_{name}'] = (df['prediction_score'] >= thresh).astype(int)

    # Add confidence indicator (how far from threshold)
    for name, thresh in thresholds.items():
        df[f'distance_from_{name}_threshold'] = df['prediction_score'] - thresh

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df
