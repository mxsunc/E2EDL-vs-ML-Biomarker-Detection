#%%
"""MSI RF - mutation catalogue RandomForest. 5-fold CV."""

import os
import sys
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

cwd = "..."

tcga_sbs_df = pd.read_csv(cwd + '/tcga_sbs_mutation_catalogues_norm.csv')
tcga_sbs_df = tcga_sbs_df.rename(columns={"Patient_ID": "bcr_patient_barcode"})
tcga_id_df = pd.read_csv(cwd + '/tcga_id_mutation_catalogues_norm.csv')
tcga_id_df = tcga_id_df.rename(columns={"Patient_ID": "bcr_patient_barcode"})
tcga_df = pd.merge(tcga_sbs_df, tcga_id_df, on='bcr_patient_barcode', how='inner')
tcga_df = tcga_df[[col for col in tcga_df.columns if col != 'bcr_patient_barcode'] + ['bcr_patient_barcode']]

cptac_sbs_df = pd.read_csv(cwd + '/cptac_combined_sbs_mutation_catalogues_norm.csv')
cptac_sbs_df = cptac_sbs_df.rename(columns={"Patient_ID": "Patient ID"})
cptac_id_df = pd.read_csv(cwd + '/cptac_combined_id_mutation_catalogues_norm.csv')
cptac_id_df = cptac_id_df.rename(columns={"Patient_ID": "Patient ID"})
cptac_df = pd.merge(cptac_sbs_df, cptac_id_df, on='Patient ID', how='inner')
cptac_df = cptac_df[[col for col in cptac_df.columns if col != 'Patient ID'] + ['Patient ID']]

msi_status_df = pd.read_csv(cwd + '/cptac_crc_msistatus.csv')
msi_status_df['label'] = msi_status_df['MSMutect-Fisher decision'].map({'MSI-high': 1, 'MSS': 0})
cptac_df = cptac_df.merge(msi_status_df[['Patient_ID', 'label']], left_on='Patient ID', right_on='Patient_ID', how='inner')
cptac_df = cptac_df.dropna(subset=['label'])

print(f"CPTAC samples: {len(cptac_df)}")
print(f"  MSI-H: {int(cptac_df['label'].sum())}")
print(f"  nonMSI-H: {int((1 - cptac_df['label']).sum())}")

all_predictions = []
model_name = 'RF_MSI_mut'

for j in range(5):
    print(f"\n{'='*60}")
    print(f"Fold {j+1}/5")
    print(f"{'='*60}")

    df = pd.read_csv(cwd + f"/MSIfolds{j+1}.csv")
    df = df.drop(columns={"BCR"})
    df_merged = df.merge(tcga_df, on="bcr_patient_barcode", how="inner")
    df_merged = df_merged.sample(frac=1, random_state=SEED)

    label_mapping = {'MSIH': 1, 'nonMSIH': 0}
    df_merged['label'] = df_merged['PCR'].map(label_mapping)

    train_df = df_merged[df_merged["split"] == "train"]
    train_pos = train_df[train_df["PCR"] == "MSIH"]
    train_neg = train_df[train_df["PCR"] == "nonMSIH"]
    train_resampled_pos = train_pos.sample(n=len(train_neg), replace=True, random_state=SEED)
    train_data = pd.concat([train_resampled_pos, train_neg], axis=0).sample(frac=1, random_state=SEED)

    test_data = df_merged[df_merged["split"] == "test"].sample(frac=1, random_state=SEED)

    metadata_cols = ['split', 'PCR', 'label', 'BCR', 'bcr_patient_barcode',
                     'tissue', 'Tissue', 'site', 'Source Site', 'Study Name',
                     'MSI MANTIS Score', 'MSIsensor Score', 'Mutation Count',
                     'TMB (nonsynonymous)', 'MANTIS', 'MSISENS']
    feature_cols = [c for c in train_data.columns if c not in metadata_cols]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_data[c])]
    X_train = train_data[feature_cols]
    X_test = test_data[feature_cols]
    y_train = train_data['label']
    y_test = test_data['label']

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample"
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    for idx, (score, label) in enumerate(zip(y_pred_proba, y_test)):
        all_predictions.append({
            'fold': j,
            'patient_id': test_data['bcr_patient_barcode'].iloc[idx],
            'cohort': 'TCGA_test',
            'label': int(label),
            'prediction_score': float(score),
            'model_name': model_name
        })
    print(f"  TCGA test: {len(y_pred_proba)} predictions")

    X_cptac = cptac_df.reindex(columns=feature_cols, fill_value=0)
    y_cptac = cptac_df['label'].values
    y_pred_cptac_proba = model.predict_proba(X_cptac)[:, 1]
    for idx, (score, label) in enumerate(zip(y_pred_cptac_proba, y_cptac)):
        all_predictions.append({
            'fold': j,
            'patient_id': cptac_df.iloc[idx]['Patient ID'],
            'cohort': 'CPTAC_full',
            'label': int(label),
            'prediction_score': float(score),
            'model_name': model_name
        })
    print(f"  CPTAC full: {len(y_pred_cptac_proba)} predictions")

print(f"\n{'='*60}")
print("Saving predictions...")
print(f"{'='*60}")

predictions_df = pd.DataFrame(all_predictions)
output_dir = "predictions_RF_mut/"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "predictions_MSI_RF_mut_all_folds.csv")
predictions_df.to_csv(output_path, index=False)
print(f"Saved: {output_path}")
print(f"Total predictions: {len(predictions_df)}")
print(f"Cohorts: {predictions_df['cohort'].unique().tolist()}")

for cohort in predictions_df['cohort'].unique():
    cohort_df = predictions_df[predictions_df['cohort'] == cohort]
    cohort_path = os.path.join(output_dir, f"predictions_MSI_RF_mut_{cohort}.csv")
    cohort_df.to_csv(cohort_path, index=False)
    print(f"  {cohort}: {len(cohort_df)} samples")

print(f"\n{'='*60}")
print("Prediction Summary")
print(f"{'='*60}")
for cohort in predictions_df['cohort'].unique():
    cohort_df = predictions_df[predictions_df['cohort'] == cohort]
    n_pos = cohort_df['label'].sum()
    n_neg = len(cohort_df) - n_pos
    print(f"  {cohort}: {len(cohort_df)} samples (MSI-H: {n_pos}, nonMSI-H: {n_neg})")

print(f"\n{'='*60}")
print("Next Step: Run evaluation pipeline")
print(f"{'='*60}")
print(f"python run_evaluation_pipeline.py \\")
print(f"  --input {output_path} \\")
print(f"  --output-dir evaluation_results_RF_mut/")

# %%
