#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
import sys
import random
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_curve, roc_curve, auc, roc_auc_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

tcga_sbs_df = pd.read_csv(".../tcga_sbs_mutation_catalogues_norm.csv")
tcga_sbs_df = tcga_sbs_df.rename(columns={"Patient_ID": "bcr_patient_barcode"})
tcga_id_df = pd.read_csv(".../tcga_id_mutation_catalogues_norm.csv")
tcga_id_df = tcga_id_df.rename(columns={"Patient_ID": "bcr_patient_barcode"})
tcga_df = pd.merge(tcga_sbs_df, tcga_id_df, on='bcr_patient_barcode', how='inner')
tcga_df = tcga_df[[col for col in tcga_df.columns if col != 'bcr_patient_barcode'] + ['bcr_patient_barcode']]

input_path = ".../all_cancers_cnv_matrix_int.csv"
all_cancers_df = pd.read_csv(input_path, index_col="sample_barcode")
all_cancers_df["bcr_patient_barcode"] = all_cancers_df.index.str[:12]
all_cancers_df = all_cancers_df.set_index("bcr_patient_barcode")
all_cancers_df = all_cancers_df[~all_cancers_df.index.duplicated(keep="first")].copy()

tcga_df = tcga_df.merge(all_cancers_df, left_on="bcr_patient_barcode", right_index=True, how="inner")

scarhrd_df = pd.read_csv(".../scarhrd_combined_all_cohorts.csv")
scarhrd_df = scarhrd_df.rename(columns={"case_id": "Patient ID"})
scarhrd_df = scarhrd_df.dropna(subset=["HRD_binary"]).drop_duplicates(subset="Patient ID", keep="first")
scarhrd_df["HRD_binary"] = scarhrd_df["HRD_binary"].astype(int)

cptac_ucec_df = pd.read_csv(".../cptac_combined_sbs_mutation_catalogues_norm.csv")
cptac_ucec_df = cptac_ucec_df.rename(columns={"Patient_ID": "Patient ID"})
cptac_df_id = pd.read_csv(".../cptac_combined_id_mutation_catalogues_norm.csv")
cptac_df_id = cptac_df_id.rename(columns={"Patient_ID": "Patient ID"})
cptac_df = pd.merge(cptac_ucec_df, cptac_df_id, on='Patient ID', how='inner')
cptac_df = cptac_df[[col for col in cptac_df.columns if col != 'Patient ID'] + ['Patient ID']]

input_path = ".../all_cancers_cnv_matrix_int.csv"
all_cancers_df_cptac = pd.read_csv(input_path, index_col="sample_barcode")
all_cancers_df_cptac = all_cancers_df_cptac[~all_cancers_df_cptac.index.duplicated(keep="first")].copy()
all_cancers_df_cptac["Patient ID"] = all_cancers_df_cptac.index

cptac_df = cptac_df.merge(all_cancers_df_cptac, on="Patient ID", how="inner")
cptac_df = cptac_df.merge(scarhrd_df[['Patient ID', 'HRD_binary', 'HRD-sum', 'cohort']],
                           on="Patient ID", how="inner")

print(f"CPTAC samples with scarHRD labels: {len(cptac_df)}")
print(f"  BRCA: {len(cptac_df[cptac_df['cohort'] == 'brca'])}")
print(f"  OV: {len(cptac_df[cptac_df['cohort'] == 'ov'])}")
print(f"  HRD-positive: {cptac_df['HRD_binary'].sum()}")
print(f"  HRD-negative: {(1 - cptac_df['HRD_binary']).sum()}")

all_predictions = []
model_name = 'LR_HRD_mutCNV'

for j in range(5):
    print(f"\n{'='*60}")
    print(f"Fold {j+1}/5")
    print(f"{'='*60}")

    df = pd.read_csv(f".../HRDfolds{j+1}.csv")
    df = df.drop(columns={"BCR"})
    df = df.rename(columns={"Patient ID": "bcr_patient_barcode"})

    df_merged = df.merge(tcga_df, on="bcr_patient_barcode", how="inner")
    df_merged = df_merged.sample(frac=1)

    label_mapping = {'HRD_positive': 1, 'HRD_negative': 0}
    df_merged['label'] = df_merged['HRD_binary_paper'].map(label_mapping)

    train_df = df_merged[df_merged["split"] == "train"]
    train_hrd_pos = train_df[train_df["HRD_binary_paper"] == "HRD_positive"]
    train_hrd_neg = train_df[train_df["HRD_binary_paper"] == "HRD_negative"]
    train_resampled_pos = train_hrd_pos.sample(n=len(train_hrd_neg), replace=True, random_state=SEED)
    train_data = pd.concat([train_resampled_pos, train_hrd_neg], axis=0).sample(frac=1, random_state=SEED)

    test_data = df_merged[df_merged["split"] == "test"].sample(frac=1, random_state=SEED)

    metadata_cols = ['split', 'HRD_binary_paper', 'label', 'BCR', 'bcr_patient_barcode',
                     'Study ID', 'Site', 'tissue', 'HRD_status', 'Source Site', 'Study Name',
                     'HRD_TAI_paper', 'HRD_LST_paper', 'HRD_LOH_paper', 'HRD_Sum_paper', 'cutoff']
    feature_cols = [c for c in train_data.columns if c not in metadata_cols]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_data[c])]
    X_train = train_data[feature_cols]
    X_test = test_data[feature_cols]
    y_train = train_data['label']
    y_test = test_data['label']

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train_s, y_train)

    y_pred_proba = model.predict_proba(X_test_s)[:, 1]

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
    y_cptac = cptac_df["HRD_binary"].values
    X_cptac_s = scaler.transform(X_cptac)

    y_pred_cptac_proba = model.predict_proba(X_cptac_s)[:, 1]

    for idx, (score, label) in enumerate(zip(y_pred_cptac_proba, y_cptac)):
        all_predictions.append({
            'fold': j,
            'patient_id': cptac_df.iloc[idx]['Patient ID'],
            'cohort': 'CPTAC_full',
            'label': int(label),
            'prediction_score': float(score),
            'model_name': model_name
        })

    print(f"  CPTAC full (scarHRD): {len(y_pred_cptac_proba)} predictions")

print(f"\n{'='*60}")
print("Saving predictions...")
print(f"{'='*60}")

predictions_df = pd.DataFrame(all_predictions)

output_dir = "predictions_LR_mutCNV/"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "predictions_HRD_LR_mutCNV_all_folds.csv")
predictions_df.to_csv(output_path, index=False)
print(f"Saved: {output_path}")
print(f"Total predictions: {len(predictions_df)}")
print(f"Cohorts: {predictions_df['cohort'].unique().tolist()}")

for cohort in predictions_df['cohort'].unique():
    cohort_df = predictions_df[predictions_df['cohort'] == cohort]
    cohort_path = os.path.join(output_dir, f"predictions_HRD_LR_mutCNV_{cohort}.csv")
    cohort_df.to_csv(cohort_path, index=False)
    print(f"  {cohort}: {len(cohort_df)} samples")

print(f"\n{'='*60}")
print("Prediction Summary")
print(f"{'='*60}")
for cohort in predictions_df['cohort'].unique():
    cohort_df = predictions_df[predictions_df['cohort'] == cohort]
    n_pos = cohort_df['label'].sum()
    n_neg = len(cohort_df) - n_pos
    print(f"  {cohort}: {len(cohort_df)} samples (HRD+: {n_pos}, HRD-: {n_neg})")

print(f"\n{'='*60}")
print("Next Step: Run evaluation pipeline")
print(f"{'='*60}")
print(f"python run_evaluation_pipeline.py \\")
print(f"  --input {output_path} \\")
print(f"  --output-dir evaluation_results_LR_mutCNV/")

# %%
