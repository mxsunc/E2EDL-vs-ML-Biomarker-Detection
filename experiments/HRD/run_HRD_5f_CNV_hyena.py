#%%
"""
CNV Hyena embeddings - 5-Fold CV Prediction Export (Revision)

Trains an MLP on CNV Hyena embeddings and exports predictions in a standardized
format for comparison with fusion models.

Uses:
- 128-dim mean-pooled Hyena embeddings as input
- 5-fold CV splits from HRDfolds{j}.csv
- scarHRD labels for CPTAC evaluation

Output:
- predictions_CNV_Hyena_int_all_folds.csv: Per-patient predictions for all cohorts
"""

SEED = 42
import os
os.environ['PYTHONHASHSEED'] = str(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)
import numpy as np
np.random.seed(SEED)
import random
random.seed(SEED)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[-1], True)
    tf.config.experimental.set_visible_devices(physical_devices[-1], 'GPU')
    print(f"Using GPU: {physical_devices[-1]}")
else:
    print("No GPU available, running on CPU")

import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models, layers

DROPOUT = 0.3
BATCH_SIZE = 128
EPOCHS = 300
EMBED_DIM = 128

def build_cnv_mlp(input_dim=EMBED_DIM, dropout_rate=DROPOUT):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(int(input_dim / 2), activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(2048, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(512, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def asfloat32(x):
    return np.asarray(x, dtype=np.float32)

AUTOTUNE = tf.data.AUTOTUNE

def make_balanced_ds(X_arr, y_arr, batch_size=BATCH_SIZE, shuffle_buffer=10_000):
    half = batch_size // 2
    pos = tf.data.Dataset.from_tensor_slices((X_arr[y_arr == 1], y_arr[y_arr == 1]))
    neg = tf.data.Dataset.from_tensor_slices((X_arr[y_arr == 0], y_arr[y_arr == 0]))
    pos = pos.shuffle(shuffle_buffer).repeat().batch(half, drop_remainder=True)
    neg = neg.shuffle(shuffle_buffer).repeat().batch(half, drop_remainder=True)
    merged = tf.data.Dataset.zip((pos, neg))
    def _merge(p, n):
        x = tf.concat([p[0], n[0]], axis=0)
        y_ = tf.concat([p[1], n[1]], axis=0)
        idx = tf.random.shuffle(tf.range(batch_size))
        return tf.gather(x, idx), tf.gather(y_, idx)
    return merged.map(_merge, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

print("Loading TCGA BERT embeddings...")
tcga_emb_path = ".../cnv_int_hyena_mean_embeddings_tcga_128.csv"
tcga_emb_df = pd.read_csv(tcga_emb_path, index_col="bcr_patient_barcode")
tcga_emb_df = tcga_emb_df[~tcga_emb_df.index.duplicated(keep="first")].copy()
tcga_emb_df["patient_id"] = tcga_emb_df.index.str[:12]
tcga_emb_df = tcga_emb_df.drop_duplicates(subset="patient_id").set_index("patient_id")
emb_cols = [c for c in tcga_emb_df.columns if c.startswith("mean_emb")]

print("Loading CPTAC BERT embeddings...")
cptac_emb_path = ".../cnv_int_hyena_mean_embeddings_cptac_128.csv"
cptac_emb_df = pd.read_csv(cptac_emb_path, index_col="bcr_patient_barcode")
cptac_emb_df = cptac_emb_df[~cptac_emb_df.index.duplicated(keep="first")].copy()

df_hrd_cptac = pd.read_csv(".../scarhrd_combined_all_cohorts.csv")
df_hrd_cptac = df_hrd_cptac[["case_id", "HRD_binary"]].dropna(subset=["HRD_binary"]).drop_duplicates(subset="case_id", keep="first")
df_hrd_cptac["HRD_Binary_us"] = df_hrd_cptac["HRD_binary"].astype(int)

merged_cptac = cptac_emb_df.merge(df_hrd_cptac[["case_id", "HRD_Binary_us"]], left_index=True, right_on="case_id", how="inner")
X_cptac = asfloat32(merged_cptac[emb_cols].values)
y_cptac = merged_cptac["HRD_Binary_us"]
cptac_patient_ids = merged_cptac["case_id"].tolist()
print(f"CPTAC samples: {len(X_cptac)}, HRD+: {int(y_cptac.sum())}")

all_predictions = []
model_name = 'CNV_Hyena_int'

for j in range(5):
    print(f"\n{'='*60}")
    print(f"Fold {j+1}/5")
    print(f"{'='*60}")

    df = pd.read_csv(f".../HRDfolds{j+1}.csv")
    df = df.drop(columns={"BCR"})
    df = df.rename(columns={"Patient ID": "patient_id"})
    df = df.dropna(subset=["HRD_binary_paper"])
    df["HRD_status"] = df["HRD_binary_paper"].map({"HRD_negative": 0, "HRD_positive": 1})

    common_ids = set(tcga_emb_df.index).intersection(df["patient_id"])
    emb_fold = tcga_emb_df.loc[tcga_emb_df.index.isin(common_ids)]
    df_fold = df[df["patient_id"].isin(common_ids)]
    merged_df = emb_fold.merge(df_fold[["patient_id", "HRD_status", "split"]], left_index=True, right_on="patient_id", how="inner")

    def split_xy(sub):
        return asfloat32(sub[emb_cols].values), sub["HRD_status"].astype(int).values

    X_train, y_train = split_xy(merged_df[merged_df["split"] == "train"])
    X_val, y_val = split_xy(merged_df[merged_df["split"] == "validation"])
    X_test, y_test = split_xy(merged_df[merged_df["split"] == "test"])
    test_patient_ids = merged_df[merged_df["split"] == "test"]["patient_id"].tolist()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    X_cptac_s = scaler.transform(X_cptac)

    INPUT_DIM = X_train.shape[1]
    print(f"Input dimension: {INPUT_DIM}")

    model = build_cnv_mlp(input_dim=INPUT_DIM, dropout_rate=DROPOUT)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=30, restore_best_weights=True)

    train_ds = make_balanced_ds(X_train_s.astype(np.float32), y_train.astype(np.float32), batch_size=BATCH_SIZE)

    model.fit(
        train_ds,
        steps_per_epoch=max(1, len(y_train) // BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_val_s, y_val),
        callbacks=[callback],
        verbose=1
    )

    y_pred_test = model.predict(X_test_s, verbose=0)
    for pid, score, label in zip(test_patient_ids, y_pred_test.flatten(), y_test):
        all_predictions.append({
            'fold': j, 'patient_id': pid, 'cohort': 'TCGA_test',
            'label': int(label), 'prediction_score': float(score),
            'model_name': model_name
        })
    print(f"TCGA Test - {len(y_pred_test)} predictions")

    y_pred_cptac = model.predict(X_cptac_s, verbose=0)
    for pid, score, label in zip(cptac_patient_ids, y_pred_cptac.flatten(), y_cptac.values):
        all_predictions.append({
            'fold': j, 'patient_id': pid, 'cohort': 'CPTAC_full',
            'label': int(label), 'prediction_score': float(score),
            'model_name': model_name
        })
    print(f"CPTAC Full - {len(y_pred_cptac)} predictions")

print(f"\n{'='*60}")
print("Saving predictions...")
print(f"{'='*60}")

predictions_df = pd.DataFrame(all_predictions)
output_dir = "predictions_CNV_Hyena_int/"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "predictions_CNV_Hyena_int_all_folds.csv")
predictions_df.to_csv(output_path, index=False)
print(f"Saved: {output_path}")
print(f"Total predictions: {len(predictions_df)}")
print(f"Cohorts: {predictions_df['cohort'].unique().tolist()}")

for cohort in predictions_df['cohort'].unique():
    cohort_df = predictions_df[predictions_df['cohort'] == cohort]
    cohort_path = os.path.join(output_dir, f"predictions_CNV_Hyena_int_{cohort}.csv")
    cohort_df.to_csv(cohort_path, index=False)
    print(f"Saved: {cohort_path} ({len(cohort_df)} samples)")

print(f"\n{'='*60}")
print("Prediction Summary")
print(f"{'='*60}")
for cohort in predictions_df['cohort'].unique():
    cohort_df = predictions_df[predictions_df['cohort'] == cohort]
    n_pos = cohort_df['label'].sum()
    print(f"  {cohort}: {len(cohort_df)} samples (HRD+: {n_pos}, HRD-: {len(cohort_df) - n_pos})")

print(f"\n{'='*60}")
print("Next Step: Run evaluation pipeline")
print(f"{'='*60}")
print(f"python run_evaluation_pipeline.py --input {output_path} --output-dir evaluation_results_CNV_Hyena_int/")
# %%
