#%%
"""
CNV Variational Autoencoder (VAE) - 5-Fold CV Prediction Export (Revision)

Trains a VAE on CNV data and exports predictions in a standardized format
for comparison with fusion models.

Uses:
- GISTIC genes as input
- 5-fold CV splits from HRDfolds{j}.csv
- scarHRD labels for CPTAC evaluation

Output:
- predictions_CNV_VAE_all_folds.csv: Per-patient predictions for all cohorts
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
from sklearn.metrics import roc_curve, precision_recall_curve, auc, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, regularizers, Model, Input

LATENT_DIM = 256
DROPOUT = 0.4
L2_WEIGHT = 1e-4
L1_LATENT = 1e-6
KL_WEIGHT = 0.001
BATCH_SIZE = 64
EPOCHS = 200
CLF_LOSS_WEIGHT = 3.0
LEARNING_RATE = 3e-4
AUTOTUNE = tf.data.AUTOTUNE

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

def make_balanced_ds(X_arr, y_arr, batch_size=64, shuffle_buffer=10_000):
    """Create balanced dataset for training."""
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
        x, y_ = tf.gather(x, idx), tf.gather(y_, idx)
        return x, {"reconstruction": x, "hrd_pred": y_}

    return merged.map(_merge, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

print("Loading TCGA CNV data...")
input_path = ".../all_cancers_cnv_matrix_int.csv"
all_cancers_df = pd.read_csv(input_path, index_col="sample_barcode")
all_cancers_df = all_cancers_df[~all_cancers_df.index.duplicated(keep="first")].copy()
all_cancers_df["patient_id"] = all_cancers_df.index.str[:12]
all_cancers_df = all_cancers_df.set_index("patient_id")

print("Loading CPTAC CNV data...")
input_path_cptac = ".../all_cancers_cnv_matrix_int.csv"
all_cancers_df_cptac = pd.read_csv(input_path_cptac, index_col="sample_barcode")
all_cancers_df_cptac = all_cancers_df_cptac[~all_cancers_df_cptac.index.duplicated(keep="first")].copy()

df_hrd_cptac = pd.read_csv(".../scarhrd_combined_all_cohorts.csv")
df_hrd_cptac = df_hrd_cptac[["case_id", "HRD_binary"]].dropna(subset=["HRD_binary"]).copy()
df_hrd_cptac = df_hrd_cptac.drop_duplicates(subset="case_id", keep="first")
df_hrd_cptac["HRD_Binary_us"] = df_hrd_cptac["HRD_binary"].astype(int)

y_label_cptac_df = df_hrd_cptac[["case_id", "HRD_Binary_us"]].rename(columns={"case_id": "sample_barcode"})
merged_cptac = all_cancers_df_cptac.merge(y_label_cptac_df, left_index=True, right_on="sample_barcode", how="inner")
X_cptac = merged_cptac.drop(columns=["sample_barcode", "HRD_Binary_us"]).astype(float).values
y_cptac = merged_cptac["HRD_Binary_us"].values
cptac_patient_ids = merged_cptac["sample_barcode"].tolist()

print(f"CPTAC samples: {len(X_cptac)}, HRD+: {sum(y_cptac)}")

all_predictions = []
model_name = 'CNV_VAE_gistic'

for j in range(5):
    print(f"\n{'='*60}")
    print(f"Fold {j+1}/5")
    print(f"{'='*60}")

    df_hrd = pd.read_csv(f".../HRDfolds{j+1}.csv")
    df_hrd = df_hrd.dropna(subset=["HRD_binary_paper"]).copy()
    df_hrd["HRD_status"] = df_hrd["HRD_binary_paper"].map({"HRD_negative": 0, "HRD_positive": 1})

    common_ids = set(all_cancers_df.index).intersection(df_hrd["Patient ID"])
    all_cancers_df_fold = all_cancers_df.loc[all_cancers_df.index.isin(common_ids)]
    df_hrd_fold = df_hrd[df_hrd["Patient ID"].isin(common_ids)]

    y_label_df = df_hrd_fold[["Patient ID", "HRD_status", "split"]].rename(columns={"Patient ID": "patient_id"})
    merged_df = all_cancers_df_fold.merge(y_label_df, left_index=True, right_on="patient_id", how="inner")

    X_train = merged_df[merged_df["split"]=="train"].drop(columns=["patient_id", "HRD_status", "split"]).astype(float)
    y_train = merged_df[merged_df["split"]=="train"]["HRD_status"].astype(int)

    X_val = merged_df[merged_df["split"]=="validation"].drop(columns=["patient_id", "HRD_status", "split"]).astype(float)
    y_val = merged_df[merged_df["split"]=="validation"]["HRD_status"].astype(int)

    X_test = merged_df[merged_df["split"]=="test"].drop(columns=["patient_id", "HRD_status", "split"]).astype(float)
    y_test = merged_df[merged_df["split"]=="test"]["HRD_status"].astype(int)
    test_patient_ids = merged_df[merged_df["split"]=="test"]["patient_id"].tolist()

    INPUT_DIM = X_train.shape[1]

    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)
    X_val_z = scaler.transform(X_val)
    X_test_z = scaler.transform(X_test)
    X_cptac_z = scaler.transform(X_cptac)

    train_ds = make_balanced_ds(X_train_z, y_train, batch_size=BATCH_SIZE)
    val_ds = tf.data.Dataset.from_tensor_slices(
        (X_val_z, {"reconstruction": X_val_z, "hrd_pred": y_val})
    ).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    steps_per_epoch = int(np.ceil(len(X_train) / BATCH_SIZE))

    cnv_inputs = layers.Input(shape=(INPUT_DIM,), name="gene_cnv")
    x = layers.GaussianNoise(0.1)(cnv_inputs)
    x = layers.Dense(512, activation="relu",
                    kernel_regularizer=regularizers.l2(L2_WEIGHT))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT)(x)

    z_mean = layers.Dense(LATENT_DIM, name="z_mean",
                          activity_regularizer=regularizers.l1(L1_LATENT))(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    clf = layers.Dense(256, activation="relu",
                    kernel_regularizer=regularizers.l2(L2_WEIGHT))(z)
    clf = layers.Dropout(DROPOUT)(clf)
    hrd_pred = layers.Dense(1, activation="sigmoid", name="hrd_pred")(clf)

    decoder_hidden = layers.Dense(512, activation="relu")(z)
    reconstruction = layers.Dense(INPUT_DIM, name="reconstruction")(decoder_hidden)

    vae = Model(cnv_inputs, [reconstruction, hrd_pred], name="cnv_vae")

    kl_loss = -0.5 * tf.reduce_mean(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    )
    vae.add_loss(KL_WEIGHT * kl_loss)

    vae.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss={
            "reconstruction": "mse",
            "hrd_pred": "binary_crossentropy"
        },
        loss_weights={
            "reconstruction": 1.0,
            "hrd_pred": CLF_LOSS_WEIGHT
        },
        metrics={"hrd_pred": "accuracy"}
    )

    vae.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_hrd_pred_accuracy",
            mode="max",
            patience=30,
            restore_best_weights=True
        )],
        verbose=1
    )

    y_pred_test = vae.predict((X_test_z,), verbose=0)[1]

    for idx, (score, label) in enumerate(zip(y_pred_test.flatten(), y_test.values.flatten())):
        all_predictions.append({
            'fold': j,
            'patient_id': test_patient_ids[idx],
            'cohort': 'TCGA_test',
            'label': int(label),
            'prediction_score': float(score),
            'model_name': model_name
        })

    print(f"TCGA Test - {len(y_pred_test)} predictions")

    y_pred_cptac = vae.predict((X_cptac_z,), verbose=0)[1]

    for idx, (score, label) in enumerate(zip(y_pred_cptac.flatten(), y_cptac.flatten())):
        all_predictions.append({
            'fold': j,
            'patient_id': cptac_patient_ids[idx],
            'cohort': 'CPTAC_full',
            'label': int(label),
            'prediction_score': float(score),
            'model_name': model_name
        })

    print(f"CPTAC Full - {len(y_pred_cptac)} predictions")

print(f"\n{'='*60}")
print("Saving predictions...")
print(f"{'='*60}")

predictions_df = pd.DataFrame(all_predictions)

output_dir = "predictions_CNV_VAE_gistic/"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "predictions_CNV_VAE_gistic_all_folds.csv")
predictions_df.to_csv(output_path, index=False)
print(f"Saved: {output_path}")
print(f"Total predictions: {len(predictions_df)}")
print(f"Cohorts: {predictions_df['cohort'].unique().tolist()}")

for cohort in predictions_df['cohort'].unique():
    cohort_df = predictions_df[predictions_df['cohort'] == cohort]
    cohort_path = os.path.join(output_dir, f"predictions_CNV_VAE_gistic_{cohort}.csv")
    cohort_df.to_csv(cohort_path, index=False)
    print(f"Saved: {cohort_path} ({len(cohort_df)} samples)")

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
print(f"python run_evaluation_pipeline.py --input {output_path} --output-dir evaluation_results_CNV_VAE_gistic/")
