#%%
"""
CNV Masked Autoencoder (MAE) - 5-Fold CV Prediction Export (Revision)

Trains a MAE on CNV data and exports predictions in a standardized format
for comparison with fusion models.

Uses:
- GISTIC genes as input
- 5-fold CV splits from HRDfolds{j}.csv
- scarHRD labels for CPTAC evaluation

Output:
- predictions_CNV_MAE_all_folds.csv: Per-patient predictions for all cohorts
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
import math
from tensorflow.keras import layers, regularizers, Model

AUTOTUNE = tf.data.AUTOTUNE

def asfloat32(x):
    return np.asarray(x, dtype=np.float32)

def align_dfB_to_dfA(dfA: pd.DataFrame, dfB: pd.DataFrame) -> pd.DataFrame:
    """Align columns of dfB to match dfA column order, interpolating missing columns."""
    cols = list(dfA.columns)
    out = dfB.copy()

    for i, col in enumerate(cols):
        if col not in out.columns:
            left = out[cols[i-1]] if i > 0 and cols[i-1] in out.columns else None
            right = out[cols[i+1]] if i < len(cols)-1 and cols[i+1] in out.columns else None

            if left is not None and right is not None:
                l = pd.to_numeric(left, errors='coerce')
                r = pd.to_numeric(right, errors='coerce')
                newcol = (l + r) / 2
                newcol = newcol.combine_first(left).combine_first(right)
            elif left is not None:
                newcol = left
            elif right is not None:
                newcol = right
            else:
                newcol = pd.Series(np.nan, index=out.index)
            out[col] = newcol

    out = out.reindex(columns=cols)
    return out

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

class MaskedCNVModel(tf.keras.Model):
    def __init__(
        self,
        input_dim=21559,
        latent_dim=512,
        mask_ratio=0.3,
        l2_weight=1e-4,
        dropout=0.3,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio

        self.noise = layers.GaussianNoise(0.1)
        self.enc_dense = layers.Dense(
            latent_dim, activation="relu",
            kernel_regularizer=regularizers.l2(l2_weight),
            activity_regularizer=regularizers.l1(L1_LATENT),
            name="encoded_dense"
        )
        self.enc_bn    = layers.BatchNormalization(name="encoded_bn")
        self.enc_drop  = layers.Dropout(dropout, name="encoded")

        self.dec_dense = layers.Dense(input_dim, name="reconstruction")

        self.clf_int   = layers.Dense(
            latent_dim//2, activation="relu",
            kernel_regularizer=regularizers.l2(l2_weight),
            name="clf_intermediate"
        )
        self.clf_drop  = layers.Dropout(dropout)
        self.clf_out   = layers.Dense(1, activation="sigmoid", name="hrd_pred")

        self.recon_loss_fn = tf.keras.losses.MeanSquaredError(reduction="none")
        self.clf_loss_fn  = tf.keras.losses.BinaryCrossentropy()

        self.acc_metric = tf.keras.metrics.BinaryAccuracy(name="accuracy")

    def call(self, inputs, training=False):
        if training:
            mask = tf.cast(
                tf.random.uniform(tf.shape(inputs)) > self.mask_ratio,
                tf.float32
            )
        else:
            mask = tf.ones_like(inputs)

        x_masked = inputs * mask
        x = self.noise(x_masked, training=training)

        z = self.enc_dense(x)
        z = self.enc_bn(z, training=training)
        z = self.enc_drop(z, training=training)

        recon = self.dec_dense(z)

        c = self.clf_int(z)
        c = self.clf_drop(c, training=training)
        pred = self.clf_out(c)

        return recon, pred, mask

    def train_step(self, data):
        X, y = data
        X        = tf.cast(X,        tf.float32)
        y_recon  = tf.cast(y["reconstruction"], tf.float32)
        y_class  = tf.cast(y["hrd_pred"],       tf.float32)

        with tf.GradientTape() as tape:
            recon, pred, mask = self(X, training=True)

            visible    = 1.0 - mask
            sq_err     = tf.square(y_recon - recon)
            masked_mse = tf.reduce_sum(sq_err * visible) / tf.reduce_sum(visible)

            clf_loss   = self.clf_loss_fn(y_class, pred)
            total_loss = masked_mse + CLF_LOSS_WEIGHT * clf_loss + sum(self.losses)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.acc_metric.update_state(y_class, pred)

        return {
            "loss":       total_loss,
            "masked_mse": masked_mse,
            "clf_loss":   clf_loss,
            "accuracy":   self.acc_metric.result(),
        }

    def test_step(self, data):
        X, y = data
        X        = tf.cast(X,        tf.float32)
        y_recon  = tf.cast(y["reconstruction"], tf.float32)
        y_class  = tf.cast(y["hrd_pred"],       tf.float32)

        recon, pred, mask = self(X, training=False)
        full_mse = tf.reduce_mean(tf.square(y_recon - recon))
        clf_loss = self.clf_loss_fn(y_class, pred)
        total    = full_mse + CLF_LOSS_WEIGHT * clf_loss + sum(self.losses)

        self.acc_metric.update_state(y_class, pred)

        return {
            "loss":     total,
            "full_mse": full_mse,
            "clf_loss": clf_loss,
            "accuracy": self.acc_metric.result(),
        }


LATENT_DIM = 256
DROPOUT = 0.3
L2_WEIGHT = 1e-4
L1_LATENT = 1e-6
BATCH_SIZE = 64
EPOCHS = 200
CLF_LOSS_WEIGHT = 3.0
LEARNING_RATE = 1e-4

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
df_hrd_cptac = df_hrd_cptac[["case_id", "HRD_binary"]].dropna(subset=["HRD_binary"]).drop_duplicates(subset="case_id", keep="first")
df_hrd_cptac["HRD_Binary_us"] = df_hrd_cptac["HRD_binary"].astype(int)

y_label_cptac_df = df_hrd_cptac[["case_id", "HRD_Binary_us"]].rename(columns={"case_id": "sample_barcode"})
merged_cptac = all_cancers_df_cptac.merge(y_label_cptac_df, left_index=True, right_on="sample_barcode", how="inner")
X_cptac_df = merged_cptac.drop(columns=["sample_barcode", "HRD_Binary_us"]).astype(float)
y_cptac = merged_cptac["HRD_Binary_us"]
cptac_patient_ids = merged_cptac["sample_barcode"].tolist()
X_cptac = asfloat32(X_cptac_df.values)

print(f"CPTAC samples: {len(X_cptac)}, HRD+: {int(y_cptac.sum())}")

all_predictions = []
model_name = 'CNV_MAE_gistic'

for j in range(5):
    print(f"\n{'='*60}")
    print(f"Fold {j+1}/5")
    print(f"{'='*60}")

    df = pd.read_csv(f".../HRDfolds{j+1}.csv")
    df = df.drop(columns={"BCR"})
    df = df.rename(columns={"Patient ID":"patient_id"})
    df = df.dropna(subset=["HRD_binary_paper"])
    df["HRD_status"] = df["HRD_binary_paper"].map({"HRD_negative": 0, "HRD_positive": 1})

    common_ids = set(all_cancers_df.index).intersection(df["patient_id"])
    all_cancers_df_fold = all_cancers_df.loc[all_cancers_df.index.isin(common_ids)]
    df_fold = df[df["patient_id"].isin(common_ids)]

    y_label_df = df_fold[["patient_id", "HRD_status", "split"]]
    merged_df = all_cancers_df_fold.merge(y_label_df, left_index=True, right_on="patient_id", how="inner")

    X_train = merged_df[merged_df["split"]=="train"].drop(columns=["patient_id", "HRD_status", "split"]).astype(float)
    y_train = merged_df[merged_df["split"]=="train"]["HRD_status"].astype(int).values
    X_val = merged_df[merged_df["split"]=="validation"].drop(columns=["patient_id", "HRD_status", "split"]).astype(float)
    y_val = merged_df[merged_df["split"]=="validation"]["HRD_status"].astype(int).values
    X_test = merged_df[merged_df["split"]=="test"].drop(columns=["patient_id", "HRD_status", "split"]).astype(float)
    y_test = merged_df[merged_df["split"]=="test"]["HRD_status"].astype(int).values
    test_patient_ids = merged_df[merged_df["split"]=="test"]["patient_id"].tolist()

    X_cptac_fold_df = X_cptac_df.reindex(columns=X_train.columns)
    X_cptac_fold = X_cptac_fold_df.values

    INPUT_DIM = X_train.shape[1]
    print(f"Input dimension: {INPUT_DIM}")

    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)
    X_val_z = scaler.transform(X_val)
    X_test_z = scaler.transform(X_test)
    X_cptac_z = scaler.transform(X_cptac_fold)

    train_ds = make_balanced_ds(X_train_z, y_train, batch_size=BATCH_SIZE)
    val_ds = tf.data.Dataset.from_tensor_slices(
        (X_val_z, {"reconstruction": X_val_z, "hrd_pred": y_val})
    ).batch(BATCH_SIZE).prefetch(AUTOTUNE)

    steps_per_epoch = math.ceil(len(X_train) / BATCH_SIZE)

    masked_cnv = MaskedCNVModel(
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        mask_ratio=0.3,
        dropout=DROPOUT,
    )

    masked_cnv.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        metrics={"hrd_pred": tf.keras.metrics.BinaryAccuracy(name="accuracy")}
    )

    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_clf_loss",
        mode="min",
        patience=30,
        restore_best_weights=True,
    )

    masked_cnv.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        callbacks=[callback],
        verbose=1
    )

    recons, y_pred_test, masks = masked_cnv.predict(X_test_z, verbose=0)

    for idx, (score, label) in enumerate(zip(y_pred_test.flatten(), y_test.flatten())):
        all_predictions.append({
            'fold': j,
            'patient_id': test_patient_ids[idx],
            'cohort': 'TCGA_test',
            'label': int(label),
            'prediction_score': float(score),
            'model_name': model_name
        })

    print(f"TCGA Test - {len(y_pred_test)} predictions")

    recons, y_pred_cptac, masks = masked_cnv.predict(X_cptac_z, verbose=0)

    for idx, (score, label) in enumerate(zip(y_pred_cptac.flatten(), y_cptac.values.flatten())):
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

output_dir = "predictions_CNV_MAE_gistic/"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "predictions_CNV_MAE_gistic_all_folds.csv")
predictions_df.to_csv(output_path, index=False)
print(f"Saved: {output_path}")
print(f"Total predictions: {len(predictions_df)}")
print(f"Cohorts: {predictions_df['cohort'].unique().tolist()}")

for cohort in predictions_df['cohort'].unique():
    cohort_df = predictions_df[predictions_df['cohort'] == cohort]
    cohort_path = os.path.join(output_dir, f"predictions_CNV_MAE_gistic_{cohort}.csv")
    cohort_df.to_csv(cohort_path, index=False)
    print(f"Saved: {cohort_path} ({len(cohort_df)} samples)")

print(f"\n{'='*60}")
print("Prediction Summary")
print(f"{'='*60}")
for cohort in predictions_df['cohort'].unique():
    cohort_df = predictions_df[predictions_df['cohort'] == cohort]
    n_pos = int(cohort_df['label'].sum())
    n_neg = len(cohort_df) - n_pos
    print(f"  {cohort}: {len(cohort_df)} samples (HRD+: {n_pos}, HRD-: {n_neg})")

print(f"\n{'='*60}")
print("Next Step: Run evaluation pipeline")
print(f"{'='*60}")
print(f"python run_evaluation_pipeline.py --input {output_path} --output-dir evaluation_results_CNV_MAE_gistic/")
