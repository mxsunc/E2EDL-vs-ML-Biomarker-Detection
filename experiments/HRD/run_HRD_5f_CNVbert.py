import numpy as np
import pandas as pd
import tensorflow as tf
from builtins import min
from tensorflow.keras import layers, models
from sklearn.metrics import roc_curve, auc

# MLP model
def build_cnv_mlp(input_dim=512, dropout_rate=0.3):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(input_dim/2, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# features TCGA/CPTAC
input_path = ".../cnv_seq_mean_embeddings_tcga_512_all.csv"
all_cancers_df = pd.read_csv(input_path,sep=",")
all_cancers_df = all_cancers_df.set_index("bcr_patient_barcode")
all_cancers_df = all_cancers_df[~all_cancers_df.index.duplicated(keep="first")].copy()

input_path = ".../cnv_seq_mean_embeddings_cptac_512_all.csv"
all_cancers_df_cptac = pd.read_csv(input_path,sep=",")
all_cancers_df_cptac = all_cancers_df_cptac.rename(columns={"Patient ID":"PATIENT"})

df_hrd_cptac = pd.read_csv(".../cptac_HRDstatus.csv")
df_hrd_cptac = df_hrd_cptac[df_hrd_cptac["tissue"].isin(["Pancreas","Bronchus and lung","Uterus, NOS"])]
df_hrd_cptac = df_hrd_cptac.dropna(subset=["HRD_Binary_us"]).copy()
df_hrd_cptac["HRD_Binary_us"] = df_hrd_cptac["HRD_Binary_us"].map({"WT": 0, "MUT": 1})

df_merged_cptac = all_cancers_df_cptac.merge(df_hrd_cptac, on="PATIENT", how="inner")
X_cptac, y_cptac = df_merged_cptac.iloc[:, 1:-11], df_merged_cptac['HRD_Binary_us']

# 5-fold CV
for j in range(5):
    df = pd.read_csv(f".../HRDfolds{j+1}_withlungucec.csv")
    df = df.drop(columns={"BCR"})
    df = df.rename(columns={"Patient ID":"bcr_patient_barcode"})
    df = df.dropna(subset=["HRD_status"])
    df["HRD_status"] = df["HRD_status"].map({"HRD_negative": 0, "HRD_positive": 1})
    df_merged_together = df.merge(all_cancers_df, on="bcr_patient_barcode", how="right")
    train_df = df_merged_together[df_merged_together["split"]=="train"]
    train_msih = train_df[train_df["HRD_status"]==1]
    train_nonmsih  = train_df[train_df["HRD_status"]==0]
    train_resampled_nonmsih = train_nonmsih.sample(n=len(train_msih))
    train_df = pd.concat([train_msih, train_resampled_nonmsih], axis=0)
    train_data = train_df.sample(frac=1)
    val_df = df_merged_together[df_merged_together["split"]=="validation"]
    val_msih = val_df[val_df["HRD_status"]==1]
    val_nonmsih  = val_df[val_df["HRD_status"]==0]
    val_resampled_nonmsih = val_nonmsih.sample(n=len(val_msih))
    val_df = pd.concat([val_msih, val_resampled_nonmsih], axis=0)
    val_data = val_df.sample(frac=1)
    test_df = df_merged_together[df_merged_together["split"]=="test"]
    test_data = test_df.sample(frac=1)
    X_train = train_data.iloc[:, 15:]
    y_train = train_data[train_data["split"]=="train"]["HRD_status"].astype(int).values
    X_val = val_data.iloc[:, 15:]
    y_val = val_data[val_data["split"]=="validation"]["HRD_status"].astype(int).values
    X_test = test_data.iloc[:, 15:]
    y_test = test_data[test_data["split"]=="test"]["HRD_status"].astype(int).values

    INPUT_DIM = X_train.shape[1]

    model = build_cnv_mlp(input_dim=INPUT_DIM, dropout_rate=0.3)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", 
        mode="max",
        patience=50,
        restore_best_weights=True,
    )

    model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=500,
        validation_data=(X_val, y_val),
        callbacks=[callback]
    )

    y_pred = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    youden_j = tpr - fpr
    best_threshold_index = np.argmax(youden_j)
    best_threshold = thresholds[best_threshold_index]
    y_pred_binary = (y_pred >= best_threshold).astype(int)

    y_pred_cptac = model.predict(X_cptac)
    y_pred_binary_cptac = (y_pred_cptac >= best_threshold).astype(int)