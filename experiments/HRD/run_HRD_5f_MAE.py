import math
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_curve
from tensorflow.keras import Model, Input
from tensorflow.keras import layers, regularizers
from experiments.helpers.align_dfs import align_dfB_to_dfA
from experiments.helpers.balance_dataset import make_balanced_ds
from AE.MAE import MaskedCNVModel

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[-1], True)
tf.config.experimental.set_visible_devices(physical_devices[-1], 'GPU')

def asfloat32(x):
    return np.asarray(x, dtype=np.float32)

RANDOM_STATE = 42 
LATENT_DIM  = 512
DROPOUT     = 0.3
L2_WEIGHT   = 1e-4
BATCH_SIZE  = 64
EPOCHS      = 200
AUTOTUNE    = tf.data.AUTOTUNE

input_path = ".../tcga_cnv_seg_matrix.tsv"
all_cancers_df = pd.read_csv(input_path, sep="\t")
all_cancers_df = all_cancers_df.set_index("patient_id")
all_cancers_df = all_cancers_df[~all_cancers_df.index.duplicated(keep="first")].copy()
all_cancers_df["bcr_patient_barcode"] = all_cancers_df.index

gene_pos_df = pd.read_csv(".../CNV_pos_embeddings_100k.csv")

chrom_order = {**{str(i): i for i in range(1,23)}, "X": 23, "Y": 24, "MT": 25}
genes_ordered = gene_pos_df["gene"].tolist()

input_path = ".../cptac_cnv_seg_matrix.tsv"
all_cancers_df_cptac = pd.read_csv(input_path, sep="\t")

df_hrd_cptac = pd.read_csv(".../cptac_HRDstatus.csv")
df_hrd_cptac = df_hrd_cptac[df_hrd_cptac["tissue"].isin(["Pancreas","Bronchus and lung","Uterus, NOS"])]
df_hrd_cptac = df_hrd_cptac.dropna(subset=["HRD_Binary_us"]).copy()
df_hrd_cptac["HRD_Binary_us"] = df_hrd_cptac["HRD_Binary_us"].map({"WT": 0, "MUT": 1})

df_merged_cptac = all_cancers_df_cptac.merge(df_hrd_cptac, on="PATIENT", how="inner")
X_cptac_df, y_cptac = df_merged_cptac.iloc[:, 1:-10], df_merged_cptac['HRD_Binary_us']
X_cptac = asfloat32(X_cptac_df.values)

for j in range(5):
    df = pd.read_csv(f"/mnt/bulk-uranus/michaela/ATGC/data/PCAWG/HRDfolds{j+1}_withlungucec.csv")
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
    X_train = train_data.iloc[:, 14:-1]
    y_train = train_data[train_data["split"]=="train"]["HRD_status"].astype(int).values

    shared = [g for g in genes_ordered if g in X_train.columns]
    X_train = X_train.loc[:, shared].copy()
    X_val = val_data.iloc[:, 14:-1]
    y_val = val_data[val_data["split"]=="validation"]["HRD_status"].astype(int).values
    X_val = X_val.loc[:, shared].copy()

    X_test = test_data.iloc[:, 14:-1]
    y_test = test_data[test_data["split"]=="test"]["HRD_status"].astype(int).values
    X_test = X_test.loc[:, shared].copy()

    X_cptac = align_dfB_to_dfA(X_train,X_cptac_df)
    INPUT_DIM = X_train.shape[1]

    train_ds = make_balanced_ds(X_train, y_train, batch_size=BATCH_SIZE)

    val_ds = (tf.data.Dataset.from_tensor_slices(
                (X_val, {"reconstruction": X_val, "hrd_pred": y_val}))
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))
    
    steps_per_epoch = math.ceil(len(X_train) / BATCH_SIZE)

    masked_cnv = MaskedCNVModel(
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        mask_ratio=0.3,
    )

    masked_cnv.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics={"hrd_pred": tf.keras.metrics.BinaryAccuracy(name="accuracy")}
    )

    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_clf_loss", 
        mode="min",
        patience=30,
        restore_best_weights=True,
    )

    x_b, y_b = next(iter(train_ds))
    y_batch  = y_b["hrd_pred"].numpy().ravel()

    raw_pred = masked_cnv.predict(x_b, batch_size=64)[1].ravel()

    initial_acc = masked_cnv.evaluate(
        train_ds.take(steps_per_epoch),
        return_dict=True, verbose=0
    )["hrd_pred_accuracy"]

    history = masked_cnv.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        callbacks=callback
    )

    cnv_input = tf.keras.Input(shape=(INPUT_DIM,), name="cnv_input")
    recon, pred, mask = masked_cnv(cnv_input, training=False)
    encoder_out = masked_cnv.get_layer("encoded")(
        masked_cnv.get_layer("batch_normalization_1")(
        masked_cnv.get_layer("encoded_dense")(cnv_input), 
        training=False
        ), 
        training=False
    )
    cnv_encoder = tf.keras.Model(cnv_input, encoder_out, name="cnv_encoder")

    model.save_weights(f".../weights/fold{j+1}_mae_full_weights_seg_512.h5")

   encoder = Model(
        inputs=vae.input,
        outputs=vae.get_layer("encoded").output,
        name="cnv_encoder"
    )

    z_mean_layer = vae.get_layer("z_mean")
    latent_dim   = z_mean_layer.output_shape[-1]

    latent_in = tf.keras.Input(shape=(latent_dim,), name="latent_input")
    preclf = vae.get_layer("clf_intermediate")(latent_in)
    preclassifier = tf.keras.Model(inputs=latent_in, outputs=preclf,
                                name="preclassifier_intermediate")

    encoder.save_weights(f".../weights/fold{j+1}_mae_encoder_weights_seg_512.h5")
    preclassifier.save_weights(f".../weights/fold{j+1}_mae_preclassifier_weights_seg_512.h5")

    recons, y_pred, masks = masked_cnv.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    youden_j = tpr - fpr
    best_threshold_index = np.argmax(youden_j)
    best_threshold = thresholds[best_threshold_index]
    y_pred_binary = (y_pred >= best_threshold).astype(int)

    recons, y_pred, masks = masked_cnv.predict(X_cptac)
    y_pred_binary = (y_pred >= best_threshold).astype(int)
