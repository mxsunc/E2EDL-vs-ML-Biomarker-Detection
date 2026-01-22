import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve
from experiments.helpers.balance_dataset import make_balanced
from experiments.helpers.dataloading_segment_bags import importmake_dataset

hrd_folds_template = ".../HRDfolds{fold}_withlungucec.csv"

external_bags_path = "/mnt/bulk-uranus/michaela/ATGC/ATGC/CNV/CNVMIL/cptac_cnv_mil_bags_HRD.pkl"  # adjust path
weights_dir = "/mnt/bulk-uranus/michaela/ATGC/ATGC/CNV/CNVMIL/mil_models_hrd"
hrd_folds_template = "/mnt/bulk-uranus/michaela/ATGC/data/PCAWG/HRDfolds{fold}_withlungucec.csv"

batch_size = 64
random_state = 42
num_folds = 5

with open(external_bags_path, "rb") as f:
    ext_bags = pickle.load(f)

ext_ids_all = list(ext_bags.keys())
ext_num_features = ext_bags[ext_ids_all[0]]["x_cont"].shape[1]

num_features = ext_num_features

for fold_idx in range(num_folds):
    df = pd.read_csv(hrd_folds_template.format(fold=fold_idx + 1))
    df = df.drop(columns={"BCR"})
    df = df.rename(columns={"Patient ID": "bcr_patient_barcode"})
    df = df.dropna(subset=["HRD_status"])
    df["HRD_status"] = df["HRD_status"].map({"HRD_negative": 0, "HRD_positive": 1})

    train_df_raw = df[df["split"] == "train"]
    val_df_raw = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"].sample(
        frac=1.0, random_state=random_state
    )
    train_df = make_balanced(train_df_raw)
    val_df = make_balanced(val_df_raw)
    train_ids = [pid for pid in train_df["bcr_patient_barcode"].unique() if pid in bag_ids]
    val_ids = [pid for pid in val_df["bcr_patient_barcode"].unique() if pid in bag_ids]
    test_ids = [pid for pid in test_df["bcr_patient_barcode"].unique() if pid in bag_ids]

    train_ds = make_dataset(train_ids, bags, batch_size, shuffle=True)
    val_ds = make_dataset(val_ids, bags, batch_size, shuffle=False)
    test_ds = make_dataset(test_ids, bags, batch_size, shuffle=False)

    ext_ids = ext_ids_all

    num_cont = num_features - 1
    num_chroms = 24
    d_chrom = 8
    hidden_dim = 512
    att_hid = 256

    reg = keras.regularizers.l2(1e-4) 
    inputs_x = keras.Input(shape=(None, num_features), dtype=tf.float32, name="x_cont")
    inputs_len = keras.Input(shape=(), dtype=tf.int32, name="length")
    x_cont = inputs_x[..., :num_cont]
    chrom_idx = tf.cast(inputs_x[..., -1], tf.int32)
    chrom_emb = layers.Embedding(num_chroms, d_chrom, name="chrom_embedding",embeddings_regularizer=reg)(chrom_idx)
    x = tf.concat([x_cont, chrom_emb], axis=-1)
    h = layers.Dense(128, activation="relu", name="inst_dense1",kernel_regularizer=reg)(x)
    h = layers.Dense(hidden_dim, activation="relu", name="inst_dense2",kernel_regularizer=reg)(h)
    a_h = layers.Dense(att_hid, activation="tanh", name="att_dense1",kernel_regularizer=reg)(h)
    a_logits = layers.Dense(1, name="att_dense2",kernel_regularizer=reg)(a_h)
    max_len = tf.shape(a_logits)[1]
    mask = tf.sequence_mask(inputs_len, maxlen=max_len)
    mask = tf.expand_dims(mask, -1)
    padding = (1.0 - tf.cast(mask, tf.float32)) * -1e9
    a_masked = a_logits + padding
    a = tf.nn.softmax(a_masked, axis=1)
    bag_emb = tf.reduce_sum(a * h, axis=1)
    z = layers.Dense(512, activation="relu", name="bag_dense1",kernel_regularizer=reg)(bag_emb)
    z = layers.Dense(64, activation="relu", name="bag_dense2",kernel_regularizer=reg)(z)
    out = layers.Dense(1, activation="sigmoid", name="pred",kernel_regularizer=reg)(z)

    model = keras.Model(
        inputs={"x_cont": inputs_x, "length": inputs_len},
        outputs=out,
        name=f"cnv_mil_model_fold{fold_idx+1}",
    )

    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", 
        mode="max",
        patience=50,
        restore_best_weights=True,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )

    weights_path = os.path.join(weights_dir, f"cnv_mil_fold{fold_idx+1}.weights.h5")
    model.load_weights(weights_path)

    test_ds_thr = make_dataset(test_ids, bags, batch_size, shuffle=False)

    ext_ds  = make_dataset(ext_ids, ext_bags, batch_size, shuffle=False)

    y_test_pred = model.predict(test_ds_thr).ravel()
    y_test = np.array([bags[sid]["label"] for sid in test_ids], dtype=np.int32)

    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
    youden = tpr - fpr
    best_threshold = thresholds_train[np.argmax(youden)]

    y_ext_pred = model.predict(ext_ds).ravel()
    y_ext = np.array([ext_bags[sid]["label"] for sid in ext_ids], dtype=np.int32)
    y_ext_bin = (y_ext_pred >= best_threshold).astype(int)