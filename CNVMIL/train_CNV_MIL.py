import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve
from experiments.helpers.balance_dataset import make_balanced
from experiments.helpers.dataloading_segment_bags import importmake_dataset

bags_path = ".../tcga_cnv_mil_bags_gene_HRD.pkl"

hrd_folds_template = ".../HRDfolds{fold}_withlungucec.csv"

batch_size = 64
random_state = 42
num_folds = 5

with open(bags_path, "rb") as f:
    bags = pickle.load(f)
bag_ids = set(bags.keys())
sample_ids_all = list(bag_ids)
num_features = bags[sample_ids_all[0]]["x_cont"].shape[1]

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

    num_cont = num_features - 1
    num_chroms = 24
    d_chrom = 8
    hidden_dim = 1025
    att_hid = 512

    reg = keras.regularizers.l2(1e-3) 
    inputs_x = keras.Input(shape=(None, num_features), dtype=tf.float32, name="x_cont")
    inputs_len = keras.Input(shape=(), dtype=tf.int32, name="length")
    x_cont = inputs_x[..., :num_cont]
    chrom_idx = tf.cast(inputs_x[..., -1], tf.int32)
    chrom_emb = layers.Embedding(num_chroms, d_chrom, name="chrom_embedding",embeddings_regularizer=reg)(chrom_idx)
    x = tf.concat([x_cont, chrom_emb], axis=-1)
    h = layers.Dense(256, activation="relu", name="inst_dense1",kernel_regularizer=reg)(x)
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
    z = layers.Dense(1024, activation="relu", name="bag_dense1",kernel_regularizer=reg)(bag_emb)
    z = layers.Dense(128, activation="relu", name="bag_dense2",kernel_regularizer=reg)(z)
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

    model.fit(train_ds, validation_data=val_ds, epochs=200, verbose=1,callbacks=[callback],)
    weights_dir = ".../weights/"
    fold_weights_path = os.path.join(weights_dir, f"cnv_mil_fold{fold_idx+1}.weights.h5")
    model.save_weights(fold_weights_path)