#%%
"""
CNV Autoencoder (GISTIC genes) - Integrated Gradients Explainability (Revision)

Trains the AE (same architecture/hyperparameters as run_HRD_5f_CNV_AE.py) on CNV
data, then attributes HRD predictions to input genes via Integrated Gradients and
produces gene- and chromosome-level importance plots.

Uses:
- GISTIC genes as input (integer CNV matrix)
- 5-fold CV splits from HRDfolds{j}.csv (explainability run on fold 1)
- scarHRD labels for CPTAC evaluation

Output:
- Per-gene IG importance + Manhattan / chromosome distribution plots
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

import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model, Input

LATENT_DIM = 256
DROPOUT = 0.2
L2_WEIGHT = 1e-4
L1_LATENT = 0.0
NOISE_STD = 0.1
BATCH_SIZE = 64
EPOCHS = 200
CLF_LOSS_WEIGHT = 3.0
LEARNING_RATE = 5e-4
AUTOTUNE = tf.data.AUTOTUNE

def asfloat32(x):
    return np.asarray(x, dtype=np.float32)

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

def _ig_batch(model, x_batch, baseline, steps=64):
    alphas = tf.linspace(0.0, 1.0, steps)[:, None, None]
    x_diff = x_batch - baseline
    x_interp = baseline + alphas * x_diff
    x_interp = tf.reshape(x_interp, [-1, x_batch.shape[-1]])

    with tf.GradientTape() as tape:
        tape.watch(x_interp)
        preds = model(x_interp, training=False)
    grads = tape.gradient(preds, x_interp)
    grads = tf.reshape(grads, [tf.shape(alphas)[0], -1, x_batch.shape[-1]])
    avg_grads = tf.reduce_mean(grads, axis=0)
    ig = x_diff * avg_grads
    return ig

def integrated_gradients(model, X, baseline, steps=64, batch_size=128):
    X = X.astype(np.float32)
    out = np.empty_like(X, dtype=np.float32)
    for i in range(0, X.shape[0], batch_size):
        xb = tf.convert_to_tensor(X[i:i+batch_size])
        igb = _ig_batch(model, xb, tf.convert_to_tensor(baseline), steps=steps)
        out[i:i+batch_size] = igb.numpy()
    return out

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
X_cptac_df = merged_cptac.drop(columns=["sample_barcode", "HRD_Binary_us"]).astype(float)
y_cptac = merged_cptac["HRD_Binary_us"]
cptac_patient_ids = merged_cptac["sample_barcode"].tolist()
X_cptac = asfloat32(X_cptac_df.values)

print(f"CPTAC samples: {len(X_cptac)}, HRD+: {int(y_cptac.sum())}")

gene_pos_df = pd.read_csv(".../CNV_positions.csv")
genes_ordered = gene_pos_df["gene"].tolist()

for j in range(1):
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

    steps_per_epoch = math.ceil(len(X_train) / BATCH_SIZE)

    # Build AE model (same architecture as run_HRD_5f_CNV_AE.py)
    inputs = Input(shape=(INPUT_DIM,), name="gene_cnv")
    x = tf.keras.layers.GaussianNoise(NOISE_STD)(inputs)
    encoded = tf.keras.layers.Dense(
        LATENT_DIM, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT),
        activity_regularizer=tf.keras.regularizers.l1(L1_LATENT),
        name="encoded_dense"
    )(x)
    encoded = tf.keras.layers.BatchNormalization(name="encoded_bn")(encoded)
    encoded = tf.keras.layers.Dropout(DROPOUT, name="encoded")(encoded)
    decoded = tf.keras.layers.Dense(INPUT_DIM, name="reconstruction")(encoded)

    # Classifier head (extra depth: latent//2 -> latent//4 -> 1, per sweep v3 winning config)
    clf_intermediate = tf.keras.layers.Dense(
        int(LATENT_DIM/2), activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT),
        name="clf_intermediate"
    )(encoded)
    clf_intermediate = tf.keras.layers.BatchNormalization(name="clf_bn")(clf_intermediate)
    clf_intermediate = tf.keras.layers.Dropout(DROPOUT, name="clf_dropout")(clf_intermediate)
    clf_intermediate = tf.keras.layers.Dense(
        int(LATENT_DIM/4), activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(L2_WEIGHT),
        name="clf_intermediate2"
    )(clf_intermediate)
    clf_intermediate = tf.keras.layers.Dropout(DROPOUT, name="clf_dropout2")(clf_intermediate)
    clf_out = tf.keras.layers.Dense(1, activation="sigmoid", name="hrd_pred")(clf_intermediate)

    model = Model(inputs=inputs, outputs=[decoded, clf_out])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss={"reconstruction": "mse", "hrd_pred": "binary_crossentropy"},
        loss_weights={"reconstruction": 1.0, "hrd_pred": CLF_LOSS_WEIGHT},
        metrics={"hrd_pred": ["accuracy"]}
    )

    es_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_hrd_pred_accuracy", mode="max",
        patience=30, restore_best_weights=True,
    )
    model.fit(train_ds, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
              validation_data=val_ds, callbacks=[es_cb], verbose=1)

    # ==========================================================================
    # Integrated Gradients attribution
    # ==========================================================================
    clf_model = Model(inputs=model.input,
                      outputs=model.get_layer("hrd_pred").output,
                      name="clf_only")

    X_test_np = np.asarray(X_test_z, dtype=np.float32)
    # Baseline: mean CNV (better than zeros)
    baseline = X_test_np.mean(axis=0, keepdims=True).astype(np.float32)

    ig_vals = integrated_gradients(clf_model, X_test_np, baseline, steps=64, batch_size=128)
    mean_abs_ig = np.mean(np.abs(ig_vals), axis=0)

# %%
# Top-30 genes by mean absolute IG
feature_names = X_train.columns.tolist()
mean_abs_ig = np.mean(np.abs(ig_vals), axis=0)
k = 30
order = np.argsort(-mean_abs_ig)[:k]
genes_top = np.array(feature_names)[order]
vals_top  = mean_abs_ig[order]

plt.figure(figsize=(8, 10))
plt.barh(range(k), vals_top)
plt.yticks(range(k), genes_top)
plt.gca().invert_yaxis()
plt.xlabel("Mean |IG| across test samples")
plt.title("Global gene importance (Integrated Gradients)")
plt.tight_layout()
plt.show()

# %%
# Merge gene IG importance with genomic positions and sum importance per chromosome
importance_df = pd.DataFrame({
    "gene": feature_names,
    "mean_abs_IG": np.mean(np.abs(ig_vals), axis=0)
})
importance_df["gene"] = importance_df["gene"].astype(str)
gene_pos_df["gene"]   = gene_pos_df["gene"].astype(str)
merged = importance_df.merge(gene_pos_df, on="gene", how="left")
merged = merged.sort_values("mean_abs_IG", ascending=False)
chr_summary = (
    merged.groupby("chr")["mean_abs_IG"]
    .sum()
    .reset_index()
    .sort_values("mean_abs_IG", ascending=False)
)

plt.figure(figsize=(10,5))
plt.bar(chr_summary["chr"].astype(str), chr_summary["mean_abs_IG"])
plt.xlabel("Chromosome")
plt.ylabel("Total |IG| importance")
plt.title("Chromosomal distribution of HRD-predictive signal")
plt.tight_layout()
plt.show()

# %%
# Manhattan plot of top genes using cumulative genomic positions and log10(|IG|)
top50_df = merged.head(200)
chr_map = {str(i): i for i in range(1,23)}
chr_map.update({"X": 23, "Y": 24, "x":23, "y":24})
for df in (top50_df, gene_pos_df):
    df["chr"] = df["chr"].astype(str)
    df["chr_num"] = df["chr"].map(chr_map)

chrom_lengths = (gene_pos_df
                 .dropna(subset=["chr_num","start"])
                 .groupby("chr_num")["start"]
                 .max()
                 .sort_index())

chrom_offsets = chrom_lengths.cumsum() - chrom_lengths  # 1st chr gets 0
chrom_offsets = chrom_offsets.astype(np.int64)

t50 = top50_df.dropna(subset=["chr_num","start","mean_abs_IG"]).copy()
t50["cum_pos"] = t50.apply(lambda r: int(r["start"]) + int(chrom_offsets.loc[int(r["chr_num"])]), axis=1)
t50["log10_IG"] = np.log10(t50["mean_abs_IG"] + 1e-12)

bounds = []
running = 0
for chr_num, length in chrom_lengths.items():
    start_bp = running
    end_bp   = running + int(length)
    bounds.append((chr_num, start_bp, end_bp))
    running = end_bp

tick_pos  = [ (s+e)//2 for (_,s,e) in bounds ]
tick_label = [ ("X" if c==23 else "Y" if c==24 else str(int(c))) for (c,_,_) in bounds ]

colors = {c: ("#4C72B0" if (int(c)%2)==0 else "#55A868") for (c,_,_) in bounds}
t50["color"] = t50["chr_num"].apply(lambda c: colors[int(c)])

plt.figure(figsize=(14,6))
plt.scatter(t50["cum_pos"], t50["log10_IG"], s=60, c=t50["color"], edgecolor="k", linewidth=0.5)
plt.ylabel("log₁₀(|IG|)")
plt.title("Manhattan plot — top 50 HRD-predictive genes")
for _, s, e in bounds:
    plt.axvline(s, color="lightgray", lw=0.8, ls="--", alpha=0.6)
plt.xticks(tick_pos, tick_label)
plt.xlabel("Chromosome")
plt.tight_layout()
plt.show()

# %%
# Manhattan plot for genes above 95th-percentile IG threshold
thr = importance_df["mean_abs_IG"].quantile(0.95)
topQ = importance_df[importance_df["mean_abs_IG"] >= thr].copy()
topQ["gene"] = topQ["gene"].astype(str)
gene_pos_df["gene"] = gene_pos_df["gene"].astype(str)
topQ = topQ.merge(gene_pos_df[["gene","chr","start"]], on="gene", how="left")

chr_map = {str(i): i for i in range(1,23)}
chr_map.update({"X":23, "Y":24, "x":23, "y":24})
for df in (topQ, gene_pos_df):
    df["chr"] = df["chr"].astype(str)
    df["chr_num"] = df["chr"].map(chr_map)

topQ = topQ.dropna(subset=["chr_num","start"]).copy()
chrom_lengths = (gene_pos_df.dropna(subset=["chr_num","start"])
                 .groupby("chr_num")["start"].max().sort_index())
chrom_offsets = (chrom_lengths.cumsum() - chrom_lengths).astype(np.int64)
topQ["cum_pos"] = topQ.apply(lambda r: int(r["start"]) + int(chrom_offsets.loc[int(r["chr_num"])]), axis=1)
topQ["log10_IG"] = np.log10(topQ["mean_abs_IG"] + 1e-12)

bounds, run = [], 0
for c, L in chrom_lengths.items():
    s, e = run, run + int(L)
    bounds.append((int(c), s, e))
    run = e
tick_pos  = [(s+e)//2 for (_,s,e) in bounds]
tick_lab  = [("X" if c==23 else "Y" if c==24 else str(c)) for (c,_,_) in bounds]
colors = {c: ("#BC1F2C" if c % 2 == 0 else "#185C86") for (c,_,_) in bounds}
topQ["color"] = topQ["chr_num"].astype(int).map(colors)

plt.figure(figsize=(10,3))
plt.scatter(topQ["cum_pos"], topQ["log10_IG"], s=18, alpha=0.8,
            c=topQ["color"], edgecolor="none")
for _, s, _e in bounds:
    plt.axvline(s, color="lightgray", lw=0.6, ls="--", alpha=0.5)
plt.xticks(tick_pos, tick_lab)
plt.xlabel("Chromosome")
plt.ylabel("log$_{10}$(|Integrated Gradient|)")
plt.xlim([-50000000,3128232943])
plt.title(f"Manhattan plot — genes in top quartile (≥ {thr:.3g})")
plt.tight_layout()
plt.show()

# %%
# Count unique top-percentile genes per chromosome
thr = importance_df["mean_abs_IG"].quantile(0.95)
topQ = importance_df[importance_df["mean_abs_IG"] >= thr].copy()
pos_uniq = gene_pos_df.drop_duplicates(subset=["gene"])
topQ_pos = topQ.merge(pos_uniq[["gene","chr"]], on="gene", how="left").dropna(subset=["chr"])
chr_counts = (topQ_pos.groupby("chr")["gene"]
              .nunique()
              .sort_values(ascending=False)
              .rename("n_top_genes")
              .reset_index())
chr_counts["normalized"] = 1 - ((chr_counts["n_top_genes"] - chr_counts["n_top_genes"].min()) / (chr_counts["n_top_genes"].max() - chr_counts["n_top_genes"].min()))

# %%
# Per-chromosome top-gene counts, normalize by assayed genes and chromosome length
thr = importance_df["mean_abs_IG"].quantile(0.95)
topQ = importance_df[importance_df["mean_abs_IG"] >= thr].copy()
pos_uniq = gene_pos_df.drop_duplicates(subset=["gene"]).copy()
topQ_pos = (topQ.merge(pos_uniq[["gene","chr","start"]], on="gene", how="left")
                 .dropna(subset=["chr"]))
chr_counts = (topQ_pos.groupby("chr")["gene"]
              .nunique()
              .rename("n_top_genes")
              .reset_index())

assayed_per_chr = (pos_uniq.dropna(subset=["chr"])
                   .groupby("chr")["gene"]
                   .nunique()
                   .rename("n_assayed")
                   .reset_index())

chr_len = (pos_uniq.dropna(subset=["chr","start"])
           .groupby("chr")["start"].max()
           .rename("chr_len_bp")
           .reset_index())
chr_len["chr_len_Mb"] = chr_len["chr_len_bp"] / 1e6

out = (chr_counts.merge(assayed_per_chr, on="chr", how="left")
               .merge(chr_len, on="chr", how="left"))

out["n_assayed"]  = out["n_assayed"].replace(0, np.nan)
out["chr_len_Mb"] = out["chr_len_Mb"].replace(0, np.nan)
out["top_rate"] = out["n_top_genes"] / out["n_assayed"]
out["top_per_Mb"] = out["n_top_genes"] / out["chr_len_Mb"]

def scale_05_1(x):
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
        return 1 - ((x - xmin) / (xmax - xmin))

out["top_rate_scaled"]  = scale_05_1(out["top_rate"].values)
out["top_perMb_scaled"] = scale_05_1(out["top_per_Mb"].values)

out["chr"] = out["chr"].astype(str).str.replace("^chr", "", regex=True)
chr_order = [str(i) for i in range(1,23)] + ["X","Y"]
out["chr_cat"] = pd.Categorical(out["chr"], categories=chr_order, ordered=True)
out = out.sort_values("chr_cat").drop(columns="chr_cat").reset_index(drop=True)

rank_by_count   = out.sort_values("n_top_genes", ascending=False)
rank_by_rate    = out.sort_values("top_rate", ascending=False)
rank_by_density = out.sort_values("top_per_Mb", ascending=False)

print("Ranked by raw count:\n", rank_by_count[["chr","n_top_genes"]])
print("\nRanked by coverage-normalized rate:\n", rank_by_rate[["chr","top_rate"]])
print("\nRanked by density per Mb:\n", rank_by_density[["chr","top_per_Mb"]])
# %%
