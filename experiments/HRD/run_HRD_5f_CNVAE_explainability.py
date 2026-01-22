#%%
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from tensorflow.keras import Model, Input
from experiments.helpers.align_dfs import align_dfB_to_dfA
from experiments.helpers.balance_dataset import make_balanced_ds
from AE.AE import CNVAE

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[-1], True)
tf.config.experimental.set_visible_devices(physical_devices[-1], 'GPU')

def asfloat32(x):
    return np.asarray(x, dtype=np.float32)

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

for j in range(1):
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
                (X_val, {"reconstruction": X_vaL, "hrd_pred": y_val}))
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))

    steps_per_epoch = math.ceil(len(X_train) / BATCH_SIZE)

    cnv = CNVAE(INPUT_DIM, LATENT_DIM, DROPOUT, L2_WEIGHT)
    model = cnv.model

    x_b, y_b = next(iter(train_ds))
    y_batch  = y_b["hrd_pred"].numpy().ravel()
    raw_pred = model.predict(x_b, batch_size=64)[1].ravel()

    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_hrd_pred_accuracy",
        mode="max",
        patience=30,
        restore_best_weights=True,
    )

    initial_acc = model.evaluate(
        train_ds.take(steps_per_epoch),
        return_dict=True, verbose=0
    )["hrd_pred_accuracy"]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        callbacks=[callback],
        verbose=2,
    )

    latent_dim = model.get_layer("encoded").output_shape[-1]
    latent_in = Input(shape=(latent_dim,), name="latent_input")
    clf_intermediate_layer = model.get_layer("clf_intermediate")
    preclf = clf_intermediate_layer(latent_in)

    clf_model = Model(inputs=model.input,
                    outputs=model.get_layer("hrd_pred").output,
                    name="clf_only")

    X_train_np = np.asarray(X_train, dtype=np.float32)
    X_test_np  = np.asarray(X_test,    dtype=np.float32)

    baseline = X_train_np.mean(axis=0, keepdims=True).astype(np.float32)
    ig_vals = integrated_gradients(clf_model, X_test_np, baseline, steps=64, batch_size=128)
    mean_abs_ig = np.mean(np.abs(ig_vals), axis=0)

# %%
# compute top-30 genes by mean absolute IG
feature_names= X_train.columns.tolist()
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
# merge gene IG importance with genomic positions and sum importance per chromosome
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
# create Manhattan plot of top genes using cumulative genomic positions and log10(|IG|)
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
plt.ylabel("log\u2081\u2080(|IG|)")
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
# count unique top-percentile genes per chromosome
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
# per-chromosome top-gene counts, normalize by assayed genes and chromosome length
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
