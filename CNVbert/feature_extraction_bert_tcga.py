#%%
"""
Extract CNV BERT embeddings (TCGA) - 128-dim mean-pooled.

Tokenizes the integer CNV matrix with CNVTokenizerLinear (9 linear bins) and
runs the pretrained BulkCNVBertEncoder2 (embed_dim=128) to produce per-sample
embeddings.

Output:
- embeddings/cnv_int_bert_{method}_embeddings_tcga_128.csv
"""

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from cnv_tokenizer import CNVTokenizerLinear
from bulk_cnv_bert_encoder import BulkCNVBertEncoder2
from helpers.extract_features import extract_cnv_features

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[-1], True)
    tf.config.experimental.set_visible_devices(physical_devices[-1], 'GPU')
    print(f"Using GPU: {physical_devices[-1]}")
else:
    print("No GPU available, running on CPU")

CNV_INT_PATH = ".../all_cancers_cnv_matrix_int.csv"
GENE_POS_PATH = ".../CNV_pos.csv"
WEIGHTS_PATH = ".../cnv_int_encoder_weights_128.pkl"
OUT_DIR = ".../embeddings"
os.makedirs(OUT_DIR, exist_ok=True)

METHOD = "mean"

print("Loading TCGA integer CNV...")
all_cancers_df = pd.read_csv(CNV_INT_PATH, index_col="sample_barcode")
all_cancers_df = all_cancers_df[~all_cancers_df.index.duplicated(keep="first")].copy()
all_cancers_df["bcr_patient_barcode"] = all_cancers_df.index

gene_pos_df = pd.read_csv(GENE_POS_PATH)
genes_ordered = gene_pos_df["gene"].tolist()

pos_map = dict(zip(gene_pos_df["gene"], gene_pos_df["global_bin_id"]))
shared_genes = sorted(
    [g for g in genes_ordered if g in all_cancers_df.columns],
    key=lambda g: pos_map[g],
)
print(f"Using {len(shared_genes)} genes (full set, genome-ordered)")

X_all = all_cancers_df[shared_genes].copy()
print(f"TCGA integer CNV matrix: {X_all.shape}")

gene_to_pos = dict(zip(gene_pos_df["gene"], gene_pos_df["global_bin_id"]))
pos_ids = [gene_to_pos[gene] for gene in shared_genes]
n_pos_bins = max(pos_ids) + 1

L = len(shared_genes)
SEQ_LEN = L + 1
CLS_GENE_ID = L

tokenizer = CNVTokenizerLinear(
    n_bins=9, min_cnv_value=-2.0, max_cnv_value=2.0,
    prepend_cls_token=True, reserve_mask_token=True,
)
vocab_size = tokenizer.get_vocab_size()
print(f"vocab_size={vocab_size}, CLS={tokenizer.cls_id}, MASK={tokenizer.mask_id}")

X_tokens_all = tokenizer.batch_tokenize(np.asarray(X_all))

gene_ids_row = np.concatenate([[CLS_GENE_ID], np.arange(L, dtype=np.int32)])
gene_ids_all = np.broadcast_to(gene_ids_row, X_tokens_all.shape).astype(np.int32)

print("Building encoder and loading weights...")
encoder = BulkCNVBertEncoder2(
    n_genes=SEQ_LEN, vocab_size=vocab_size, embed_dim=128,
    num_heads=4, ff_dim=512, num_layers=2,
    n_pos_bins=n_pos_bins, use_pos_enc=False, use_gene_emb=True,
)

dummy_tokens = tf.zeros((1, SEQ_LEN), dtype=tf.int32)
dummy_gene_ids = tf.constant(gene_ids_row[None, :], dtype=tf.int32)
_ = encoder(dummy_tokens, gene_ids=dummy_gene_ids, training=False)

with open(WEIGHTS_PATH, "rb") as f:
    weights = pickle.load(f)
encoder.set_weights(weights)
encoder.trainable = False
print(f"Loaded weights from {WEIGHTS_PATH}")

print("Extracting features...")
sample_names = X_all.index.tolist()
cnv_feature_df = extract_cnv_features(
    encoder, tokens=X_tokens_all, gene_ids=gene_ids_all,
    pos_ids=None, sample_names=sample_names, method=METHOD,
    batch_size=4, pad_id=0,
)

cnv_feature_df["bcr_patient_barcode"] = sample_names
cnv_feature_df = cnv_feature_df.drop_duplicates(subset="bcr_patient_barcode").set_index("bcr_patient_barcode")
out_path = f"{OUT_DIR}/cnv_int_bert_{METHOD}_embeddings_tcga_128.csv"
cnv_feature_df.to_csv(out_path)
print(f"Saved TCGA embeddings ({cnv_feature_df.shape}) -> {out_path}")
# %%
