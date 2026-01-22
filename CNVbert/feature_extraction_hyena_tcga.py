import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
from cnv_tokenizer import CNVTokenizer2 as CNVTokenizer
from bulk_cnv_hyena_encoder import BulkCNVHyenaEncoder
from helpers.extract_features_hyena import extract_cnv_features_hyena
from helpers.pretrain_utils import apply_fixed_mask

# load data
input_path = ".../tcga_cnv_seg_matrix.tsv"
all_cancers_df = pd.read_csv(input_path, sep="\t")
all_cancers_df = all_cancers_df.set_index("patient_id")
all_cancers_df = all_cancers_df[~all_cancers_df.index.duplicated(keep="first")].copy()
all_cancers_df["bcr_patient_barcode"] = all_cancers_df.index

# load gene positions
gene_pos_df = pd.read_csv(".../CNV_pos_embeddings_100k.csv")
chrom_order = {**{str(i): i for i in range(1,23)}, "X": 23, "Y": 24, "MT": 25}
genes_ordered = gene_pos_df["gene"].tolist()

# sort columns by gene positions
shared = [g for g in genes_ordered if g in all_cancers_df.columns]
X_all = X_all.loc[:, shared].copy()

gene_to_pos = dict(zip(gene_pos_df["gene"], gene_pos_df["global_bin_id"]))
pos_ids = [gene_to_pos[gene] for gene in shared]
n_pos_bins = max(pos_ids) + 1

gene_to_pos = dict(zip(gene_pos_df["gene"], gene_pos_df["global_bin_id"]))
pos_ids = [gene_to_pos[gene] for gene in shared]
n_pos_bins = max(pos_ids) + 1
n_genes = len(shared)

# set parameters
N_BINS = 64
L = len(X_all.columns) 
SEQ_LEN = len(X_all.columns) +1
BATCH_SIZE = 1
EPOCHS = 10
vocab_size = 67
L_real  = SEQ_LEN - 1
CLS_GENE_ID = L_real

# tokenize
tokenizer = CNVTokenizer(n_bins=64, prepend_cls_token=True, reserve_mask_token=True)
tokenizer.fit(X_all)

X_all_tokens = tokenizer.batch_tokenize(np.array(X_all))

vocab_size = tokenizer.get_vocab_size()
CLS_ID     = tokenizer.cls_id
MASK_ID    = tokenizer.mask_id 

mask_token_id = tokenizer.mask_id if hasattr(tokenizer, "mask_id") else tokenizer.get_vocab_size()  
masked_tokens, mask, labels = apply_fixed_mask(X_all_tokens, mask_token_id, mask_ratio=0.3)

cls_position_id = 0
pos_ids_with_cls = [cls_position_id] + pos_ids

position_ids = np.tile(pos_ids_with_cls, (X_all.shape[0], 1))

gene2idx = {g:i for i,g in enumerate(genes_ordered)}
CLS_GENE_ID = len(genes_ordered)
gene_ids_row_subset = np.array([CLS_GENE_ID] + [gene2idx[g] for g in shared], dtype=np.int32)
gene_ids = np.broadcast_to(gene_ids_row_subset, (len(X_all), len(shared)+1))

# load hyena model
encoder = BulkCNVHyenaEncoder(
    n_genes=SEQ_LEN,
    vocab_size=tokenizer.get_vocab_size() + 1,
    embed_dim=512,
    num_layers=2,
    filter_len=8192,
    expand=3,
    dropout_rate=0.4,
    use_pos_enc=False,
    use_gene_emb=True,
    name="cnv_hyena_encoder"
)

with open("/weights/cnv_seq_hyenaencoder_weights_512.pkl", "rb") as f:
    weights = pickle.load(f)

# extract features
dummy_tokens = tf.zeros((1, SEQ_LEN), dtype=tf.int32)
dummy_gene_ids_row = np.concatenate([[CLS_GENE_ID], np.arange(L_real, dtype=np.int32)])
dummy_gene_ids = tf.constant(dummy_gene_ids_row[None, :], dtype=tf.int32)
use_pos_enc = getattr(encoder, "use_pos_enc", False)
if use_pos_enc:
    dummy_pos = tf.range(SEQ_LEN, dtype=tf.int32)[None, :]
    _ = encoder(dummy_tokens, gene_ids=dummy_gene_ids, position_ids=dummy_pos, training=False)
else:
    _ = encoder(dummy_tokens, gene_ids=dummy_gene_ids, training=False)

encoder.set_weights(weights)
encoder.trainable = False

X_cnv_all   = np.asarray(Xall)
X_tokens_all = tokenizer.batch_tokenize(X_cnv_all)

gene_ids_row = np.concatenate([[CLS_GENE_ID], np.arange(L_real, dtype=np.int32)])
gene_ids_all = np.broadcast_to(gene_ids_row, X_tokens_all.shape).astype(np.int32)

if use_pos_enc:
    pos_ids_with_cls = np.arange(SEQ_LEN, dtype=np.int32)
    pos_ids_all = np.broadcast_to(pos_ids_with_cls, X_tokens_all.shape).astype(np.int32)
else:
    pos_ids_all = None

me = "cls"  # or "mean"
sample_names = X_all.index

cnv_feature_df = extract_cnv_features_hyena(
    encoder,
    tokens=X_tokens_all,
    gene_ids=gene_ids_all,
    sample_names=sample_names,
    method=me,
    batch_size=BATCH_SIZE,
    pad_id=0
)

# save features
cnv_feature_df["bcr_patient_barcode"] = df_merged_together["bcr_patient_barcode"].values
cnv_feature_df.to_csv(".../embeddings/cnv_seq_hyena_"+me+"_embeddings_tcga_512.csv")