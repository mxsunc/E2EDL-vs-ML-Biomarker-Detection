import numpy as np
import pickle 
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
from cnv_tokenizer import CNVTokenizer2 as CNVTokenizer
from bulk_cnv_bert_encoder import BulkCNVBertEncoder2
from helpers.extract_features import extract_cnv_features
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

# set parameters
BATCH_SIZE = 1
EPOCHS = 10
vocab_size = 67
SEQ_LEN = X_all.shape[1] + 1
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
masked_tokens, mask, labels = apply_fixed_mask(X_all_tokens, mask_token_id, mask_ratio=0.15)

cls_position_id = 0
pos_ids_with_cls = [cls_position_id] + pos_ids

position_ids = np.tile(pos_ids_with_cls, (X_all.shape[0], 1))
gene_ids_row = np.arange(SEQ_LEN, dtype=np.int32) 

# load bert model
encoder = BulkCNVBertEncoder2(
    n_genes=SEQ_LEN,
    vocab_size=tokenizer.get_vocab_size() + 1,
    embed_dim=512,
    num_heads=4,
    ff_dim=1024,
    num_layers=2,
    n_pos_bins=n_pos_bins,
    use_pos_enc=False,                   
    use_gene_emb=True
)

with open(".../weights/cnv_seq_encoder_weights_512_genesorted.pkl", "rb") as f:
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

X_cnv_all   = np.asarray(X_all)
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

cnv_feature_df = extract_cnv_features(
    encoder,
    tokens=X_tokens_all,
    gene_ids=gene_ids_all,
    pos_ids=pos_ids_all,
    sample_names=sample_names,
    method=me,
    batch_size=1,
    pad_id=0
)

# save features
cnv_feature_df["bcr_patient_barcode"] = df_merged_together["bcr_patient_barcode"].values
cnv_feature_df.to_csv(".../embeddings/cnv_seq_"+me+"_embeddings_tcga_512_genesorted.csv")