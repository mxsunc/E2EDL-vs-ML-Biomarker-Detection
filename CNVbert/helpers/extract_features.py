import tensorflow as tf
import from tqdm import tqdm
import numpy as np
import pandas as pd

def extract_cnv_features(encoder, tokens, gene_ids, pos_ids=None, sample_names=None,
                         method="cls", batch_size=32, pad_id=0):
    """
    method:
      - 'cls' : take embedding at position 0 (CLS)
      - 'mean': mean-pool over tokens 1..L (excludes CLS) and ignores PAD (id==pad_id)
    """
    features = []
    N = tokens.shape[0]
    for i in tqdm(range(0, N, batch_size)):
        btok = tf.convert_to_tensor(tokens[i:i+batch_size], dtype=tf.int32)
        bgid = tf.convert_to_tensor(gene_ids[i:i+batch_size], dtype=tf.int32)
        if pos_ids is not None:
            bpos = tf.convert_to_tensor(pos_ids[i:i+batch_size], dtype=tf.int32)
            enc = encoder(btok, gene_ids=bgid, position_ids=bpos, training=False)
        else:
            enc = encoder(btok, gene_ids=bgid, training=False)

        if method == "cls":
            emb = enc[:, 0, :]
        elif method == "mean":
            enc_no_cls = enc[:, 1:, :]
            tok_no_cls = btok[:, 1:]
            mask = tf.cast(tf.not_equal(tok_no_cls, pad_id), tf.float32)
            w = tf.expand_dims(mask, -1)
            summed = tf.reduce_sum(enc_no_cls * w, axis=1)
            denom = tf.reduce_sum(mask, axis=1, keepdims=True)
            emb = tf.math.divide_no_nan(summed, denom)
        else:
            raise ValueError("method must be 'cls' or 'mean'.")

        features.append(emb.numpy())

    all_emb = np.vstack(features)
    cols = [f"{method}_emb_{j}" for j in range(all_emb.shape[1])]
    df = pd.DataFrame(all_emb, columns=cols)
    if sample_names is not None:
        df.index = sample_names
    return df

def extract_cnv_features_hyena(
    encoder,
    tokens,
    gene_ids,
    sample_names=None,
    method="cls",
    batch_size=32,
    pad_id=0,
):
    """
    'cls' = take embedding at position 0,
    'mean' = mean-pool over tokens 1..L (excludes CLS, ignores PAD).
    """
    features = []
    N = tokens.shape[0]

    for i in tqdm(range(0, N, batch_size)):
        btok = tf.convert_to_tensor(tokens[i:i+batch_size], dtype=tf.int32)
        bgid = tf.convert_to_tensor(gene_ids[i:i+batch_size], dtype=tf.int32)

        enc = encoder(btok, gene_ids=bgid, training=False)

        if method == "cls":
            emb = enc[:, 0, :]
        elif method == "mean":
            enc_no_cls = enc[:, 1:, :]
            tok_no_cls = btok[:, 1:]
            mask = tf.cast(tf.not_equal(tok_no_cls, pad_id), tf.float32)
            w = tf.expand_dims(mask, -1)
            summed = tf.reduce_sum(enc_no_cls * w, axis=1)
            denom = tf.reduce_sum(mask, axis=1, keepdims=True)
            emb = tf.math.divide_no_nan(summed, denom)
        else:
            raise ValueError("method must be 'cls' or 'mean'.")

        features.append(emb.numpy())

    all_emb = np.vstack(features)
    cols = [f"{method}_emb_{j}" for j in range(all_emb.shape[1])]
    df = pd.DataFrame(all_emb, columns=cols)

    if sample_names is not None:
        df.index = sample_names
    return df