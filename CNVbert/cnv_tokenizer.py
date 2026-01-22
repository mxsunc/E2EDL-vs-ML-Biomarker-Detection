import numpy as np
from typing import List, Optional

class CNVTokenizer:
    def __init__(
        self,
        n_bins: int = 32,
        min_cnv_value: float = -2.0,
        max_cnv_value: float = 2.0,
        use_max_normalization: bool = True,
        normalization_factor: float = 1.0,
        prepend_cls_token: bool = False,
        fixed_sequence_length: Optional[int] = None,
        pad_token: int = 0,
    ):
        self.n_bins = n_bins
        self.use_max_normalization = use_max_normalization
        self.normalization_factor = normalization_factor
        self.prepend_cls_token = prepend_cls_token
        self.fixed_sequence_length = fixed_sequence_length
        self.pad_token = pad_token
        self.bin_edges = np.linspace(min_cnv_value, max_cnv_value, n_bins)

    def tokenize_sample(self, cnv_vector: np.ndarray) -> np.ndarray:
        vector = cnv_vector.copy()
        if self.use_max_normalization:
            vector /= self.normalization_factor
        tokens = np.digitize(vector, self.bin_edges).astype(np.int32)
        if self.prepend_cls_token:
            tokens = np.concatenate([[self.n_bins], tokens]) 
        if self.fixed_sequence_length:
            padded = np.full(self.fixed_sequence_length, self.pad_token, dtype=np.int32)
            length = min(len(tokens), self.fixed_sequence_length)
            padded[:length] = tokens[:length]
            tokens = padded
        return tokens

    def batch_tokenize(self, cnv_matrix: np.ndarray) -> np.ndarray:
        return np.vstack([self.tokenize_sample(vec) for vec in cnv_matrix])

    def get_vocab_size(self) -> int:
        return self.n_bins + 1 if self.prepend_cls_token else self.n_bins



class CNVTokenizer2:
    """
    Per-gene quantile binning for CNV values.

    IDs:
      0              -> PAD
      1..n_bins      -> value bins
      n_bins + 1     -> [CLS] (if prepend_cls_token=True)
      n_bins + 2     -> [MASK] (if reserve_mask_token=True)
    """
    def __init__(
        self,
        n_bins: int = 64,
        prepend_cls_token: bool = True,
        fixed_sequence_length: Optional[int] = None,
        pad_token: int = 0,
        reserve_mask_token: bool = True,
        nan_to_pad: bool = True,
    ):
        self.n_bins = int(n_bins)
        self.prepend_cls_token = prepend_cls_token
        self.fixed_sequence_length = fixed_sequence_length
        self.pad_token = int(pad_token)
        self.reserve_mask_token = reserve_mask_token
        self.nan_to_pad = nan_to_pad
        self.bin_edges = None
        self.n_genes = None

    @property
    def cls_id(self) -> int:
        return self.n_bins + 1

    @property
    def mask_id(self) -> int:
        return self.n_bins + 2

    def get_vocab_size(self) -> int:
        vocab = 1 + self.n_bins  # PAD + bins
        if self.prepend_cls_token:
            vocab += 1           # CLS
        if self.reserve_mask_token:
            vocab += 1           # MASK
        return vocab

    def fit(self, X_train: np.ndarray) -> "CNVTokenizer":
        """
        X_train: (N, G) float array (train split only!)
        Learns per-gene interior quantile edges so bins are ~equiprobable per gene.
        """
        if X_train.ndim != 2:
            raise ValueError("X_train must be 2D: (n_samples, n_genes)")
        N, G = X_train.shape
        self.n_genes = G

        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.bin_edges = np.quantile(X_train, qs, axis=0)
        if np.isnan(self.bin_edges).any():
            X = X_train.copy()
            col_med = np.nanmedian(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_med, inds[1])
            self.bin_edges = np.quantile(X, qs, axis=0)
        return self

    def tokenize_sample(self, cnv_vector: np.ndarray) -> np.ndarray:
        if self.bin_edges is None:
            raise RuntimeError("Call .fit(...) first to compute quantile edges.")
        v = np.asarray(cnv_vector, dtype=np.float64)
        if v.ndim != 1:
            raise ValueError("cnv_vector must be 1D (n_genes,)")

        if v.size != self.n_genes:
            raise ValueError(f"Expected {self.n_genes} genes, got {v.size}")

        edges_T = self.bin_edges.T  # (G, n_bins-1)
        tok = (v[:, None] > edges_T).sum(axis=1).astype(np.int32) + 1

        if self.nan_to_pad and np.isnan(v).any():
            tok[np.isnan(v)] = self.pad_token

        if self.prepend_cls_token:
            tok = np.concatenate(([self.cls_id], tok)).astype(np.int32)

        if self.fixed_sequence_length is not None:
            padded = np.full(self.fixed_sequence_length, self.pad_token, dtype=np.int32)
            L = min(len(tok), self.fixed_sequence_length)
            padded[:L] = tok[:L]
            tok = padded
        return tok

    def batch_tokenize(self, cnv_matrix: np.ndarray) -> np.ndarray:
        X = np.asarray(cnv_matrix)
        if X.ndim == 1:
            return self.tokenize_sample(X)[None, :]
        return np.vstack([self.tokenize_sample(row) for row in X])