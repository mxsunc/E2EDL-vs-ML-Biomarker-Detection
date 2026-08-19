import numpy as np
from typing import Optional


class CNVTokenizerLinear:
    """
    Fixed linear-edge binning for integer/thresholded CNV values.

    Token-ID layout (n_bins = N):
        0           -> PAD            (unique; never collides with a value bin)
        1 .. N      -> value bins     (digitize result + 1, offset past PAD)
        N + 1       -> [CLS]          (if prepend_cls_token=True)
        N + 2       -> [MASK]         (if reserve_mask_token=True)

    Unlike CNVTokenizer2 (per-gene quantile edges learned via .fit()), this
    tokenizer uses FIXED linear edges, so no fitting is required. It is the
    right choice for CNV matrices on a known discrete scale (e.g. TCGA/CPTAC
    integer CN states on [-2, 2] in 0.5 steps -> 9 distinct values -> n_bins=9).

    IDs
    ----
    pad_id  = 0
    cls_id  = n_bins + 1           (if prepend_cls_token)
    mask_id = n_bins + 2           (if reserve_mask_token)
    """
    def __init__(
        self,
        n_bins: int = 9,
        min_cnv_value: float = -2.0,
        max_cnv_value: float = 2.0,
        prepend_cls_token: bool = True,
        fixed_sequence_length: Optional[int] = None,
        pad_token: int = 0,
        reserve_mask_token: bool = True,
        nan_to_pad: bool = True,
    ):
        self.n_bins = int(n_bins)
        self.min_cnv_value = float(min_cnv_value)
        self.max_cnv_value = float(max_cnv_value)
        self.prepend_cls_token = prepend_cls_token
        self.fixed_sequence_length = fixed_sequence_length
        self.pad_token = int(pad_token)
        self.reserve_mask_token = reserve_mask_token
        self.nan_to_pad = nan_to_pad

        # Interior edges: n_bins-1 cuts that split [min, max] into n_bins bins.
        # For [-2, 2] with n_bins=9 these are the 8 midpoints between the 9
        # distinct half-integer values, so each distinct CNV state -> one bin.
        self.bin_edges = np.linspace(
            self.min_cnv_value, self.max_cnv_value, self.n_bins + 1
        )[1:-1]

    # ----- Special token IDs -----
    @property
    def cls_id(self) -> int:
        return self.n_bins + 1

    @property
    def mask_id(self) -> int:
        return self.n_bins + 2

    def get_vocab_size(self) -> int:
        # PAD (0) is implicit; Embedding.input_dim must be max_id + 1.
        vocab = 1 + self.n_bins  # PAD + bins -> ids 0..n_bins
        if self.prepend_cls_token:
            vocab += 1           # CLS
        if self.reserve_mask_token:
            vocab += 1           # MASK
        return vocab  # e.g. 9 -> 12; used DIRECTLY as input_dim (no +1)

    # ----- Tokenize a single vector -----
    def tokenize_sample(self, cnv_vector: np.ndarray) -> np.ndarray:
        v = np.asarray(cnv_vector, dtype=np.float64)
        if v.ndim != 1:
            raise ValueError("cnv_vector must be 1D (n_genes,)")

        # digitize returns 0..n_bins; +1 offsets past PAD=0 -> bins 1..n_bins+1.
        # Values == max land in bin n_bins; +1 -> n_bins+1 is fine (within vocab).
        tok = np.digitize(v, self.bin_edges).astype(np.int32) + 1

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

    # ----- Batch -----
    def batch_tokenize(self, cnv_matrix: np.ndarray) -> np.ndarray:
        X = np.asarray(cnv_matrix)
        if X.ndim == 1:
            return self.tokenize_sample(X)[None, :]
        return np.vstack([self.tokenize_sample(row) for row in X])
