import numpy as np
import tensorflow as tf
import math
from tensorflow.keras.utils import Sequence

IGNORE_IDX = -100

def apply_fixed_mask(tokens, mask_token_id, mask_ratio=0.15):
    input_tokens = tokens.copy()
    labels = np.full_like(tokens, fill_value=-100)
    mask = np.zeros_like(tokens, dtype=bool)

    for i in range(tokens.shape[0]):
        seq_len = tokens.shape[1]
        candidate_indices = np.arange(1, seq_len)
        num_to_mask = max(1, int(mask_ratio * len(candidate_indices)))

        masked_indices = np.random.choice(candidate_indices, num_to_mask, replace=False)
        
        input_tokens[i, masked_indices] = mask_token_id
        labels[i, masked_indices] = tokens[i, masked_indices]
        mask[i, masked_indices] = True

    return input_tokens, mask, labels

def apply_random_mask(tokens, mask_token_id, mask_ratio=0.15):
    input_tokens = tokens.copy()
    labels = tokens.copy()
    mask_positions = np.random.rand(*tokens.shape) < mask_ratio
    mask_positions[:, 0] = False
    input_tokens[mask_positions] = mask_token_id
    labels[~mask_positions] = -100

    return input_tokens, mask_positions, labels

class MaskedTokenAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="acc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="correct", initializer="zeros", dtype=tf.float32)
        self.total   = self.add_weight(name="total",   initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        mask   = tf.not_equal(y_true, IGNORE_IDX)
        pred   = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        y_true_masked = tf.boolean_mask(y_true, mask)
        pred_masked   = tf.boolean_mask(pred,   mask)
        correct = tf.reduce_sum(tf.cast(tf.equal(y_true_masked, pred_masked), tf.float32))
        total   = tf.cast(tf.size(y_true_masked), tf.float32)
        self.correct.assign_add(correct)
        self.total.assign_add(total)

    def result(self):
        return tf.math.divide_no_nan(self.correct, self.total)

    def reset_states(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

class MaskedTopKAccuracy(tf.keras.metrics.Metric):
    def __init__(self, k=5, name=None, **kwargs):
        super().__init__(name=name or f"top{k}", **kwargs)
        self.k = k
        self.correct = self.add_weight("correct", initializer="zeros", dtype=tf.float32)
        self.total   = self.add_weight("total",   initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        mask   = tf.not_equal(y_true, IGNORE_IDX)
        topk = tf.math.top_k(y_pred, k=self.k).indices
        y_true_exp = tf.expand_dims(y_true, axis=-1)
        in_topk = tf.reduce_any(tf.equal(topk, y_true_exp), axis=-1)
        in_topk = tf.boolean_mask(in_topk, mask)
        correct = tf.reduce_sum(tf.cast(in_topk, tf.float32))
        total   = tf.cast(tf.size(in_topk), tf.float32)
        self.correct.assign_add(correct)
        self.total.assign_add(total)

    def result(self):
        return tf.math.divide_no_nan(self.correct, self.total)

    def reset_states(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

class MaskedPerplexity(tf.keras.metrics.Metric):
    """exp(mean masked cross-entropy)"""
    def __init__(self, from_logits=True, name="ppl", **kwargs):
        super().__init__(name=name, **kwargs)
        self.from_logits = from_logits
        self.loss_sum = self.add_weight("loss_sum", initializer="zeros", dtype=tf.float32)
        self.w_sum    = self.add_weight("w_sum",    initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        per_tok_ce = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=self.from_logits
        )  # (B, L)
        mask = tf.cast(tf.not_equal(y_true, IGNORE_IDX), tf.float32)
        self.loss_sum.assign_add(tf.reduce_sum(per_tok_ce * mask))
        self.w_sum.assign_add(tf.reduce_sum(mask))

    def result(self):
        mean_ce = tf.math.divide_no_nan(self.loss_sum, self.w_sum)
        return tf.exp(mean_ce)

    def reset_states(self):
        self.loss_sum.assign(0.0)
        self.w_sum.assign(0.0)


class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps, min_lr=0.0):
        super().__init__()
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        w = tf.cast(self.warmup_steps, tf.float32)
        T = tf.cast(self.total_steps, tf.float32)

        def lr_warmup():
            return self.base_lr * (step / tf.maximum(w, 1.0))

        def lr_cosine():
            t = (step - w) / tf.maximum(T - w, 1.0)
            cos_decay = 0.5 * (1.0 + tf.cos(math.pi * tf.minimum(1.0, t)))
            return self.min_lr + (self.base_lr - self.min_lr) * cos_decay

        return tf.where(step < w, lr_warmup(), lr_cosine())

    def get_config(self):
        return dict(base_lr=self.base_lr, warmup_steps=self.warmup_steps,
                    total_steps=self.total_steps, min_lr=self.min_lr)



def mask_once(x_tokens, mask_id, mask_ratio, special_ids=(0,), force_min_one=True, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    L = x_tokens.shape[0]
    specials = np.zeros(L, dtype=bool)
    if special_ids:
        for sid in special_ids:
            specials |= (x_tokens == sid)
    cand_idx = np.where(~specials)[0]
    if cand_idx.size == 0 or mask_ratio <= 0.0:
        return x_tokens.copy(), np.full(L, -100, dtype=np.int32)
    k = int(round(mask_ratio * cand_idx.size))
    if force_min_one:
        k = max(1, k)
    k = min(k, cand_idx.size)
    chosen = rng.choice(cand_idx, size=k, replace=False)
    masked = x_tokens.copy()
    labels = np.full(L, -100, dtype=np.int32)
    labels[chosen] = x_tokens[chosen]
    masked[chosen] = mask_id
    return masked, labels

class MaskingSequence(Sequence):
    def __init__(self, X_tokens, gene_ids, batch_size, mask_id,
                 start_ratio=0.05, end_ratio=0.15, warmup_epochs=2,
                 special_ids=(), shuffle=True, seed=42):
        self.X_tokens = X_tokens
        self.gene_ids = gene_ids
        self.N = X_tokens.shape[0]
        self.L = X_tokens.shape[1]
        self.batch_size = int(batch_size)
        self.mask_id = int(mask_id)
        self.start_ratio = float(start_ratio)
        self.end_ratio = float(end_ratio)
        self.warmup_epochs = int(warmup_epochs)
        self.special_ids = tuple(int(s) for s in special_ids)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(self.N)
        self.current_epoch = 0
        self._update_mask_ratio()

    def _update_mask_ratio(self):
        if self.warmup_epochs <= 0:
            t = 1.0
        else:
            t = min(1.0, self.current_epoch / float(self.warmup_epochs))
        self.mask_ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * t

    def __len__(self):
        return int(np.ceil(self.N / self.batch_size))

    def on_epoch_begin(self):
        self._update_mask_ratio()
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def on_epoch_end(self):
        self.current_epoch += 1

    def __getitem__(self, idx):
        sl = slice(idx * self.batch_size, min((idx+1) * self.batch_size, self.N))
        batch_ix = self.indices[sl]
        Xb = self.X_tokens[batch_ix]
        Gb = self.gene_ids[batch_ix]
        B = Xb.shape[0]
        masked_b = np.empty_like(Xb)
        labels_b = np.full_like(Xb, fill_value=-100)
        for i in range(B):
            masked_i, labels_i = mask_once(
                Xb[i], mask_id=self.mask_id, mask_ratio=self.mask_ratio,
                special_ids=self.special_ids, rng=self.rng
            )
            masked_b[i] = masked_i
            labels_b[i] = labels_i
        return [masked_b, Gb], labels_b