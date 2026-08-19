import tensorflow as tf
from tensorflow.keras import layers

def _next_pow2(n):
    # n: scalar/int Tensor
    n = tf.cast(tf.maximum(n, 1), tf.int32)
    log2_n = tf.math.log(tf.cast(n, tf.float32)) / tf.math.log(tf.constant(2.0, tf.float32))
    k = tf.cast(tf.math.ceil(log2_n), tf.int32)               # ceil(log2(n))
    return tf.bitwise.left_shift(tf.constant(1, tf.int32), k) # 1 << k
class FFTLongConv1D(layers.Layer):
    """
    Depthwise long convolution via FFT, 'same' padding.
    Each channel (feature dim) has its own learned filter.
    u: (B, L, D) -> y: (B, L, D)
    """
    def __init__(self, d_model, filter_len=8192, name=None):
        super().__init__(name=name)
        self.D = int(d_model)
        self.K = int(filter_len)
        # Per-channel real filter. Initialize small.
        self.h = self.add_weight(
            "h", shape=(self.D, self.K),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True, dtype=tf.float32
        )

    def call(self, u):
        # u: (B,L,D) -> (B*D, L)
        u = tf.cast(u, tf.float32)
        B = tf.shape(u)[0]; L = tf.shape(u)[1]; D = tf.shape(u)[2]
        x = tf.transpose(u, [0, 2, 1])                   # (B,D,L)
        x = tf.reshape(x, [B * D, L])                    # (BD, L)

        # Pad to FFT length
        pad_len = L + self.K - 1
        N = _next_pow2(pad_len)

        # Zero-pad signals and filters to N
        x_pad = tf.pad(x, [[0, 0], [0, N - L]])          # (BD, N)
        h_pad = tf.pad(self.h, [[0, 0], [0, N - self.K]])# (D, N)
        h_pad = tf.repeat(h_pad, repeats=B, axis=0)      # (BD, N)

        # FFT, multiply, IFFT
        X = tf.signal.rfft(x_pad)                        # (BD, N/2+1) complex64
        H = tf.signal.rfft(h_pad)                        # (BD, N/2+1) complex64
        Y = X * H
        y_full = tf.signal.irfft(Y, [N])                 # (BD, N) real

        # 'same' slice centered: take first L positions aligned as conv1d SAME
        # For linear conv, same-aligned start = (K-1)//2
        start = (self.K - 1) // 2
        y = y_full[:, start:start + L]                   # (BD, L)

        # Back to (B, L, D)
        y = tf.reshape(y, [B, self.D, L])
        y = tf.transpose(y, [0, 2, 1])                   # (B, L, D)
        return y

class HyenaLiteBlock(layers.Layer):
    """
    Gated long convolutional mixer (Hyena-like), pre-norm + residual.
    """
    def __init__(self, d_model=256, filter_len=8192, expand=2, dropout=0.1, name=None):
        super().__init__(name=name)
        self.d = d_model
        self.d_inner = expand * d_model
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.in_proj = layers.Dense(2 * self.d_inner, use_bias=True)  # to (u, z)
        self.longconv = FFTLongConv1D(self.d_inner, filter_len=filter_len)
        self.dropout = layers.Dropout(dropout)
        self.out_proj = layers.Dense(d_model, dtype="float32")        # keep fp32 out

    def call(self, x, training=False):
        h = self.norm(x)                                # (B,L,D)
        u_z = self.in_proj(h)                           # (B,L,2*d_inner)
        u, z = tf.split(u_z, 2, axis=-1)                # (B,L,Dh), (B,L,Dh)
        z = tf.nn.sigmoid(z)
        u = self.longconv(u)                            # (B,L,Dh), O(L log L)
        y = u * z
        y = self.out_proj(y)                            # (B,L,D)
        y = self.dropout(y, training=training)
        return x + tf.cast(y, x.dtype)                  # residual


class BulkCNVHyenaEncoder(tf.keras.Model):
    def __init__(self, n_genes, vocab_size, embed_dim=256, num_layers=2,
                 filter_len=8192, expand=2, dropout_rate=0.1,
                 use_pos_enc=False, use_gene_emb=True, n_pos_bins=None, name=None):
        super().__init__(name=name)
        self.use_pos_enc = use_pos_enc
        self.use_gene_emb = use_gene_emb

        init = tf.keras.initializers.TruncatedNormal(stddev=0.02)
        self.token_embedding = layers.Embedding(vocab_size, embed_dim, embeddings_initializer=init, name="tok_emb")
        if use_gene_emb:
            self.gene_embedding  = layers.Embedding(n_genes, embed_dim, embeddings_initializer=init, name="gene_emb")
        if use_pos_enc:
            self.position_embedding = layers.Embedding(n_pos_bins, embed_dim, embeddings_initializer=init, name="pos_emb")

        # learnable balances
        self.alpha_tok  = self.add_weight("alpha_tok",  shape=(), initializer="ones",  trainable=True)
        self.alpha_gene = self.add_weight("alpha_gene", shape=(), initializer="ones",  trainable=True)
        self.alpha_pos  = self.add_weight("alpha_pos",  shape=(), initializer="zeros", trainable=True)

        self.embed_ln = layers.LayerNormalization(epsilon=1e-6, name="embed_ln")
        self.blocks = [
            HyenaLiteBlock(d_model=embed_dim, filter_len=filter_len, expand=expand, dropout=dropout_rate, name=f"hyena_{i}")
            for i in range(num_layers)
        ]

    def call(self, input_tokens, gene_ids=None, position_ids=None, training=False):
        x = self.alpha_tok * self.token_embedding(input_tokens)
        if self.use_gene_emb:
            if gene_ids is None: raise ValueError("gene_ids required when use_gene_emb=True")
            x = x + self.alpha_gene * self.gene_embedding(gene_ids)
        if self.use_pos_enc:
            if position_ids is None: raise ValueError("position_ids required when use_pos_enc=True")
            x = x + self.alpha_pos * self.position_embedding(position_ids)

        x = self.embed_ln(x)
        for blk in self.blocks:
            x = blk(x, training=training)
        return x  # (B,L,D)
