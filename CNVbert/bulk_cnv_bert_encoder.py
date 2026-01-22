import tensorflow as tf
from tensorflow.keras import layers

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training, mask=None):
        attn_output = self.att(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class BulkCNVBertEncoder(tf.keras.Model):

    def __init__(
        self,
        n_genes: int,
        vocab_size: int,
        n_pos_bins: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 256,
        num_layers: int = 4,
        dropout_rate: float = 0.1,
        use_pos_enc: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_pos_enc = use_pos_enc
        self.token_embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            name="token_embedding"
        )
        if self.use_pos_enc:
            self.position_embedding = tf.keras.layers.Embedding(
                input_dim=n_pos_bins,
                output_dim=embed_dim,
                name="positional_embedding",
            )

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate, name=f"transformer_block_{i}")
            for i in range(num_layers)
        ]


    def call(self, input_tokens, position_ids=None, training=False):
        x = self.token_embedding(input_tokens)
        if self.use_pos_enc:
            if position_ids is None:
                raise ValueError("position_ids must be provided when use_pos_enc=True")
            pos_embed = self.position_embedding(position_ids)
            x = x + pos_embed
        x = self.dropout(x, training=training)
        for block in self.transformer_blocks:
            x = block(x, training=training)
        return x


class BulkCNVBertEncoder2(tf.keras.Model):
    def __init__(
        self,
        n_genes: int,
        vocab_size: int,
        n_pos_bins: int = None,
        embed_dim: int = 128,
        num_heads: int = 4,
        ff_dim: int = 256,
        num_layers: int = 4,
        dropout_rate: float = 0.1,
        use_pos_enc: bool = False,
        use_gene_emb: bool = True,
        pretrained_gene_embeddings=None,
        freeze_gene_emb: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_pos_enc = use_pos_enc
        self.use_gene_emb = use_gene_emb

        self.token_embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            name="token_embedding"
        )
        if self.use_gene_emb:
            d_pre = None if pretrained_gene_embeddings is None else pretrained_gene_embeddings.shape[1]
            self.gene_embedding = layers.Embedding(
                input_dim=n_genes,
                output_dim=d_pre if d_pre is not None else embed_dim,
                name="gene_embedding",
                trainable=not freeze_gene_emb
            )
            self.gene_proj = None
            if d_pre is not None and d_pre != embed_dim:
                self.gene_proj = layers.Dense(embed_dim, name="gene_proj")

        if self.use_pos_enc:
            if n_pos_bins is None:
                raise ValueError("n_pos_bins must be set when use_pos_enc=True")
            self.position_embedding = layers.Embedding(
                input_dim=n_pos_bins,
                output_dim=embed_dim,
                name="positional_embedding",
            )

        self.alpha_tok  = self.add_weight("alpha_tok",  shape=(), initializer="ones", trainable=True)
        self.alpha_gene = self.add_weight("alpha_gene", shape=(), initializer="ones", trainable=True)
        self.alpha_pos  = self.add_weight("alpha_pos",  shape=(), initializer="zeros", trainable=True)  # start small


        self.embed_ln = layers.LayerNormalization(epsilon=1e-6, name="embed_ln")
        self.dropout = layers.Dropout(dropout_rate)

        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate, name=f"transformer_block_{i}")
            for i in range(num_layers)
        ]

        if self.use_gene_emb and pretrained_gene_embeddings is not None:
            self.gene_embedding.build((None, None))
            self.gene_embedding.set_weights([pretrained_gene_embeddings])

    def call(self, input_tokens, gene_ids=None, position_ids=None, training=False):
        """
        input_tokens: (B, L) token IDs from your CNV tokenizer
        gene_ids:     (B, L) gene IDs in [0..n_genes-1] aligned to positions
                      (often identical across batch if gene order is fixed)
        position_ids: (B, L) optional positional IDs (e.g., genomic order bins)
        """
        x = self.alpha_tok  * self.token_embedding(input_tokens)
        if self.use_gene_emb:
            g = self.gene_embedding(gene_ids)
            x = x + self.alpha_gene * g
        if self.use_pos_enc:
            p = self.position_embedding(position_ids)
            x = x + self.alpha_pos * p

        x = self.embed_ln(x) 
        x = self.dropout(x, training=training)
        for block in self.transformer_blocks:
            x = block(x, training=training)
        return x
