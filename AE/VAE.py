import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


class CNVVAE:
    def __init__(self, input_dim, latent_dim, dropout, l2_weight, beta_kl=0.01):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.l2_weight = l2_weight
        self.beta_kl = beta_kl

        cnv_inputs = layers.Input(shape=(input_dim,), name="gene_cnv")
        x = layers.GaussianNoise(0.1)(cnv_inputs)
        x = layers.Dense(
            512, activation="relu",
            kernel_regularizer=regularizers.l2(l2_weight)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)

        z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z         = Sampling()([z_mean, z_log_var])

        clf = layers.Dense(
            256, activation="relu",
            kernel_regularizer=regularizers.l2(l2_weight)
        )(z)
        clf = layers.Dropout(dropout)(clf)
        hrd_pred = layers.Dense(1, activation="sigmoid", name="hrd_pred")(clf)

        decoder_hidden = layers.Dense(512, activation="relu")(z)
        reconstruction = layers.Dense(input_dim, name="reconstruction")(decoder_hidden)

        vae = Model(cnv_inputs, [reconstruction, hrd_pred], name="cnv_vae")

        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        vae.add_loss(beta_kl * kl_loss)

        vae.compile(
            optimizer="adam",
            loss={"reconstruction": "mse", "hrd_pred": "binary_crossentropy"},
            loss_weights={"reconstruction": 1.0, "hrd_pred": 1.0},
            metrics={"hrd_pred": "accuracy"},
        )

        self.model = vae
        self.encoder = Model(inputs=cnv_inputs, outputs=z_mean, name="cnv_encoder")
        self.preclassifier = Model(inputs=z, outputs=clf, name="preclassifier_intermediate")
