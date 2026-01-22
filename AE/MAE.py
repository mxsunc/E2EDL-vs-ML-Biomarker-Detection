import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as regularizers

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

class MaskedCNVModel(tf.keras.Model):
    def __init__(
        self,
        input_dim=21559,
        latent_dim=128,
        mask_ratio=0.3,
        l2_weight=1e-4,
        dropout=0.4,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.noise = layers.GaussianNoise(0.1)
        self.enc_dense = layers.Dense(
            latent_dim, activation="relu",
            kernel_regularizer=regularizers.l2(l2_weight),
            name="encoded_dense"
        )
        self.enc_bn    = layers.BatchNormalization(name="encoded_bn")
        self.enc_drop  = layers.Dropout(dropout, name="encoded")
        self.dec_dense = layers.Dense(input_dim, name="reconstruction")
        self.clf_int   = layers.Dense(
            latent_dim//2, activation="relu",
            kernel_regularizer=regularizers.l2(l2_weight),
            name="clf_intermediate"
        )
        self.clf_drop  = layers.Dropout(dropout)
        self.clf_out   = layers.Dense(1, activation="sigmoid", name="hrd_pred")
        self.recon_loss_fn = tf.keras.losses.MeanSquaredError(reduction="none")
        self.clf_loss_fn  = tf.keras.losses.BinaryCrossentropy()
        self.acc_metric = tf.keras.metrics.BinaryAccuracy(name="accuracy")


    def call(self, inputs, training=False):
        if training:
            mask = tf.cast(
                tf.random.uniform(tf.shape(inputs)) > self.mask_ratio,
                tf.float32
            )
        else:
            mask = tf.ones_like(inputs)

        x_masked = inputs * mask
        x = self.noise(x_masked, training=training)

        z = self.enc_dense(x)
        z = self.enc_bn(z, training=training)
        z = self.enc_drop(z, training=training)

        recon = self.dec_dense(z)
        c = self.clf_int(z)
        c = self.clf_drop(c, training=training)
        pred = self.clf_out(c)

        return recon, pred, mask

    def train_step(self, data):
        X, y = data
        X        = tf.cast(X,        tf.float32)
        y_recon  = tf.cast(y["reconstruction"], tf.float32)
        y_class  = tf.cast(y["hrd_pred"],       tf.float32)

        with tf.GradientTape() as tape:
            recon, pred, mask = self(X, training=True)
            visible    = 1.0 - mask
            sq_err     = tf.square(y_recon - recon)
            masked_mse = tf.reduce_sum(sq_err * visible) / tf.reduce_sum(visible)
            clf_loss   = self.clf_loss_fn(y_class, pred)
            total_loss = masked_mse + clf_loss + sum(self.losses)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.acc_metric.update_state(y_class, pred)

        return {
            "loss":       total_loss,
            "masked_mse": masked_mse,
            "clf_loss":   clf_loss,
            "accuracy":   self.acc_metric.result(),
        }

    def test_step(self, data):
        X, y = data
        X        = tf.cast(X,        tf.float32)
        y_recon  = tf.cast(y["reconstruction"], tf.float32)
        y_class  = tf.cast(y["hrd_pred"],       tf.float32)
        recon, pred, mask = self(X, training=False)
        full_mse = tf.reduce_mean(tf.square(y_recon - recon))
        clf_loss = self.clf_loss_fn(y_class, pred)
        total    = full_mse + clf_loss + sum(self.losses)
        self.acc_metric.update_state(y_class, pred)

        return {
            "loss":     total,
            "full_mse": full_mse,
            "clf_loss": clf_loss,
            "accuracy": self.acc_metric.result(),
        }