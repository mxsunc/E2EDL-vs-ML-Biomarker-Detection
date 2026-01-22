import tensorflow as tf
from tensorflow.keras import Model, Input

class CNVAE:
    def __init__(self, input_dim, latent_dim, dropout, l2_weight):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.l2_weight = l2_weight

        inputs = tf.keras.Input(shape=(input_dim,), name="gene_cnv")

        x = tf.keras.layers.GaussianNoise(0.1)(inputs)
        encoded = tf.keras.layers.Dense(
            latent_dim, activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
            name="encoded_dense"
        )(x)
        encoded = tf.keras.layers.BatchNormalization(name="encoded_bn")(encoded)
        encoded = tf.keras.layers.Dropout(dropout, name="encoded")(encoded)

        decoded = tf.keras.layers.Dense(input_dim, name="reconstruction")(encoded)

        clf_intermediate = tf.keras.layers.Dense(
            int(latent_dim / 2),
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
            name="clf_intermediate"
        )(encoded)
        clf_intermediate = tf.keras.layers.Dropout(dropout)(clf_intermediate)

        clf_out = tf.keras.layers.Dense(
            1, activation="sigmoid", name="hrd_pred"
        )(clf_intermediate)

        model = tf.keras.Model(inputs, [decoded, clf_out], name="cnv_ae")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss={"reconstruction": "mse", "hrd_pred": "binary_crossentropy"},
            loss_weights={"reconstruction": 1.0, "hrd_pred": 1.0},
            metrics={"hrd_pred": tf.keras.metrics.BinaryAccuracy(name="accuracy")},
        )

        self.model = model
        self.encoder = Model(inputs=model.input, outputs=model.get_layer("encoded").output, name="cnv_encoder")

        latent_in = Input(shape=(latent_dim,), name="latent_input")
        preclf = model.get_layer("clf_intermediate")(latent_in)
        self.preclassifier = Model(inputs=latent_in, outputs=preclf, name="preclassifier_intermediate")