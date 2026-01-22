import numpy as np
import tensorflow as tf

def make_dataset(id_list, bags, batch_size, shuffle=True):
        def gen():
            for sid in id_list:
                x = bags[sid]["x_cont"].astype(np.float32)  # (n_segments, num_features)
                y = np.int32(bags[sid]["label"])            # scalar 0/1
                length = x.shape[0]
                yield {"x_cont": x, "length": length}, y

        output_signature = (
            {
                "x_cont": tf.TensorSpec(shape=(None, num_features), dtype=tf.float32),
                "length": tf.TensorSpec(shape=(), dtype=tf.int32),
            },
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )

        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

        def cast(inputs, y):
            return inputs, tf.cast(y, tf.float32)

        if shuffle:
            ds = ds.shuffle(2048, reshuffle_each_iteration=True)

        ds = ds.map(cast).padded_batch(
            batch_size,
            padded_shapes=(
                {"x_cont": [None, num_features], "length": []},
                [],
            ),
        ).prefetch(tf.data.AUTOTUNE)

        return ds