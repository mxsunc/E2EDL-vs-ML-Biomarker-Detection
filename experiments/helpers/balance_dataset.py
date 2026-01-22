import tensorflow as tf

def make_balanced_ds(X_arr, y_arr, batch_size=64, shuffle_buffer=10_000):
    half = batch_size // 2

    pos = tf.data.Dataset.from_tensor_slices((X_arr[y_arr == 1],
                                              y_arr[y_arr == 1]))
    neg = tf.data.Dataset.from_tensor_slices((X_arr[y_arr == 0],
                                              y_arr[y_arr == 0]))

    pos = (pos.shuffle(shuffle_buffer)
              .repeat()
              .batch(half, drop_remainder=True))
    neg = (neg.shuffle(shuffle_buffer)
              .repeat()
              .batch(half, drop_remainder=True))

    merged = tf.data.Dataset.zip((pos, neg))

    def _merge(p, n):
        x = tf.concat([p[0], n[0]], axis=0)
        y_ = tf.concat([p[1], n[1]], axis=0)
        idx = tf.random.shuffle(tf.range(batch_size))
        x, y_ = tf.gather(x, idx), tf.gather(y_, idx)
        return x, {"reconstruction": x, "hrd_pred": y_}

    return (merged.map(_merge, num_parallel_calls=AUTOTUNE)
                   .prefetch(AUTOTUNE))