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

def make_balanced(sub_df):
        pos = sub_df[sub_df["HRD_status"] == 1]
        neg = sub_df[sub_df["HRD_status"] == 0]
        if len(pos) == 0 or len(neg) == 0:
            raise ValueError("No positive or negative samples in split.")
        neg_res = neg.sample(
            n=len(pos),
            random_state=random_state,
            replace=False,
        )
        out = pd.concat([pos, neg_res], axis=0)
        return out.sample(frac=1.0, random_state=random_state)