import numpy as np
import tensorflow as tf


rng = np.random.default_rng(333)


class AssociativeRecallTask:
    def __init__(self,
                 batch_size,
                 max_items,
                 bits_per_vector,
                 lines_per_item):
        self.batch_size = batch_size
        self.max_items = max_items
        self.bits_per_vector = bits_per_vector
        self.lines_per_item = lines_per_item

    def generate_sequence(self, num_items):
        query_item_idx = rng.integers(0, high=num_items-1)
        query_key = None
        query_value = None
        seq_x = np.zeros((0, self.bits_per_vector+2))
        for i in range(num_items):
            pattern = rng.integers(0,
                                   high=2,
                                   size=(self.lines_per_item,
                                         self.bits_per_vector),
                                   )
            if i == query_item_idx:
                query_key = pattern
            if i == query_item_idx + 1:
                query_value = pattern
            item = np.concatenate([np.zeros((self.lines_per_item, 2)),
                                   pattern],
                                  axis=1)  # pattern with delim chan
            pre_delim = tf.expand_dims(tf.one_hot(1, self.bits_per_vector+2),
                                       axis=0)

            item = np.concatenate([pre_delim, item], axis=0)
            seq_x = np.concatenate([seq_x, item], axis=0)

        item = np.concatenate([np.zeros((self.lines_per_item, 2)),
                               query_key], 1)
        pre_delim = tf.expand_dims(tf.one_hot(0, self.bits_per_vector+2),
                                   axis=0)
        post_delim = tf.expand_dims(tf.one_hot(0, self.bits_per_vector+2),
                                    axis=0)
        item = np.concatenate([pre_delim, item, post_delim], axis=0)
        seq_x = np.concatenate([seq_x, item])

        seq_y = np.zeros((seq_x.shape[0], self.bits_per_vector))
        seq_y = np.concatenate([seq_y, query_value])
        seq_x = np.concatenate([seq_x, np.zeros((3, self.bits_per_vector+2))])

        seq_x = seq_x.astype(np.float32)
        seq_y = seq_y.astype(dtype=np.float32)
        return seq_x, seq_y

    def generate_batch(self):
        batch_x = []
        batch_y = []
        num_items = rng.integers(2, self.max_items)
        for i in range(self.batch_size):
            x, y = self.generate_sequence(num_items)
            batch_x.append(x)
            batch_y.append(y)
        return np.stack(batch_x), np.stack(batch_y), num_items

    def loss(self, truth, output, offset):
        return tf.reduce_mean(tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(truth[:, offset:, :],
                                                    output[:, offset:, :]),
            axis=1
        ))
