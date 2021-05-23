import numpy as np
import tensorflow as tf


rng = np.random.default_rng()


class CopyTask:
    def __init__(self,
                 batch_size,
                 max_seq_length,
                 bits_per_vector):
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.bits_per_vector = bits_per_vector

    def generate_sequence(self, seq_len):

        pattern = rng.integers(0,
                               high=2,
                               size=(seq_len, self.bits_per_vector),
                               )
        seq_x = np.concatenate([np.zeros((seq_len, 1)),
                                pattern], 1)
        assert(np.count_nonzero(seq_x[:, 0]) == 0)
        seq_x = np.concatenate([seq_x, np.ones((1, self.bits_per_vector+1))])
        assert(np.count_nonzero(seq_x[seq_len]) == self.bits_per_vector+1)
        seq_x = np.concatenate([seq_x, np.zeros((seq_len,
                                                 self.bits_per_vector+1))])
        seq_x = seq_x.astype(np.float32)
        seq_y = np.zeros_like(pattern)
        seq_y = np.concatenate([seq_y, np.zeros((1, self.bits_per_vector))])
        seq_y = np.concatenate([seq_y, pattern])
        seq_y = seq_y.astype(dtype=np.float32)
        return seq_x, seq_y

    def generate_batch(self, seq_len=None):
        batch_x = []
        batch_y = []
        if seq_len is None:
            seq_len = rng.integers(1, self.max_seq_length)
        for i in range(self.batch_size):
            x, y = self.generate_sequence(seq_len)
            batch_x.append(x)
            batch_y.append(y)
        return np.stack(batch_x), np.stack(batch_y), seq_len

    def loss(self, truth, output, offset):
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(truth[:, offset+1:, :],
                                                    output[:, offset+1:, :]),
            )
