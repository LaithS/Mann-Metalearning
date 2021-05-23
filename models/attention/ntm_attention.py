
import tensorflow as tf

from collections import namedtuple, OrderedDict
from .common import content_addressing, EPSILON

NTMAttentionState = namedtuple('NTMAttentionState', ('weightings'))
NTMAttentionConfig = namedtuple('NTMAttentionConfig', ('read_heads',
                                                       'write_heads'))


NTMinterface = namedtuple("interface", [
    "keys",
    "strengths",
    "gates",
    "shifts",
    "gammas",
    "erase_vectors",
    "add_vectors",
])


@tf.function
def pars_params(read_heads, write_heads, memory_width, interface_vector):
    heads = read_heads + write_heads
    sizes = [heads * memory_width,
             heads,
             heads,
             3 * heads,
             heads,
             memory_width * write_heads,
             memory_width * write_heads]
    fns = OrderedDict([
        ("keys",
         lambda v: tf.reshape(v, (-1,
                                  memory_width,
                                  heads),
                              name="keys")),
        ("strengths",
         lambda v: 1 + tf.nn.softplus((tf.reshape(v, (-1, heads),
                                                  name="strengths_")),
                                      name="strengths")),
        ("gates",
         lambda v: tf.nn.sigmoid(tf.reshape(v, (-1, heads),
                                            name="gates_"),
                                 name="gates")),
        ("shifts",
         lambda v: tf.nn.softmax(tf.reshape(v, (-1, 3, heads),
                                            name="shifts_"),
                                 axis=1,
                                 name="shifts")),
        ("gammas",
         lambda v: 1 + tf.nn.softplus(tf.reshape(v, (-1, heads),
                                                 name="gammas_"),
                                      name="gammas")),
        ("erase_vectors",
         lambda v: tf.nn.sigmoid(tf.reshape(v, (-1,
                                                memory_width,
                                                write_heads),
                                            name="erase_vectors_"),
                                 name="erase_vectors")),
        ("add_vectors",
         lambda v: tf.reshape(v, (-1,
                                  memory_width,
                                  write_heads),
                              name="add_vectors_")),
    ])
    indices = [[sum(sizes[:i]), sum(sizes[:i + 1])] for i in range(len(sizes))]
    zipped_items = zip(fns.keys(), fns.values(), indices)
    interface = {name: fn(interface_vector[:, i[0]:i[1]])
                 for name, fn, i in zipped_items}

    return NTMinterface(**interface)


class NTMAttention:
    def __init__(self, memory_rows, memory_width, read_heads, write_heads):
        self.memory_width = memory_width
        self.memory_rows = memory_rows
        self.read_heads = read_heads
        self.write_heads = write_heads
        self.heads = read_heads + write_heads

    def __call__(self, parameters, memory, prev_state):
        interface = pars_params(self.read_heads,
                                self.write_heads,
                                self.memory_width,
                                parameters)
        prev_w = prev_state.weightings
        with tf.name_scope("Addressing"):
            k = interface.keys
            beta = interface.strengths
            g = interface.gates
            s = interface.shifts
            gamma = interface.gammas
            with tf.name_scope('addressing_head'):
                weightings = self.addressing(k,
                                             beta,
                                             g,
                                             s,
                                             gamma,
                                             memory,
                                             prev_w)
        read_weightings = weightings[:, :, :self.read_heads]
        read_vectors = tf.matmul(memory, read_weightings, adjoint_a=True)
        write_weightings = weightings[:, :, self.read_heads:]
        M_new = memory
        erase_vectors = interface.erase_vectors
        add_vectors = interface.add_vectors
        with tf.name_scope('writing_head'):
            erase = M_new * (
                (1 - tf.einsum("bnh,bwh->bnw", write_weightings, erase_vectors)))
            write = tf.einsum("bnh,bwh->bnw", write_weightings, add_vectors)
            M_new = erase + write
        return M_new, read_vectors, NTMAttentionState(weightings=weightings)

    @tf.function
    def addressing(self, k, beta, g, s, gamma, prev_M, prev_w):
        w_c = content_addressing(prev_M, k, beta)
        g = tf.expand_dims(g, axis=1)
        w_g = g * w_c + (1 - g) * prev_w

        s_ = tf.roll(tf.pad(s, [[0, 0],
                                [0, self.memory_rows-3],
                                [0, 0]]), -1, axis=1)
        t = tf.concat([tf.reverse(s_, axis=[1]),
                       tf.reverse(s_, axis=[1])],
                      axis=1)
        s_matrix = tf.stack(
            [t[:, self.memory_rows - i - 1:self.memory_rows * 2 - i - 1]
             for i in range(self.memory_rows)],
            axis=1
            )
        w_ = tf.reduce_sum(tf.expand_dims(w_g, axis=1) * s_matrix, axis=2)
        w_sharpen = tf.pow(w_, tf.expand_dims(gamma, axis=1))
        w = w_sharpen / tf.reduce_sum(w_sharpen, axis=1, keepdims=True)

        return w

    @property
    def state_size(self):
        return NTMAttentionState(
            weightings=tf.TensorShape([self.memory_rows, self.heads])
        )

    def get_initial_state(self,
                          inputs=None,
                          batch_size=None,
                          dtype=tf.float32):
        return NTMAttentionState(
            weightings=tf.fill((batch_size, self.memory_rows, self.heads),
                               EPSILON,
                               name="attention/weightings"),
        )

    def num_interface_params(self):
        size = (self.heads * self.memory_width
                + self.heads
                + self.heads
                + 3 * self.heads
                + self.heads
                + self.memory_width*self.write_heads
                + self.memory_width*self.write_heads)
        return size

    def read_vectors_size(self):
        return tf.TensorShape([self.memory_width, self.read_heads])
