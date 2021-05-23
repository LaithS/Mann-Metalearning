
import tensorflow as tf

from collections import namedtuple, OrderedDict
from .common import content_addressing

LRUAttentionState = namedtuple('LRUAttentionState', ('wu',
                                                     'wr',)
                               )
LRUAttentionConfig = namedtuple('LRUAttentionConfig', ('read_heads', 'gamma'))


LRUinterface = namedtuple("interface", [
    "keys",
    "add_keys",
    "alphas",
])


@tf.function
def pars_params(read_heads, memory_width, interface_vector):
    sizes = [read_heads * memory_width,
             read_heads * memory_width,
             read_heads]
    fns = OrderedDict([
        ("keys",
         lambda v: tf.tanh(tf.reshape(v, (-1, memory_width, read_heads)),
                           name='keys')),  # read value by each head
        ("add_keys",
         lambda v: tf.tanh(tf.reshape(v, (-1, memory_width, read_heads)),
                           name='write_vector')),
        ("alphas",
         lambda v: tf.sigmoid(tf.reshape(v, (-1, read_heads, 1)),
                              name='alphas')),
    ])
    indices = [[sum(sizes[:i]), sum(sizes[:i + 1])] for i in range(len(sizes))]
    zipped_items = zip(fns.keys(), fns.values(), indices)
    interface = {name: fn(interface_vector[:, i[0]:i[1]])
                 for name, fn, i in zipped_items}

    return LRUinterface(**interface)


def variable_one_hot(shape, name=''):
    initial = tf.zeros(shape)
    initial[..., 0] = 1
    return tf.constant(initial, dtype=tf.float32)


class LRUAttention:
    def __init__(self, memory_rows, memory_width, read_heads, gamma):
        self.memory_width = memory_width
        self.memory_rows = memory_rows
        self.read_heads = read_heads
        self.gamma = gamma

    def __call__(self, parameters, memory, prev_state):
        interface = pars_params(self.read_heads, self.memory_width, parameters)
        indices_prev, wlu_prev = self.least_used(prev_state.wu)
        wr = content_addressing(memory, interface.keys)
        ww = self.write_head_addressing(interface.alphas,
                                        prev_state.wr,
                                        wlu_prev)
        wu = (self.gamma
              * prev_state.wu
              + tf.reduce_sum(wr, axis=1, name="wr_sum")
              + tf.reduce_sum(ww, axis=1, name="ww_sum"))
        memory = (memory
                  * tf.expand_dims(1.0
                                   - tf.one_hot(indices_prev[:, -1],
                                                self.memory_rows),
                                   axis=2))
        tmp = tf.matmul(ww, interface.add_keys, adjoint_b=True,
                        name="ww_mul_adds")
        memory = (memory + tmp)
        r = tf.matmul(memory, wr, adjoint_a=True)
        return memory, r, LRUAttentionState(wu=wu, wr=wr)

    def write_head_addressing(self, sig_alpha, wr_prev, wlu_prev):
        with tf.name_scope("write_head_addressing"):
            wlu_prev = tf.expand_dims(wlu_prev, axis=-1)
            return (sig_alpha
                    * wr_prev
                    + (1. - sig_alpha)
                    * wlu_prev)

    def least_used(self, w_u):
        _, indices = tf.nn.top_k(w_u, k=self.memory_rows)
        wlu = tf.cast(tf.slice(indices,
                               [0, self.memory_rows - self.read_heads],
                               [w_u.get_shape()[0], self.read_heads]),
                      dtype=tf.int32)
        wlu = tf.reduce_sum(tf.one_hot(wlu, self.memory_rows), axis=1)
        return indices, wlu

    @property
    def state_size(self):
        return LRUAttentionState(
            wu=tf.TensorShape([self.memory_rows]),
            wr=tf.TensorShape([self.memory_rows, self.read_heads]),

        )

    def get_initial_state(self,
                          inputs=None,
                          batch_size=None,
                          dtype=tf.float32):
        return LRUAttentionState(
            wu=tf.zeros([batch_size, self.memory_rows]),
            wr=tf.zeros([batch_size,
                         self.memory_rows,
                         self.read_heads]),

        )

    def num_interface_params(self):
        sizes = (self.read_heads * self.memory_width
                 + self.read_heads * self.memory_width
                 + self.memory_width)
        return sizes

    def read_vectors_size(self):
        return tf.TensorShape([self.memory_width, self.read_heads])
