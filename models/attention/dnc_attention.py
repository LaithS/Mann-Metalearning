import tensorflow as tf

from collections import namedtuple, OrderedDict
from .common import (content_addressing,
                     allocation_addressing,
                     temporal_addressing,
                     update_usage_vector,
                     update_link_matrix,
                     update_precedence_vector)


DNCAttentionState = namedtuple('DNCAttentionState', ('usage_vector',
                                                     'link_matrix',
                                                     'precedence_vector',
                                                     'write_weighting',
                                                     'read_weightings'
                                                     )
                               )
DNCAttentionConfig = namedtuple('DNCAttentionConfig', ('read_heads'))

EPSILON = 1e-6

DNCinterface = namedtuple("interface", [
    "read_keys",
    "read_strengths",
    "write_key",
    "write_strength",
    "erase_vector",
    "write_vector",
    "free_gates",
    "allocation_gate",
    "write_gate",
    "read_modes",
])


@tf.function
def pars_params(read_heads, memory_width, interface_vector):
    sizes = [read_heads * memory_width,
             read_heads, memory_width,
             1,
             memory_width,
             memory_width,
             read_heads,
             1,
             1,
             3 * read_heads]
    fns = OrderedDict([
        ("read_keys",
         lambda v: tf.reshape(v, (-1, memory_width, read_heads),
                              name='read_keys')),
        ("read_strengths",
         lambda v: 1 + tf.nn.softplus((tf.reshape(v, (-1, read_heads))),
                                      name='read_strengths')),
        ("write_key",
         lambda v: tf.reshape(v, (-1, memory_width, 1),
                              name='write_key')),
        ("write_strength",
         lambda v: 1 + tf.nn.softplus((tf.reshape(v, (-1, 1))),
                                      name='write_strength')),
        ("erase_vector",
         lambda v: tf.nn.sigmoid(tf.reshape(v, (-1, memory_width)),
                                 name='erase_vector')),
        ("write_vector",
         lambda v: tf.reshape(v, (-1, memory_width),
                              name='write_vector')),
        ("free_gates",
         lambda v: tf.nn.sigmoid(tf.reshape(v, (-1, read_heads)),
                                 name='free_gates')),
        ("allocation_gate",
         lambda v: tf.nn.sigmoid(tf.reshape(v, (-1, 1)),
                                 name='allocation_gate')),
        ("write_gate",
         lambda v: tf.nn.sigmoid(tf.reshape(v, (-1, 1)),
                                 name='write_gate')),
        ("read_modes",
         lambda v: tf.nn.softmax(tf.reshape(v, (-1, 3, read_heads)), axis=1,
                                 name='read_modes')),
    ])
    indices = [[sum(sizes[:i]), sum(sizes[:i + 1])] for i in range(len(sizes))]
    zipped_items = zip(fns.keys(), fns.values(), indices)
    interface = {name: fn(interface_vector[:, i[0]:i[1]])
                 for name, fn, i in zipped_items}

    return DNCinterface(**interface)


@tf.function
def read(memory_matrix, prev_read_weightings, link_matrix, interface):
    with tf.name_scope("content_addressing"):
        lookup_weighting = content_addressing(
            memory_matrix,
            interface.read_keys,
            interface.read_strengths
        )
    with tf.name_scope("temporal_link_addressing"):
        forward_weighting, backward_weighting = temporal_addressing(
            link_matrix,
            prev_read_weightings,
        )
    with tf.name_scope("blend_addressing_modes"):
        read_weightings = tf.einsum(
            "bsr,bnrs->bnr",
            interface.read_modes,
            tf.stack([backward_weighting, lookup_weighting,
                      forward_weighting], axis=3)
        )
    read_vectors = tf.matmul(memory_matrix, read_weightings, adjoint_a=True)

    return read_weightings, read_vectors


@tf.function
def write(memory_matrix, prev_state, interface):
    m = prev_state
    i = interface

    with tf.name_scope("calculate_weighting"):
        with tf.name_scope("allocation_addressing"):
            usage_vector = update_usage_vector(
                i.free_gates,
                m.read_weightings,
                m.write_weighting,
                m.usage_vector
            )
            allocation_weighting = allocation_addressing(usage_vector,
                                                         memory_matrix.shape[1]
                                                         )
        with tf.name_scope("content_addressing"):
            lookup_weighting = content_addressing(
                memory_matrix,
                i.write_key,
                i.write_strength
            )
        write_weighting = (
            i.write_gate * (i.allocation_gate
                            * allocation_weighting
                            + (1 - i.allocation_gate)
                            * tf.squeeze(lookup_weighting)
                            )
        )
    with tf.name_scope("erase_and_write"):
        erase = memory_matrix * (
            (1 - tf.einsum("bn,bw->bnw", write_weighting, i.erase_vector)))
        write = tf.einsum("bn,bw->bnw", write_weighting, i.write_vector)
        memory_matrix = erase + write

    with tf.name_scope("final_update"):
        link_matrix = update_link_matrix(
            m.link_matrix,
            m.precedence_vector,
            write_weighting
        )
        precedence_vector = update_precedence_vector(
            m.precedence_vector,
            write_weighting
        )

    return usage_vector, write_weighting, memory_matrix, \
        link_matrix, precedence_vector


class DNCAttention:
    def __init__(self, memory_rows, memory_width, read_heads):
        self.memory_width = memory_width
        self.memory_rows = memory_rows
        self.read_heads = read_heads

    def __call__(self, parameters, memory, prev_state):
        interface = pars_params(self.read_heads, self.memory_width, parameters)
        with tf.name_scope("write"):
            usage_vector, write_weighting, memory_new, link, precedence = write(
                memory,
                prev_state,
                interface,
            )

        with tf.name_scope("read"):
            read_weightings, read_vectors = read(
                memory_new,
                prev_state.read_weightings,
                link,
                interface,
            )
        return memory_new, read_vectors, DNCAttentionState(
            usage_vector=usage_vector,
            link_matrix=link,
            precedence_vector=precedence,
            write_weighting=write_weighting,
            read_weightings=read_weightings,
        )

    @property
    def state_size(self):
        return DNCAttentionState(
            usage_vector=tf.TensorShape([self.memory_rows]),
            link_matrix=tf.TensorShape([self.memory_rows, self.memory_rows]),
            precedence_vector=tf.TensorShape([self.memory_rows]),
            write_weighting=tf.TensorShape([self.memory_rows]),
            read_weightings=tf.TensorShape(
                [self.memory_rows, self.read_heads]),
        )

    def get_initial_state(self,
                          inputs=None,
                          batch_size=None,
                          dtype=tf.float32):
        return DNCAttentionState(
            usage_vector=tf.zeros([batch_size, self.memory_rows], dtype=dtype),
            link_matrix=tf.zeros(
                [batch_size, self.memory_rows, self.memory_rows], dtype=dtype),
            precedence_vector=tf.zeros(
                [batch_size, self.memory_rows], dtype=dtype),
            write_weighting=tf.fill([batch_size, self.memory_rows], EPSILON),
            read_weightings=tf.fill(
                [batch_size, self.memory_rows, self.read_heads], EPSILON),
        )

    def num_interface_params(self):
        size = (self.read_heads * self.memory_width
                + self.read_heads + self.memory_width + 1
                + self.memory_width + self.memory_width
                + self.read_heads + 1 + 1 + 3 * self.read_heads)
        return size

    def read_vectors_size(self):
        return tf.TensorShape([self.memory_width, self.read_heads])
