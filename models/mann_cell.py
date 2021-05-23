import tensorflow as tf

from collections import namedtuple
from tensorflow.python.util import nest

from .attention import NTMAttention, DNCAttention, LRUAttention

MANNState = namedtuple(
    'MANNState', ('controller_state', 'attention_state', 'M', 'read_vectors'))


def create_attention(attention_type,
                     num_memory_rows,
                     memory_width,
                     attention_config):
    if attention_type == 'NTM':
        return NTMAttention(num_memory_rows, memory_width, *attention_config)
    elif attention_type == 'DNC':
        return DNCAttention(num_memory_rows, memory_width, *attention_config)
    elif attention_type == 'LRU':
        return LRUAttention(num_memory_rows, memory_width, *attention_config)
    else:
        raise NotImplementedError


class MANNCell(tf.keras.layers.Layer):
    def __init__(self,
                 output_dim,
                 num_controller_units,
                 attention_type,
                 attention_config,
                 num_memory_rows,
                 memory_width,
                 clip_value,
                 **kwargs
                 ):
        super().__init__(name='MANNCell', **kwargs)
        self.num_controller_units = num_controller_units
        self.num_memory_rows = num_memory_rows
        self.memory_width = memory_width
        self.clip_value = clip_value
        self.attention = create_attention(
            attention_type, num_memory_rows, memory_width, attention_config)

        total_parameter_num = self.attention.num_interface_params()

        self.controller = tf.keras.layers.LSTMCell(
            num_controller_units, unit_forget_bias=True, name="controller")

        self.output_dim = output_dim

        self.controller_interface = tf.keras.layers.Dense(
            total_parameter_num, name='controller_to_interface')
        self.final_join = tf.keras.layers.Dense(output_dim, name="out_dense")
        self.flatten_read_vectors = tf.keras.layers.Flatten()

    @tf.function
    def call(self, x, prev_state):
        with tf.name_scope("inputs_to_controller"):
            prev_state = nest.pack_sequence_as(
                self.state_size_nested, prev_state)
            prev_attention_state = prev_state.attention_state
            prev_read_vectors = prev_state.read_vectors
            prev_read_vectors = self.flatten_read_vectors(prev_read_vectors)
            controller_input = tf.concat([x, prev_read_vectors], 1)
            controller_output, controller_state = self.controller(
                controller_input, prev_state.controller_state)

        with tf.name_scope("parse_interface"):
            parameters = self.controller_interface(controller_output)
            parameters = tf.clip_by_value(parameters,
                                          -self.clip_value,
                                          self.clip_value)

        with tf.name_scope("attention"):
            M, read_vectors, attention_state = self.attention(
                parameters, prev_state.M, prev_attention_state)

        with tf.name_scope("output"):
            read_vectors_flattened = self.flatten_read_vectors(read_vectors)
            mann_output = self.final_join(
                tf.concat([controller_output, read_vectors_flattened], 1))
            mann_output = tf.clip_by_value(mann_output,
                                           -self.clip_value,
                                           self.clip_value,
                                           name="final_clip")
        mann_state = MANNState(controller_state=controller_state,
                               attention_state=attention_state,
                               M=M,
                               read_vectors=read_vectors)
        return mann_output, nest.flatten(mann_state)

    @property
    def output_size(self):
        return self.output_dim

    def get_initial_state(self,
                          inputs=None,
                          batch_size=None,
                          dtype=tf.float32):
        attention_init_state = self.attention.get_initial_state(
            batch_size=batch_size, dtype=dtype)
        controller_init_state = self.controller.get_initial_state(
            batch_size=batch_size, dtype=dtype)
        M = tf.fill([batch_size, self.num_memory_rows,
                     self.memory_width], 1e-6)
        read_vectors = tf.fill(tf.TensorShape([batch_size])
                               + self.attention.read_vectors_size(), 1e-6)
        init_state = MANNState(
            controller_state=controller_init_state,
            attention_state=attention_init_state,
            M=M,
            read_vectors=read_vectors)
        return nest.flatten(init_state)

    @property
    def state_size_nested(self):
        return MANNState(
            controller_state=self.controller.state_size,
            attention_state=self.attention.state_size,
            M=tf.TensorShape([self.num_memory_rows, self.memory_width]),
            read_vectors=self.attention.read_vectors_size()
        )

    @property
    def state_size(self):
        return nest.flatten(self.state_size_nested)

    def get_config(self):
        return super().get_config()
