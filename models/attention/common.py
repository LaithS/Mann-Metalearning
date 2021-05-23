import tensorflow as tf

EPSILON = 1e-6


@tf.function
def content_addressing(memory_matrix, keys, strengths=None):
    memory_normalised = tf.math.l2_normalize(memory_matrix, 2, epsilon=EPSILON)
    keys_normalised = tf.math.l2_normalize(keys, 1, epsilon=EPSILON)
    similiarity = tf.matmul(memory_normalised, keys_normalised)
    if strengths is not None:
        strengths = tf.expand_dims(tf.math.softplus(strengths), 1)
    else:
        return tf.math.softmax(similiarity, 1)
    return tf.math.softmax(similiarity * strengths, 1)


@tf.function
def update_precedence_vector(prev_precedence_vector, write_weighting):

    write_strength = tf.reduce_sum(
        input_tensor=write_weighting, axis=1, keepdims=True)
    updated_precedence_vector = (
        1 - write_strength) * prev_precedence_vector + write_weighting

    return updated_precedence_vector


@tf.function
def update_link_matrix(prev_link_matrix,
                       prev_precedence_vector,
                       write_weighting):

    # [b x N x 1 ] duplicate columns
    write_weighting_i = tf.expand_dims(write_weighting, 2)
    write_weighting_j = tf.expand_dims(write_weighting, 1)  # [b*1*N] dupl rows
    prev_precedence_vector_j = tf.expand_dims(
        prev_precedence_vector, 1)  # [b x 1 X N]

    link_matrix = (
        (1 - write_weighting_i - write_weighting_j) * prev_link_matrix
        + (write_weighting_i * prev_precedence_vector_j)
    )
    zero_diagonal = tf.zeros_like(write_weighting, dtype=link_matrix.dtype)

    return tf.linalg.set_diag(link_matrix, zero_diagonal)


@tf.function
def temporal_addressing(link_matrix, prev_read_weightings):
    forward_weighting = tf.matmul(link_matrix, prev_read_weightings)
    backward_weighting = tf.matmul(
        link_matrix, prev_read_weightings, adjoint_a=True)

    return forward_weighting, backward_weighting


@tf.function
def update_usage_vector(free_gates,
                        prev_read_weightings,
                        prev_write_weighting,
                        prev_usage_vector):
    with tf.name_scope('update_usage'):
        retention_vector = tf.reduce_prod(
            input_tensor=(1
                          - tf.expand_dims(free_gates, 1)
                          * prev_read_weightings),
            axis=2,
        )
        usage_vector = (
            (prev_usage_vector + prev_write_weighting
                - (prev_usage_vector * prev_write_weighting))
            * retention_vector
        )
        return usage_vector


@tf.function
def batch_unsort(tensor, indices):
    indices_inverted = tf.map_fn(fn=tf.math.invert_permutation, elems=indices)
    batch_unsorted = tf.gather(tensor,
                               indices_inverted,
                               axis=1,
                               batch_dims=1)
    return batch_unsorted


@tf.function
def allocation_addressing(usage_vector, words_num):
    with tf.name_scope('allocation_addressing'):
        usage = (1 - EPSILON) * usage_vector + EPSILON
        emptiness = 1 - usage
        emptiness_sorted, free_list = tf.nn.top_k(emptiness, k=words_num)
        usage_sorted = 1 - emptiness_sorted
        allocation_sorted = (emptiness_sorted
                             * tf.math.cumprod(usage_sorted,
                                               axis=1,
                                               exclusive=True)
                             )
    return batch_unsort(allocation_sorted, free_list)
