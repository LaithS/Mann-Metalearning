import tensorflow as tf
from .mann_cell import MANNCell
from .attention import (NTMAttentionConfig,
                        DNCAttentionConfig,
                        LRUAttentionConfig)


def residual_conv_block(filters, strides=(1, 1), cut='pre'):
    def layer(input):

        x = tf.keras.layers.BatchNormalization(axis=3,
                                               epsilon=2e-5)(input)
        x = tf.keras.layers.ReLU()(x)

        if cut == 'pre':
            shortcut = input
        elif cut == 'post':
            shortcut = tf.keras.layers.Conv2D(filters,
                                              (1, 1),
                                              strides=strides,
                                              use_bias=False,
                                              kernel_initializer='he_uniform'
                                              )(x)

        x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
        x = tf.keras.layers.Conv2D(filters,
                                   (3, 3),
                                   strides=strides,
                                   use_bias=False,
                                   kernel_initializer='he_uniform')(x)

        x = tf.keras.layers.BatchNormalization(axis=3,
                                               epsilon=2e-5)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
        x = tf.keras.layers.Conv2D(filters,
                                   (3, 3),
                                   use_bias=False,
                                   kernel_initializer='he_uniform')(x)

        x = tf.keras.layers.Add()([x, shortcut])
        return x

    return layer


def create_resnet_18(image_size):
    in_image = tf.keras.layers.Input(image_size)
    x = tf.keras.layers.BatchNormalization(axis=3,
                                           epsilon=2e-5,
                                           scale=False)(in_image)
    x = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(x)
    x = tf.keras.layers.Conv2D(64,
                               (7, 7),
                               strides=(2, 2),
                               use_bias=False,
                               kernel_initializer='he_uniform')(x)
    x = tf.keras.layers.BatchNormalization(axis=3,
                                           epsilon=2e-5)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
    x = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2))(x)

    for stage in range(4):
        for block in range(2):

            filters = 64 * (2 ** stage)

            if block == 0 and stage == 0:
                x = residual_conv_block(filters, strides=(1, 1),
                                        cut='post')(x)

            elif block == 0:
                x = residual_conv_block(filters, strides=(2, 2),
                                        cut='post')(x)

            else:
                x = residual_conv_block(filters, strides=(1, 1),
                                        cut='pre')(x)

    x = tf.keras.layers.BatchNormalization(axis=3,
                                           epsilon=2e-5)(x)
    x = tf.keras.layers.ReLU()(x)
    cnn_model = tf.keras.Model(in_image, x)
    return cnn_model


def create_4layer_conv(image_size):
    in_image = tf.keras.layers.Input(image_size)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='SAME')(in_image)
    x = tf.keras.layers.BatchNormalization()(x)
    for i in range(3):
        with tf.name_scope(f"convbn_layer{i}"):
            x = tf.keras.layers.Conv2D(64, (3, 3), padding='SAME')(x)
            x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Flatten()(x)
    cnn_model = tf.keras.Model(in_image, x)
    return cnn_model


def create_cnn(cnn, image_size):
    if cnn == '4layerConv':
        return create_4layer_conv(image_size)
    elif cnn == 'ResNet':
        return create_resnet_18(image_size)
    else:
        raise NotImplementedError(cnn)


def create_cell(cell_type,
                output_dim,
                controller_units,
                memory_rows,
                memory_width,
                clip):
    if cell_type == 'LSTM':
        cell = tf.keras.layers.LSTMCell(256)
    elif cell_type == 'NTM':
        cell = MANNCell(output_dim,
                        controller_units,
                        'NTM',
                        NTMAttentionConfig(4, 1),
                        memory_rows,
                        memory_width,
                        clip)
    elif cell_type == 'DNC':
        cell = MANNCell(output_dim,
                        controller_units,
                        'DNC',
                        DNCAttentionConfig(1),
                        memory_rows,
                        memory_width,
                        clip)
    elif cell_type == 'LRU':
        cell = MANNCell(output_dim,
                        controller_units,
                        'LRU',
                        LRUAttentionConfig(1, 0.95),
                        memory_rows,
                        memory_width,
                        clip)
    elif cell_type == 'NTMold':
        cell = NTMCell(output_dim,
                       1,
                       controller_units,
                       memory_rows,
                       memory_width,
                       1,
                       1,
                       1,
                       clip)
    elif cell_type == 'NTMv2':
        cell = NTMCellV2(output_dim,
                       1,
                       controller_units,
                       memory_rows,
                       memory_width,
                       1,
                       1,
                       1,
                       clip)
    elif cell_type == 'LRUold':
        cell = LRUCell(controller_units, memory_rows, memory_width, 1,
                 gamma=0.95)
    else:
        raise NotImplementedError(cell_type)
    return cell


def create_recurrent_layer(cell,
                           batch_size,
                           output_dim,
                           controller_units,
                           memory_rows,
                           memory_width,
                           clip):
    cells = create_cell(cell,
                        output_dim,
                        controller_units,
                        memory_rows,
                        memory_width,
                        clip)
    initial_states = cells.get_initial_state(batch_size=batch_size,
                                             dtype=tf.float32)

    rnn = tf.keras.layers.RNN(cells,
                              return_sequences=True)
    return rnn, initial_states


def create_algorithmic_task_model(input_size,
                                  output_size,
                                  batch_size,
                                  cell):
    x = tf.keras.layers.Input((None, input_size), batch_size=batch_size)
    x0 = tf.stop_gradient(x)
    rnn, initial_state = create_recurrent_layer(cell,
                                                batch_size,
                                                output_size,
                                                100,
                                                128,
                                                20,
                                                20)
    y0 = rnn(x0, initial_state=initial_state)
    y1 = tf.keras.layers.Dense(output_size)(y0)
    model = tf.keras.Model(x, y1)
    return model


def create_one_hot_model(input_size, output_size, batch_size, cell):
    x = tf.keras.layers.Input((None, input_size), batch_size=batch_size)
    x0 = tf.stop_gradient(x)
    cell = create_cell(cell,
                       output_size,
                       100,
                       128,
                       20,
                       10)
    initial_state = cell.get_initial_state(batch_size=batch_size,
                                           dtype=tf.float32)
    rnn = tf.keras.layers.RNN(cell, return_sequences=True)
    y0 = rnn(x0, initial_state=initial_state)
    y1 = tf.keras.layers.Dense(output_size)(y0)
    y2 = tf.keras.layers.Softmax()(y1)
    model = tf.keras.Model(x, y2)
    return model


def create_string_model(input_size, batch_size, cell):
    x = tf.keras.layers.Input((None, input_size), batch_size=batch_size)
    x0 = tf.stop_gradient(x)
    cell = create_cell(cell,
                       25,
                       100,
                       128,
                       20,
                       10)
    initial_state = cell.get_initial_state(batch_size=batch_size,
                                           dtype=tf.float32)
    rnn = tf.keras.layers.RNN(cell, return_sequences=True)
    y0 = rnn(x0, initial_state=initial_state)
    y1 = tf.keras.layers.Dense(25)(y0)
    y2 = tf.split(y1, 5, axis=-1)
    y3 = [tf.keras.layers.Softmax()(y2[i]) for i in range(len(y2))]
    y4 = tf.concat(y3, -1)
    model = tf.keras.Model(x, y4)
    return model


def create_one_hot_model_convolutional(image_size,
                                       extra_channel,
                                       output_size,
                                       batch_size,
                                       cell,
                                       cnn):
    labels = tf.keras.layers.Input([None] + extra_channel,
                                   batch_size=batch_size)
    images = tf.keras.layers.Input([None] + image_size,
                                   batch_size=batch_size)
    labels0 = tf.stop_gradient(labels)
    images0 = tf.stop_gradient(images)
    vgg = create_cnn(cnn, image_size)
    cnn_timeseries = tf.keras.layers.TimeDistributed(vgg)
    flatten = tf.keras.layers.Flatten()
    flatten_timeseries = tf.keras.layers.TimeDistributed(flatten)
    linear = tf.keras.layers.Dense(1600)
    linear_timeseries = tf.keras.layers.TimeDistributed(linear)
    cell = create_cell(cell,
                       output_size,
                       100,
                       128,
                       20,
                       10)

    initial_state = cell.get_initial_state(batch_size=batch_size,
                                           dtype=tf.float32)
    rnn = tf.keras.layers.RNN(cell, return_sequences=True)
    cnn_out = cnn_timeseries(images0)
    flatten_out = flatten_timeseries(cnn_out)
    linear_out = linear_timeseries(flatten_out)
    cnn_out_plus_labels = tf.concat([linear_out, labels0], axis=-1)
    y1 = rnn(cnn_out_plus_labels, initial_state=initial_state)
    y2 = tf.keras.layers.Dense(output_size)(y1)
    y3 = tf.keras.layers.Softmax()(y2)
    model = tf.keras.Model([images, labels], y3)
    return model
