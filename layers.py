import tensorflow as tf
import numpy as np

def lrelu(out):
    return tf.maximum(0.05 * out, out)


def tans(out):
    return 2*tf.nn.sigmoid(out) - 1

def gelu_fast(_x):
    return 0.5 * _x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (_x + 0.044715 * tf.pow(_x, 3))))

def isru(_x, alpha=1):
    return _x / (tf.sqrt(1 + alpha * tf.square(_x)))

activations = {'selu': tf.nn.selu,
               'relu': tf.nn.relu,
               'lrelu': lrelu,
               'tanh': tf.nn.tanh,
               'tans': tans,
               'sigm': tf.nn.sigmoid,
               'elu': tf.nn.elu,
               'silu': lambda x: x*tf.sigmoid(x),
               'gelu': gelu_fast,
               'geluf': lambda x: tf.sigmoid(1.702 * x) * x,
               'isru': isru,
               'none': lambda x: x}

def dropconnect(layer_out, keep_prob):
    with tf.name_scope('dropconnect'):
        # keep_prob = tf.placeholder(tf.float32)
        return tf.nn.dropout(layer_out, keep_prob=keep_prob) * keep_prob


def dropout(layer_out, keep_prob):
    with tf.name_scope('dropout'):
        # keep_prob = tf.placeholder(tf.float32)
        return tf.nn.dropout(layer_out, keep_prob=keep_prob)


def conv1d(input, filters,
           kernel_size=9, stride=2, keep_prob=1.0, dc=False, name="conv1d", activation='selu',
           padding="SAME",
           stddev=0.02, bias=False):

    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel_size, input.get_shape().as_list()[-1], filters],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))

        if dc is True:
            w = dropconnect(w, keep_prob)

        c = tf.nn.conv1d(input, w, stride, padding=padding)

        if bias:
            b = tf.get_variable('b', [filters], initializer=tf.constant_initializer(0.0))
            act = activations[activation](c + b)
        else:
            act = activations[activation](c)

        if dc is False:
            act = dropout(act, keep_prob)

    return act


def transform_conv(x, flags, name="transform"):
    count, stride, window, batch_size, size_in, size_out = flags

    with tf.name_scope(name):
        # TODO: Подумать над размерностью
        # b = tf.Variable(tf.constant(0.1, shape=[batch_size, size_out]), name="B")
        w = tf.ones([count, stride, size_out], name="W", dtype=tf.float64)
        # w = tf.Variable(tf.truncated_normal([count, stride, size_out], stddev=0.1), name="W")
        conv = tf.nn.conv1d(x, w, stride=1, padding="VALID")

        # w = tf.ones([window, 1, size_out], name="W")
        # conv = tf.reshape(tf.nn.conv1d(input, w, stride=int(stride/4), padding="VALID"), [batch_size, -1, 1])

        # act = tf.nn.selu(conv + b)
        #
        # tf.summary.histogram("weights", w)
        # tf.summary.histogram("biases", b)
        # tf.summary.histogram("activations", act)

        return conv


def transform(x, count, stride, size_out, keep_prob, dc=False, name="transform", activation='selu', trainable=True):
    window = count*stride
    size_in = int(x.shape[-1])

    with tf.variable_scope(name):

        buf = []
        for n in range(count-1):
            slice = (x[:, stride * n:-(window - stride * (n + 1))])
            buf.append(slice)
        buf.append(x[:, stride * (count - 1):])

        concat = tf.concat(buf, axis=1)
        reshape = tf.reshape(concat, [-1, count, size_in-window+stride])
        transpose = tf.transpose(reshape, [0, 2, 1])
        out = tf.reshape(transpose, [-1, int((size_in-window+stride)/stride), 1, window])

        w = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1),
                            shape=[int(out.shape[1]), window, size_out],
                            name="W", trainable=trainable)

        if dc is True:
            w = dropconnect(w, keep_prob)

        out_shape = size_out*out.shape[1]

        b = tf.get_variable(initializer=tf.constant_initializer(0.0), shape=[out_shape],
                            name="B", trainable=trainable)

        act = activations[activation](tf.reshape(tf.map_fn(lambda sample: sample @ w, out), [-1, out_shape]) + b)

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)

        if dc is False:
            act = dropout(act, keep_prob)

        return act


def split_layer(x, flags, split, size_out, keep_prob, dc=False, name="split_layer", activation='selu', reshape=True):
    count, stride, window, batch_size, size_in, _ = flags

    with tf.name_scope(name):

        if reshape:
            x = tf.reshape(x, [batch_size, split, 1, -1])

        w = tf.Variable(tf.truncated_normal([split, int(size_in/split), size_out], stddev=0.1), name="W", trainable=True)

        if dc is True:
            w = dropconnect(w, keep_prob)

        b = tf.Variable(tf.constant(0.1, shape=[split, 1, size_out]), name="B", trainable=True)

        act = activations[activation](tf.map_fn(lambda sample: sample@w, x) + b)

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)

        if dc is False:
            act = dropout(act, keep_prob)

        return act


def fc_layer(x, size_out, keep_prob=0.5, dc=None, name="fc_layer", act='selu', reuse=False, trainable=True):

    size_in = int(x.shape[1])
    # size_in = x.shape[1]
    with tf.variable_scope(name, reuse=reuse):

        # w = tf.Variable(tf.ones([size_in, size_out]), name="W")

        # w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W", trainable=trainable)
        # b = tf.Variable(tf.constant(0.1, shape=[size_out], dtype=tf.float32), name="B", trainable=trainable)

        w = tf.get_variable(initializer=tf.truncated_normal_initializer(stddev=0.1), shape=[size_in, size_out],
                            name="W", trainable=trainable)

        if dc is True:
            w = dropconnect(w, keep_prob)

        b = tf.get_variable(initializer=tf.constant_initializer(0.1), shape=[size_out],
                            name="B", trainable=trainable)

        act = activations[act](x @ w + b)

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)

        if dc is False:
            act = dropout(act, keep_prob)

        return act


