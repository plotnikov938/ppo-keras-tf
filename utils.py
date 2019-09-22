import numpy as np
import tensorflow as tf


def make_gif():
    pass


def save_target_graph(sess, folder=None, filename=None, target_name=None):
    train_vars = tf.trainable_variables(target_name)

    if not os.path.exists(folder):
        os.makedirs(folder)

    buf = []
    for var in train_vars:
        name = var.name
        shape = var.get_shape().as_list()
        weights = var.eval(sess)

        # Convert name to numpy float array
        name_converted = str2np(name)

        # Append var_name and var_weights to the list of numpy arrays as follows:
        # 'converted var name' + 'np.nan' + 'var weights' + 'np.nan' (here np.nan is used as separator)
        buf.extend([name_converted, [np.nan], shape, [np.nan], weights.ravel(), [np.nan]])

    np.save(folder + filename, np.concatenate(buf))


def restore_target_graph(sess, path=None, target_name=None):
    arr = np.load(path)
    args = np.argwhere(np.isnan(arr)).ravel() + 1
    for name, shape, weights in zip(*[iter(np.split(arr, args))] * 3):
        name = np2str(name[:-1])
        weights = weights[:-1].reshape(shape[:-1].astype(int))
        var = tf.trainable_variables(name)[0]

        var.load(weights, sess)