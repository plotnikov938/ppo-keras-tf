import numpy as np
import tensorflow as tf
import os
from numba import jit


# TODO: Переделать для быстрой загрузки
def save_target_graph(sess, folder=None, target_name=None):
    train_vars = tf.trainable_variables(target_name)

    if not os.path.exists(folder):
        os.mkdir(folder)

    for var in train_vars:
        name = var.name
        arr = var.eval(sess)
        np.save(folder + '/' + name.replace('/', '_').rpartition(':')[0], arr)

def restore_target_graph(sess, folder=None, target_name=None):
    train_vars = tf.trainable_variables(target_name)

    for var in train_vars:
        name = var.name

        try:
            arr = np.load(folder + '/' + name.replace('/', '_').rpartition(':')[0] + '.npy')
            var.load(arr, sess)
        except FileNotFoundError as e:
            print(e)


def update_target_graph(name_from, name_to):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name_from)

    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name_to)

    op_holder = []

    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))

    return op_holder


def labels_to_binary(labels):
    left = labels.flatten()

    # split_ids = np.argwhere((left - np.roll(left, -1))) + 1
    split_ids = np.argwhere(np.diff(left)) + 1
    splited = np.split(left, split_ids.reshape(-1))

    try:
        ones = splited[1::2] if left[split_ids[0]] > left[split_ids[1]] else splited[::2]
    except IndexError:
        ones = splited[1::2] if left[0] < left[-1] else splited[::2]

    left *= 0
    for item in ones:
        item += 1

    right = np.ones_like(left) - left
    binary = np.vstack([left, right]).T*2 - 1

    return binary


def labels_from_binary(labels, prices):
    import matplotlib.pyplot as plt
    args = np.argwhere(np.diff(labels) != 0).ravel() + 1

    labels = np.zeros_like(labels, dtype=np.float32)
    splited = np.split(labels, args)

    for split, price in zip(splited, prices[args-1]):
        split += price

    return labels[:args[-1]]


def get_classic_set(param, num, max, stock_exchanger, pair, spred, price_selector, pivot_count, ts, step):
    # Получаем точки разворота
    pivot_points = PivotFunc(stock_exchanger,
                             pair,
                             spred,
                             price_selector=price_selector,
                             pivot_count=pivot_count,
                             ts=ts,
                             step=step,
                             start=0,
                             end=None,
                             test=False)

    df = pivot_points.df[-max:]

    pivot_points.set_df(df[df.index >= 0].copy())
    pointers, labels = pivot_points.get_random_pointers(n=-1)

    labels = labels_to_binary(labels)

    # df = df[price_selector].values
    df = df[[*param.keys()]].values.reshape(len(df), -1)

    samples = np.empty([*df.shape, num])
    for x in range(num):
        samples[:, :, x] = np.roll(df, -x, axis=0)

    # df = labels[:, 0]
    # for x in range(num):
    #     samples[:, :, num + x] = np.roll(df, -x, axis=0)

    samples = samples[:-num + 1]
    labels = labels[num - 1:]

    prices = samples[:, 0, -1].copy()

    # samples[:, 0, 0] = samples[:, 0, -1] * 0.994
    # samples[:, 0, 1] = samples[:, 0, -1] * 1.006
    # samples[:, 0, 2] = samples[:, 0, -1] * 0.998
    # samples[:, 0, 3] = samples[:, 0, -1] * 1.002

    for i, func in enumerate(param.values()):
        samples[:, i] = func(samples[:, i])

    # Делаем structured_array для удобства пользования
    # dtype = [(fieldname, float, num) for fieldname in param.keys()]
    # buf = np.zeros(samples.shape[0], dtype=dtype)
    # for name, assign_from in zip(buf.dtype.names, samples.transpose(1, 0, 2)):
    #     buf[name] = assign_from
    # samples = buf

    return samples, labels, prices


def standardization(samples, _max=None, _min=None, axis=1):
    if _max is None:
        _max = samples.max(axis=axis).reshape([-1, 1])
    if _min is None:
        _min = samples.min(axis=axis).reshape([-1, 1])
    samples = 2 * (samples - _min) / (_max - _min + 1e-8) - 1

    return samples


def spred_standartization(_set, spred):
    _max = _set.max(axis=1).reshape(-1, 1)
    _min = _set.min(axis=1).reshape(-1, 1)
    _set = (2 * (_set - _min) / (_max - _min + 1e-8) - 1) * ((_max - _min + 1e-8) / _set[:, -1].reshape(-1, 1) / spred)

    return _set


def standardization_beg(samples, num, c=1):
    std = samples[:, num - 1].reshape(-1, 1).copy()
    samples /= std
    samples /= samples.std() / c
    mean = samples[:, num - 1].reshape(-1, 1).copy()
    samples -= mean

    return samples#, mean


def normalization(_set):
    # _set -= _set.mean(axis=1)
    # _set /= _set.std(axis=1)
    _set -= _set.mean(axis=1).reshape([-1, 1])
    _set /= _set.std(axis=1).reshape([-1, 1])

    return _set


def animate(data_gen, labels, prices, frames=None):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(bottom=0.1)

    line1, = ax.plot(labels, 'r.-', animated=False, lw=1)
    line2, = ax.plot(prices, 'b-', animated=False, lw=1)
    line3, = ax.plot([], 'g', animated=True, lw=2)

    def init():
        return line3,

    def update(data):
        line3.set_data(line1.get_xdata(), data.ravel())

        return line3,

    ani = FuncAnimation(fig, update, init_func=init, frames=data_gen, interval=10, blit=True, repeat=False)

    plt.show()


def hysteresis(distribution, epsilon=0.5):
    distribution -= distribution.min()
    distribution /= distribution.max()

    filt_1 = distribution > (1 - epsilon)
    filt_2 = distribution <= epsilon
    distribution[filt_1] = 1
    distribution[filt_2] = -1
    distribution[~(filt_1 | filt_2)] = 0
    for i in range(1, distribution.shape[0], 1):
        if distribution[i] == 0:
            distribution[i] = distribution[i - 1]

    return distribution


# @jit(nopython=True)
def moving_average(a, n=3):

    ret = np.cumsum(a, axis=0)
    ret[n:] = ret[n:] - ret[:-n]

    return ret[n-1:] / n


def numpy_ewma(data, window):

    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))
    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum(axis=0)
    out = offset + cumsums*scale_arr[::-1]

    return out


class SampleGen:
    def __init__(self, _set, labels=None, classes=None, batch_size=64, pivot_count=2):
        self.set = _set[:]
        self.labels = labels
        self.classes = classes
        self.mini_batch_size = batch_size
        self.pivot_count = pivot_count

        self.buf_arr = np.arange(_set.shape[0])
        self.cut_size = _set.shape[0] // batch_size
        self.counter = 0

        self.indexes = []

    def get_batch(self, random=True):
        #TODO: Принимать и возвращать произвольное колличество пакетов
        # Обновляем заново массив индексов
        if self.counter == 0:
            if random:
                np.random.shuffle(self.buf_arr)
            self.indexes = self.buf_arr[-self.cut_size*self.mini_batch_size:].reshape([-1, self.mini_batch_size])[::-1]
            self.counter = self.cut_size

        self.counter -= 1

        indexes = self.indexes[self.counter]

        samples = self.set[indexes]

        try:
            labels = self.labels[indexes]
        except TypeError as e:
            labels = None

        try:
            classes = self.classes[indexes]
        except TypeError as e:
            classes = None

        return samples, labels, classes

    def get_current_pos(self):
        try:
            if self.indexes[self.counter][1] - self.indexes[self.counter][0] == 1:
                return self.indexes[self.counter][0]
            else:
                return None
        except IndexError as e:
            return None

    def get_indexes(self):
        return self.indexes[self.counter]
