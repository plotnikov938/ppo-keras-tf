import tensorflow as tf
import numpy as np


def cross_entropy(p, q):
    return -tf.reduce_sum(p * tf.log(q + 1e-10), axis=-1)


class Categorical:
    def __init__(self, logits):
        self.logits = logits
        self.probs = tf.nn.softmax(logits)

    def sample(self):
        action = tf.random.categorical(tf.log(self.probs), num_samples=1)
        return tf.squeeze(action)

    def neglogp(self, x):
        if x.dtype not in (tf.uint8, tf.int32, tf.int64):
            print('Casting dtype of x to tf.int32')
            x = tf.cast(x, tf.int32)

        return cross_entropy(tf.one_hot(x, self.logits.shape[-1]), self.probs)

    def entropy(self):
        return cross_entropy(self.probs, self.probs)
        # return tf.reduce_mean(cross_entropy(self.probs, self.probs), axis=-1)


class Normal:
    def __init__(self, mean=.0, logstd=1.0):
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / (self.std + 1e-10)) +
                                   np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], tf.float32) +
                                   2 * self.logstd, axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)


# class MultiCategorical:
#     def __init__(self, vec, n):
#         self.vec = vec
#         self.n = n
#
#         self.distributions = list(map(Categorical, tf.split(vec, n, axis=-1)))
#
#     def sample(self):
#         return tf.cast(tf.stack([p.sample() for p in self.distributions], axis=-1), tf.int32)
#
#     def neglogp(self, x):
#         return tf.add_n([p.neglogp(_x) for p, _x in zip(self.distributions, tf.unstack(x, axis=-1))])
#
#     def entropy(self):
#         return tf.add_n([p.entropy() for p in self.distributions])

# class Multi:
#     def __init__(self, distributions):
#         self.distributions = distributions
#
#     def sample(self):
#         # return tf.cast(tf.stack([p.sample() for p in self.distributions], axis=-1), tf.int32)
#         return tf.stack([p.sample() for p in self.distributions], axis=-1)
#
#     def neglogp(self, x):
#         return tf.add_n([p.neglogp(_x) for p, _x in zip(self.distributions, tf.unstack(x, axis=-1))])
#
#     def entropy(self):
#         return tf.add_n([p.entropy() for p in self.distributions])
#
#
# class MultiCategorical(Multi):
#     def __init__(self, vec):
#         self.vec = vec
#         n = self.vec.shape[0]
#
#         categoricals = list(map(Categorical, tf.split(vec, n, axis=0)))
#         super().__init__(categoricals)
#
#     def sample(self):
#         return tf.cast(super().sample(), tf.int32)
#
#
# class MultiNormal(Multi):
#     def __init__(self,  means, logstds):
#         assert means.shape == logstds.shape
#
#         self.means = means
#         self.logstds = logstds
#         n = self.means.shape[0]
#
#         normals = list(map(Normal, *[tf.split(vec, n, axis=-1) for vec in [means, logstds]]))
#         # normals = [Normal(means[idx], logstds[idx]) for idx in range(n)]
#         super().__init__(normals)
#

# with tf.Session() as sess:
#     means = tf.Variable(np.array([[0., 1., 2.]]), dtype=tf.float32)
#     logstds = tf.Variable(np.array([-2, -2, -2]), dtype=tf.float32)
#     var = tf.Variable([2])
#     print(means.shape)
#     print(var.shape)
#     # result = Normal(mean=means, logstd=logstds)
#     result = Categorical(means)
#     # result = MultiCategorical(means)
#     s, e, n = result.sample(), result.entropy(), result.neglogp(var)
#     sess.run(tf.global_variables_initializer())
#     for x in range(10):
#         print(sess.run(s))
#         print(sess.run(e))
#         print(sess.run(n))
