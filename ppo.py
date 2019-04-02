import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout
import numpy as np
from layers import fc_layer
from common import save_target_graph, restore_target_graph
import gym


class CategoricalOld:
    def __init__(self, logits):
        self.logits = logits

    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits))

        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)

    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0

        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

    def neglogp(self, x):
        if x.dtype not in (tf.uint8, tf.int32, tf.int64):
            print('Casting dtype of x')
            x = tf.cast(x, tf.int32)

        return tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits,
            labels=tf.one_hot(x, self.logits.shape[-1]))


def cross_entropy(p, q):
    return -tf.reduce_sum(p * tf.log(q + 1e-10), axis=-1)


# class Categorical:
#     def __init__(self, logits):
#         self.logits = logits
#         self.probs = tf.nn.softmax(logits)
#
#     def sample(self):
#         try:
#             action = tf.random.categorical(tf.log(self.probs), num_samples=1)
#         except ValueError:
#             action = tf.stack([tf.random.categorical(tf.log(self.probs[idx]),
#                                                      num_samples=1) for idx in range(self.probs.shape[0])], axis=0)
#
#         return tf.squeeze(action, axis=-1)
#
#     def neglogp(self, x):
#         if x.dtype not in (tf.uint8, tf.int32, tf.int64):
#             print('Casting dtype of x to tf.int32')
#             x = tf.cast(x, tf.int32)
#
#         return tf.reduce_sum(cross_entropy(tf.one_hot(x, self.logits.shape[-1]), self.probs), axis=-1)
#
#     def entropy(self):
#         return tf.reduce_mean(cross_entropy(self.probs, self.probs), axis=-1)

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

def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = action_space.n
    logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits, 1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi

class Normal:
    def __init__(self, mean=.0, logstd=1.0):
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], tf.float32) \
               + tf.reduce_sum(self.logstd, axis=-1)

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


class Agent:
    # TODO: Remove 'stochastic' from func definition
    def __init__(self, env, actor_net, critic_net, cliprange=0.1, max_grad_norm=None, stochastic=False):
        def get_space_shape(space):
            try:
                return space.shape[0]
            except IndexError:
                return space.n

        # Environment parameters
        self.act_space = env.action_space
        self.act_shape = get_space_shape(self.act_space)
        self.obs_shape = get_space_shape(env.observation_space)

        self.actor_net = actor_net
        self.critic_net = critic_net

        # Reset the graph
        tf.reset_default_graph()

        # Init
        self.states = tf.placeholder(tf.float32, shape=[None, self.obs_shape], name='states')
        self.actions_old = tf.placeholder(tf.float32, shape=[None], name='actions')
        self.rewards = tf.placeholder(tf.float32, shape=[None], name='rewards')
        self.gaes = tf.placeholder(tf.float32, shape=[None], name='advantage')
        self.values_old = tf.placeholder(tf.float32, shape=[None], name='values_old')
        self.neglogps_old = tf.placeholder(tf.float32, shape=[None], name='neglogps_old')
        self.q_values = tf.placeholder(tf.float32, shape=[None], name='estimation')

        self.cliprange = cliprange
        self.keep_prob = tf.Variable(1.0, dtype=tf.float32)
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        """Build the model"""
        # Agent for acting
        with tf.variable_scope('old_agent') as scope:
            self.actions_distrs = self.actor(self.states, keep_prob=self.keep_prob)
            self.values = self.critic(self.states, keep_prob=self.keep_prob)
            policy_vars_old = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name)

        # Agent for training
        with tf.variable_scope('new_agent') as scope:
            self.actions_distrs_new = self.actor(self.states)
            self.values_new = self.critic(self.states)
            policy_vars_new = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name)

        # Sample actions from given distribution
        self.actions = self.actions_distrs.sample()
        try:
            self.actions = tf.clip_by_value(tf.squeeze(self.actions, axis=0), self.act_space.low, self.act_space.high)
        except AttributeError:
            pass

        self.neglogp = self.actions_distrs.neglogp(self.actions)
        self.neglogp_new = self.actions_distrs_new.neglogp(self.actions_old)

        self.synchronize_op = [old_p.assign(p) for old_p, p in zip(policy_vars_old, policy_vars_new)]

        """Losses"""
        with tf.variable_scope('critic_loss'):
            self.values_cliped = self.values_old + \
                                 tf.clip_by_value(self.values_new - self.values_old, -cliprange, cliprange)
            critic_loss = tf.square(self.q_values - self.values)
            critic_loss_clipped = tf.square(self.q_values - self.values_cliped)

            self.critic_loss = 0.5*tf.reduce_mean(tf.maximum(critic_loss, critic_loss_clipped))

        with tf.variable_scope('actor_loss'):

            ratio = tf.exp(self.neglogps_old - self.neglogp_new)

            actor_loss = self.gaes * ratio
            actor_loss_clipped = self.gaes * tf.clip_by_value(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
            self.actor_loss = -tf.reduce_mean(tf.minimum(actor_loss, actor_loss_clipped))

        with tf.variable_scope('entropy_loss'):
            entropy = self.actions_distrs_new.entropy()
            self.entropy_loss = tf.reduce_mean(entropy, axis=0)
            # self.entropy_loss = entropy


        with tf.variable_scope('total_loss'):
            self.loss = self.critic_loss + self.actor_loss# - 0.001*self.entropy_loss

        # Define all trainable variables
        self.train_vars = policy_vars_new

        # Choose the optimizer
        # self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss=self.loss, var_list=self.train_vars)

        # Calculate the gradients
        grads = tf.gradients(self.loss, self.train_vars)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, self.train_vars))

        trainer = tf.train.AdamOptimizer(self.learning_rate)

        # Backpropagate
        self.train_op = trainer.apply_gradients(grads)

        # Define additional features
        pass

        # Merge all the summaries and write them out to selected folder
        self.merged = tf.summary.merge_all()

        # self.var_names = list(self.__dict__.keys())

        # TODO: Доделать
        # buf = {}
        # for k, v in self.__dict__.copy().items():
        #     if k.startswith('loss_') or k.startswith('accuracy_'):
        #         buf.update({k: v})
        # self.loss_dict = buf

    @staticmethod
    def restore(self, sess, folder):
        # Загружаем ранее натренированную модель
        restore_target_graph(sess, folder)

    @staticmethod
    def save(self, sess, folder):
            # Сохраняем модель
            save_target_graph(sess, folder)

    def get_actions_distribution(self, sess, samples, keep_prob):
        return sess.run(self.actions_distrs, {self.states: samples, self.keep_prob: keep_prob})

    def get_value(self, sess, samples, keep_prob):
        return sess.run(self.values, {self.states: samples, self.keep_prob: keep_prob})

    # TODO: Use here distr.sample() func instead of tf.multinomial()
    def get_action(self, sess, samples, keep_prob, stochastic=True):
        if stochastic:
            action = tf.multinomial(tf.log(self.actions_distrs + 1e-10), num_samples=1)
            action = tf.reshape(self.action, shape=[-1])
        else:
            action = tf.argmax(self.actions_distrs, axis=1)

    def evaluate_model(self, sess, samples, keep_prob):
        return sess.run([self.actions, self.values, self.neglogp], feed_dict={self.states: samples, self.keep_prob: keep_prob})

    def synchronize_policies(self, sess):
        sess.run(self.synchronize_op)

    def train_agent(self, sess, states, actions, values, neglogps, gaes, q_values, learning_rate, keep_prob):
        return sess.run([self.loss, self.train_op], feed_dict={self.states: states,
                                                               self.actions_old: actions,
                                                               self.values_old: values,
                                                               self.neglogps_old: neglogps,
                                                               self.gaes: gaes,
                                                               self.q_values: q_values,
                                                               self.learning_rate: learning_rate,
                                                               self.keep_prob: keep_prob})

    def actor(self, states, name='actor', keep_prob=0.5, reuse=False, trainable=True):
        with tf.variable_scope(name, reuse=reuse) as scope:
            features = self.actor_net(states, trainable=trainable)

            # TODO: Доделать
            if isinstance(self.act_space, gym.spaces.Discrete):
                logits = Dense(self.act_shape, None, trainable=trainable, name="layer_logits")(features)
                distribution = Categorical(logits)

            else:
                mu = Dense(self.act_shape, None, trainable=trainable, name='mu')(features)
                sigma = Dense(self.act_shape, None, trainable=trainable, name='sigma')(features)
                distribution = Normal(mean=mu, logstd=sigma)

            return distribution

    def critic(self, states, name='critic', keep_prob=0.5, reuse=False, trainable=True):
        with tf.variable_scope(name, reuse=reuse):
            features = self.critic_net(states, trainable=trainable)

            value = Dense(1, None, trainable=trainable, name="layer_logits")(features)

        return tf.squeeze(value)