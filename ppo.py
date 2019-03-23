import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from layers import fc_layer
from common import save_target_graph, restore_target_graph
import gym


class Categorical_v0:
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


class CategoricalNew:
    def __init__(self, logits):
        self.logits = logits
        self.probs = tf.nn.softmax(logits)

    def sample(self):
        action = tf.random.categorical(tf.log(self.probs), num_samples=1)
        return tf.reshape(action, shape=[-1])

    @staticmethod
    def cross_entropy(p, q):
        return -tf.reduce_sum(p * tf.log(q + 1e-10), axis=1)

    def neglogp(self, x):
        if x.dtype not in (tf.uint8, tf.int32, tf.int64):
            print('Casting dtype of x to tf.int32')
            x = tf.cast(x, tf.int32)

        return self.cross_entropy(tf.one_hot(x, self.logits.shape[-1]), self.probs)

    def entropy(self):
        return self.cross_entropy(self.probs, self.probs)


class Agent:
    def __init__(self, env, cliprange=0.1, max_grad_norm=None, stochastic=False):
        # Environment parameters
        self.act_space = env.action_space
        try:
            self.act_shape = env.action_space.shape[0]
        except IndexError:
            self.act_shape = env.action_space.n
        self.obs_shape = env.observation_space.shape[0]

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
            self.actions_distrs, self.actor_logits = self.actor(self.states, keep_prob=self.keep_prob)
            self.values, self.critic_feature = self.critic(self.states, keep_prob=self.keep_prob)
            policy_vars_old = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name)

        # Agent for training
        with tf.variable_scope('new_agent') as scope:
            self.actions_distrs_new, self.actor_logits_new = self.actor(self.states)
            self.values_new, self.critic_feature_new = self.critic(self.states)
            policy_vars_new = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name)

        self.actions = self.actions_distrs.sample()
        self.neglogp = self.actions_distrs.neglogp(self.actions)

        self.synchronize_op = [old_p.assign(p) for old_p, p in zip(policy_vars_old, policy_vars_new)]

        """Losses"""
        with tf.variable_scope('critic_loss'):
            self.values_cliped = self.values_old + \
                                 tf.clip_by_value(self.values_new - self.values_old, -cliprange, cliprange)
            critic_loss = tf.square(self.q_values - self.values)
            critic_loss_clipped = tf.square(self.q_values - self.values_cliped)

            self.critic_loss = 0.5*tf.reduce_mean(tf.maximum(critic_loss, critic_loss_clipped))

        with tf.variable_scope('actor_loss'):
            neglogps_new = self.actions_distrs_new.neglogp(self.actions_old)

            ratio = tf.exp(self.neglogps_old - neglogps_new)

            actor_loss = self.gaes * ratio
            actor_loss_clipped = self.gaes * tf.clip_by_value(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
            self.actor_loss = -tf.reduce_mean(tf.minimum(actor_loss, actor_loss_clipped))

        with tf.variable_scope('entropy_loss'):
            entropy = self.actions_distrs_new.entropy()
            self.entropy_loss = tf.reduce_mean(entropy, axis=0)

        with tf.variable_scope('total_loss'):
            self.loss = self.critic_loss + self.actor_loss - 0.001*self.entropy_loss

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

        # Backpropagation
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

    def get_action(self, sess, samples, keep_prob, stochastic=True):
        if stochastic:
            action = tf.multinomial(tf.log(self.actions_distrs), num_samples=1)
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
        dc = None
        act = 'silu'

        with tf.variable_scope(name, reuse=reuse) as scope:
            out = fc_layer(states, 60, keep_prob, dc=dc, name="layer_0", act=act, trainable=trainable)
            # out = fc_layer(out, 150, keep_prob, dc=dc, name="layer_1", act=act, trainable=trainable)

            out = fc_layer(out, 20, keep_prob, dc=None, name="layer_2", act=act, trainable=trainable)

            # TODO: Доделать
            if isinstance(self.act_space, gym.spaces.Discrete):
                logits = fc_layer(out, self.act_shape, keep_prob, dc=None, name="layer_logits", act='none',
                                  trainable=trainable)

                actions_probability = CategoricalNew(logits)

            else:
                mu = logits = 2 * tf.contrib.layers.fully_connected(inputs=out, num_outputs=self.act_shape,
                                                           activation_fn=tf.nn.tanh,
                                                           trainable=trainable,
                                                           scope='mu')

                # fc_layer(out, self.act_shape, keep_prob, dc=None, name="mu", act='tanh',
                #          trainable=trainable)

                sigma = tf.contrib.layers.fully_connected(inputs=out, num_outputs=self.act_shape,
                                                          activation_fn=tf.nn.softplus,
                                                          trainable=trainable,
                                                          scope='sigma')
                actions_probability = tfp.distributions.Normal(loc=mu, scale=sigma)
                # fc_layer(out, self.act_shape, keep_prob, dc=None, name="sigma", act='none',
                #          trainable=trainable)

                # self.sample_op = \
                # tf.clip_by_value(tf.squeeze(pi.sample(1), axis=0), self.action_bound[0], self.action_bound[1])[0]

            return actions_probability, logits

    def critic(self, states, name='critic', keep_prob=0.5, reuse=False, trainable=True):
        dc = None
        act = 'silu'

        with tf.variable_scope(name, reuse=reuse):
            out = fc_layer(states, 60, keep_prob, dc=dc, name="layer_0", act=act, trainable=trainable)
            # out = fc_layer(out, 150, keep_prob, dc=dc, name="layer_1", act=act, trainable=trainable)

            features = fc_layer(out, 20, keep_prob, dc=None, name="layer_2", act=act, trainable=trainable)

            value = fc_layer(features, 1, keep_prob, dc=None, name="layer_logits", act='none',
                             trainable=trainable)

        return tf.squeeze(value), features