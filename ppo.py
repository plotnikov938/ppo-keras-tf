import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Dropout
import numpy as np
import gym

from common import save_target_graph, restore_target_graph
from distributions import Categorical, Normal


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
        self.drop_rate = tf.Variable(1.0, dtype=tf.float32)
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        """Build the model"""
        # Agent for acting
        with tf.variable_scope('old_agent') as scope:
            self.actions_distrs = self.actor(self.states)
            self.values = self.critic(self.states)
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
        return sess.run(self.actions_distrs, {self.states: samples, self.drop_rate: keep_prob})

    def get_value(self, sess, samples, keep_prob):
        return sess.run(self.values, {self.states: samples, self.drop_rate: keep_prob})

    # TODO: Use here distr.sample() func instead of tf.multinomial()
    def get_action(self, sess, samples, keep_prob, stochastic=True):
        if stochastic:
            action = tf.multinomial(tf.log(self.actions_distrs + 1e-10), num_samples=1)
            action = tf.reshape(self.action, shape=[-1])
        else:
            action = tf.argmax(self.actions_distrs, axis=1)

    def evaluate_model(self, sess, samples, keep_prob):
        return sess.run([self.actions, self.values, self.neglogp], feed_dict={self.states: samples, self.drop_rate: keep_prob})

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
                                                               self.drop_rate: keep_prob})

    def actor(self, states, name='actor', reuse=False, trainable=True):
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

    def critic(self, states, name='critic', reuse=False, trainable=True):
        with tf.variable_scope(name, reuse=reuse):
            features = self.critic_net(states, trainable=trainable)

            value = Dense(1, None, trainable=trainable, name="layer_logits")(features)

        return tf.squeeze(value)
