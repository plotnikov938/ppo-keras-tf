import tensorflow as tf

from tensorflow.python.keras.layers import Dense, Dropout

from datetime import datetime
import glob
import os

import numpy as np
import gym

from common import save_target_graph, restore_target_graph
from distributions import Categorical, Normal


class Agent:

    def __init__(self, env, actor_net, critic_net, ent_coef=0.001, cliprange=0.1, max_grad_norm=None):
        def get_space_shape(space):
            try:
                return None, space.shape[0]
            except IndexError:
                return None,

        # Environment parameters
        self.act_space = env.action_space
        self.act_shape = get_space_shape(self.act_space)
        self.obs_shape = get_space_shape(env.observation_space)

        self.actor_net = actor_net
        self.critic_net = critic_net

        # Reset the graph
        tf.reset_default_graph()

        # Init
        self.states = tf.placeholder(tf.float32, shape=self.obs_shape, name='states')
        self.actions_old = tf.placeholder(tf.float32, shape=self.act_shape, name='actions')
        self.values_old = tf.placeholder(tf.float32, shape=(None,), name='values_old')
        self.neglogps_old = tf.placeholder(tf.float32, shape=(None,), name='neglogps_old')
        self.gaes = tf.placeholder(tf.float32, shape=(None,), name='advantage')
        self.q_values = tf.placeholder(tf.float32, shape=(None,), name='estimation')

        self.drop_rate = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.actor_lr = tf.placeholder(tf.float32, name='actor_lr')
        self.critic_lr = tf.placeholder(tf.float32, name='critic_lr')

        # Build the agent
        with tf.variable_scope('agent') as scope:
            self.actions_distrs = self.actor(self.states)
            self.values = self.critic(self.states)
            policy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name)

        # Sample actions from the given distribution
        self.actions = self.actions_distrs.sample()

        self.neglogp = self.actions_distrs.neglogp(self.actions)
        self.neglogp_new = self.actions_distrs.neglogp(self.actions_old)

        try:
            self.actions = tf.clip_by_value(self.actions, self.act_space.low, self.act_space.high)
        except AttributeError:
            pass

        """Losses"""
        with tf.variable_scope('critic_loss'):
            self.values_cliped = self.values_old + \
                                 tf.clip_by_value(self.values - self.values_old, -cliprange, cliprange)
            critic_loss = tf.square(self.q_values - self.values)
            critic_loss_clipped = tf.square(self.q_values - self.values_cliped)

            self.critic_loss = tf.reduce_mean(tf.maximum(critic_loss, critic_loss_clipped))

        with tf.variable_scope('actor_loss'):
            ratio = tf.exp(self.neglogps_old - self.neglogp_new)
            actor_loss = self.gaes * ratio
            actor_loss_clipped = self.gaes * tf.clip_by_value(ratio, 1.0 - cliprange, 1.0 + cliprange)
            self.actor_loss = -tf.reduce_mean(tf.minimum(actor_loss, actor_loss_clipped))

        with tf.variable_scope('entropy_loss'):
            entropy = self.actions_distrs.entropy()
            try:
                self.entropy_loss = tf.reduce_mean(entropy, axis=0)
            except:
                self.entropy_loss = entropy

        with tf.variable_scope('total_loss'):
            self.loss = self.critic_loss + self.actor_loss - self.entropy_loss*ent_coef

        # Define all trainable variables
        self.train_vars = policy_vars

        # Choose the optimizer
        # self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss=self.loss, var_list=self.train_vars)

        # Calculate the gradients
        # grads = tf.gradients(self.loss, self.train_vars)
        # if max_grad_norm is not None:
        #     grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        # grads = list(zip(grads, self.train_vars))
        #
        # trainer = tf.train.AdamOptimizer(self.learning_rate)

        # Backpropagate
        # self.train_op = trainer.apply_gradients(grads)

        # Calculate the gradients
        def get_grads(loss):
            grads = tf.gradients(loss, self.train_vars)
            if max_grad_norm is not None:
                grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

            return list(zip(grads, self.train_vars))

        # Backpropagate
        self.train_op_critic = tf.train.AdamOptimizer(self.critic_lr).apply_gradients(get_grads(self.critic_loss))
        self.train_op_actor = tf.train.AdamOptimizer(self.actor_lr).apply_gradients(
            get_grads(self.actor_loss - self.entropy_loss*ent_coef))

        # Merge all the summaries
        # self.merged = tf.summary.merge_all()

    @staticmethod
    def restore_weights(sess, path=None, lastpath=None):

        if lastpath is None:
            paths = glob.glob(path + '/*.npy')
            lastpath = paths[-1]#.replace('/', '\\')

        # Restore weights if they exists
        try:
            restore_target_graph(sess, lastpath)
            print('******** Model was restored! *********')
        except FileNotFoundError:
            print("******** Model not found! *********")

    @staticmethod
    def save_weights(sess, path, global_step=0, max_keep=10):
        index = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')

        # Save weights
        save_target_graph(sess, path, '/weights_{}_{:04d}'.format(index, global_step))
        print('******** Model was saved! *********')

        # Remove earlier files to keep their total number equal to max_keep
        remove_paths = glob.glob(path + '/*.npy')[:-max_keep]
        # Lazy way to loop over files and remove them
        _ = [*map(os.remove, remove_paths)]

    def get_actions_distribution(self, sess, samples, keep_prob):
        return sess.run(self.actions_distrs, {self.states: samples, self.drop_rate: keep_prob})

    def get_value(self, sess, states, drop_rate):
        return sess.run(self.values, {self.states: states, self.drop_rate: drop_rate})

    # TODO: Add deterministic policy
    def get_action(self, sess, states, drop_rate=0, stochastic=False):
        # # Sample actions from the given distribution
        # self.action = self.actions_distrs.sample(stochastic)
        #
        # # Clip range of the action if allowed
        # try:
        #     self.action = tf.clip_by_value(self.action, self.act_space.low, self.act_space.high)
        # except AttributeError:
        #     pass

        return sess.run(self.actions, {self.states: states, self.drop_rate: drop_rate})

    def evaluate_model(self, sess, samples):
        return sess.run([self.actions, self.values, self.neglogp], feed_dict={self.states: samples})

    def train_agent(self, sess, states, actions, values, neglogps, gaes, q_values):
        return sess.run([self.loss, self.train_op], feed_dict={self.states: states,
                                                               self.actions_old: actions,
                                                               self.values_old: values,
                                                               self.neglogps_old: neglogps,
                                                               self.gaes: gaes,
                                                               self.q_values: q_values})

    def train_actor(self, sess, states, actions, values, neglogps, gaes, q_values, lr, drop_rate):
        return sess.run([self.actor_loss, self.train_op_actor], feed_dict={self.states: states,
                                                                           self.actions_old: actions,
                                                                           self.values_old: values,
                                                                           self.neglogps_old: neglogps,
                                                                           self.gaes: gaes,
                                                                           self.q_values: q_values,
                                                                           self.actor_lr: lr,
                                                                           self.drop_rate: drop_rate})

    def train_critic(self, sess, states, actions, values, neglogps, gaes, q_values, lr, drop_rate):
        return sess.run([self.critic_loss, self.train_op_critic], feed_dict={self.states: states,
                                                                             self.actions_old: actions,
                                                                             self.values_old: values,
                                                                             self.neglogps_old: neglogps,
                                                                             self.gaes: gaes,
                                                                             self.q_values: q_values,
                                                                             self.critic_lr: lr,
                                                                             self.drop_rate: drop_rate})

    def actor(self, states, name='actor', reuse=False, trainable=True):
        with tf.variable_scope(name, reuse=reuse):
            features = self.actor_net(states, trainable=trainable)

            if isinstance(self.act_space, gym.spaces.Discrete):
                logits = Dense(self.act_space.n, None, trainable=trainable, name="layer_logits")(features)
                distribution = Categorical(logits)

            else:
                mean = Dense(self.act_space.shape[0], None, trainable=trainable, name='mean')(features)
                logstd = tf.get_variable('logstd', initializer=-0.5*np.ones(self.act_space.shape[0], dtype=np.float32))

                # logstd = Dense(self.act_space.shape[0], None, trainable=trainable, name='logstd')(features)
                distribution = Normal(mean=mean, logstd=logstd)

            return distribution

    def critic(self, states, name='critic', reuse=False, trainable=True):
        with tf.variable_scope(name, reuse=reuse):
            features = self.critic_net(states, trainable=trainable)

            value = Dense(1, None, trainable=trainable, name="layer_logits")(features)

        return tf.squeeze(value, axis=1)


class Agent3:
    # TODO: Remove 'stochastic' from func definition
    def __init__(self, env, actor_net, critic_net, ent_coef=0.0001, cliprange=0.1, max_grad_norm=None):
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
        self.states = tf.placeholder(tf.float32, shape=(None, self.obs_shape), name='states')
        self.actions_old = tf.placeholder(tf.float32, shape=[None, 1], name='actions')
        self.gaes = tf.placeholder(tf.float32, shape=[None], name='advantage')
        self.values_old = tf.placeholder(tf.float32, shape=[None], name='values_old')
        self.neglogps_old = tf.placeholder(tf.float32, shape=[None], name='neglogps_old')
        self.q_values = tf.placeholder(tf.float32, shape=[None], name='estimation')

        self.cliprange = cliprange
        self.drop_rate = tf.Variable(1.0, dtype=tf.float32)
        self.actor_lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.critic_lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        # Agent for training
        with tf.variable_scope('new_agent') as scope:
            self.actions_distrs_new = self.actor(self.states)
            self.values = self.critic(self.states)
            policy_vars_new = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name)

        # Sample actions from given distribution
        self.actions = self.actions_distrs_new.sample()

        self.neglogp = self.actions_distrs_new.neglogp(self.actions)
        self.neglogp_new = self.actions_distrs_new.neglogp(self.actions_old)

        # self.actions = self.actions_distrs.sample()
        try:
            self.actions = tf.clip_by_value(self.actions, self.act_space.low, self.act_space.high)
        except AttributeError:
            pass

        """Losses"""
        with tf.variable_scope('critic_loss'):
            self.values_cliped = self.values_old + \
                                 tf.clip_by_value(self.values - self.values_old, -cliprange, cliprange)
            critic_loss = tf.square(self.q_values - self.values)
            critic_loss_clipped = tf.square(self.q_values - self.values_cliped)

            self.critic_loss = tf.reduce_mean(tf.maximum(critic_loss, critic_loss_clipped))
            # self.critic_loss = tf.reduce_mean(tf.square(self.values - self.q_values))

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
            self.loss = self.critic_loss + self.actor_loss# - 0.001*self.entropy_loss*0

        # Define all trainable variables
        self.train_vars = policy_vars_new

        # Choose the optimizer
        # self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss=self.loss, var_list=self.train_vars)

        # Calculate the gradients
        # grads = tf.gradients(self.loss, self.train_vars)
        # if max_grad_norm is not None:
        #     grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        # grads = list(zip(grads, self.train_vars))
        #
        # trainer = tf.train.AdamOptimizer(self.learning_rate)

        # Backpropagate
        # self.train_op = trainer.apply_gradients(grads)
        # self.train_op_critic = tf.train.AdamOptimizer(0.018).minimize(self.critic_loss)
        # self.train_op_actor = tf.train.AdamOptimizer(0.0006).minimize(self.actor_loss - self.entropy_loss*0.001)
        # self.train_op = tf.group(self.train_op_critic, self.train_op_actor)

        # Calculate the gradients
        def get_grads(loss):
            grads = tf.gradients(loss, self.train_vars)
            if max_grad_norm is not None:
                grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)

            return list(zip(grads, self.train_vars))

        # Backpropagate
        self.train_op_critic = tf.train.AdamOptimizer(0.018).apply_gradients(get_grads(self.critic_loss))
        self.train_op_actor = tf.train.AdamOptimizer(0.0006).apply_gradients(
            get_grads(self.actor_loss - self.entropy_loss * 0.0001))


        # Define additional features
        pass

        # Merge all the summaries and write them out to selected folder
        self.merged = tf.summary.merge_all()

    @staticmethod
    def restore_weights(sess, path=None, lastpath=None):

        if lastpath is None:
            paths = glob.glob(path + '/*.npy')[-1]
            lastpath = paths[-1]

        # Restore weights if they exists
        try:
            restore_target_graph(sess, lastpath)
            print('******** Model was restored! *********')
        except FileNotFoundError:
            print("******** Model not found! *********")

    @staticmethod
    def save_weights(sess, path, global_step=0, max_keep=10):
        index = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')

        # Save weights
        save_target_graph(sess, path, '/weights_{}_{:04d}'.format(index, global_step))
        print('******** Model was saved! *********')

        # Remove earlier files to keep their total number equal to max_keep
        remove_paths = glob.glob(path + '/*.npy')[:-max_keep]
        # Lazy way to loop over files and remove them
        _ = [*map(os.remove, remove_paths)]

    def get_actions_distribution(self, sess, samples, keep_prob):
        return sess.run(self.actions_distrs, {self.states: samples, self.drop_rate: keep_prob})

    def get_value(self, sess, states, drop_rate):
        return sess.run(self.values, {self.states: states, self.drop_rate: drop_rate})

    # TODO: Use here distr.sample() func instead of tf.multinomial()
    def get_action(self, sess, samples, keep_prob, stochastic=False):
        # Sample actions from the given distribution
        self.action = self.actions_distrs.sample(stochastic)

        # Clip range of the action if allowed
        try:
            self.action = tf.clip_by_value(self.action, self.act_space.low, self.act_space.high)
        except AttributeError:
            pass

        return self.action

    def evaluate_model(self, sess, samples):
        return sess.run([self.actions, self.values, self.neglogp], feed_dict={self.states: samples})

    def train_agent(self, sess, states, actions, values, neglogps, gaes, q_values):
        return sess.run([self.loss, self.train_op], feed_dict={self.states: states,
                                                               self.actions_old: actions,
                                                               self.values_old: values,
                                                               self.neglogps_old: neglogps,
                                                               self.gaes: gaes,
                                                               self.q_values: q_values})

    def train_actor(self, sess, states, actions, values, neglogps, gaes, q_values, lr, drop_rate):
        return sess.run([self.actor_loss, self.train_op_actor], feed_dict={self.states: states,
                                                                           self.actions_old: actions,
                                                                           self.values_old: values,
                                                                           self.neglogps_old: neglogps,
                                                                           self.gaes: gaes,
                                                                           self.q_values: q_values,
                                                                           self.actor_lr: lr,
                                                                           self.drop_rate: drop_rate})

    def train_critic(self, sess, states, actions, values, neglogps, gaes, q_values, lr, drop_rate):
        return sess.run([self.critic_loss, self.train_op_critic], feed_dict={self.states: states,
                                                                             self.actions_old: actions,
                                                                             self.values_old: values,
                                                                             self.neglogps_old: neglogps,
                                                                             self.gaes: gaes,
                                                                             self.q_values: q_values,
                                                                             self.critic_lr: lr,
                                                                             self.drop_rate: drop_rate})

    def actor(self, states, name='actor', reuse=False, trainable=True):
        with tf.variable_scope(name, reuse=reuse):
            features = self.actor_net(states, trainable=trainable)

            # TODO: Доделать
            if isinstance(self.act_space, gym.spaces.Discrete):
                logits = Dense(self.act_shape, None, trainable=trainable, name="layer_logits")(features)
                distribution = Categorical(logits)

            else:
                mean = Dense(self.act_shape, None, trainable=trainable, name='mean')(features)

                # logstd = Dense(self.act_shape, None, trainable=trainable, name='logstd')(features)
                logstd = tf.get_variable('logstd', initializer=-0.5*np.ones([self.act_shape, 1], dtype=np.float32))

                distribution = Normal(mean=mean, logstd=logstd)

            return distribution

    def critic(self, states, name='critic', reuse=False, trainable=True):
        with tf.variable_scope(name, reuse=reuse):
            features = self.critic_net(states, trainable=trainable)

            value = Dense(1, None, trainable=trainable, name="layer_logits")(features)

        return tf.squeeze(value, axis=1)
