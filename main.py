import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import gym

from itertools import count
from collections import deque
import time

from ppo import Agent as Agent
from networks import actor, critic


def get_one_hot(a, classes):
    shape = a.shape[0]
    b = np.zeros((shape, classes))
    b[np.arange(shape), a] = 1

    return b


def get_q_values(episode_rewards, gamma):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    return discounted_episode_rewards


def choice_actions_vectorized(prob_matrix):
    s = prob_matrix if prob_matrix.shape[1] == 2 else prob_matrix.cumsum(axis=1)
    s[:, -1] = 1.0
    r = np.random.rand(prob_matrix.shape[0]).reshape(-1, 1)
    return (s < r).sum(axis=1)


def get_gaes(rewards, values, values_next, gamma, lam):
    deltas = rewards + gamma * values_next - values

    gaes = deltas.copy()

    for t in reversed(range(len(gaes) - 1)):
        gaes[t] = gaes[t] + gamma * lam * gaes[t + 1]

    return gaes


# TODO: Add an option to choose whether use gaes or q_values
class ExpBuffer:
    """
    A Buffer that is used to store experience of an agent and
    provides the gaes calculation after each episode completed.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.states = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, act_dim), dtype=np.float32)
        self.gaes = np.zeros(size, dtype=np.float32)
        self.q_values = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size + 1, dtype=np.float32)
        self.neglogps = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size + 1, dtype=np.float32)
        self.episode_pointers = [0]

        self.gamma, self.lam = gamma, lam
        self.ptr, self.start_ptr, self.max_size, self.args = 0, 0, size, None

    def append(self, state, action, value, neglogp, reward):
        """Appends agent experience to the buffer"""

        assert self.ptr < self.max_size  # not enough buffer size

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.values[self.ptr] = value
        self.neglogps[self.ptr] = neglogp
        self.rewards[self.ptr] = reward
        self.ptr += 1

    def cutoff_episode(self, last_val=0):
        """A method that cuts off episode of an agent interaction with the
        environment and calculates the gaes ​​for a completed episode.

        Call this at the end of an episode, or when one gets cut off
        by an epoch ending.
        """

        self.rewards[self.ptr] = last_val
        self.values[self.ptr] = last_val

        episode_slice = slice(self.start_ptr, self.ptr)

        self.gaes[episode_slice] = get_gaes(
            self.rewards[episode_slice], self.values[episode_slice], self.values[1:][episode_slice],
            gamma=self.gamma, lam=self.lam)

        self.q_values[episode_slice] = get_q_values(
            self.rewards[self.start_ptr:self.ptr+1], self.gamma)[:self.ptr - self.start_ptr]

        self.episode_pointers.append(self.ptr)
        self.start_ptr = self.ptr

    def get_episode(self, num=0):
        """Use this method to get training data for a specific episode with the 'num' number.

        Args:
            num (int, optional): The number of a specific episode where you want to train an agent. Defaults to zero.

        Returns:
            (:obj:`list` of :obj:`numpy.ndarray`): List consist of
            [states, actions, values, neglogps, gaes, q_values] numpy arrays.

        Raises:
            AssertionError: In case of the absence of the stored comlited episodes in the buffer.
        """

        assert len(self.episode_pointers) != 1  # there are no stored episodes in the buffer

        episode_slice = slice(self.episode_pointers[num], self.episode_pointers[num + 1])

        return [x[episode_slice] for x in [self.states, self.actions, self.values, self.neglogps,
                                           self.gaes, self.q_values]]

    def get_all(self):
        """Use this method to get all training data in a list of numpy arrays.

        Returns:
            (:obj:`list` of :obj:`numpy.ndarray`): List consists of the
            [states, actions, values, neglogps, gaes, q_values] numpy arrays.

        Raises:
            AssertionError: In case of insufficient buffer filling
        """

        assert self.ptr == self.max_size    # buffer has to be full before you can get the data

        self.gaes = (self.gaes - self.gaes.mean()) / (self.gaes.std() + 1e-10)

        plt.plot(self.gaes, 'r')
        plt.plot(self.values[:-1])
        plt.show()

        return [self.states, self.actions, self.values[:-1], self.neglogps, self.gaes, self.q_values]

    def get_batches(self, batch_size, num):
        """Use this method in case of batch training to get random batch generator.
        Generator produces a number of batches determined by parameter 'count'.

        Args:
            batch_size (int): The size of the batch. Must be less than or equal to the buffer size.
            count (int): Number of batches being produced by generator

        Yields:
            list of `numpy.ndarray`: The next batch out of 'num' batches from the buffer.
                Batch consists of the [states, actions, values, neglogps, gaes, q_values] numpy arrays.

        Raises:
            AssertionError: In case of insufficient buffer filling and incorrect batch size
        """

        assert self.ptr == self.max_size    # buffer has to be full before you can get the data
        assert batch_size < self.max_size    # batch size must be less than or equal to the buffer size

        self.gaes = (self.gaes - self.gaes.mean()) / (self.gaes.std() + 1e-10)
        for _ in range(num):
            self.args = np.random.randint(0, self.ptr - 1, batch_size)

            yield [x[self.args] for x in [self.states, self.actions, self.values, self.neglogps,
                                          self.gaes, self.q_values]]

    def reset(self):
        """Call this at the end of an epoch to reset buffer for
        storing the data of the next epoch
        """

        self.ptr, self.start_ptr, self.episode_pointers = 0, 0, [0]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--restore', '-r', action='store_true')
    parser.add_argument('--last', '-l', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--mov_av_size', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.99)
    parser.add_argument('--cliprange', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=None)
    parser.add_argument('--drop_rate', type=float, default=0.5)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--trans_per_epoch', type=int, default=200)
    parser.add_argument('--episode_training', action='store_true')
    parser.add_argument('--batch_training', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--actor_train_epochs', type=int, default=10)
    parser.add_argument('--critic_train_epochs', type=int, default=10)
    parser.add_argument('--ent_coef', type=float, default=0.001)
    parser.add_argument('--actor_lr', type=float, default=0.001)
    parser.add_argument('--critic_lr', type=float, default=0.001)

    #  --data_dir /Users/Eli/Desktop/Pendulum --dt
    if True:
        args = parser.parse_args('--env Pendulum-v0 --epochs 400 --gamma 0.992 --lam 0.999 --cliprange 0.2'
                                 ' --trans_per_epoch 12000 --critic_lr 0.018 --actor_lr 0.0006 --seed 50 '
                                 '--actor_train_epochs 80 --critic_train_epochs 80 --dir PPO_exp -r -l'.split())
    else:
        args = parser.parse_args('--env CartPole-v0 --mov_av_size 10 --epochs 4000 --gamma 0.999 --lam 0.998 '
                                 '--cliprange 0.1 --trans_per_epoch 400 --critic_lr 0.001 --actor_lr 0.001 --seed 50 '
                                 '--actor_train_epochs 2 --critic_train_epochs 2 --dir PPO_exp'.split())

    if args.episode_training and args.batch_training:
        parser.error('either batch_training or episode_training can be selected')

    TRANSITIONS_PER_EPOCH = args.trans_per_epoch
    EPOCHS = args.epochs
    ACTOR_TRAIN_EPOCHS = args.actor_train_epochs
    CRITIC_TRAIN_EPOCHS = args.critic_train_epochs
    BATCH_TRAIN = args.batch_training
    EPISODE_TRAIN = args.episode_training
    PATH = '{}/{}'.format(args.dir, args.env)
    seed = args.seed

    # Init the env
    env = gym.make(args.env)
    # env.seed(1000)

    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Create the model
    model = Agent(env, actor_net=actor, critic_net=critic, ent_coef=args.ent_coef,
                  cliprange=args.cliprange, max_grad_norm=args.max_grad_norm)

    # TODO: Убрать при отладке
    for var in tf.trainable_variables():
        print(var)

    ma_ep_len = deque(maxlen=args.mov_av_size)
    ma_ep_rew = deque(maxlen=args.mov_av_size)
    buf = ExpBuffer(model.obs_shape[-1], model.act_shape[-1], TRANSITIONS_PER_EPOCH, args.gamma, args.lam)

    def main():
        # Experience buffer
        local_steps_per_epoch = MAX_DURATION

        buf = ExpBuffer(model.obs_shape, 1, MAX_DURATION, gamma, lam)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Setup model saving
        def update():
            phs = [model.states, model.actions_old, model.values_old, model.neglogps_old, model.gaes, model.q_values]
            gets = buf.get_all()
            inputs = {k: v for k, v in zip(phs, gets)}
            pi_l_old, v_l_old = sess.run([model.actor_loss, model.critic_loss], feed_dict=inputs)
            print(pi_l_old, v_l_old)

            # Training
            for i in range(TRAIN_EPOCHS):
                model.train_actor(sess, *gets, learning_rate, keep_prob)
            for _ in range(TRAIN_EPOCHS):
                model.train_critic(sess, *gets, learning_rate, keep_prob)

        start_time = time.time()
        state, reward, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        from collections import deque
        mn = deque(maxlen=100)

        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(EPOCHS):
            buf.reset()
            for t in range(local_steps_per_epoch):
                action, value, neglogp = model.evaluate_model(sess, state.reshape(1, -1), keep_prob=1.0)

                # save and log
                buf.append(state, action, value, neglogp, reward)

                state, reward, done, _ = env.step(action[0])
                ep_ret += reward
                ep_len += 1

                terminal = done or (ep_len == 1000)
                if terminal or (t == local_steps_per_epoch - 1):
                    if not terminal:
                        print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
                    buf.cutoff_episode()

                    mn.append(ep_ret)

                    state, reward, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0

            print('Epoch: ', epoch)
            print("Mean_reward_per_epoch:", np.mean(mn))
            print('Time:', time.time() - start_time)
            start_time = time.time()

            # Perform PPO update!
            update()

    def main2():

        mov_av = deque(maxlen=100)

        buf = ExpBuffer(model.obs_shape, 1, MAX_DURATION, gamma, lam)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(EPOCHS):

                buf.reset()
                start_time = time.time()
                state, reward, ep_rew, ep_len = env.reset(), 0, 0, 0
                for j in count(1):
                    # if episode > 500:
                    #     env.render()

                    action, value, neglogp = model.evaluate_model(sess, state.reshape(1, -1), keep_prob=1.0)

                    buf.append(state, action, value, neglogp, reward)

                    state, reward, done, info = env.step(action.reshape(-1, 1))

                    ep_rew += reward
                    ep_len += 1

                    # Switching between discrete and cont action_space on the fly
                    # try:
                    #     next_obs, rewards[j], done, info = env.step(actions[j].reshape(-1, 1))
                    # except AssertionError:
                    #     actions = actions.astype(np.int)
                    #     next_obs, rewards[j], done, info = env.step(actions[j].reshape(-1, 1))

                    # buf.append(states[j], actions[j], values[j], neglogps[j], rewards[j])

                    # print(actions[j], values[j], neglogps[j])
                    if done:
                        buf.cutoff_episode()
                        mov_av.append(ep_rew)
                        state, reward, ep_rew, ep_len = env.reset(), 0, 0, 0

                        # rewards[j] = 0

                    if j >= TRANSITIONS_PER_EPOCH:
                        buf.cutoff_episode()
                        break


                # q_values = get_q_values(rewards[:end], gamma=gamma)
                #
                # # Нормализуем оценночные значения
                # q_values = (q_values - q_values.mean()) / (q_values.std() + 1e-8)
                #
                # gaes = q_values - values[:end]

                # Переназначаем параметры нового агента старому
                # model.synchronize_policies(sess)

                # Тренируем нового агента
                if BATCH_TRAIN:
                    for inputs in buf.get_batches(batch_size, TRAIN_EPOCHS):
                        loss, _ = model.train_agent(sess,
                                                    inputs,
                                                    learning_rate,
                                                    keep_prob)

                else:
                    inputs = buf.get_all()

                    # plt.plot(neglogps)
                    # plt.show()
                    for _ in range(TRAIN_EPOCHS):
                        loss, _ = model.train_actor(sess, *inputs, learning_rate, keep_prob)

                    for _ in range(TRAIN_EPOCHS):
                        loss2, _ = model.train_critic(sess, *inputs, learning_rate, keep_prob)

                print("==========================================")
                print("Epoch: ", epoch)
                # print("Len of Episode", end)
                print("Loss: ", loss, loss2)
                print("MA: ", np.mean(mov_av))
                print('Time:', time.time() - start_time, '\n')

    def main3():

        # TODO: Добавить длину очереди в качестве параметра для метрики
        mov_av = deque(maxlen=100)

        buf = ExpBuffer(model.obs_shape, 1, TRANSITIONS_PER_EPOCH, args.gamma, args.lam)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(EPOCHS):
                buf.reset()
                start_time = time.time()
                state, reward, ep_rew, ep_len = env.reset(), 0, 0, 0

                for j in count(1):
                    # if episode > 500:
                    #     env.render()

                    action, value, neglogp = model.evaluate_model(sess, state.reshape(1, -1))

                    buf.append(state, action, value, neglogp, reward)

                    state, reward, done, info = env.step(action[0])

                    ep_rew += reward
                    ep_len += 1

                    if done:
                        buf.cutoff_episode(last_val=-1)
                        mov_av.append(ep_rew)
                        print(ep_len)
                        state, reward, ep_rew, ep_len = env.reset(), 0, 0, 0
                        if EPISODE_TRAIN:
                            break

                    if j >= TRANSITIONS_PER_EPOCH:
                        last_val = model.get_value(sess, [state], args.drop_rate)
                        if not done:
                            buf.cutoff_episode(last_val=last_val)
                        break

                # Training process
                if EPISODE_TRAIN:
                    inputs = buf.get_episode(0)
                    inputs[-2] = (inputs[-2] - inputs[-2].mean()) / (inputs[-2].std() + 1e-10)
                    # plt.plot(inputs[-1])
                    # plt.plot(inputs[-4])
                    # plt.plot(inputs[-2], 'r')
                    # plt.show()
                    for _ in range(ACTOR_TRAIN_EPOCHS):
                        loss, _ = model.train_actor(sess, *inputs, args.actor_lr, args.drop_rate)
                    for _ in range(CRITIC_TRAIN_EPOCHS):
                        loss2, _ = model.train_critic(sess, *inputs, args.critic_lr, args.drop_rate)
                elif BATCH_TRAIN:
                    for inputs in buf.get_batches(batch_size, ACTOR_TRAIN_EPOCHS):
                        loss, _ = model.train_actor(sess, *inputs, args.actor_lr, args.drop_rate)
                    for inputs in buf.get_batches(batch_size, CRITIC_TRAIN_EPOCHS):
                        loss2, _ = model.train_critic(sess, *inputs, args.critic_lr, args.drop_rate)
                else:
                    inputs = buf.get_all()
                    # plt.plot(inputs[-1])
                    # plt.plot(inputs[-4])
                    # plt.plot(inputs[-2], 'r')
                    # plt.show()
                    for _ in range(ACTOR_TRAIN_EPOCHS):
                        loss, _ = model.train_actor(sess, *inputs, args.actor_lr, args.drop_rate)
                    for _ in range(CRITIC_TRAIN_EPOCHS):
                        loss2, _ = model.train_critic(sess, *inputs, args.critic_lr, args.drop_rate)

                print("==========================================")
                print("Epoch: ", epoch)
                # print("Len of Episode", end)
                print("Loss: ", loss, loss2)
                print("MA: ", np.mean(mov_av))
                print('Time:', time.time() - start_time, '\n')

    def main4():

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            if PATH is not None:
                if args.last:
                    # Restore model from the most recently writen file to the specified path
                    model.restore_weights(sess, path=PATH)
                else:
                    # Restore model from file in path
                    model.restore_weights(sess, path=PATH)

            for epoch in range(EPOCHS):
                buf.reset()
                start_time = time.time()
                state, reward, ep_rew, ep_len = env.reset(), 0, 0, 0

                for j in count(1):
                    if args.render:
                        env.render()

                    action, value, neglogp = model.evaluate_model(sess, state.reshape(1, -1))

                    buf.append(state, action, value, neglogp, reward)

                    state, reward, done, info = env.step(action[0])

                    ep_rew += reward
                    ep_len += 1

                    if done:
                        buf.cutoff_episode(last_val=reward)
                        ma_ep_len.append(ep_len)
                        ma_ep_rew.append(ep_rew)

                        if EPISODE_TRAIN:
                            break

                        state, reward, ep_rew, ep_len = env.reset(), 0, 0, 0

                    if j >= TRANSITIONS_PER_EPOCH:
                        last_val = model.get_value(sess, state[None, :], args.drop_rate)
                        if not done:
                            buf.cutoff_episode(last_val=last_val)
                        break

                # Training process
                if EPISODE_TRAIN:
                    inputs = buf.get_episode(0)
                    inputs[-2] = (inputs[-2] - inputs[-2].mean()) / (inputs[-2].std() + 1e-10)

                    for _ in range(ACTOR_TRAIN_EPOCHS):
                        actor_loss, _ = model.train_actor(sess, *inputs, args.actor_lr, args.drop_rate)
                    for _ in range(CRITIC_TRAIN_EPOCHS):
                        critic_loss, _ = model.train_critic(sess, *inputs, args.critic_lr, args.drop_rate)

                elif BATCH_TRAIN:
                    for inputs in buf.get_batches(args.batch_size, ACTOR_TRAIN_EPOCHS):
                        actor_loss, _ = model.train_actor(sess, *inputs, args.actor_lr, args.drop_rate)
                    for inputs in buf.get_batches(args.batch_size, CRITIC_TRAIN_EPOCHS):
                        critic_loss, _ = model.train_critic(sess, *inputs, args.critic_lr, args.drop_rate)

                else:
                    inputs = buf.get_all()
                    # inputs = [inputs[i] for i in [0, 1, 2, -1, 3, 4]]

                    for _ in range(ACTOR_TRAIN_EPOCHS):
                        actor_loss, _ = model.train_actor(sess, *inputs, args.actor_lr, args.drop_rate)
                    for _ in range(CRITIC_TRAIN_EPOCHS):
                        critic_loss, _ = model.train_critic(sess, *inputs, args.critic_lr, args.drop_rate)

                # TODO: Добавить как константу в argsparse
                if not epoch % 20 and not epoch:
                    model.save_weights(sess, PATH, epoch)

                print("==========================================")
                print("Epoch: ", epoch)
                # print("Len of Episode", end)
                print("Losses: ", actor_loss, critic_loss)
                print("MA_ep_len: ", np.mean(ma_ep_len))
                print("MA_ep_rew: ", np.mean(ma_ep_rew))
                print('Time:', time.time() - start_time, '\n')

                if np.mean(ma_ep_rew) >= 200:
                    model.save_weights(sess, PATH, epoch)
                    break

    main4()
