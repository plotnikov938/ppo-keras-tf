import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import gym

from itertools import count
from collections import deque

from ppo import Agent
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


if __name__ == '__main__':
    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    # parser.add_argument('--hid', type=int, default=64)
    # parser.add_argument('--l', type=int, default=2)
    # parser.add_argument('--gamma', type=float, default=0.99)
    # parser.add_argument('--seed', '-s', type=int, default=0)
    # parser.add_argument('--cpu', type=int, default=4)
    # parser.add_argument('--steps', type=int, default=4000)
    # parser.add_argument('--epochs', type=int, default=50)
    # parser.add_argument('--exp_name', type=str, default='ppo')
    # args = parser.parse_args()
    #
    # mpi_fork(args.cpu)  # run parallel code with mpi
    #
    # from spinup.utils.run_utils import setup_logger_kwargs
    #
    # logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    #
    # ppo(lambda: gym.make(args.env), actor_critic=core.mlp_actor_critic,
    #     ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
    #     seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
    #     logger_kwargs=logger_kwargs)

    MAX_DURATION = 2000  # 22320
    TRAIN_EPISODES = 20000
    TRAIN_EPOCHS = 4
    TRESHOLD = 0.6
    PRINT = False
    FEE = 0.001
    START_POS = 0
    BATCH_TRAIN = True
    learning_rate = 0.0001

    # Инициализируем среду
    env = gym.make('CartPole-v0')
    # env = gym.make('LunarLanderContinuous-v2')
    # env = gym.make('LunarLander-v2')
    env = gym.make('Pendulum-v0')
    # env = gym.make('FrozenLake-v0')
    env.seed(0)

    # Create the model
    model = Agent(env, actor_net=actor, critic_net=critic,
                  cliprange=0.2, max_grad_norm=0.5, stochastic=True)

    keep_prob = 1.0
    gamma, lam = 0.999, 0.998
    batch_size = 64

    mov_av = deque(maxlen=100)

    def main():
        # Create some arrays to store episodes information
        states = np.zeros((MAX_DURATION, model.obs_shape), dtype=np.float32)
        actions = np.zeros(MAX_DURATION, dtype=np.float32)
        values = np.zeros(MAX_DURATION, dtype=np.float32)
        neglogps = np.zeros(MAX_DURATION, dtype=np.float32)
        rewards = np.zeros(MAX_DURATION, dtype=np.float32)
        values_next = np.zeros(MAX_DURATION, dtype=np.float32)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            obs = env.reset()

            success_num = 0
            for episode in range(TRAIN_EPISODES):
                for j in count():
                    # if episode > 500:
                    #     env.render()

                    states[j] = obs

                    actions[j], values[j], neglogps[j] = model.evaluate_model(sess,
                                                                              obs.reshape(1, -1),
                                                                              keep_prob=1.0)
                    try:
                        next_obs, rewards[j], done, info = env.step(actions[j])
                    except AssertionError:
                        actions = actions.astype(np.int)
                        next_obs, rewards[j], done, info = env.step(actions[j])

                    # print(actions[j], values[j], neglogps[j])
                    if done:
                        obs = env.reset()
                        # rewards[j] = 0
                        break
                    else:
                        obs = next_obs

                # if rewards[:j + 1].sum() >= 195:
                #     success_num += 1
                #     if success_num >= 10:
                #         print('Clear!! Model saved.')
                #         break
                # else:
                #     success_num = 0

                end = j + 1
                values_next[:end - 1] = values[1:end]
                values_next[end - 1] = 0

                mov_av.append(sum(rewards[:end]))

                # Выбираем способ оценки
                gaes = True
                if gaes:
                    gaes = get_gaes(rewards[:end], values[:end], values_next[:end],
                                    gamma=gamma, lam=lam)

                    # Нормализуем оценночные значения
                    gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

                    q_values = gaes + values[:end]

                else:
                    q_values = get_q_values(rewards[:end], gamma=gamma)

                    # Нормализуем оценночные значения
                    q_values = (q_values - q_values.mean()) / (q_values.std() + 1e-8)

                    gaes = q_values - values[:end]

                # Переназначаем параметры нового агента старому
                model.synchronize_policies(sess)

                # Тренируем нового агента
                if BATCH_TRAIN:
                    for epoch in range(TRAIN_EPOCHS):
                        args = np.random.randint(0, end-1, batch_size)
                        loss, _ = model.train_agent(sess,
                                                    states[args],
                                                    actions[args],
                                                    values[args],
                                                    neglogps[args],
                                                    gaes[args],
                                                    q_values[args],
                                                    learning_rate,
                                                    keep_prob)
                else:
                    loss, _ = model.train_agent(sess,
                                                states[:end],
                                                actions[:end],
                                                values[:end],
                                                neglogps[:end],
                                                gaes,
                                                q_values,
                                                learning_rate,
                                                keep_prob)

                print("==========================================")
                print("Episode: ", episode)
                print("Len of Episode", end)
                print("Loss: ", loss)
                print("MA: ", np.mean(mov_av), '\n')

    main()
