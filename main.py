import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from itertools import count

from ppo import Agent


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
    MAX_DURATION = 2000  # 22320
    TRAIN_EPISODES = 20000
    TRAIN_EPOCHS = 4
    TRESHOLD = 0.6
    PRINT = False
    FEE = 0.001
    START_POS = 0
    BATCH_TRAIN = True
    learning_rate = 0.0005

    # Создаем скелет модели
    model = Agent(4, cliprange=0.1, max_grad_norm=0.5, stochastic=True)

    keep_prob = 1.0
    gamma, lam = 0.99, 0.98
    batch_size = 64

    actions = np.zeros((MAX_DURATION, 2), dtype=np.int8)
    values = np.zeros((MAX_DURATION, 2), dtype=np.float32)
    neglogps = np.zeros((MAX_DURATION, 2), dtype=np.float32)

    real_actions = np.zeros_like(actions[:, 0], dtype=np.int8)
    real_values = np.zeros_like(values[:, 0], dtype=np.float32)
    real_neglogps = np.zeros_like(neglogps[:, 0], dtype=np.float32)
    rewards = np.zeros_like(actions[:, 0], dtype=np.float32)
    values_next = np.zeros(MAX_DURATION, dtype=np.float32)

    def main():
        env = gym.make('CartPole-v0')
        # env = gym.make('LunarLanderContinuous-v2')
        # env = gym.make('Pendulum-v0')
        print(env.action_space)
        print(env.action_space)
        input(env.observation_space)
        env.seed(0)
        states = np.zeros((MAX_DURATION, env.observation_space.shape[0]), dtype=np.float32)

        model = Agent(4, cliprange=0.1, max_grad_norm=0.5, stochastic=True)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            obs = env.reset()
            success_num = 0
            for episode in range(TRAIN_EPISODES):
                for j in count():
                    # env.render()

                    real_actions[j], real_values[j], real_neglogps[j] = model.evaluate_model(sess, obs.reshape(1, -1), 1.0)

                    states[j] = obs
                    next_obs, rewards[j], done, info = env.step(real_actions[j])

                    if done:
                        obs = env.reset()
                        rewards[j] = -1
                        break
                    else:
                        obs = next_obs

                if rewards[:j + 1].sum() >= 195:
                    success_num += 1
                    if success_num >= 10:
                        print('Clear!! Model saved.')
                        break
                else:
                    success_num = 0

                end = j + 1
                values_next[:end - 1] = real_values[1:end]
                values_next[end - 1] = 0

                # Выбираем способ оценки
                gaes = True
                if gaes:
                    gaes = get_gaes(rewards[:end], real_values[:end], values_next[:end],
                                    gamma=gamma, lam=lam)

                    # Нормализуем оценночные значения
                    gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

                    q_values = gaes + real_values[:end]

                else:
                    q_values = get_q_values(rewards[:end], gamma=gamma)

                    # Нормализуем оценночные значения
                    q_values = (q_values - q_values.mean()) / (q_values.std() + 1e-8)

                    gaes = q_values - real_values[:end]

                # Переназначаем параметры нового агента старому
                model.synchronize_policies(sess)

                # Тренируем нового агента
                if BATCH_TRAIN:
                    for epoch in range(TRAIN_EPOCHS):
                        args = np.random.randint(0, len(states[:end]), batch_size)
                        loss, _ = model.train_agent(sess,
                                                    states[args],
                                                    real_actions[args],
                                                    real_values[args],
                                                    real_neglogps[args],
                                                    gaes[args],
                                                    q_values[args],
                                                    learning_rate,
                                                    keep_prob)
                else:
                    loss, _ = model.train_agent(sess,
                                                states[:end],
                                                real_actions[:end],
                                                real_values[:end],
                                                real_neglogps[:end],
                                                gaes,
                                                q_values,
                                                learning_rate,
                                                keep_prob)

                print("==========================================")
                print("Episode: ", episode)
                print("Len of Episode", len(states[:end]))
                print("Loss: ", loss, '\n')

    main()
