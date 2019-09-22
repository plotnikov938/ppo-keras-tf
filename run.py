import os
import time
from collections import deque

import numpy as np
import tensorflow as tf
import gym
import imageio
from ppo import Agent
from networks import actor, critic


def play():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if PATH is not None:
            if args.last:
                # Restore model from the most recently writen file to the specified path
                model.restore_weights(sess, path=PATH)
            else:
                # Restore model from file in path
                model.restore_weights(sess, path=PATH)

        renders = []
        for episode in range(EPISODES):
            start_time = time.time()
            state, reward, ep_rew, ep_len = env.reset(), 0, 0, 0

            while True:
                if args.render:
                    if args.save_gif:
                        renders.append(env.render(mode='rgb_array'))
                    else:
                        env.render()

                action = model.get_action(sess, state[None, :])

                state, reward, done, info = env.step(action[0])

                ep_rew += reward
                ep_len += 1

                if done:
                    ma_ep_len.append(ep_len)
                    ma_ep_rew.append(ep_rew)

                    break

            print("==========================================")
            print("Episode: ", episode)
            print("Episode reward: ", ep_rew)
            print("Episode len: ", ep_len)
            print("MA_ep_len_{}: ".format(args.mov_av_size), np.mean(ma_ep_len))
            print("MA_ep_rew_{}: ".format(args.mov_av_size), np.mean(ma_ep_rew))
            print('Time:', time.time() - start_time, '\n')

    # Generate and save a gif
    if args.save_gif:
        try:
            os.makedirs('{}/results'.format(args.dir))
        except FileExistsError:
            pass
        imageio.mimsave(PATH_TO_GIF, renders[::args.save_gif])


if __name__ == '__main__':
    import argparse

    # TODO: DONE add helps
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0', help='environment to use')
    parser.add_argument('--dir', '-d', type=str, required=False,
                        help="directory for saving/loading agent weights")
    parser.add_argument('--render', '-r', action='store_true')
    parser.add_argument('--save_gif', '-g', default=10, help='generate and save a gif for each N frames')
    parser.add_argument('--last', '-l', action='store_true', help='restore the model from the last writen file')
    parser.add_argument('--episodes', type=int, default=4, help='episodes to play')
    parser.add_argument('--mov_av_size', type=int, default=100, help='the window size for the moving average')
    parser.add_argument('--seed', '-s', type=int, default=0, help='seed for reproducibility')

    if True:
        args = parser.parse_args('--env Pendulum-v0 --dir PPO_exp -r -l'.split())
    else:
        args = parser.parse_args('--env CartPole-v0 --dir PPO_exp -r -l'.split())

    args = parser.parse_args('--env LunarLander-v2 --dir PPO_exp -r -l -g'.split())

    EPISODES = args.episodes
    try:
        PATH = '{}/{}'.format(args.dir, args.env)
        PATH_TO_GIF = '{}/results/{}.gif'.format(args.dir, args.env)
    except:
        PATH = PATH_TO_GIF = None
        print("Model weights won't be loaded! Please, set `--dir` parameter if you want to test your trained agent!")

    seed = args.seed

    # Init the env
    env = gym.make(args.env)

    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Create the model
    model = Agent(env, actor_net=actor, critic_net=critic)

    ma_ep_len = deque(maxlen=args.mov_av_size)
    ma_ep_rew = deque(maxlen=args.mov_av_size)

    play()
