from itertools import count
from collections import deque
import time

import numpy as np
import tensorflow as tf
import gym

from ppo import Agent
from networks import actor, critic
from buffer import ExpBuffer


def train():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if args.restore:
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

                    if EPISODE_TRAINING:
                        break

                    state, reward, ep_rew, ep_len = env.reset(), 0, 0, 0

                if j >= TRANSITIONS_PER_EPOCH:
                    last_val = model.get_value(sess, state[None, :], args.drop_rate)
                    if not done:
                        buf.cutoff_episode(last_val=last_val)
                    break

            # Training process
            if EPISODE_TRAINING:
                inputs = buf.get_episode(0)

                for _ in range(ACTOR_TRAIN_EPOCHS):
                    actor_loss, _ = model.train_actor(sess, *inputs, args.actor_lr, args.drop_rate)
                for _ in range(CRITIC_TRAIN_EPOCHS):
                    critic_loss, _ = model.train_critic(sess, *inputs, args.critic_lr, args.drop_rate)

            elif BATCH_TRAINING:
                for inputs in buf.get_batches(args.batch_size, ACTOR_TRAIN_EPOCHS):
                    actor_loss, _ = model.train_actor(sess, *inputs, args.actor_lr, args.drop_rate)
                for inputs in buf.get_batches(args.batch_size, CRITIC_TRAIN_EPOCHS):
                    critic_loss, _ = model.train_critic(sess, *inputs, args.critic_lr, args.drop_rate)

            else:
                inputs = buf.get_all()

                for _ in range(ACTOR_TRAIN_EPOCHS):
                    actor_loss, _ = model.train_actor(sess, *inputs, args.actor_lr, args.drop_rate)
                for _ in range(CRITIC_TRAIN_EPOCHS):
                    critic_loss, _ = model.train_critic(sess, *inputs, args.critic_lr, args.drop_rate)

            # TODO: Добавить как константу в argsparse
            if not epoch % args.save_epoch and epoch:
                model.save_weights(sess, PATH, epoch)

            print("==========================================")
            print("Epoch: ", epoch)
            print("Losses: ", actor_loss, critic_loss)
            print("MA_ep_len: ", np.mean(ma_ep_len))
            print("MA_ep_rew: ", np.mean(ma_ep_rew))
            print('Time:', time.time() - start_time, '\n')

            # # TODO: Criterion
            # if np.mean(ma_ep_rew) >= 200:
            #     model.save_weights(sess, PATH, epoch)
            #     break


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--dir', type=str, required=True, help='directory for saving/loading agent weights')
    parser.add_argument('--restore', '-r', action='store_true')
    parser.add_argument('--last', '-l', action='store_true',
                        help='restore the model from the last writen file if `--restore` is True')
    parser.add_argument('--max_to_keep', type=int, default=10,
                        help='the maximum number of recent saves to keep. Older files will be deleted')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--mov_av_size', type=int, default=100,
                        help='the window size for the moving average')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount rate for future rewards')
    parser.add_argument('--lam', type=float, default=0.99,
                        help='discount rate for gaes')
    parser.add_argument('--cliprange', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=None)
    parser.add_argument('--drop_rate', type=float, default=0.5)
    parser.add_argument('--seed', '-s', type=int, default=0, help='seed for reproducability')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument("--save_epoch", type=float, default=200, help="save weights every N epochs")
    parser.add_argument('--trans_per_epoch', type=int, default=200,
                        help='number of transitions per episode')
    parser.add_argument('--episode_training', action='store_true',
                        help='train the agent after the end of each episode')
    parser.add_argument('--batch_training', action='store_true',
                        help='randomly sample batches from the experience buffer and train the agent on them')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--actor_train_epochs', type=int, default=10)
    parser.add_argument('--critic_train_epochs', type=int, default=10)
    parser.add_argument('--ent_coef', type=float, default=0.001)
    parser.add_argument('--actor_lr', type=float, default=0.001)
    parser.add_argument('--critic_lr', type=float, default=0.001)

    # 'LunarLanderContinuous-v2'
    if True:
        args = parser.parse_args('--env LunarLander-v2 --epochs 400 --gamma 0.999 --lam 0.999 --cliprange 0.1 '
                                 '--trans_per_epoch 4000 --critic_lr 0.0024 --actor_lr 0.002 --seed 50 '
                                 '--actor_train_epochs 80 --critic_train_epochs 80 --dir PPO_exp -l'.split())

    elif False:
        args = parser.parse_args('--env Pendulum-v0  --batch_training --epochs 400 --gamma 0.96 --lam 0.999 --cliprange 0.2 '
                                 '--trans_per_epoch 12000 --critic_lr 0.018 --actor_lr 0.0006 --seed 50 '
                                 '--actor_train_epochs 80 --critic_train_epochs 80 --dir PPO_exp -l'.split())

    else:
        args = parser.parse_args('--env CartPole-v0 --mov_av_size 10 --epochs 4000 --gamma 0.98 --lam 0.99 '
                                 '--cliprange 0.1 --trans_per_epoch 400 --critic_lr 0.001 --actor_lr 0.001 --seed 50 '
                                 '--actor_train_epochs 2 --critic_train_epochs 2 --dir PPO_exp'.split())

    if args.episode_training and args.batch_training:
        parser.error('either batch_training or episode_training can be selected')

    TRANSITIONS_PER_EPOCH = args.trans_per_epoch
    EPOCHS = args.epochs
    ACTOR_TRAIN_EPOCHS = args.actor_train_epochs
    CRITIC_TRAIN_EPOCHS = args.critic_train_epochs
    BATCH_TRAINING = args.batch_training
    EPISODE_TRAINING = args.episode_training
    PATH = '{}/{}'.format(args.dir, args.env)
    seed = args.seed

    # Init the env
    env = gym.make(args.env)

    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Create the model
    model = Agent(env, actor_net=actor, critic_net=critic, ent_coef=args.ent_coef,
                  cliprange=args.cliprange, max_grad_norm=args.max_grad_norm, saver_max_to_keep=args.max_to_keep)

    ma_ep_len = deque(maxlen=args.mov_av_size)
    ma_ep_rew = deque(maxlen=args.mov_av_size)

    buf = ExpBuffer(model.obs_shape[-1], model.act_shape[-1], TRANSITIONS_PER_EPOCH, args.gamma, args.lam)

    # Run the training process
    train()
