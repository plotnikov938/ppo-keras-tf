import argparse


# TODO: Redefine args for my purposes
def get_train_args():
    train_params = argparse.ArgumentParser()

    # Environment parameters
    train_params.add_argument("--env", type=str, default='Pendulum-v0',
                              help="Environment to use (must have low dimensional state space (i.e. not image) and continuous action space)")
    train_params.add_argument("--render", type=bool, default=False,
                              help="Whether or not to display the environment on the screen during training")
    train_params.add_argument("--random_seed", type=int, default=99999999, help="Random seed for reproducability")

    # Training parameters
    train_params.add_argument("--batch_size", type=int, default=64)
    train_params.add_argument("--num_eps_train", type=int, default=50000, help="Number of episodes to train for")
    train_params.add_argument("--max_ep_length", type=int, default=1000, help="Maximum number of steps per episode")
    train_params.add_argument("--replay_mem_size", type=int, default=1000000, help="Maximum size of replay memory")
    train_params.add_argument("--initial_replay_mem_size", type=int, default=50000,
                              help="Initial size of replay memory (populated by random actions) before learning starts")
    train_params.add_argument("--noise_scale", type=float, default=0.1,
                              help="Scale of exploration noise range (as a fraction of action space range), e.g. for a noise_scale=0.1, the noise range is a tenth of the action space range")
    train_params.add_argument("--discount_rate", type=float, default=0.99,
                              help="Discount rate (gamma) for future rewards.")

    # Network parameters
    train_params.add_argument("--critic_learning_rate", type=float, default=0.001)
    train_params.add_argument("--actor_learning_rate", type=float, default=0.0001)
    train_params.add_argument("--critic_l2_lambda", type=float, default=0.0,
                              help="Coefficient for L2 weight regularisation in critic - if 0, no regularisation is performed")
    train_params.add_argument("--dense1_size", type=int, default=400, help="Size of first hidden layer in networks")
    train_params.add_argument("--dense2_size", type=int, default=300, help="Size of second hidden layer in networks")
    train_params.add_argument("--final_layer_init", type=float, default=0.003,
                              help="Initialise networks' final layer weights in range +/-final_layer_init")
    train_params.add_argument("--tau", type=float, default=0.001, help="Parameter for soft target network updates")
    train_params.add_argument("--use_batch_norm", type=bool, default=False,
                              help="Whether or not to use batch normalisation in the networks")

    # Files/Directories
    train_params.add_argument("--save_ckpt_step", type=float, default=200, help="Save checkpoint every N episodes")
    train_params.add_argument("--ckpt_dir", type=str, default='./ckpts',
                              help="Directory for saving/loading checkpoints")
    train_params.add_argument("--ckpt_file", type=str, default=None,
                              help="Checkpoint file to load and resume training from (if None, train from scratch)")
    train_params.add_argument("--log_dir", type=str, default='./logs/train',
                              help="Directory for saving Tensorboard logs")

    return train_params.parse_args()
