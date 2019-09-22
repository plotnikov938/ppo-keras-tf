import numpy as np


def get_q_values(episode_rewards, gamma):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    return discounted_episode_rewards


def get_gaes(rewards, values, values_next, gamma, lam):
    deltas = rewards + gamma * values_next - values

    gaes = deltas.copy()

    for t in reversed(range(len(gaes) - 1)):
        gaes[t] = gaes[t] + gamma * lam * gaes[t + 1]

    return gaes


class ExpBuffer:
    """A Buffer that is used to store experience of the agent and
    provides the gaes calculation after each episode completed.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        if act_dim is None:
            act_dim = ()
        else:
            act_dim = tuple(act_dim)
        self.states = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((size, *act_dim), dtype=np.float32)
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
        """Use this method to get training data for an episode specified by the number.

        Args:
            num (int, optional): The number of a specific episode you want to use to train an agent. Defaults to zero.

        Returns:
            (:obj:`list` of :obj:`numpy.ndarray`): List consist of
            [states, actions, values, neglogps, gaes, q_values] numpy arrays.

        Raises:
            AssertionError: In case of the absence of the stored comlited episodes in the buffer.
        """

        assert len(self.episode_pointers) != 1  # there are no stored episodes in the buffer

        episode_slice = slice(self.episode_pointers[num], self.episode_pointers[num + 1])

        ret_gaes = self.gaes[episode_slice]
        ret_gaes = (ret_gaes - ret_gaes) / (ret_gaes + 1e-10)

        return [x[episode_slice] for x in [self.states, self.actions, self.values, self.neglogps,
                                           ret_gaes, self.q_values]]

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

        return [self.states, self.actions, self.values[:-1], self.neglogps, self.gaes, self.q_values]

    def get_batches(self, batch_size, num):
        """Use this method in case of batch training to get random batch generator.

        Generator produces a number of batches determined by the parameter 'num'.

        Args:
            batch_size (int): The size of the batch. Must be less than or equal to the buffer size.
            num (int): Number of batches being produced by generator

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
        """Call this at the end of an epoch to reset buffer to
        store the data of the next epoch
        """

        self.ptr, self.start_ptr, self.episode_pointers = 0, 0, [0]

