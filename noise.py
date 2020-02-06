import numpy as np


# Взят отсюда: https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUStrategy:
    """
    This strategy implements the Ornstein-Uhlenbeck process, which adds
    time-correlated noise to the actions taken by the deterministic policy.
    The OU process satisfies the following stochastic differential equation:
    dxt = theta*(mu - xt)*dt + sigma*dWt
    where Wt denotes the Wiener process
    """

    def __init__(self, action_space, mu=0, theta=0.15, max_sigma=0.3,
                 min_sigma=None, decay_period=100000):
        if min_sigma is None:
            min_sigma = max_sigma
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma

        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.dim = np.prod(action_space.low.shape)
        self.low = action_space.low
        self.high = action_space.high
        self.state = np.ones(self.dim) * self.mu

    def reset(self):
        self.state = np.ones(self.dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

    def get_action_from_raw_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = (self.max_sigma - (self.max_sigma - self.min_sigma)
                      * min(1.0, t * 1.0 / self.decay_period))
        return np.clip(action + ou_state, self.low, self.high)
