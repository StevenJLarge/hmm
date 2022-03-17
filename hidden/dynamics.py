import numpy as np
from typing import Iterable, Optional


# How should this be structured? As an HMM class? with inference and
# filtering routines as member functions?
class HMM:
    def __init__(self, dim_sys: int, dim_obs: int):
        # Initialize the bayesian filter to the initial observation
        self.n_sys = dim_sys
        self.n_obs = dim_obs

        self.A = np.zeros((self.n_sys, self.n_sys))
        self.B = np.zeros((self.n_obs, self.n_obs))

        self.state_tracker = None
        self.obs_tracker = None

    def _initialize_tracking(self):
        self.state_tracker = []
        self.obs_tracker = []

    def init_uniform_cycle(
        self, trans_rate: Optional[int] = 0.3,
        error_rate: Optional[int] = 0.2
    ):
        self.A = np.zeros((self.n_sys, self.n_sys))
        self.B = np.zeros((self.n_obs, self.n_obs))

        # NOTE there must be a better way of doing this...
        for i in range(self.n_sys):
            for j in range(self.n_sys):
                self.A[i, j] = (lambda x: trans_rate * self.n_sys if x == 1 else 0)(np.abs(i - j))

        for i in range(self.n_obs):
            for j in range(self.n_obs):
                self.B[i, j] = (lambda x: error_rate * self.n_obs if x == 1 else 0)(np.abs(i - j))

            self.A[i, i] = 1 - np.sum(self.A[:, i])
            self.B[i, i] = 1 - np.sum(self.B[:, i])

    # Dynamics routines
    def run_dynamics(
        self, n_steps: Optional[int] = 100, init_state: Optional[int] = None
    ):
        if init_state is None:
            self._set_state(np.random.randint(0, self.n_sys))
        self._set_obs(np.argmax(self.state))

        if self.state_tracker is None or self.obs_tracker is None:
            self._initialize_tracking()

        for _ in range(n_steps):
            self.state_tracker.append(self.state)
            self.obs_tracker.append(self.obs)
            self.step_dynamics()

    def step_dynamics(self):
        trans_prob = self.A[:, np.argmax(self.state)]
        obs_prob = self.B[:, np.argmax(self.state)]

        # Kinetic monte-carlo dynamics
        w1 = np.random.uniform()
        w2 = np.random.uniform()

        csum_sys = np.cumsum(trans_prob)
        self.state = self._set_state(w1 < csum_sys)

        csum_obs = np.cumsum(obs_prob)
        self.obs = self._set_obs(w2 < csum_obs)

    def _set_state(self, new_state):
        self.state = np.zeros(self.n_sys)
        self.state[new_state] = 1

    def _set_obs(self, new_obs):
        self.obs = np.zeros(self.n_obs)
        self.obs[new_obs] = 1

    def get_state_ts(self):
        return [np.argmax(s) for s in self.state_tracker]

    def get_obs_ts(self):
        return [np.argmax(o) for o in self.obs_tracker]


def plot_traj(state: Iterable, observation: Iterable):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='darkgrid')
    Pal = sns.color_palette('hls', 2)

    _, ax = plt.subplots(1, 1, figsize=(6, 3.5))

    ax.plot(state, 'o', markersize=7, color=Pal[0], label='State')
    ax.plot(observation, 'o', markersize=4, color=Pal[1], label='Observation')
    ax.set_xlabel(r"Time", fontsize=15)
    ax.set_ylabel(r"State", fontsize=15)
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":

    # Setup
    obj = HMM(2, 2)
    obj.init_uniform_cycle(0.1)

    # Generate synthetic dynamics:
    obj.run_dynamics(n_steps=25)
    state = obj.get_state_ts()
    obs = obj.get_obs_ts()

    plot_traj(state, obs)

    # Perform EM algo to infer system parameters
    # Instantiate differeny object to infer state
    # obj.baum_welch()

    # TESTING PHASE
    # generate dynamics
    # obj.run_dynamics(n_steps=50)
    # Run forward algorithm for determining hidden state
    # obj.forward_algo() 


