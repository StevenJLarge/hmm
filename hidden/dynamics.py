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

    def initialize_dynamics(self, A: np.ndarray, B: np.ndarray):
        self.A = A
        self.B = B

    def init_uniform_cycle(
        self, trans_rate: Optional[int] = 0.3,
        error_rate: Optional[int] = 0.1
    ):
        self.A = np.zeros((self.n_sys, self.n_sys))
        self.B = np.zeros((self.n_obs, self.n_obs))

        # NOTE that this assumes a symmetric cycle (fwd rate = back-rate)
        for i in range(self.n_sys):
            for j in range(i + 1, self.n_sys):
                self.A[i, j] = (lambda x: trans_rate if x == 1 else 0)(np.abs(i - j))
                self.A[j, i] = self.A[i, j]
            self.A[i, i] = 1 - np.sum(self.A[:, i])

        for i in range(self.n_obs):
            for j in range(i + 1, self.n_obs):
                self.B[i, j] = (lambda x: error_rate if x == 1 else 0)(np.abs(i - j))
                self.B[j, i] = self.B[i, j]
            self.B[i, i] = 1 - np.sum(self.B[:, i])

    # Dynamics routines
    def run_dynamics(
        self, n_steps: Optional[int] = 100, init_state: Optional[int] = None
    ):
        if init_state is None:
            self._set_state(np.random.randint(0, self.n_sys))
        # Assume initial obervation is accurate
        self._set_obs(np.argmax(self.state))

        if self.state_tracker is None or self.obs_tracker is None:
            self._initialize_tracking()

        for _ in range(n_steps):
            self.state_tracker.append(self.state)
            self.obs_tracker.append(self.obs)
            self.step_dynamics()

    def step_dynamics(self):
        # Kinetic monte-carlo dynamics
        w_sys = np.random.uniform()
        w_obs = np.random.uniform()

        trans_prob = self.A[:, np.argmax(self.state)]
        new_sys = np.argmax(w_sys < np.cumsum(trans_prob))
        self._set_state(new_sys)

        obs_prob = self.B[:, np.argmax(self.state)]
        new_obs = np.argmax(w_obs < np.cumsum(obs_prob))
        self._set_obs(new_obs)

    def _set_state(self, new_state):
        self.state = np.zeros(self.n_sys)
        self.state[new_state] = 1

    def _set_obs(self, new_obs):
        self.obs = np.zeros(self.n_obs)
        cumul_trans = np.cumsum(self.B[:, new_obs])
        rand = np.random.uniform(0, cumul_trans[-1])
        init_obs = np.argmax(rand < cumul_trans)
        self.obs[init_obs] = 1

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
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":

    # Setup
    obj = HMM(3, 3)
    obj.init_uniform_cycle(0.1)

    # Generate synthetic dynamics:
    obj.run_dynamics(n_steps=25)
    state = obj.get_state_ts()
    obs = obj.get_obs_ts()

    plot_traj(state, obs)

