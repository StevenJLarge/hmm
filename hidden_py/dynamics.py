import numpy as np
from typing import Iterable, Optional


class HMM:
    def __init__(self, dim_sys: int, dim_obs: int):
        # Initialize the bayesian filter to the initial observation
        self.n_sys = dim_sys
        self.n_obs = dim_obs

        self.A = np.zeros((self.n_sys, self.n_sys))
        self.B = np.zeros((self.n_obs, self.n_sys))

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
        # NOTE This routine assumes that the observation matrix dimension
        # is equal to the transitionmatrix dimension
        if self.n_sys != self.n_obs:
            raise NotImplementedError(
                "Currently, cycle default matrices can only be instantiated "
                "when `n_sys` = `n_obs`"
            )
        self.A = np.zeros((self.n_sys, self.n_sys))
        self.B = np.zeros((self.n_sys, self.n_sys))

        # NOTE that this assumes a symmetric cycle (fwd rate = back-rate)
        for i in range(self.n_sys):
            for j in range(i + 1, self.n_sys):
                self.A[i, j] = (lambda x: trans_rate if x == 1 else 0)(np.abs(i - j))
                self.A[j, i] = self.A[i, j]
            # For PBCs
            if i == 0: self.A[self.n_sys - 1, i] = trans_rate
            if i == self.n_sys - 1: self.A[0, i] = trans_rate

            self.A[i, i] = 1 - np.sum(self.A[:, i])

        for i in range(self.n_obs):
            for j in range(i + 1, self.n_obs):
                self.B[i, j] = (lambda x: error_rate if x == 1 else 0)(np.abs(i - j))
                self.B[j, i] = self.B[i, j]
            # For PBCs
            if i == 0: self.B[self.n_obs - 1, i] = error_rate
            if i == self.n_obs - 1: self.B[0, i] = error_rate

            self.B[i, i] = 1 - np.sum(self.B[:, i])

    def _validate_dynamics_matrices(self):
        A_condition = self.A.sum(axis=0)
        B_condition = self.B.sum(axis=0)

        if not (A_condition == 1).all():
            raise ValueError(
                f"Invalid transition matrix A : {self.A}"
            )
        if  not (B_condition == 1).all():
            raise ValueError(
                f"Invalid observation matrix B : {self.B}"
            )

    # Dynamics routines
    def run_dynamics(
        self, n_steps: Optional[int] = 100, init_state: Optional[int] = None
    ):
        self._validate_dynamics_matrices()

        if init_state is None:
            self._set_state(np.random.randint(0, self.n_sys))
        else:
            self._set_state(init_state)

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
        if new_state < 0 or new_state > self.n_sys:
            raise IndexError(
                f"New state `{new_state}` out of bounds for system of "
                f"dimension `{self.n_sys}`"
            )
        self.state = np.zeros(self.n_sys)
        self.state[new_state] = 1

    def _set_obs(self, new_obs):
        if new_obs < 0 or new_obs > self.n_obs:
            raise IndexError(
                f"New observation `{new_obs}` out of bounds for system with "
                f"dimension `{self.n_obs}`"
            )
        self.obs = np.zeros(self.n_obs)
        self.obs[new_obs] = 1

    def get_state_ts(self):
        return [np.argmax(s) for s in self.state_tracker]

    def get_obs_ts(self):
        return [np.argmax(o) for o in self.obs_tracker]

    @property
    def current_state(self):
        return self.state

    @property
    def current_obs(self):
        return self.obs

    @property
    def steady_state(self):
        e_vals, e_vecs = np.linalg.eig(self.A)
        ss_idx = np.argmin(np.abs(1 - e_vals))
        return e_vecs[:, ss_idx] / np.sum(e_vecs[:, ss_idx])


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

