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
        self, dim: int, trans_rate: Optional[int] = 0.2,
        error_rate: Optional[int] = 0.1
    ):
        self.A = np.zeros((dim, dim))
        self.B = np.zeros_like(self.A)

        for i in range(dim):
            for j in range(dim):
                self.A[i, j] = (lambda x: trans_rate * dim if x == 1 else 0)(np.abs(i - j))
                self.B[i, j] = (lambda x: error_rate * dim if x == 1 else 0)(np.abs(i - j))

            self.A[i, i] = 1 - np.sum(self.A[:, i])
            self.B[i, i] = 1 - np.sum(self.B[:, i])

    # Dynamics routines
    def run_dynamics(
        self, n_steps: Optional[int] = 100, init_state: Optional[int] = None
    ):
        if init_state is None:
            self._set_state(np.random.randint(0, self.n_sys))
        self._set_obs()

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

    def _set_obs(self):
        self.obs = self.state

    def get_state_ts(self):
        return [np.argmax(s) for s in self.state_tracker]

    def get_obs_ts(self):
        return [np.argmax(o) for o in self.obs_tracker]


if __name__ == "__main__":

    # Setup
    obj = HMM(2, 2)
    obj.init_uniform_cycle(0.1)

    # TRAINING PHASE
    # Run dynamics
    obj.run_dynamics(n_steps=500)
    state = obj.get_state_ts()
    obs = obj.get_obs_ts()

    # Perform EM algo to infer system parameters
    # Instantiate differeny object to infer state
    obj.baum_welch()

    # TESTING PHASE
    # generate dynamics
    obj.run_dynamics(n_steps=50)
    # Run forward algorithm for determining hidden state
    obj.forward_algo()


