# File to contain the class definitions and routines for inferring the
# properties of the HMM

import numpy as np
from typing import Iterable, Optional


class MarkovInfer:
    def __init__(self, dim_sys: int, dim_obs: int):
        self.forward_tracker = []
        self.backward_tracker = []

        self.n_sys = dim_sys
        self.n_obs = dim_obs

        self.bayes_filter = None
        self.backward_filter = None
        self.bayes_smoother = None

    def _initialize_bayes_tracker(self):
        self.bayes_filter = np.ones(self.n_sys) / self.n_sys
        self.forward_tracker = [self.bayes_filter]

    def _initialize_back_tracker(self):
        self.back_filter = self.forward_tracker[-1]
        self.backward_tracker = [self.back_filter]

    # Filtering and processing routines
    def bayesian_filter(self, obs: int, A: np.ndarray, B: np.ndarray):
        self.bayes_filter = np.matmul(A, self.bayes_filter)
        self.bayes_filter = B[:, obs] * self.bayes_filter
        self.bayes_filter /= np.sum(self.bayes_filter)

    def bayesian_back(self, obs: int, A: np.ndarray, B: np.ndarray):
        self.back_filter = np.matmul(A, self.back_filter)
        self.back_filter = B[:, obs] * self.back_filter
        self.back_filter /= np.sum(self.back_filter)

    def forward_algo(
        self, observations: Iterable[int],
        trans_matrix: np.ndarray,
        obs_matrix: np.ndarray
    ):
        # Runs the fwd algo over an entire sequence
        self._initialize_bayes_tracker()

        for obs in observations:
            self.bayesian_filter(obs, trans_matrix, obs_matrix)
            self.forward_tracker.append(self.bayes_filter)

    def backward_algo(
        self,
        observations: Iterable[int],
        trans_matrix: np.ndarray,
        obs_matrix: np.ndarray
    ):
        self._initialize_back_tracker()

        for obs in np.flip(observations):
            self.bayesian_back(obs, trans_matrix, obs_matrix)
            self.backward_tracker.append(self.back_filter)

        self.backward_tracker = np.flip(self.backward_tracker)

    def bayesian_smooth(self, A: np.ndarray):
        self.bayes_smoother = [[]] * len(self.forward_tracker)
        self.bayes_smoother[-1] = np.array(self.forward_tracker[-1])

        for i in range(len(self.forward_tracker) - 1):
            prediction = np.matmul(A, self.forward_tracker[-(i + 2)])
            summand = [np.sum(self.bayes_smoother[-(i+1)] * A[:, j] / prediction) for j in range(A.shape[1])]
            self.bayes_smoother[-(i + 2)] = self.forward_tracker[-(i + 2)] * np.array(summand)

    def discord(self, obs: Iterable) -> float:
        if len(self.filter_tracker) - 1 != len(obs):
            raise ValueError(
                "You must run `forward_algo(...)` before `discord`..."
            )

        error = [1 if f == o else -1 for f, o in zip(self.filter_tracker, obs)]
        return 1 - np.mean(error)

    def error_rate(self, pred_ts: Iterable, state_ts: Iterable) -> float:
        return np.sum([pred_ts == state_ts])/len(state_ts)

    # Inferrence routines
    def expectation(self):
        pass

    def maximization(self):
        pass

    def baum_welch(self):
        pass


if __name__ == "__main__":
    from hidden import dynamics
    hmm = dynamics.HMM(2, 2)
    hmm.init_uniform_cycle()
    hmm.run_dynamics(100)

    state_ts = hmm.get_state_ts()
    obs_ts = hmm.get_obs_ts()

    BayesInfer = MarkovInfer(2, 2)

    BayesInfer.forward_algo(obs_ts, hmm.A, hmm.B)
    BayesInfer.backward_algo(obs_ts, hmm.A, hmm.B)
    BayesInfer.bayesian_smooth(hmm.A)

