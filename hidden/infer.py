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
        self.back_filter = self.bayes_filter[-1]
        self.back_tracker = [0] * len(self.bayes_filter)
        self.back_tracker[-1] = self.back_filter

    # Filtering and processing routines
    def bayesian_filter(self, obs: int, A: np.ndarray, B: np.ndarray):
        self.bayes_prob = np.matmul(A, self.bayes_prob)
        self.bayes_filter = B[:, obs] * self.bayes_prob
        self.bayes_filter /= np.sum(self.bayes_filter)

    def bayesian_back(self, obs: int, A: np.ndarray, B: np.ndarray):
        pass

    def forward_algo(
        self, observations: Iterable[int],
        trans_matrix: np.ndarray,
        obs_matrix: np.ndarray
    ):
        # Runs the fwd algo over an entire sequence
        self._initialize_bayes_tracker()

        for obs in observations:
            self.bayesian_filter(obs, trans_matrix, obs_matrix)
            self.bayes_tracker.append(self.bayes_filter)

    def backward_algo(
        self,
        observations: Iterable[int],
        trans_matrix: np.ndarray,
        obs_matrix: np.ndarray
    ):
        self._initialize_back_tracker()

        for i, obs in enumerate(np.flip(observations)):
            pass


    def bayesian_smooth(self):
        pass

    def forward_backward_algo(self):
        pass

    def discord(self):
        pass

    def error_rate(self):
        pass

    # Inferrence routines
    def expectation(self):
        pass

    def maximization(self):
        pass

    def baum_welch(self):
        pass
