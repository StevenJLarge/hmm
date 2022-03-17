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

    # Filtering and processing routines
    def bayesian_filter(self, obs: int):
        self.bayes_prob = np.matmul(self.A, self.bayes_prob)
        self.bayes_filter = self.B[:, obs] * self.bayes_prob
        self.bayes_filter /= np.sum(self.bayes_filter)

    def forward_algo(self, observations: Iterable[int]):
        # Runs the fwd algo over an entire sequence
        self._initialize_bayes_tracker()

        for obs in observations:
            self.bayesian_filter(obs)
            self.bayes_tracker.append(self.bayes_filter)

    def backward_algo(self, oservations: Iterable[int]):
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
