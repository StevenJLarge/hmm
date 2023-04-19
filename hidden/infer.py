# File to contain the class definitions and routines for inferring the
# properties of the HMM

import numpy as np
import scipy.optimize as so
from scipy.optimize import OptimizeResult
from typing import Iterable, Optional, Tuple, Dict


class MarkovInfer:
    # Type hints for instance variables
    forward_tracker: Iterable
    backward_tracker: Iterable
    predictions: Iterable
    n_sys: int
    n_obs: int

    bayes_filter: Iterable
    backward_filter: Iterable
    bayes_smoother: Iterable
 
    def __init__(self, dim_sys: int, dim_obs: int):
        # Tracker lists for forward and backward estimates
        self.forward_tracker = []
        self.backward_tracker = []
        self.predictions = []

        # Dimension of target system and observation vector
        self.n_sys = dim_sys
        self.n_obs = dim_obs

        # Default initialization values for bayes filer, backward filter, and
        # bayes smoother instance variables
        self.bayes_filter = None
        self.backward_filter = None
        self.bayes_smoother = None

    def _initialize_bayes_tracker(self):
        # Initialize a naive bayes filter and initialize the forward_tracker
        # list with it
        # NOTE is this the best initial condition? Its more an agnostic prior
        # rather than a naive prior...
        self.bayes_filter = np.ones(self.n_sys) / self.n_sys
        self.forward_tracker = []
        self.backward_tracker = []

    def _initialize_pred_tracker(self, mode: Optional[str] = "forwards"):
        if mode == "forwards":
            self.predictions = []
        elif mode == "backwards":
            self.predictions_back = []
        else:
            raise ValueError(
                "Invalid direction in prediction initializer, must "
                "be `forwards` or `backwards`"
            )

    def _initialize_back_tracker(self):
        # 'initial' value for the back-filter is the final value in the forward
        # filter
        self.back_filter = self.forward_tracker[-1]
        # self.backward_tracker = [self.back_filter]
        self.backward_tracker = []

    def bayesian_filter(
        self, obs: int, A: np.ndarray, B: np.ndarray,
        prediction_tracker: bool
    ):
        # Two-step bayesian filter equations, updates the current bayes_filter
        self.bayes_filter = np.matmul(A, self.bayes_filter)
        if prediction_tracker:
            self.predictions.append(self.bayes_filter)
        self.bayes_filter = B[:, obs] * self.bayes_filter
        self.bayes_filter /= np.sum(self.bayes_filter)

    def bayesian_back(
        self, obs: int, A: np.ndarray, B: np.ndarray,
        prediction_tracker: bool
    ):
        # Two-step update for back-propagated bayesian filter
        self.back_filter = np.matmul(A.T, self.back_filter)
        if prediction_tracker:
            self.predictions_back.append(self.back_filter)
        self.back_filter = B[:, obs] * self.back_filter
        self.back_filter /= np.sum(self.back_filter)

    def forward_algo(
        self,
        observations: Iterable[int],
        trans_matrix: np.ndarray,
        obs_matrix: np.ndarray,
        prediction_tracker: Optional[bool] = False
    ):
        # Runs the fwd algo over an entire sequence
        self._initialize_bayes_tracker()

        if prediction_tracker:
            self._initialize_pred_tracker()

        for obs in observations:
            self.bayesian_filter(obs, trans_matrix, obs_matrix, prediction_tracker)
            self.forward_tracker.append(self.bayes_filter)

    def backward_algo(
        self,
        observations: Iterable[int],
        trans_matrix: np.ndarray,
        obs_matrix: np.ndarray,
        prediction_tracker: Optional[bool] = False
    ):
        if len(self.forward_tracker) != len(observations):
            self.forward_algo(observations, trans_matrix, obs_matrix)
        self._initialize_back_tracker()

        if prediction_tracker:
            self._initialize_pred_tracker(mode="backwards")

        for obs in np.flip(observations):
            self.bayesian_back(obs, trans_matrix, obs_matrix, prediction_tracker)
            self.backward_tracker.append(self.back_filter)

        # NOTE modification here
        self.backward_tracker = np.flip(self.backward_tracker, axis=0)

    def alpha(self, A: np.ndarray, B: np.ndarray, obs_ts: np.ndarray):
        alpha = B[:, obs_ts[0]]
        self.alpha_tracker = [alpha]
        for obs in obs_ts[1:]:
            # update alpha term
            alpha = (A @ alpha) * B[:, obs]
            self.alpha_tracker.append(alpha)

    def beta(self, A: np.ndarray, B: np.ndarray, obs_ts: np.ndarray):
        beta = np.ones(2)
        self.beta_tracker = [beta]
        for obs in obs_ts[-1::-1]:
            beta = A.T @ (beta * B[:, obs])
            self.beta_tracker.append(beta)
        # reverse ordering
        self.beta_tracker = self.beta_tracker[::-1][1:]

    def bayesian_smooth(self, A: np.ndarray):
        # Check to ensure that forward and backward algos have been run before this
        if (len(self.forward_tracker) == 0):
            raise ValueError(
                'forward_tracker is empty, you must run forward_algo before '
                + 'bayesian_smooth'
            )

        if (len(self.backward_tracker) == 0):
            raise ValueError(
                'backward_tracker is empty, you must run backward_algo before '
                + 'bayesian_smooth'
            )

        # Combine forward and backward algos to calculate bayesian smoother results
        self.bayes_smoother = [[]] * len(self.forward_tracker)
        self.bayes_smoother[-1] = np.array(self.forward_tracker[-1])

        for i in range(len(self.forward_tracker) - 1):
            prediction = np.matmul(A.T, self.forward_tracker[-(i + 2)])
            summand = [np.sum(self.bayes_smoother[-(i+1)] * A[:, j] / prediction) for j in range(A.shape[1])]
            self.bayes_smoother[-(i + 2)] = self.forward_tracker[-(i + 2)] * np.array(summand)

    def discord(self, obs: Iterable, filter_est: Iterable) -> float:
        # calculates the discord order parameter, given knowledge of the true
        # underlying states and opbserved sequence
        error = [1 if f == o else -1 for f, o in zip(filter_est, obs)]
        return 1 - np.mean(error)

    def error_rate(self, pred_ts: Iterable, state_ts: Iterable) -> float:
        return 1 - np.mean([p == s for p, s in zip(pred_ts, state_ts)])

    # Optimization routine: now this will compose the optimizer from the
    # available optimizers in the optimization submodule
    def optimize(self):
        pass


if __name__ == "__main__":
    from hidden import dynamics
    hmm = dynamics.HMM(2, 2)
    hmm.init_uniform_cycle()
    hmm.run_dynamics(1000)

    state_ts = hmm.get_state_ts()
    obs_ts = hmm.get_obs_ts()

    BayesInfer = MarkovInfer(2, 2)
    param_init = (0.15, 0.15)

    BayesInfer.forward_algo(obs_ts, hmm.A, hmm.B)
    BayesInfer.backward_algo(obs_ts, hmm.A, hmm.B)
    BayesInfer.bayesian_smooth(hmm.A)

    BayesInfer.alpha(hmm.A, hmm.B, obs_ts)
    BayesInfer.beta(hmm.A, hmm.B, obs_ts)

    res_loc = BayesInfer.max_likelihood(param_init, obs_ts, mode='local')
    res_glo = BayesInfer.max_likelihood(param_init, obs_ts, mode='global')

    # res_bw = BayesInfer.baum_welch(param_init, obs_ts, maxiter=10)

    print("--DONE--")
