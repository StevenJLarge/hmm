# File to contain the class definitions and routines for inferring the
# properties of the HMM
from typing import Iterable, Optional, Tuple, Dict
from operator import mul
import numpy as np
from pandas import DataFrame, Series
from hidden.optimize.registry import OPTIMIZER_REGISTRY
from hidden.optimize.base import OptClass
from hidden.optimize.results import OptimizationResult
from hidden.filters import bayesian


class MarkovInfer:
    # Type hints for instance variables
    forward_tracker: Iterable
    backward_tracker: Iterable
    predictions: Iterable
    predictions_back: Iterable
    bayes_smooth: Iterable
    alpha_tracker: Iterable
    beta_tracker: Iterable
    n_sys: int
    n_obs: int
 
    def __init__(self, dim_sys: int, dim_obs: int):
        # Tracker lists for forward and backward estimates
        self.forward_tracker = None
        self.backward_tracker = None
        self.predictions = None
        self.predictions_back = None
        self.bayes_smooth = None
        self.alpha_tracker = None
        self.beta_tracker = None

        # Dimension of target system and observation vector
        self.n_sys = dim_sys
        self.n_obs = dim_obs

    @staticmethod
    def _validate_input(obs_ts):
        # We want this to support input lists as well as pandas Series
        if isinstance(obs_ts, np.ndarray):
            if 1 in obs_ts.shape or len(obs_ts.shape) == 1:
                return obs_ts.flatten()
            else:
                raise ValueError("Input observations must be 1-D...")

        if isinstance(obs_ts, list):
            return np.array(obs_ts)
        if isinstance(obs_ts, (Series, DataFrame)):
            if 1 in obs_ts.shape or len(obs_ts.shape) == 1:
                return obs_ts.to_numpy().flatten()
            else:
                raise ValueError("Input observations must be 1-D...")

        else:
            raise NotImplementedError(
                "observation timeseries must be list or pandas Series/DataFrame"
            )

    def forward_algo(
        self,
        observations: Iterable[int],
        trans_matrix: np.ndarray,
        obs_matrix: np.ndarray,
    ):
        observations = self._validate_input(observations)
        # This is now just an interface for the filter/bayesian methods
        self.forward_tracker, self.prediction_tracker = bayesian.forward_algo(
            observations, trans_matrix, obs_matrix
        )

    def backward_algo(
        self,
        observations: Iterable[int],
        trans_matrix: np.ndarray,
        obs_matrix: np.ndarray,
    ):
        observations = self._validate_input(observations)
        self.backward_tracker, self.predictions_back = bayesian.backward_algo(
            observations, trans_matrix, obs_matrix
        )

    def alpha(
        self, observations: np.ndarray, trans_matrix: np.ndarray,
        obs_matrix: np.ndarray, norm: Optional[bool] = False
    ):
        observations = self._validate_input(observations)
        self.alpha_tracker = bayesian.alpha_prob(
            observations, trans_matrix, obs_matrix, norm=norm
        )

    def beta(
        self, observations: np.ndarray, trans_matrix: np.ndarray,
        obs_matrix: np.ndarray, norm: Optional[bool] = True
    ):
        observations = self._validate_input(observations)
        self.beta_tracker = bayesian.beta_prob(
            observations, trans_matrix, obs_matrix, norm=norm
        )

    def bayesian_smooth(
        self, observations: np.ndarray, trans_matrix: np.ndarray,
        obs_matrix: np.ndarray
    ):
        observations = self._validate_input(observations)
        self.bayes_smooth = bayesian.bayes_estimate(
            observations, trans_matrix, obs_matrix
        )

    def discord(self, state: Iterable, filter_est: Iterable) -> float:
        error = [1 if f == o else -1 for f, o in zip(filter_est, state)]
        return 1 - np.mean(error)

    def error_rate(self, pred_ts: Iterable, state_ts: Iterable) -> float:
        return 1 - np.mean([p == s for p, s in zip(pred_ts, state_ts)])

    def optimize(
        self, observations: Iterable, trans_init: np.ndarray,
        obs_init: np.ndarray, symmetric: Optional[bool] = False,
        opt_type: Optional[OptClass] = OptClass.Local,
        algo_opts: Optional[Dict] = {}
    ) -> OptimizationResult:
        if not isinstance(opt_type, OptClass):
            raise ValueError(
                'Invalid `opt_class`, must be a member of OptClass enum...'
            )
        observations = self._validate_input(observations)
        # For the global optimizer, I need n_params, dim_tuple, and
        optimizer = OPTIMIZER_REGISTRY[opt_type](**algo_opts)
        if (opt_type is OptClass.Global):
            dim_tuple = (trans_init.shape, obs_init.shape)
            return optimizer.optimize(observations, dim_tuple, symmetric=symmetric)

        return optimizer.optimize(observations, trans_init, obs_init, symmetric)


if __name__ == "__main__":
    from hidden import dynamics
    hmm = dynamics.HMM(2, 2)
    hmm.init_uniform_cycle()
    hmm.run_dynamics(1000)

    state_ts = hmm.get_state_ts()
    obs_ts = hmm.get_obs_ts()

    BayesInfer = MarkovInfer(2, 2)
    param_init = (0.15, 0.15)
    A_init = np.array([[0.85, 0.15], [0.15, 0.85]])
    B_init = np.array([[0.85, 0.15], [0.15, 0.85]])

    BayesInfer.forward_algo(obs_ts, hmm.A, hmm.B)
    BayesInfer.backward_algo(obs_ts, hmm.A, hmm.B)
    BayesInfer.bayesian_smooth(hmm.A)

    BayesInfer.alpha(hmm.A, hmm.B, obs_ts)
    BayesInfer.beta(hmm.A, hmm.B, obs_ts)

    res_loc = BayesInfer.optimize(obs_ts, A_init, B_init, symmetric=True)
    res_glo = BayesInfer.optimize(
        obs_ts, A_init, B_init, symmetric=True, opt_type=OptClass.Global
    )
    # res_bw = BayesInfer.baum_welch(param_init, obs_ts, maxiter=10)

    print("--DONE--")
