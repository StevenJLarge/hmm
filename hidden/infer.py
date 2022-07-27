# File to contain the class definitions and routines for inferring the
# properties of the HMM

import numpy as np
import scipy.optimize as so
from scipy.optimize import OptimizeResult
from typing import Iterable, Optional, Tuple, Dict


class LikelihoodOptResult:
    result: np.ndarray
    type: str
    success: bool
    likelihood: float
    metadata: Dict

    def __init__(self, res: OptimizeResult, method_type: str, **kwargs):
        self.result = res.x
        self.type = method_type
        self.success = res.sucess
        self.likelihood = res.fun
        self.metadata = {"message": res.message}

        for key, val in kwargs.items():
            self.metadata[key] = val

class MarkovInfer:
    #Type hints for instance variables
    forward_tracker: Iterable
    backward_tracker: Iterable
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

        # Dimension of tareg systema nd observation vector
        self.n_sys = dim_sys
        self.n_obs = dim_obs

        # Default initialization values for bayes filer, backward filter, and
        # bayessmoother instance variables
        self.bayes_filter = None
        self.backward_filter = None
        self.bayes_smoother = None

    def _initialize_bayes_tracker(self):
        # Initialize a naive bayes filter and initialize the forawrd_tracker
        # list with it
        self.bayes_filter = np.ones(self.n_sys) / self.n_sys
        self.forward_tracker = []

    def _initialize_pred_tracker(self):
        self.predictions = []

    def _initialize_back_tracker(self):
        # 'initial' value for the back-filter is the final value in the forward
        # filter
        self.back_filter = self.forward_tracker[-1]
        self.backward_tracker = [self.back_filter]

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

    def bayesian_back(self, obs: int, A: np.ndarray, B: np.ndarray):
        # Two-step update for back-propagated bayesian filter
        self.back_filter = np.matmul(A.T, self.back_filter)
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
        obs_matrix: np.ndarray
    ):
        self._initialize_back_tracker()

        for obs in np.flip(observations):
            self.bayesian_back(obs, trans_matrix, obs_matrix)
            self.backward_tracker.append(self.back_filter)

        # NOTE modification here
        self.backward_tracker = np.flip(self.backward_tracker[:-1])

    def bayesian_smooth(self, A: np.ndarray):
        # Combine forward and backward algos to calculate bayesian smoother results
        self.bayes_smoother = [[]] * len(self.forward_tracker)
        self.bayes_smoother[-1] = np.array(self.forward_tracker[-1])

        for i in range(len(self.forward_tracker) - 1):
            # NOTE I think there is a transpose on the A term here? This wont
            # matter for symmetric dynamics, but will when that is not the case
            prediction = np.matmul(A.T, self.forward_tracker[-(i + 2)])
            summand = [np.sum(self.bayes_smoother[-(i+1)] * A[:, j] / prediction) for j in range(A.shape[1])]
            self.bayes_smoother[-(i + 2)] = self.forward_tracker[-(i + 2)] * np.array(summand)

    def discord(self, obs: Iterable) -> float:
        # calculates the discord order parameter, given knowledge of the true
        # underlying states and opbserved sequence
        if len(self.filter_tracker) - 1 != len(obs):
            raise ValueError(
                "You must run `forward_algo(...)` before `discord`..."
            )

        error = [1 if f == o else -1 for f, o in zip(self.filter_tracker, obs)]
        return 1 - np.mean(error)

    def error_rate(self, pred_ts: Iterable, state_ts: Iterable) -> float:
        return np.sum([pred_ts == state_ts])/len(state_ts)

    # Total Likelihood calculation (Brute)
    def calc_likelihood(self, B: np.ndarray, obs_ts: Iterable[int]) -> float:
        likelihood = 0
        for bayes, obs in zip(self.predictions, obs_ts):
            inner = bayes @ B[:, obs]
            likelihood -= np.log(inner)
        return likelihood

    def _extract_hmm_parameters(
        theta: np.ndarray, symmetric: Optional[bool] = False
    ) -> Tuple[np.ndarray]:
        # NOTE this is strictly for a 2-D HMM
        # Builds the matrix terms from a theta vector. For symmetric models,
        # the theta variables is in the form [A_{0,1}, B_{0, 1}]
        # For a non-symmetric model, they are of the form
        # theta = [A_{0,1}, A_{1,0}, B_{0,1}, B_{1,0}]
        if not symmetric:
            if len(theta) != 4:
                raise ValueError(
                    "Need to provide len=4 parameter vector when not symmetric"
                )
            A = np.zeros((len(theta), len(theta)))
            B = np.zeros((len(theta), len(theta)))
            A[0, 0], A[1, 1] = 1 - theta[0], 1 - theta[1]
            A[0, 1], A[1, 0] = theta[0], theta[1]

            B[0, 0], B[1, 1] = 1 - theta[2], 1 - theta[3]
            B[0, 1], B[1, 0] = theta[2], theta[3]

            return A, B

        A = theta[0] * np.ones((2, 2))
        B = theta[1] * np.ones((2, 2))

        A[0, 0], A[1, 1] = 1 - theta[0], 1 - theta[0]
        B[0, 0], B[1, 1] = 1 - theta[1], 1 - theta[1]

        return A, B

    # Likelihood optimizers:
    def maximize_likleihood(
        self, obs_ts: Iterable, method: Optional[str] = "local"
    ) -> OptimizeResult:
        # NOTE Should I pass in optimization parameters for each submethod?

        if method == 'baum-welch':
            raise NotImplementedError(
                '`baum-welch` algorithm not currently implemented'
            )
        if method == 'local':
            return self._optimize_likelihood_local(obs_ts)
        if method == 'global':
            return self._optimize_likelihood_global(obs_ts)

        raise ValueError(
            f"Method {method} is invalid, must be `local` or `global`..."
        )

    def _calc_likeihood_optimizer(
        self, param_arr: Iterable, obs_ts: Iterable
    ) -> float:
        # NOTE Currently this only works for a 2D HMM
        A, B = self._extract_hmm_parameters(param_arr)
        self.forward_algo(obs_ts, A, B, prediction_tracker=True)
        
        likelihood = 0
        for bayes, obs, in zip(self.predictions, obs_ts):
            inner = bayes @ B[:, obs]
            likelihood -= np.log(inner)
        return likelihood

    def _build_optimization_bounds(
        self, param_init: Iterable,
        lower_lim: Optional[float] = 0, upper_lim: Optional[float] = 1
    ) -> Iterable:
        return [lower_lim, upper_lim] * len(param_init)

    def _optimize_likelihood_local(
        self, obs_ts: Iterable, param_init: Iterable,
        method: Optional[str]="SLSQP"
    ) -> OptimizeResult:
        # Local optimization of likelihood using local methds
        bnds = self._build_optimization_bounds(param_init)
        res = so.minimize(
            self._calc_likeihood_optimizer,
            param_init,
            args=(obs_ts, self),
            method=method,
            bounds=bnds,
        )

        return LikelihoodOptResult(res, 'local', method=method)

    def _optimize_likelihood_global(
        self, obs_ts: Iterable, param_init: Iterable,
        sampling_method: Optional[str] = 'sobol'
    ) -> OptimizeResult:
        # Global optimization of likelihood using SHGO algorithm
        if sampling_method not in ['sobol', 'simplical', 'halton']:
            raise ValueError(
                "`sampling_method` must be one of `sobol`, `simplical`, or "
                "`halton`"
            )

        bnds = self._build_optimization_bounds(param_init)
        res = so.shgo(
            self._calc_likelihood_optimizer,
            bounds=bnds,
            args=(obs_ts, self),
            sampling_method=sampling_method
        )

        return LikelihoodOptResult(res, 'global', sampling_method=sampling_method)

    # Inferrence routines
    def expectation(self):
        pass

    def maximization(self):
        pass

    def baum_welch(self):
        pass


# ANCHOR Potentially move this logic to a separate `optimize` module
# Optimizers
def local_likelihood_optimizer(
    est: MarkovInfer, obs_ts: Iterable, method: Optional[str] = "SLSQP"
):
    res = so.minimize()


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

