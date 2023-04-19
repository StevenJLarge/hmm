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
        self.success = res.success
        self.likelihood = res.fun
        self.metadata = {"message": res.message}

        for key, val in kwargs.items():
            self.metadata[key] = val

    def __str__(self):
        return (
            f"result:\t{self.result}\n"
            f"method_type:\t{self.type}\n"
            f"success:\t{self.success}\n"
            f"metadata:\t{self.metadata}\n"
        )


class ExpectationResult:

    def __init__(
        self, bayes_smooth: np.ndarray, bayes_fwd: np.ndarray,
        bayes_back: np.ndarray, A: np.ndarray, B: np.ndarray
    ):
        self.gamma = bayes_smooth
        self.alpha = bayes_fwd
        self.beta = bayes_back
        self.A = A
        self.B = B
        self.dim = A.shape[0]

    def alpha_k(self, state: int):
        return np.array([elem[state] for elem in self.alpha])

    def beta_k(self, state: int):
        return np.array([elem[state] for elem in self.beta])

    def gamma_k(self, state: int):
        return np.array([elem[state] for elem in self.gamma])


class MaximizationResult:

    def __init__(
        self, A_updated: np.ndarray, B_updated: np.ndarray,
        A_prev: np.ndarray, B_prev: np.ndarray
    ):
        self.A = A_updated
        self.B = B_updated
        self.A_prev = A_prev
        self.B_prev = B_prev


class BaumWelchOptimizationResult:

    def __init__(
        self, maxim_res: MaximizationResult, lkly_tracker: Iterable,
        iterations: int, runtime: Optional[str] = None
    ):
        self.A_opt = maxim_res.A
        self.B_opt = maxim_res.B
        self.lkly_tracker = lkly_tracker
        self.iterations = iterations
        self.runtime = runtime


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

    def likelihood_alpha_beta(self):
        if self.alpha_tracker is None:
            raise ValueError(
                "Must run alpha(...) before calculating liklihood..."
            )

        if self.beta_tracker is None:
            raise ValueError(
                "Must run beta(...) before calculating likelihood..."
            )

        self.likelihood_tracker_ab = []
        for a, b in zip(self.alpha_tracker, self.beta_tracker):
            self.likelihood_tracker_ab.append(np.sum(a * b))

    def bayesian_smooth_alpha_beta(self):
        if self.beta_tracker is None or self.alpha_tracker is None or self.likelihood_tracker_ab is None:
            raise ValueError(
                "Must run alpha(...), beta(...), and "
                "likelihood_alpha_beta(...) before calcualting smoothed "
                "estimate..."
            )
        self.bayes_alpha_beta = []
        for a, b, l in zip(self.alpha_tracker, self.beta_tracker, self.likelihood_tracker_ab):
            self.bayes_alpha_beta.append(a * b / l)

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

    # Total Likelihood calculation (Brute)
    # TODO Can probably vectorize this
    def calc_likelihood(self, B: np.ndarray, obs_ts: Iterable[int]) -> float:
        likelihood = 0
        for bayes, obs in zip(self.predictions, obs_ts):
            inner = bayes @ B[:, obs]
            likelihood -= np.log(inner)
        return likelihood

    def _calc_likeihood_optimizer(
        self, param_arr: Iterable, obs_ts: Iterable, *args
    ) -> float:
        # NOTE Currently this only works for a 2D HMM
        A, B = self._extract_hmm_parameters(param_arr)
        # This populates the self.predictions array, which is used by the
        # calc_likelihood below
        self.forward_algo(obs_ts, A, B, prediction_tracker=True)
        return self.calc_likelihood(B, obs_ts)

    def _calc_likelihood_baum_welch(
        self, param_arr: Iterable, obs_ts: Iterable, state_ts: Iterable
    ) -> float:
        # Calculate likelihood assuming full knowledge of hidden state sequence
        _, B = self._extract_hmm_parameters(param_arr)
        likelihood = 0
        for state, obs in zip(state_ts, obs_ts):
            inner = state @ B[:, obs]
            likelihood -= np.log(inner)
        return likelihood

    def _extract_hmm_parameters(
        self, theta: np.ndarray, symmetric: Optional[bool] = True
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

    def _build_optimization_bounds(
        self, param_init: Iterable,
        lower_lim: Optional[float] = 1e-3, upper_lim: Optional[float] = 1 - 1e-3
    ) -> Iterable:
        return [(lower_lim, upper_lim)] * len(param_init)

    def _validate_optimizer_input(
        self, mode: str, obs_ts: Iterable, **kwargs
    ) -> Tuple:

        # Check mode of optimization
        if mode not in ['global', 'baum-welch']:
            raise NotImplementedError(
                f"Mode `{mode}` not implemented, must be one of global "
                "(default) or baum-welch"
            )

        if mode == 'baum-welch':
            if "state_ts" not in kwargs:
                raise ValueError(
                    "baum-welch mode requires input kwarg `state_ts`"
                )
            cost_func = self._calc_likelihood_baum_welch
            opt_args = (obs_ts, kwargs['state_ts'])
        else:
            cost_func = self._calc_likeihood_optimizer
            opt_args = (obs_ts)

        return cost_func, opt_args

    def _optimize_likelihood_local(
        self, obs_ts: Iterable, param_init: Iterable,
        method: Optional[str]="L-BFGS-B", mode: Optional[str] = 'local',
        **kwargs
    ) -> LikelihoodOptResult:

        cost_func, opt_args = self._validate_optimizer_input(mode, obs_ts, **kwargs)

        # Local optimization of likelihood using local methods
        bnds = self._build_optimization_bounds(param_init)
        res = so.minimize(
            cost_func,
            param_init,
            args=opt_args,
            method=method,
            bounds=bnds,
        )

        modifier = (lambda: "-baum-welch" if mode == "baum-welch" else "")()
        return LikelihoodOptResult(res, f'local{modifier}', method=method)

    def _optimize_likelihood_global(
        self, obs_ts: Iterable, param_init: Iterable,
        sampling_method: Optional[str] = 'sobol',
        mode: Optional[str] = 'global',
        **kwargs
    ) -> LikelihoodOptResult:
        # Global optimization of likelihood using SHGO algorithm
        if sampling_method not in ['sobol', 'simplical', 'halton']:
            raise ValueError(
                "`sampling_method` must be one of `sobol`, `simplical`, or "
                "`halton`"
            )

        cost_func, opt_args = self._validate_optimizer_input(mode, obs_ts, **kwargs)

        bnds = self._build_optimization_bounds(param_init)
        res = so.shgo(
            cost_func,
            bounds=bnds,
            # NOTE there is a discrepancy here between how shgo and minimize
            # unpack the args, so we need to force this to be a tuple-nested
            # list
            args=tuple([opt_args]),
            sampling_method=sampling_method,
        )

        modifier = (lambda: "-baum-welch" if mode == 'baum-welch' else '')()

        return LikelihoodOptResult(
            res, f'global{modifier}', sampling_method=sampling_method,
            local_minima=res.xl, local_likelihoods=res.funl
        )

    # Inferrence routines
    def expectation(
        self, obs_ts: Iterable, A_est: np.ndarray, B_est: np.ndarray
    ) -> ExpectationResult:
        # Prediction trackers need to be true for likelihood calcualtion
        self.forward_algo(obs_ts, A_est, B_est, prediction_tracker=True)
        self.backward_algo(obs_ts, A_est, B_est, prediction_tracker=True)
        self.bayesian_smooth(A_est)

        return ExpectationResult(
            self.bayes_smoother, self.forward_tracker, self.backward_tracker,
            A_est, B_est
        )

    def maximization(
        self, exp_result: ExpectationResult, obs_ts: Iterable
    ) -> MaximizationResult:
        # Calc xi term from expectation result
        xi = self._calc_xi_term(exp_result, obs_ts)
        # Update matrices
        return self._update_matrices(exp_result, xi, obs_ts)

    def _calc_xi_term(
        self, exp: ExpectationResult, obs_ts: np.ndarray
    ) -> np.ndarray:
        xi = np.zeros((exp.dim, exp.dim, len(exp.gamma) - 1))
        normalization = np.zeros(xi.shape[2])
        for i in range(exp.dim):
            for j in range(exp.dim):
                xi[i, j, :] = exp.alpha_k(j)[:-1] * exp.A[i, j] * exp.beta_k(i)[1:] * exp.B[i, obs_ts[1:]]
                normalization += xi[i, j, :]

        return xi / normalization

    def _update_matrices(
        self, exp: ExpectationResult, xi: np.ndarray, obs_ts: Iterable
    ) -> MaximizationResult:
        A_new = np.zeros_like(exp.A)
        B_new = np.zeros_like(exp.B)

        for i in range(exp.dim):
            for j in range(exp.dim):
                A_new[i, j] = np.sum(xi[i, j, :]) / np.sum(exp.gamma_k(i)[:-1])
                B_new[i, j] = np.sum(exp.gamma_k(i)[np.array(obs_ts) == j]) / np.sum(exp.gamma_k(i))

        # normalize results to enforce probability conservation
        for col in range(A_new.shape[1]):
            A_new[:, col] = A_new[:, col] / np.sum(A_new[:, col])
            B_new[:, col] = B_new[:, col] / np.sum(B_new[:, col])

        return MaximizationResult(A_new, B_new, exp.A[:, :], exp.B[:, :])

    # ANCHOR READY FOR TESTING
    def baum_welch(
        self, param_init: Iterable, obs_ts: Iterable,
        maxiter: Optional[int] = 100, tolerance: Optional[float] = 1e-8
    ) -> Iterable:
        # Iterate through steps of self.expectation, self.maximization
        opt_param = param_init
        A_est, B_est = self._extract_hmm_parameters(opt_param)
        param_tracker = []

        # TODO Add in tolerance checks based on parameter updates (we have no measure fo CDLL, right?)
        for iteration in range(maxiter):
            exp = self.expectation(obs_ts, A_est, B_est)
            maxim = self.maximization(exp, obs_ts)
            A_est, B_est = maxim.A, maxim.B
            param_tracker.append({
                'iteration': iteration,
                'A_est': A_est[:, :],
                'B_est': B_est[:, :]
            })

        return BaumWelchOptimizationResult(maxim, param_tracker, iteration)

    def _update_hidden_estimate(self, A_est: np.ndarray, trans_rate: float):
        A_est = np.diag([trans_rate] * self.n_sys)
        A_est += np.fliplr(np.diag([1 - trans_rate] * self.n_sys))
        return A_est

    # ANCHOR TBC
    def max_likelihood(
        self, param_init: Iterable, obs_ts: Iterable,
        mode: Optional[str] = 'local'
    ) -> LikelihoodOptResult:
        if mode == 'local':
            return self._optimize_likelihood_local(obs_ts, param_init)
        elif mode == 'global':
            return self._optimize_likelihood_global(obs_ts, param_init)


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
