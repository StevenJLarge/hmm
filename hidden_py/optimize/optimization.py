from typing import Iterable, Tuple, Optional, Union
import warnings
import numba
from operator import mul
import numpy as np
import scipy.optimize as so
import scipy.linalg as sl

from hidden_py.optimize.results import LikelihoodOptimizationResult, EMOptimizationResult
from hidden_py.optimize.base import LikelihoodOptimizer, CompleteLikelihoodOptimizer
from hidden_py.filters import bayesian


class LocalLikelihoodOptimizer(LikelihoodOptimizer):
    def __init__(
        self, algorithm: Optional[str] = "SLSQP"
    ) -> None:
        """Contructor for LicalLikelihoodOptimizer class

        Args:
            algorithm (Optional[str], optional): Algorithm to use in numerical
                optimization. Defaults to "SLSQP" (Sequential Least-SQuareds
                Programming).
        """
        self.algo = algorithm
        super().__init__()

    def optimize(
        self, obs_ts: Iterable, A_guess: np.ndarray, B_guess: np.ndarray,
        symmetric: Optional[bool] = False
    ) -> LikelihoodOptimizationResult:
        """Routine to run actual optimization of the model. Passes off the
        objective function and encoded parameter array to a scipy optimizer

        Args:
            obs_ts (Iterable): integer sequence of observation values
            A_guess (np.ndarray): Initial guess at transition matrix
            B_guess (np.ndarray): Initial guess at observation matrix
            symmetric (Optional[bool], optional): Whether or not the model
                (A and B matrices) are assumed to be symmetric.
                Defaults to False.

        Returns:
            LikelihoodOptimizationResult: Container object for model results
        """
        # Cast observations to numpy array if they are a list
        obs_ts = np.array(obs_ts)

        # Encode model parameters into parameter vector
        if symmetric:
            param_init, dim_tuple = self._encode_parameters_symmetric(A_guess, B_guess)
        else:
            param_init, dim_tuple = self._encode_parameters(A_guess, B_guess)

        # Additional arguments to pass into optimizer
        opt_args = (dim_tuple, obs_ts, symmetric)
        # Parameter bounds in optimization
        bnds = self._build_optimization_bounds(len(param_init))
        if any(dim_tuple > 2):
            const = self._build_optimization_constraints()

        # run optimizer
        self.result = so.minimize(
            fun=LikelihoodOptimizer.calc_likelihood,
            x0=param_init,
            args=opt_args,
            method=self.algo,
            bounds=bnds
        )

        # Return results
        if symmetric:
            return LikelihoodOptimizationResult(
                self,
                *self._extract_parameters_symmetric(self.result.x, *dim_tuple)
            )
        return LikelihoodOptimizationResult(
            self, *self._extract_parameters(self.result.x, *dim_tuple)
        )


class GlobalLikelihoodOptimizer(LikelihoodOptimizer):
    def __init__(
        self, sampling_algorithm: Optional[str] = "sobol"
    ):
        """Constructor for GlobalLikelihoodOptimizer class

        Args:
            sampling_algorithm (Optional[str], optional): algorithm to use for
                sampling initial random points in optimization algorithm.
                Defaults to "sobol".
        """
        self.algo = sampling_algorithm
        super().__init__()

    def _get_num_params(self, dim_tuple: Tuple, symmetric: bool) -> int:
        """Utility function to determine the number fo parameters from the input
        matrix dimensions.

        Args:
            dim_tuple (Tuple): Tuple containing the size tupled for A and B
            symmetric (bool): Whether the problem is assumed ot be symmetric

        Returns:
            int: total number of parametsr in the problem
        """
        N_a = mul(*dim_tuple[0])
        N_b = mul(*dim_tuple[1])

        N_a = N_a - dim_tuple[0][1]
        N_b = N_b - dim_tuple[1][1]

        if symmetric:
            N_a //= 2
            N_b //= 2

        return N_a + N_b

    def optimize(
        self, obs_ts: Iterable, dim_tuple: Tuple,
        symmetric: Optional[bool] = False
    ) -> LikelihoodOptimizationResult:
        """Routine to run actual optimization of the model. Passes off the
        objective function and encoded parameter array to a scipy optimizer.

        This global optimizer uses the 'Simplical Homology Global Optimizer'
        `shgo` algorithm in the scipy library.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.shgo.html
        or Ref[1] for an overview of the method

        REFERENCES:
            [1] Endres, SC, Sandrock, C, Focke, WW (2018) “A simplicial
            homology algorithm for lipschitz optimisation”, Journal of Global
            Optimization.

        Args:
            obs_ts (Iterable): Integer sequence of observations
            dim_tuple (Tuple): dimensions of A and B matrices
            symmetric (Optional[bool], optional): Whether or not the model
                parameters are assumed to be symmetric.
                Defaults to False.

        Returns:
            LikelihoodOptimizationResult: Container object for model results
        """
        # Additional arguments
        obs_ts = np.array(obs_ts)
        opt_args = (dim_tuple, obs_ts, symmetric)
        n_params = self._get_num_params(dim_tuple, symmetric)
        bnds = self._build_optimization_bounds(n_params)

        self.result = so.shgo(
            func=LikelihoodOptimizer.calc_likelihood,
            bounds=bnds,
            args=opt_args,
            sampling_method=self.algo
        )

        if symmetric:
            return LikelihoodOptimizationResult(
                self,
                *self._extract_parameters_symmetric(self.result.x, *dim_tuple),
                metadata={"local_min": self.result.xl}
            )
        return LikelihoodOptimizationResult(
            self, *self._extract_parameters(self.result.x, *dim_tuple),
            metadata={"local_min": self.result.xl}
        )


class EMOptimizer(CompleteLikelihoodOptimizer):
    def __init__(
        self, threshold: Optional[float] = 1e-8, maxiter: Optional[int] = 5000,
        track_optimization: Optional[Union[bool, int]] = False,
        tracking_interval: Optional[int] = 100,
        tracking_norm: Optional[str] = 'fro',
        **kwargs
    ) -> None:
        """Constructor for Expectation-Maximization optimizer

        Args:
            threshold (Optional[float], optional): update threshold below which
                the iteration procedure for BW optimization will terminate.
                Defaults to 1e-8.
            maxiter (Optional[int], optional): Maximum number of iterations
                performed before the optimizer terminates. Defaults to 5000.
            track_optimization (Optional[Union[bool, int]], optional): Flag as
            to whether or not internal updates to the transition and
            observation matrices are recorded/tracked. Defaults to False.
            tracking_interval (Optional[int], optional): Number of steps
                between recording tracked values, only has an impact if
                `track_optimization` is set to `True`. Defaults to 100.
            tracking_norm (Optional[str], optional): Norm used to track matrix
                update sizes. Can be any of the supported norms used in
                `scipy.linalg.norm`
                (https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.norm.html).
                Defaults to 'fro' (Frobenius).
        """
        if len(kwargs) > 0:
            warnings.warn(
                f"Unrecognized optimizer options {kwargs}, proceeding without "
                "using them. See documentation for valid initialization "
                "options for EM algorithm...", RuntimeWarning
            )

        super().__init__()
        self._opt_threshold = threshold
        self._max_iter = maxiter
        self._track = track_optimization
        self._interval = tracking_interval
        self._update_norm = tracking_norm

    @staticmethod
    # @numba.jit(nopython=True)
    def xi_matrix(
        obs_ts: np.ndarray, trans_matrix: np.ndarray, obs_matrix: np.ndarray,
        alpha_norm: np.ndarray, beta_norm: np.ndarray, bayes: np.ndarray
    ) -> np.ndarray:
        """Routine to calculate the `xi` matrix, which represents the joint
        probability term in the BW algorithm caluclation p(x_t, x_t-1 | Y^T)
        For a more complete description of the term rationale, see supporting
        documentation

        Args:
            obs_ts (np.ndarray): time series of observation values
            trans_matrix (np.ndarray): transition matrix
            obs_matrix (np.ndarray): observation matrix
            alpha_norm (np.ndarray): normalized (at point-in-time) alpha values
                over time (dim: (n_obs, n_hidden_states))
            beta_norm (np.ndarray): normalized (at point-in-time) beta values
                over time (dim: (n_obs, n_hidden_states))
            bayes (np.ndarray): Bayesian smoothed estimate of hidden state
                probabilities (dim: (n_obs, n_hidden_states))

        Returns:
            np.ndarray: final xi-matrix, shape will be = shape(trans_matrix)
        """
        _shape = trans_matrix.shape
        # Iniaitalize the xi array with all zeros
        xi = np.zeros((*_shape, len(obs_ts) - 1))

        for t in range(1, len(obs_ts)):
            # Stack the column of observation matrix to repeat
            stacked_obs = np.repeat(obs_matrix[:, obs_ts[t]], obs_matrix.shape[0]).reshape(*_shape)

            # Outer products calculate the beta_i * alpha_j terms in the xi
            # matrix term equation
            numer_outer = np.outer(
                beta_norm[t, :], (alpha_norm[t - 1, :] * bayes[t - 1, :])
            )
            outer_denom = np.outer(
                beta_norm[t, :], alpha_norm[t - 1, :]
            )

            # Calculate the elements of numerator and denominator
            numer = trans_matrix * stacked_obs * numer_outer
            denom = np.repeat(
                np.sum(trans_matrix * stacked_obs * outer_denom, axis=0),
                obs_matrix.shape[0]
            ).reshape(numer.shape).T

            # Set the time-t element of the resulting xi matrix
            xi[:, :, t - 1] = numer / denom

        # Return the sum over all points in time
        return xi.sum(axis=2)

    @staticmethod
    def _gamma_numer(obs: int, t: int, bayes: np.ndarray) -> np.ndarray:
        """Helper routine to return a matrix with the bayesan estimate in the
        row corresponding to the time-t observation, and zeros in the other

        Args:
            obs (int): observation value at time t
            t (int): time index value
            bayes (np.ndarray): bayesian state estimate at time t

        Returns:
            np.ndarray: n_state x n_state array with row `obs` replaced with
            the bayesian estimate
        """
        obs_num = np.zeros((bayes.shape[1], bayes.shape[1]))
        obs_num[obs, :] = bayes[t, :]
        return obs_num

    @staticmethod
    # @numba.jit(nopython=True)
    def _update_transition_matrix(
        obs_ts: np.ndarray, trans_matrix: np.ndarray, obs_matrix: np.ndarray,
        alpha: np.ndarray, beta: np.ndarray, bayes: np.ndarray
    ) -> np.ndarray:
        """Baum-Welch update to transition matrix. This routine calls the
        calculation of the xi-matrix (joint probability term) and divides the
        result by the summation over all bayesian probability estimates, summed
        over time (see supporting notebooks/documentation for a mathematical
        rationale for each of these terms)

        Args:
            obs_ts (np.ndarray): time-series of HMM observations
            trans_matrix (np.ndarray): transition matrix
            obs_matrix (np.ndarray): observation matrix
            alpha (np.ndarray): (normed) alpha value over time
                (dim: (n_observations, n_hidden_states))
            beta (np.ndarray): (normed) beta value over time
                (dim: (n_observations, n_hidden_states))
            bayes (np.ndarray): bayesian state estimate over time
                (dim: (n_observations, n_hidden_states))

        Returns:
            np.ndarray: Updated transition matrix
                (dim: (n_hidden_states, n_hidden_states))
        """
        ratio = EMOptimizer.xi_matrix(
            obs_ts, trans_matrix, obs_matrix, alpha, beta, bayes
        )

        bayes_matrix = np.repeat(
            (1 / bayes[:-1, :].sum(axis=0)).reshape(1, trans_matrix.shape[0]),
            trans_matrix.shape[1],
            axis=0
        )

        trans_matrix_updated = ratio * bayes_matrix
        return trans_matrix_updated

    @staticmethod
    def _update_observation_matrix(
        obs_ts: np.ndarray, bayes: np.ndarray
    ) -> np.ndarray:
        """Routine to update the observation matrix, based on recorded
        observations and the bayesian state estiamtes. See supporting
        documentation and notebooks for a mathematical description of this term

        Args:
            obs_ts (np.ndarray): observation time-series
            bayes (np.ndarray): bayesian state estiamtes

        Returns:
            np.ndarray: updated observation matrix
                (dim: (n_possible_observation_values, n_hidden_states))
        """
        # This is almost certainly vectorizable, but I cant think of how to do it ATM...
        gamma_mat = np.zeros((bayes.shape[1], bayes.shape[1], len(obs_ts)))

        for i, obs in enumerate(obs_ts):
            gamma_mat[:, :, i] = EMOptimizer._gamma_numer(obs, i, bayes)
        gamma_denom = np.vstack(bayes.shape[1] * [bayes.T]).reshape(gamma_mat.shape)

        return (gamma_mat.sum(axis=2) / gamma_denom.sum(axis=2))

    def baum_welch_step(
        self, trans_matrix: np.ndarray, obs_matrix: np.ndarray,
        obs_ts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single-iteration of the Baum-Welch EM algorithm: Calculate expected
        quantities first, and then update transition matrix and observation
        matrix estimates.

        Args:
            trans_matrix (np.ndarray): current transition matrix estimate
            obs_matrix (np.ndarray): current observation matrix estimate
            obs_ts (np.ndarray): observation timeseries

        Returns:
            Tuple[np.ndarray, np.ndarray]: Updated transition and observation
                matrices (respectively)
        """
        # Expectation step: calculate quantities
        _alpha = bayesian.alpha_prob(obs_ts, trans_matrix, obs_matrix, norm=True)
        _beta = bayesian.beta_prob(obs_ts, trans_matrix, obs_matrix, norm=True)
        _bayes = bayesian.bayes_estimate(obs_ts, trans_matrix, obs_matrix)

        # Maximization step: update matrices
        trans_matrix_updated = EMOptimizer._update_transition_matrix(
            obs_ts, trans_matrix, obs_matrix, _alpha, _beta, _bayes
        )
        obs_matrix_updated = EMOptimizer._update_observation_matrix(
            obs_ts, _bayes
        )

        return trans_matrix_updated, obs_matrix_updated

    def optimize(
        self, obs_ts: np.ndarray, trans_matrix: np.ndarray,
        obs_matrix: np.ndarray
    ) -> EMOptimizationResult:
        """Main entrypoint for executing on Baum-Welch optimization procedure
        this routine will iterate updates to the input transition and
        observation matrices until EITHER the iteration limit is hit OR the
        maximum change to the A or B matrix during the previous step is below
        a threshold (both the threshold level and maximum iteration number)
        are specified in the constructor. Here, update size is quantified by a
        matrix norm. THe defualt is the Frobenius norm, but can be changes to
        any of the norms supported by `scipy.linalg.norm`
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.norm.html)

        Args:
            obs_ts (np.ndarray): time series of observations
            trans_matrix (np.ndarray): transition matrix
            obs_matrix (np.ndarray): observation matrix

        Returns:
            EMOptimizationResult: Optimization result
        """
        iter_count = 0
        update_size = self._opt_threshold + 1
        update_tracker = []
        if self._track:
            trans_mat_tracker = []
            obs_mat_tracker = []

        while update_size > self._opt_threshold and iter_count < self._max_iter:
            prev_trans, prev_obs = trans_matrix, obs_matrix

            # Perform single-step update
            trans_matrix, obs_matrix = self.baum_welch_step(
                trans_matrix, obs_matrix, obs_ts
            )

            # Calculate update 'size'
            update_size = np.max([
                sl.norm(prev_trans - trans_matrix, ord=self._update_norm),
                sl.norm(prev_obs - obs_matrix, ord=self._update_norm)
            ])
            update_tracker.append(update_size)

            # Record the transition and observation matrices if tracking is
            # specified
            if self._track and (iter_count % self._interval == 0):
                trans_mat_tracker.append(trans_matrix)
                obs_mat_tracker.append(obs_matrix)
            iter_count += 1

        meta_dict = {}
        if self._track:
            meta_dict = {
                "trans_tracker": trans_mat_tracker,
                "obs_tracker": obs_mat_tracker
            }

        return EMOptimizationResult(
            trans_matrix, obs_matrix, update_tracker, iter_count,
            metadata=meta_dict
        )


if __name__ == "__main__":
    import time
    # import os
    from hidden_py import dynamics, infer
    from hidden_py.optimize.base import OptClass
    # testing routines here, lets work with symmetric ''true' matrices
    A = np.array([
        [0.7, 0.3],
        [0.3, 0.7]
    ])

    A_3 = np.array([
        [0.80, 0.05, 0.05],
        [0.15, 0.85, 0.20],
        [0.05, 0.10, 0.75]
    ])

    B = np.array([
        [0.9, 0.1],
        [0.1, 0.9]
    ])

    B_3 = np.array([
        [0.90, 0.05, 0.10],
        [0.05, 0.85, 0.05],
        [0.05, 0.10, 0.85]
    ])

    hmm = dynamics.HMM(2, 2)
    hmm3 = dynamics.HMM(3, 3)
    hmm.initialize_dynamics(A, B)
    hmm3.initialize_dynamics(A_3, B_3)
    hmm.run_dynamics(500)
    hmm3.run_dynamics(500)
    obs_ts = hmm.get_obs_ts()
    obs_ts_3 = hmm3.get_obs_ts()

    analyzer = infer.MarkovInfer(2, 2)
    analyzer3 = infer.MarkovInfer(3, 3)

    A_test = np.array([
        [0.75, 0.25],
        [0.25, 0.75]
    ])

    A_test_sym = np.array([
        [0.8, 0.2],
        [0.2, 0.8]
    ])

    B_test = np.array([
        [0.95, 0.20],
        [0.05, 0.80]
    ])

    B_test_sym = np.array([
        [0.95, 0.05],
        [0.05, 0.95]
    ])

    A_test_3 = np.array([
        [0.80, 0.10, 0.15],
        [0.15, 0.85, 0.10],
        [0.05, 0.05, 0.75]
    ])

    B_test_3 = B_3

    param_init_legacy = [0.2, 0.05]
    start_leg = time.time()
    # legacy_res = analyzer.max_likelihood(param_init_legacy, obs_ts)
    end_leg = time.time()
    opt = LocalLikelihoodOptimizer(algorithm="SLSQP")
    opt_em = EMOptimizer()

    start_new_nonsym = time.time()
    res_nosym = opt.optimize(obs_ts, A_test, B_test)
    res_nosym3 = opt.optimize(obs_ts_3, A_test_3, B_test_3)
    end_new_nonsym = time.time()

    start_new_sym = time.time()
    res = opt.optimize(obs_ts, A_test_sym, B_test_sym, symmetric=True)
    end_new_sym = time.time()

    start_new_em = time.time()
    res_em = opt_em.optimize(np.array(obs_ts), A_test, B_test)
    res_em_2 = analyzer.optimize(obs_ts, A_test, B_test, opt_type=OptClass.ExpMax)
    end_new_em = time.time()

    print(f"Time Leg    : {end_leg - start_leg}")
    print(f"Time NonSym : {end_new_nonsym - start_new_nonsym}")
    print(f"Time Sym    : {end_new_sym - start_new_sym}")

    print("--DONE--")
