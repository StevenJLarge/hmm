from typing import Iterable, Tuple, Optional, Union
import warnings
import numba
import itertools
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

    def __repr__(self):
        return f"LocalLikelihoodOptimizer(algorithm={self.algo})"

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
        if any(d > 2 for d in itertools.chain(*dim_tuple)):
            const = self._build_optimization_constraints(dim_tuple, symmetric)
        else:
            const = ()

        # run optimizer
        self.result = so.minimize(
            fun=LikelihoodOptimizer.calc_likelihood,
            x0=param_init,
            args=opt_args,
            method=self.algo,
            bounds=bnds,
            constraints=const
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

    def __repr__(self):
        return f"GlobalLikelihoodOptimizer(sampling_algorithm={self.algo})"

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
        if any(d > 2 for d in itertools.chain(*dim_tuple)):
            const = self._build_optimization_constraints(dim_tuple, symmetric)
        else:
            const = None

        self.result = so.shgo(
            func=LikelihoodOptimizer.calc_likelihood,
            constraints=const,
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
        self, threshold: float = 1e-8, maxiter: int = 5000,
        track_optimization: Union[bool, int] = False,
        tracking_interval: int = 100,
        tracking_norm: str = 'fro',
        laplace_factor: float=1e-10,
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
            laplace_factor float: laplace smoothing factor used for handling
                scenarios where a state is unobserved
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
        self._laplace = laplace_factor

    def __repr__(self):
        return (
            f"EMOptimizer(threshold={self._opt_threshold}, "
            f"maxiter={self._max_iter}, track_optimization={self._track})"
        )

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
            # TODO -- fix this warning so that we never actually set anything to NaN
            xi[:, :, t - 1] = numer / denom
            # In cases where the denominator goes to zero, the neumerator is
            # also zero and we can just define those as zero elements
            if np.isnan(xi[:, :, t-1]).any():
                xi[:, :, t-1] = np.nan_to_num(xi[:, :, t-1], nan=0.0)

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
        alpha: np.ndarray, beta: np.ndarray, bayes: np.ndarray, laplace: float
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

        # Laplace smoothing for scenarios where inferred transition rate is zero
        if np.any(trans_matrix_updated == 0):
            trans_matrix_updated += laplace
            trans_matrix_updated /= trans_matrix_updated.sum(axis=0)

        return trans_matrix_updated

    @staticmethod
    def _update_observation_matrix(
        obs_ts: np.ndarray, bayes: np.ndarray, laplace: float
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

        updated = (gamma_mat.sum(axis=2) / gamma_denom.sum(axis=2))

        # Laplace smoothing for scenarios where there is an unobserved state
        if np.any(updated == 0):
            updated += laplace
            updated /= updated.sum(axis=0)

        return updated

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
            obs_ts, trans_matrix, obs_matrix, _alpha, _beta, _bayes,
            self._laplace
        )
        obs_matrix_updated = EMOptimizer._update_observation_matrix(
            obs_ts, _bayes, self._laplace
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
    import hidden_py as hp
    from hidden_py.optimize import optimization
    # for warning handling
    import warnings

    # replicating test
    n_iterations = 100

    # Force warnings to be raised as errors
    warnings.simplefilter("error", category=RuntimeWarning)

    A_test_2 = np.array([[0.7, 0.2], [0.3, 0.8]])
    B_test_2 = np.array([[0.9, 0.01], [0.1, 0.99]])

    A_test_3 = np.array([
        [0.8, 0.1, 0.2],
        [0.1, 0.7, 0.2],
        [0.1, 0.2, 0.6]
    ])
    B_test_3 = np.array([
        [0.98, 0.1, 0.4],
        [0.01, 0.7, 0.3],
        [0.01, 0.2, 0.3]
    ])


    opt2 = optimization.EMOptimizer()
    opt3 = optimization.EMOptimizer()

    for i in range(n_iterations):
        hmm = hp.dynamics.HMM(2, 2)
        hmm3 = hp.dynamics.HMM(3, 3)
        
        hmm.init_uniform_cycle()
        hmm3.init_uniform_cycle()

        hmm.run_dynamics(10)
        hmm3.run_dynamics(10)

        obs_ts = np.array(hmm.get_obs_ts())
        obs_ts_3 = np.array(hmm3.get_obs_ts())

        A_new_2, B_new_2 = opt2.baum_welch_step(A_test_2, B_test_2, obs_ts)
        A_new_3, B_new_3 = opt3.baum_welch_step(A_test_3, B_test_3, obs_ts)

    print("--DONE--")
