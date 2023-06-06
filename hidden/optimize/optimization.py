from typing import Iterable, Tuple, Optional, Union
from operator import mul
import numpy as np
import scipy.optimize as so
import scipy.linalg as sl

from hidden.optimize.results import LikelihoodOptimizationResult
from hidden.optimize.base import LikelihoodOptimizer, CompleteLikelihoodOptimizer
from hidden.filters import bayesian


class LocalLikelihoodOptimizer(LikelihoodOptimizer):
    def __init__(
        self, algorithm: Optional[str] = "L-BFGS-B"
    ) -> None:
        """Contructor for LicalLikelihoodOptimizer class

        Args:
            algorithm (Optional[str], optional): Algorithm to use in numerical
                optimization. Defaults to "L-BFGS-B" (Limited-Memory
                Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm).
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
    def __init__(self):
        pass

    @staticmethod
    def _get_xi_matrix(
        obs: np.ndarray, trans_matrix: np.ndarray, obs_matrix: np.ndarray,
        alpha_norm: np.ndarray, beta_norm: np.ndarray, bayes: np.ndarray
    ):
        # Declare xi array (joint probability numerator)
        xi = np.zeros((*trans_matrix.shape, len(obs_ts) - 1))

        for t in range(1, len(obs_ts)):
            xi[:, :, t - 1] = (
                trans_matrix
                * np.vstack([obs_matrix[:, obs[t]], obs_matrix[:, obs[t]]]).T
                * np.outer(
                    beta_norm[t, :], (alpha_norm[t - 1, :] * bayes[t - 1, :])
                )
            )

        return xi

    @staticmethod
    def _get_denom_matrix(
            obs: np.ndarray, trans_matrix: np.ndarray, obs_matrix: np.ndarray,
            alpha_norm: np.ndarray, beta_norm: np.ndarray
        ):
        xi_denom = np.zeros((1, trans_matrix.shape[0], len(obs_ts) - 1))

        for t in range(1, len(obs_ts)):
            xi_denom[:, :, t-1] = np.sum(
                trans_matrix
                * np.vstack([obs_matrix[:, obs[t]], obs_matrix[:, obs[t]]]).T
                * np.outer(
                    beta_norm[t, :], alpha_norm[t - 1, :]
                )
            , axis=0)

        # And then we sum over the columns
        xi_denom = np.vstack([xi_denom, xi_denom])

        return xi_denom

    @staticmethod
    def _gamma_numer(obs: int, t: int, bayes: np.ndarray):
        # NOTE this must be vectorizable...
        obs_num = np.zeros((bayes.shape[1], bayes.shape[1]))

        target_row = obs
        obs_num[target_row] = bayes[t, :]
        return obs_num

    @staticmethod
    def _update_transition_matrix(
        obs_ts: np.ndarray, trans_matrix: np.ndarray, obs_matrix: np.ndarray,
        alpha: np.ndarray, beta: np.ndarray, bayes: np.ndarray
    ):
        xi_numer = EMOptimizer._get_xi_matrix(
            obs_ts, trans_matrix, obs_matrix, alpha, beta, bayes
        )

        xi_denom = EMOptimizer._get_denom_matrix(
            obs_ts, trans_matrix, obs_matrix, alpha, beta
        )

        ratio = xi_numer / xi_denom
        bayes_matrix = np.repeat(
            (1 / bayes[:-1, :].sum(axis=0)).reshape(1, trans_matrix.shape[0]),
            trans_matrix.shape[1],
            axis=0
        )

        trans_matrix_updated = np.sum(ratio, axis=2) * bayes_matrix
        return trans_matrix_updated

    @staticmethod
    def _update_observation_matrix(obs_ts: np.ndarray, bayes: np.ndarray):
        gamma_mat = np.zeros((bayes.shape[1], bayes.shape[1], len(obs_ts)))

        for i, obs in enumerate(obs_ts):
            gamma_mat[:, :, i] = EMOptimizer._get_gamma_num(obs, i, bayes)
        gamma_denom = np.vstack([bayes.T, bayes.T]).reshape(gamma_mat.shape)

        return (gamma_mat.sum(axis=2) / gamma_denom.sum(axis=2))

    def baum_welch_step(
        self, trans_matrix: np.ndarray, obs_matrix: np.ndarray,
        obs_ts: np.ndarray
    ):
        # Expectation step: calculate quantities
        alpha_ts = self.analyzer.alpha(obs_ts, trans_matrix, obs_matrix, norm=True)
        beta_ts = self.analyzer.beta(obs_ts, trans_matrix, obs_matrix, norm=True)
        bayes_ts = self.analyzer.bayesian_smooth(obs_ts, trans_matrix, obs_matrix)

        # Maximization step: update matrices
        trans_matrix_updated = EMOptimizer._update_transition_matrix(
            obs_ts, trans_matrix, obs_matrix, alpha_ts, beta_ts, bayes_ts
        )
        obs_matrix_updated = EMOptimizer._update_observation_matrix(
            obs_ts, bayes_ts
        )

        return trans_matrix_updated, obs_matrix_updated

    def optimize(
        self, obs_ts: np.ndarray, trans_matrix: np.ndarray,
        obs_matrix: np.ndarray, norm: Optional[str] = 'fro',
        threshold: Optional[float] = 1e-8, maxiter: Optional[int] = 1000,
        tracking: Optional[Union[bool, int]] = False
    ):

        obs_ts = np.array(obs_ts)
        self.analyzer = infer.MarkovInfer(
            trans_matrix.shape[0], obs_matrix.shape[0]
        )
        iter_count = 0
        update_size = threshold + 1
        update_tracker = []
        if tracking:
            trans_mat_tracker = []
            obs_mat_tracker = []

        while update_size > threshold and iter_count < maxiter:
            prev_trans, prev_obs = trans_matrix, obs_matrix

            trans_matrix, obs_matrix = self.baum_welch_step(
                trans_matrix, obs_matrix, obs_ts
            )

            update_size = np.max(
                [sl.norm(prev_trans - trans_matrix, ord=norm),
                sl.norm(prev_obs - obs_matrix, ord=norm)]
            )
            update_tracker.append(update_size)

            iter_count += 1
            if tracking:
                trans_mat_tracker.append(trans_matrix)
                obs_mat_tracker.append(obs_matrix)

        return trans_matrix, obs_matrix


if __name__ == "__main__":
    import time
    from hidden import dynamics, infer
    # testing routines here, lets work with symmetric ''true' matrices
    A = np.array([
        [0.7, 0.3],
        [0.3, 0.7]
    ])

    B = np.array([
        [0.9, 0.1],
        [0.1, 0.9]
    ])

    hmm = dynamics.HMM(2, 2)
    hmm.initialize_dynamics(A, B)
    hmm.run_dynamics(500)
    obs_ts = hmm.get_obs_ts()

    analyzer = infer.MarkovInfer(2, 2)

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

    param_init_legacy = [0.2, 0.05]
    start_leg = time.time()
    # legacy_res = analyzer.max_likelihood(param_init_legacy, obs_ts)
    end_leg = time.time()
    opt = LocalLikelihoodOptimizer(algorithm="SLSQP")
    opt_em = EMOptimizer()

    start_new_nonsym = time.time()
    # res_nosym = opt.optimize(obs_ts, A_test, B_test)
    end_new_nonsym = time.time()

    start_new_sym = time.time()
    # res = opt.optimize(obs_ts, A_test_sym, B_test_sym, symmetric=True)
    end_new_sym = time.time()

    start_new_em = time.time()
    res_em = opt_em.optimize(obs_ts, A_test, B_test)
    end_new_em = time.time()

    print(f"Time Leg    : {end_leg - start_leg}")
    print(f"Time NonSym : {end_new_nonsym - start_new_nonsym}")
    print(f"Time Sym    : {end_new_sym - start_new_sym}")

    print("--DONE--")

