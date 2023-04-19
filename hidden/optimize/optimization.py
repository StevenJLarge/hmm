from typing import Iterable, Tuple, Optional
import numpy as np
import scipy.optimize as so

from hidden.optimize.results import LikelihoodOptimizationResult
from hidden.optimize.base import LikelihoodOptimizer


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
            return LikelihoodOptimizationResult(self, *self._extract_parameters_symmetric(self.result.x, *dim_tuple))
        return LikelihoodOptimizationResult(self, *self._extract_parameters(self.result.x, *dim_tuple))


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

    def optimize(
        self, obs_ts: Iterable, n_params: int, dim_tuple: Tuple,
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
            n_params (int): number of parameters in the model
            dim_tuple (Tuple): dimensions of A and B matrices
            symmetric (Optional[bool], optional): Whether or not the model
                parameters are assumed to be symmetric.
                Defaults to False.

        Returns:
            LikelihoodOptimizationResult: Container object for model results
        """
        # Additional arguments
        opt_args = (obs_ts, symmetric)
        bnds = self._build_optimization_bounds(n_params)

        self.result = so.shgo(
            fun=LikelihoodOptimizer.calc_likelihood,
            bounds=bnds,
            args=opt_args,
            sampling_method=self.sampling_algo
        )

        return LikelihoodOptimizationResult(self, *self._extract_parameters(self.result.x, *dim_tuple))


# class EMOptimizer(BaseOptimizer):
    # pass


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
    hmm.run_dynamics(10000)
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
    legacy_res = analyzer.max_likelihood(param_init_legacy, obs_ts)
    end_leg = time.time()
    opt = LocalLikelihoodOptimizer(algorithm="SLSQP")

    start_new_nonsym = time.time()
    res_nosym = opt.optimize(obs_ts, A_test, B_test)
    end_new_nonsym = time.time()

    start_new_sym = time.time()
    res = opt.optimize(obs_ts, A_test_sym, B_test_sym, symmetric=True)
    end_new_sym = time.time()

    print(f"Time Leg    : {end_leg - start_leg}")
    print(f"Time NonSym : {end_new_nonsym - start_new_nonsym}")
    print(f"Time Sym    : {end_new_sym - start_new_sym}")

    print("--DONE--")

