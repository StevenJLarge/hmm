# Routines for running optimizations in likelihood calcualtions
from abc import ABC, abstractmethod
from operator import mul
import os
from typing import Iterable, Tuple, Optional, Union

import numpy as np
import numba
import scipy.optimize as so

from hidden.infer import MarkovInfer
from hidden.optimize.results import LikelihoodOptimizationResult


class BaseOptimizer(ABC):
    def __init__(self):
        self.status = 0
        self.result = None
        self.bayes_filter = None
        self.predictions = None

    def __repr__(self):
        return f"{self.__name__}(status={self.status})"

    @staticmethod
    # @numba.jit(nopython=True)
    def _forward_algo(observations, trans_matrix, obs_matrix):
        bayes_track = np.zeros((trans_matrix.shape[0], len(observations)))
        pred_track = np.zeros((trans_matrix.shape[0], len(observations)))
        bayes_ = np.ones(trans_matrix.shape[0]) / trans_matrix.shape[0]

        for i, obs in enumerate(observations):
            bayes_, pred_ = BaseOptimizer._bayesian_filter(obs, trans_matrix, obs_matrix, bayes_)
            bayes_track[:, i] = bayes_
            pred_track[:, i] = pred_

        return bayes_track, pred_track

    @staticmethod
    # @numba.jit(nopython=True)
    def _bayesian_filter(obs: int, A: np.ndarray, B: np.ndarray, bayes_: np.ndarray):
        bayes_ = A @ bayes_
        pred_ = bayes_.copy()
        bayes_ = B[:, obs] * bayes_
        bayes_ /= np.sum(bayes_)
        return bayes_, pred_

    @abstractmethod
    def optimize(self):
        pass


class LikelihoodOptimizer(BaseOptimizer):

    def _encode_parameters(self, A: np.ndarray, B: np.ndarray):
        encoded = np.zeros(mul(*A.shape) + mul(*B.shape) - A.shape[0] - B.shape[0])
        dim_tuple = (A.shape, B.shape)
        # Compress the diagonal entries out of A and B
        A_compressed = np.triu(A, k=1)[:, 1:] + np.tril(A, k=-1)[:, :-1]
        B_compressed = np.triu(B, k=1)[:, 1:] + np.tril(B, k=-1)[:, :-1]
        # Encode the off-diagonals into a vector
        encoded[: mul(*A.shape) - A.shape[0]] = np.ravel(A_compressed)
        encoded[mul(*A.shape) - A.shape[0]:] = np.ravel(B_compressed)
        return encoded, dim_tuple

    # @numba.jit(nopython=True)
    # This cant be numba-optimized, becuase we need to cast the dimensional
    # parameters to integers to use as indices.
    @staticmethod
    def _extract_parameters(param_arr: Union[np.ndarray, Tuple], A_dim: Tuple, B_dim: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        # If this is passed in as a tuple, cast to numpy array
        # if isinstance(param_arr, Tuple):
        param_arr = np.array(param_arr)
        # This is not totally true, need to set diagonal elements as sum to
        # preserve cons of prob
        # Take hte dimension to be the 'true' dimension, less the diagonal terms
        A_size = A_dim[0] * A_dim[1] - A_dim[0]

        trans_mat = param_arr[: A_size]
        obs_mat = param_arr[A_size:]

        trans_mat = trans_mat.reshape(A_dim[0], A_dim[1] - 1)
        obs_mat = obs_mat.reshape(B_dim[0], B_dim[1] - 1)

        # Now reconstruct the trans matrix diagonal elements: first the
        # following line will add a diagonal of zeros, note this assumes that
        # the matrix os condensed along axis 1
        trans_mat = (
            np.hstack((np.zeros((A_dim[0], 1)), np.triu(trans_mat)))
            + np.hstack((np.tril(trans_mat, k=-1), np.zeros((A_dim[0], 1))))
        )

        obs_mat = (
            np.hstack((np.zeros((B_dim[0], 1)), np.triu(obs_mat)))
            + np.hstack((np.tril(obs_mat, k=-1), np.zeros((B_dim[0], 1))))
        )
        # Add in diagonal terms so that sum(axis=0) = 1
        trans_mat += np.eye(trans_mat.shape[0], M=trans_mat.shape[1]) - np.diag(trans_mat.sum(axis=0))
        obs_mat += np.eye(obs_mat.shape[0], M=obs_mat.shape[1]) - np.diag(obs_mat.sum(axis=0))

        return trans_mat, obs_mat

    @staticmethod
    # @numba.jit(nopython=True)
    def _likelihood(
        predictions: np.ndarray, obs_ts: np.ndarray, B: np.ndarray
    ) -> float:
        likelihood = 0
        # Transpose on predictions will return each column (which is what we
        # # want here)
        for bayes, obs in zip(predictions.T, obs_ts):
            # Numba tells me that below is faster than using '@'
            inner = bayes @ B[:, obs]
            likelihood -= np.log(inner)
        return likelihood

    @staticmethod
    # @numba.jit(nopython=True)
    def calc_likelihood(param_arr: Iterable, dim: Tuple, obs_ts: Iterable) -> float:
        A_dim, B_dim = dim
        A, B = LikelihoodOptimizer._extract_parameters(param_arr, A_dim, B_dim)
        _, pred = BaseOptimizer._forward_algo(obs_ts, A, B)
        return LikelihoodOptimizer._likelihood(pred, obs_ts, B)

    def _build_optimization_bounds(self, n_params: int, lower_lim: Optional[float] = 1e-3, upper_lim: Optional[float] = 1 - 1e-3) -> Iterable:
        return [(lower_lim, upper_lim)] * n_params

    # Empty method for testing
    def optimize(self):
        pass

class LocalLikelihoodOptimizer(LikelihoodOptimizer):
    def __init__(
        self, algorithm: Optional[str] = "L-BFGS-B"
    ):
        self.algo = algorithm
        super().__init__()

    def optimize(self, obs_ts, A_guess, B_guess) -> LikelihoodOptimizationResult:
        param_init, dim_tuple = self._encode_parameters(A_guess, B_guess)
        opt_args = (dim_tuple, obs_ts)
        bnds = self._build_optimization_bounds(len(param_init))

        # NOTE There is an error here somewhere...
        self.result = so.minimize(
            fun=LikelihoodOptimizer.calc_likelihood,
            x0=param_init,
            args=opt_args,
            method=self.algo,
            bounds=bnds
        )

        return LikelihoodOptimizationResult(self, *self._extract_parameters(self.result.x, *dim_tuple))


class GlobalLikelihoodOptimizer(LikelihoodOptimizer):
    def __init__(
        self, analyzer: MarkovInfer, algorithm: Optional[str] = "sobol"
    ):
        self.algo = algorithm
        super().__init__(analyzer)

    def optimize(self, obs_ts: Iterable, n_params: int, dim_tuple: Tuple) -> LikelihoodOptimizationResult:
        opt_args = (obs_ts,)
        bnds = self._build_optimization_bounds(n_params)

        self.result = so.shgo(
            fun=LikelihoodOptimizer.calc_likelihood,
            bounds=bnds,
            args=opt_args,
            sampling_method=self.sampling_algo
        )

        return LikelihoodOptimizationResult(self, *self._extract_parameters(self.result.x, *dim_tuple))


class EMOptimizer(BaseOptimizer):
    pass


if __name__ == "__main__":
    from hidden import dynamics
    # testing routines here
    A = np.array([
        [0.7, 0.2],
        [0.3, 0.8]
    ])

    B = np.array([
        [0.9, 0.15],
        [0.1, 0.85]
    ])

    hmm = dynamics.HMM(2, 2)
    hmm.initialize_dynamics(A, B)
    hmm.run_dynamics(500)

    obs_ts = hmm.get_obs_ts()
    A_test = np.array([
        [0.75, 0.15],
        [0.25, 0.85]
    ])

    B_test = np.array([
        [0.95, 0.20],
        [0.05, 0.80]
    ])

    opt = LocalLikelihoodOptimizer()

    res = opt.optimize(obs_ts, A_test, B_test)

    print("--DONE--")
