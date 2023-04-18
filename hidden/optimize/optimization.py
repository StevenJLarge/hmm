# Routines for running optimizations in likelihood calcualtions
from abc import ABC, abstractmethod
from operator import mul
import os
from typing import Iterable, Tuple, Optional, Union, Iterator
from itertools import islice

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

    @staticmethod
    def _encode_parameters_symmetric(
        A: np.ndarray, B: np.ndarray
    ) -> Tuple[np.ndarray, Tuple]:
        if A.shape[0] != A.shape[1] or B.shape[0] != B.shape[1]:
            raise ValueError("Input matrix not square...")

        if not np.all(A == A.T) or not np.all(B == B.T):
            raise ValueError(
                'Input matrix `A` or `B` is not symmetric...'
            )

        dim_tuple = (A.shape, B.shape)
        A_entries = (mul(*A.shape) - A.shape[0]) // 2
        B_entries = (mul(*B.shape) - B.shape[0]) // 2
        encoded = np.zeros(A_entries + B_entries)

        encoded[:A_entries] = A[np.triu_indices(A.shape[0], k=1)]
        encoded[A_entries:] = B[np.triu_indices(B.shape[0], k=1)]

        return encoded, dim_tuple

    @staticmethod
    def _encode_parameters(
        A: np.ndarray, B: np.ndarray
    ) -> Tuple[np.ndarray, Tuple]:

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
    @staticmethod
    def _extract_parameters_symmetric(
        param_arr: Union[np.ndarray, Tuple], A_dim: Tuple, B_dim: Tuple
    ) -> Tuple[np.ndarray, np.ndarray]:

        def _build_upper_tri(dim: Tuple, param_iter: Iterator):
            mat_ = np.zeros(dim)
            for c in range(dim[0] - 1):
                mat_[c, c+1:] = list(islice(param_iter, dim[0] - 1 - c))
            return mat_

        A_size = A_dim[0] * A_dim[1] - A_dim[0]
        A_size //= 2

        param_iter = iter(param_arr)
        trans_mat = _build_upper_tri(A_dim, param_iter)
        obs_mat = _build_upper_tri(B_dim, param_iter)

        trans_mat += trans_mat.T
        obs_mat += obs_mat.T
        trans_mat += np.diag(1 - trans_mat.sum(axis=1))
        obs_mat += np.diag(1 - obs_mat.sum(axis=1))
        return trans_mat, obs_mat

    # @numba.jit(nopython=True)
    @staticmethod
    def _extract_parameters(
        param_arr: Union[np.ndarray, Tuple], A_dim: Tuple, B_dim: Tuple,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # If this is passed in as a tuple, cast to numpy array
        # if isinstance(param_arr, Tuple):
        param_arr = np.array(param_arr)

        # Take the dimension to be the 'true' dimension, less the diagonal terms
        A_size = A_dim[0] * A_dim[1] - A_dim[0]

        # How do I recompose the symmetric matrix?
        trans_mat = param_arr[:A_size]
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
            inner = bayes @ B[:, obs]
            likelihood -= np.log(inner)
        return likelihood

    @staticmethod
    # @numba.jit(nopython=True)
    def calc_likelihood(
        param_arr: Iterable, dim: Tuple, obs_ts: Iterable,
        symmetric: Optional[bool] = False
    ) -> float:
        A_dim, B_dim = dim
        if symmetric:
            A, B = LikelihoodOptimizer._extract_parameters_symmetric(param_arr, A_dim, B_dim)
        else:
            A, B = LikelihoodOptimizer._extract_parameters(param_arr, A_dim, B_dim)
        _, pred = BaseOptimizer._forward_algo(obs_ts, A, B)
        return LikelihoodOptimizer._likelihood(pred, obs_ts, B)

    @staticmethod
    def _build_optimization_bounds(
        n_params: int, lower_lim: Optional[float] = 1e-3,
        upper_lim: Optional[float] = 1 - 1e-3
    ) -> Iterable:
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

    def optimize(
        self, obs_ts: Iterable, A_guess: np.ndarray, B_guess: np.ndarray,
        symmetric: Optional[bool] = False
    ) -> LikelihoodOptimizationResult:
        if symmetric:
            param_init, dim_tuple = self._encode_parameters_symmetric(A_guess, B_guess)
        else:
            param_init, dim_tuple = self._encode_parameters(A_guess, B_guess)

        opt_args = (dim_tuple, obs_ts, symmetric)
        bnds = self._build_optimization_bounds(len(param_init))
        _ = LikelihoodOptimizer.calc_likelihood(param_init, *opt_args)

        # NOTE There is an error here somewhere...
        self.result = so.minimize(
            fun=LikelihoodOptimizer.calc_likelihood,
            x0=param_init,
            args=opt_args,
            method=self.algo,
            bounds=bnds
        )

        if symmetric:
            return LikelihoodOptimizationResult(self, *self._extract_parameters_symmetric(self.result.x, *dim_tuple))
        return LikelihoodOptimizationResult(self, *self._extract_parameters(self.result.x, *dim_tuple))


class GlobalLikelihoodOptimizer(LikelihoodOptimizer):
    def __init__(
        self, algorithm: Optional[str] = "sobol"
    ):
        self.algo = algorithm
        super().__init__()

    def optimize(
        self, obs_ts: Iterable, n_params: int, dim_tuple: Tuple,
        symmetric: Optional[bool] = False
    ) -> LikelihoodOptimizationResult:
        opt_args = (obs_ts, symmetric)
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
    from hidden import dynamics, infer
    # testing routines here, lets work with symmetric matrices
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
    hmm.run_dynamics(1000)
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

    legacy_res = analyzer.max_likelihood(param_init_legacy, obs_ts)

    opt = LocalLikelihoodOptimizer(algorithm="SLSQP")

    res = opt.optimize(obs_ts, A_test, B_test)
    res = opt.optimize(obs_ts, A_test_sym, B_test_sym, symmetric=True)

    print("--DONE--")

