# Routines for running optimizations in likelihood calcualtions
from abc import ABC, abstractmethod
from operator import mul
import os
from typing import Iterable, Tuple, Optional, Union

import numpy as np
import numba
import scipy.optimize as so

from hidden.infer import MarkovInfer
from hidden.optimize.results import LocalOptimizationResult, GlobalOptimizationResult

# Dont think I actually need this? Ill just do rwhatever the convention for ravel is...
_MTX_ENCODING = "ROW_MAJOR"


class BaseOptimizer(ABC):
    def __init__(self, mtx_encoding: Optional[str] = _MTX_ENCODING):
        self.status = 0
        self.result = None
        self.bayes_filter = None
        self.predictions = None
        self.matrix_encoding = mtx_encoding

    def __repr__(self):
        return f"{self.__name__}(status={self.status})"

    @staticmethod
    @numba.jit(nopython=True)
    def _forward_algo(observations, trans_matrix, obs_matrix):
        bayes_track = np.zeros((trans_matrix.shape[0], len(observations)))
        pred_track = np.zeros((trans_matrix.shape[0], len(observations)))
        bayes_ = np.ones(trans_matrix.shape[0]) / trans_matrix.shape[0]

        for i, obs in enumerate(observations):
            bayes_, pred_ = BaseOptimizer._bayesian_filter(obs, trans_matrix, obs_matrix, bayes_)
            bayes_track[: i] = bayes_
            pred_track[:, i] = pred_

        return bayes_track, pred_track

    # NOTE Look into this, not sure this is actually working....
    @staticmethod
    @numba.jit(nopython=True)
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

    def _encode_parameters(self, A, B):
        encoded = np.zeros(4 + mul(*A.shape) + mul(*B.shape) - A.shape[0] - B.shape[0])
        encoded[: 2] = A.shape
        encoded[2: 4] = B.shape
        # Compress the diagonal entries out of A and B
        A_compressed = np.triu(A, k=1)[:, 1:] + np.tril(A, k=-1)[:, :-1]
        B_compressed = np.triu(B, k=1)[:, 1:] + np.tril(B, k=-1)[:, :-1]
        # Encode the off-diagonals into a vector
        encoded[4: 4 + mul(*A.shape) - A.shape[0]] = np.ravel(A_compressed)
        encoded[4 + mul(*A.shape) - A.shape[0]:] = np.ravel(B_compressed)
        return encoded

    @staticmethod
    @numba.jit(forceobj=True)
    def _extract_parameters(param_arr: Union[np.ndarray, Tuple]) -> Tuple[np.ndarray, np.ndarray]:
        # If this is passed in as a tuple, cast to numpy array
        # if isinstance(param_arr, Tuple):
        param_arr = np.array(param_arr)
        # This is not totally true, need to set diagonal elements as sum to
        # preserve cons of prob
        dim_config = param_arr[:4].astype(int)
        # Take hte dimension to be the 'true' dimension, less the diagonal terms
        A_size = dim_config[0] * dim_config[1] - dim_config[0]

        trans_mat = param_arr[4: 4 + A_size]
        obs_mat = param_arr[4 + A_size:]

        trans_mat = trans_mat.reshape(dim_config[0], dim_config[1] - 1)
        obs_mat = obs_mat.reshape(dim_config[2], dim_config[3] - 1)

        # Now reconstruct the trans matrix diagonal elements: first the
        # following line will add a diagonal of zeros, note this assumes that
        # the matrix os condensed along axis 1
        trans_mat = (
            np.hstack((np.zeros((dim_config[0], 1)), np.triu(trans_mat)))
            + np.hstack((np.tril(trans_mat, k=-1), np.zeros((dim_config[0], 1))))
        )

        obs_mat = (
            np.hstack((np.zeros((dim_config[2], 1)), np.triu(obs_mat)))
            + np.hstack((np.tril(obs_mat, k=-1), np.zeros((dim_config[2], 1))))
        )
        # Add in diagonal terms so that sum(axis=0) = 1
        trans_mat += np.eye(trans_mat.shape[0], M=trans_mat.shape[1]) - np.diag(trans_mat.sum(axis=0))
        obs_mat += np.eye(obs_mat.shape[0], M=obs_mat.shape[1]) - np.diag(obs_mat.sum(axis=0))

        return trans_mat, obs_mat

    @staticmethod
    @numba.jit(nopython=True)
    def _likelihood(
        predictions: np.ndarray, obs_ts: np.ndarray, B: np.ndarray
    ) -> float:
        likelihood = 0
        for bayes, obs in zip(predictions, obs_ts):
            inner = bayes @ B[:, obs]
            likelihood -= np.log(inner)
        return likelihood

    @staticmethod
    @numba.jit(nopython=True)
    def calc_likelihood(param_arr: Iterable, obs_ts: Iterable) -> float:
        A, B = LikelihoodOptimizer._extract_hmm_parameters(param_arr)
        _, pred = BaseOptimizer._forward_algo(obs_ts, A, B)
        return LikelihoodOptimizer._likelihood(pred, B, obs_ts)

    def _build_optimization_bounds(self, n_params: int, lower_lim: Optional[float] = 1e-3, upper_lim: Optional[float] = 1 - 1e-3) -> Iterable:
        return [(lower_lim, upper_lim)] * n_params

    # Empty method for testing
    def optimize(self):
        pass

class LocalLikelihoodOptimizer(LikelihoodOptimizer):
    def __init__(
        self, algorithm: Optional[str] = "Nelder-Mead"
    ):
        self.algo = algorithm
        super().__init__()

    def optimize(self, obs_ts, A_guess, B_guess) -> LocalOptimizationResult:
        param_init = self._encode_parameters(A_guess, B_guess)
        opt_args = (obs_ts,)
        bnds = self._build_optimization_bounds(param_init)

        self.result = so.minimize(
            fun=LikelihoodOptimizer.calc_likelihood,
            x0=param_init,
            args=opt_args,
            method=self.algo,
            bounds=bnds
        )

        return LocalOptimizationResult(self)


class GlobalLikelihoodOptimizer(LikelihoodOptimizer):
    def __init__(
        self, analyzer: MarkovInfer, algorithm: Optional[str] = "sobol"
    ):
        self.sampling_algo = algorithm
        super().__init__(analyzer)

    def optimize(self, obs_ts, *args):
        opt_args = (obs_ts,)
        # NOTE update this to pass in an actual parameter number
        bnds = self._build_optimization_bounds()

        self.result = so.shgo(
            fun=LikelihoodOptimizer.calc_likelihood,
            bounds=bnds,
            args=opt_args,
            sampling_method=self.sampling_algo
        )

        return GlobalOptimizationResult(self)


class EMOptimizer(BaseOptimizer):
    pass


if __name__ == "__main__":
    # testing routines here
    pass
