# Routines for running optimizations in likelihood calcualtions
from abc import ABC, abstractmethod
from operator import mul
import os
from typing import Iterable, Tuple, Optional

import numpy as np
import numba
import scipy.optimize as so

from hidden.infer import MarkovInfer
from hidden.optimize.results import LocalOptimizationResult, GlobalOptimizationResult

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
        encoded = np.zeros(4 + mul(*A.shape) + mul(*B.shape))
        encoded[: 2] = A.shape
        encoded[2: 4] = B.shape
        encoded[4: 4 + mul(*A.shape)] = np.ravel(A)
        encoded[4 + mul(*A.shape):] = np.ravel(B)
        return encoded

    @staticmethod
    @numba.jit(nopython=True)
    def _extract_parameters(param_arr: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        # This is not totally true, need to set diagonal elements as sum to
        # preserve cons of prob
        dimension_config = param_arr[:4]
        A_size = dimension_config[0] * dimension_config[1]

        trans_mat = param_arr[4:A_size]
        obs_mat = param_arr[4 + A_size:]

        trans_mat = trans_mat.reshape(dimension_config[0], dimension_config[1])
        obs_mat = obs_mat.reshape(dimension_config[2], dimension_config[3])
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


class LocalLikelihoodOptimizer(LikelihoodOptimizer):
    def __init__(
        self, analyzer: MarkovInfer, algorithm: Optional[str] = "Nelder-Mead"
    ):
        self.algo = algorithm
        super().__init__(analyzer)

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
