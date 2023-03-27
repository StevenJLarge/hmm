# Routines for running optimizations in likelihood calcualtions
from abc import ABC, abstractmethod
import os
from typing import Iterable, Tuple, Optional

import numpy as np
import numba
import scipy.optimize

from hidden.infer import MarkovInfer
from hidden.optimize.results import LocalOptimizationResult

_MTX_ENCODING = "ROW_MAJOR"


class BaseOptimizer(ABC):
    def __init__(self, analyzer: MarkovInfer, mtx_encoding: Optional[str] = _MTX_ENCODING):
        self.status = 0
        self.analyzer = analyzer
        self.bayes_filter = None
        self.predictions = None
        self.matrix_encoding = mtx_encoding

    def __repr__(self):
        return f"{self.__name__}(status={self.status})"

    def _extract_parameters(self, param_arr: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        dimension_config = param_arr[:4]
        A_size = dimension_config[0] * dimension_config[1]

        trans_mat = param_arr[4:A_size]
        obs_mat = param_arr[4 + A_size:]

        trans_mat = trans_mat.reshape(dimension_config[0], dimension_config[1])
        obs_mat = obs_mat.reshape(dimension_config[2], dimension_config[3])
        return trans_mat, obs_mat

    def _forward_algo(self):
        # Runs the fwd algo over an entire sequence
        self._initialize_bayes_tracker()

        if prediction_tracker:
            self._initialize_pred_tracker()

        for obs in observations:
            self.bayesian_filter(obs, trans_matrix, obs_matrix, prediction_tracker)
            self.forward_tracker.append(self.bayes_filter)

    @abstractmethod
    def optimize(self):
        pass


class LikelihoodOptimizer(BaseOptimizer):
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
    def _forward_algo(
        observations: Iterable[int], trans_matrix: np.ndarray,
        obs_matrix: np.ndarray
    ) -> np.ndarray:
        pass

    # TODO make this into a static method
    def _calc_likelihood(
        self, param_arr: Iterable, obs_ts: Iterable, *args
    ) -> float:
        # NOTE Currently this only works for a 2D HMM
        A, B = self._extract_hmm_parameters(param_arr)
        # This populates the self.predictions array, which is used by the
        # calc_likelihood below
        self.forward_algo(obs_ts, A, B, prediction_tracker=True)
        return self.calc_likelihood(B, obs_ts)


class LocalLikelihoodOptimizer(BaseOptimizer):
    def __init__(
        self, analyzer: MarkovInfer, algorithm: Optional[str] = "Nelder-Mead"
    ):
        self.algo
        super().__init__(analyzer)

    def optimize(self) -> LocalOptimizationResult:
        pass


class GlobalLikelihoodOptimizer(BaseOptimizer):
    def __init__(
        self, analyzer: MarkovInfer, algorithm: Optional[str] = "sobol"
    ):
        self.algo = algorithm
        super().__init__(analyzer)

    def optimize(self):
        pass

