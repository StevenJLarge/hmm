# Routines for running optimizations in likelihood calcualtions
from abc import ABC, abstractmethod
import os
from typing import Iterable, Tuple, Optional

import numpy as np
import numba
import scipy.optimize

from hidden.infer import MarkovInfer
from hidden.optimize.results import LocalOptimizationResult


class BaseOptimizer(ABC):
    def __init__(self, analyzer: MarkovInfer):
        self.status = 0
        self.analyzer = analyzer

    def __repr__(self):
        return f"{self.__name__}(status={self.status})"

    def _extract_parameters(self, param_arr: Tuple):
        pass

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

