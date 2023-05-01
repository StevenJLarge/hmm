from abc import ABC, abstractmethod
from typing import Optional, Dict
from scipy.optimize import OptimizeResult
import numpy as np


class OptimizationResult(ABC):
    def __init__(self, success: bool, algo_name: str, results: OptimizeResult):
        self._success = success
        self._algo_name = algo_name
        self._results = results
        self._report = None

    def __repr__(self):
        return f"{self.__name__}(algo={self._algo_name}, success={self.success})"

    @abstractmethod
    def package_results(self):
        pass

    @property
    def success(self) -> bool:
        return self._success

    @property
    def report(self) -> Dict:
        if self._report is None:
            return self.package_results()
        else:
            return self._report


class LikelihoodOptimizationResult(OptimizationResult):
    def __init__(
        self, optimizer: OptimizeResult, A_opt: np.ndarray, B_opt: np.ndarray,
        metadata: Optional[Dict] = {}
    ):
        super().__init__(optimizer.result.success, optimizer.algo, optimizer.result)
        self.likelihood = optimizer.result.fun
        self._optimal_params = optimizer.result.x
        self.A = A_opt
        self.B = B_opt
        self.metadata = metadata

    def package_results(self) -> Dict:
        self._report = {
            "A_opt": self.A,
            "B_opt": self.B,
            "final_likelihood": self.likelihood,
            "success": self.success,
            "metadata": self.metadata
        }
        return self._report


class EMOptimizationResult(OptimizationResult):
    def __init__(self):
        pass

    def package_results(self):
        pass
