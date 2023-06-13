from abc import ABC, abstractmethod
from typing import Optional, Dict, Iterable
from scipy.optimize import OptimizeResult
import numpy as np


class OptimizationResult(ABC):
    def __init__(
        self, success: bool, algo_name: str,
        results: Optional[OptimizeResult] = None
    ):
        self._success = success
        self._algo_name = algo_name
        self._results = results
        self._report = None

    @abstractmethod
    def package_results(self):
        pass

    @property
    def success(self) -> bool:
        return self._success

    @property
    def summary(self) -> Dict:
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
            "trans_matrix_opt": self.A_opt,
            "obs_matrix_opt": self.B_opt,
            "final_likelihood": self.likelihood,
            "success": self.success,
            "metadata": list(self.metadata.keys())
        }
        return self._report


class EMOptimizationResult(OptimizationResult):
    def __init__(
        self, A_opt: np.ndarray, B_opt: np.ndarray, change_tracker: Iterable,
        iteration_count: int, metadata: Optional[Dict] = {}
    ):
        super().__init__(True, "baum-welch")
        self.update_size_tracking = change_tracker
        self._iterations = iteration_count
        self.A = A_opt
        self.B = B_opt
        self.metadata = metadata

    def package_results(self) -> Dict:
        self._report = {
            "trans_matrix_opt": self.A,
            "obs_matrix_opt": self.B,
            "iterations": self._iterations,
            "final_iteration_norm": self.update_size_tracking[-1],
            "success": True,
            "metadata": list(self.metadata.keys())
        }
        return self._report
