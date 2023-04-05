from abc import ABC, abstractmethod
from scipy.optimize import OptimizeResult
import numpy as np


class BaseOptimizationResult(ABC):
    def __init__(self, success: bool, algo_name: str, results: OptimizeResult):
        self._success = success
        self._algo_name = algo_name
        self._results = results
        self.report = None

    def __repr__(self):
        return f"{self.__name__}(algo={self._algo_name}, success={self.success})"

    @abstractmethod
    def package_results(self):
        pass

    @property
    def success(self):
        return self._success

    @property
    def report(self):
        if self.report is None:
            return self.package_results()
        else:
            return self.report


class LikelihoodOptimizationResult(BaseOptimizationResult):
    def __init__(self, optimizer: OptimizeResult, A_opt: np.ndarray, B_opt: np.ndarray):        
        super().__init__(optimizer.result.success, optimizer.algo, optimizer.result)
        self.likelihood = optimizer.result.fun
        self._optimal_params = optimizer.result.x
        self.A = A_opt
        self.B = B_opt

    def package_results(self):
        self.report = {
            "A_opt": self.A,
            "B_opt": self.B,
            "final_likelihood": self.likelihood,
            "success": self.success,
        }
        return self.report


class EMOptimizationResult(BaseOptimizationResult):
    def __init__(self):
        pass

    def package_results(self):
        pass
