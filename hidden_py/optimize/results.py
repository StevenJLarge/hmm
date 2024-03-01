from abc import ABC
from typing import Optional, Dict, Iterable
from scipy.optimize import OptimizeResult
import numpy as np


class OptimizationResult(ABC):
    """
    Represents the result of an optimization process.
    """
    def __init__(
        self, success: bool, algo_name: str,
        results: Optional[OptimizeResult] = None
    ):
        """
        Initialize the Results object.

        Args:
            success (bool): Indicates whether the optimization was successful.
            algo_name (str): The name of the optimization algorithm used.
            results (OptimizeResult, optional): The optimization results. Defaults to None.
        """
        self._success = success
        self._algo_name = algo_name
        self._results = results
        self._report = None

    @property
    def success(self) -> bool:
        """
        Check if the operation was successful.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        return self._success


class LikelihoodOptimizationResult(OptimizationResult):
    """
    Represents the result of likelihood optimization.

    Attributes:
        likelihood (float): The optimized likelihood value.
        _optimal_params (ndarray): The optimal parameters obtained from optimization.
        A (ndarray): The optimized transition matrix.
        B (ndarray): The optimized emission matrix.
        metadata (dict): Additional metadata associated with the optimization result.
    """

    def __init__(
        self, optimizer: OptimizeResult, A_opt: np.ndarray, B_opt,
        metadata: Optional[Dict] = {}
    ):
        """
        Initialize the Results object.

        Args:
            optimizer (OptimizeResult): The optimizer result object.
            A_opt (np.ndarray): The optimized transition matrix.
            B_opt: The optimized emission matrix.
            metadata (Optional[Dict], optional): Additional metadata. Defaults to {}.
        """
        super().__init__(optimizer.result.success, optimizer.algo, optimizer.result)
        self.likelihood = optimizer.result.fun
        self._optimal_params = optimizer.result.x
        self.A = A_opt
        self.B = B_opt
        self.metadata = metadata

    def __repr__(self):
        """
        Returns a string representation of the LikelihoodOptimizationResult object.
        
        The string representation includes the success status and the algorithm name.
        
        Returns:
            str: A string representation of the LikelihoodOptimizationResult object.
        """
        return (
            f'LikelihoodOptimizationResult(success={self.success}, '
            f'algorithm={self._algo_name})'
        )


class EMOptimizationResult(OptimizationResult):
    """
    Represents the result of an EM optimization process.

    Attributes:
        A_opt (np.ndarray): The optimized transition matrix.
        B_opt (np.ndarray): The optimized observation matrix.
        change_tracker (Iterable): An iterable containing the change in parameters at each iteration.
        iteration_count (int): The number of iterations performed during the optimization process.
        metadata (Optional[Dict]): Optional metadata associated with the optimization result.

    Methods:
        __repr__(): Returns a string representation of the optimization result.
        package_results(): Packages the optimization result into a dictionary.

    """
    def __init__(
        self, A_opt: np.ndarray, B_opt: np.ndarray, change_tracker: Iterable,
        iteration_count: int, metadata: Optional[Dict] = {}
    ):
        """
        Initialize the Results object.

        Args:
            A_opt (np.ndarray): The optimized transition matrix.
            B_opt (np.ndarray): The optimized emission matrix.
            change_tracker (Iterable): The change tracker for size updates.
            iteration_count (int): The number of iterations performed.
            metadata (Optional[Dict], optional): Additional metadata. Defaults to {}.
        """
        super().__init__(True, "baum-welch")
        self.update_size_tracking = change_tracker
        self._iterations = iteration_count
        self.A = A_opt
        self.B = B_opt
        self.metadata = metadata

    def __repr__(self):
        return (
            f"EMOptimizationResult(success={self.success}, "
            f"algorithm={self._algo_name}, iterations={self._iterations})"
        )

    def package_results(self) -> Dict:
        """
        Packages the optimized results into a dictionary.

        Returns:
            dict: A dictionary containing the optimized transition matrix, observation matrix,
                  number of iterations, final iteration norm, success status, and metadata keys.
        """
        self._report = {
            "trans_matrix_opt": self.A,
            "obs_matrix_opt": self.B,
            "iterations": self._iterations,
            "final_iteration_norm": self.update_size_tracking[-1],
            "success": True,
            "metadata": list(self.metadata.keys())
        }
        return self._report
