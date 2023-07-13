from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Optional, Iterable, Tuple, Union, Iterator
from operator import mul
from itertools import islice
import numpy as np
import numba

from hidden_py.filters import bayesian


class OptClass(Enum):
    Local = auto()
    Global = auto()
    ExpMax = auto()


class BaseOptimizer(ABC):
    def __init__(self):
        self.status = 0
        self.result = None
        self.bayes_filter = None
        self.predictions = None

    # def __repr__(self):
    #     return f"{self.name}(status={self.status})"

    @staticmethod
    def _build_optimization_bounds(
        n_params: int, lower_lim: Optional[float] = 1e-3,
        upper_lim: Optional[float] = 1 - 1e-3
    ) -> Iterable:
        """Routine to generate a list of optimization bounds for input into an
        optimization function.

        Args:
            n_params (int): number of model parameters
            lower_lim (Optional[float], optional): lower bound on model
                parameters. Defaults to 1e-3.
            upper_lim (Optional[float], optional): upper bound on model
            parameters. Defaults to 1-1e-3.

        Returns:
            Iterable: Variable-length iterable, with each element containing
                the same upper and lower bounds on a model parameters.
        """
        return [(lower_lim, upper_lim)] * n_params

    def _build_optimization_constraints(self, n_params: int, dim_tuple: Tuple):
        pass

    @abstractmethod
    def optimize(self):
        pass


class LikelihoodOptimizer(BaseOptimizer):

    @staticmethod
    def _encode_parameters_symmetric(
        A: np.ndarray, B: np.ndarray
    ) -> Tuple[np.ndarray, Tuple]:
        """Encoding logic for input HMM transition (A) and abservation (B)
        matrices. Takes in (symmetric) matrices A and B and encodes them into
        a parameter array theta = (p1, p2, ...) and also returns a tuple
        with the dimensions of each input matrix

        Note that because the input matrices are both stochastic matrices, the
        columns are normalized and so the diagonal entries are not independent
        parameters, so they do not get included in the encoded vector

        EXAMPLE:

        INPUTS:
        A = [0.8, 0.2]      B = [0.9, 0.1]
            [0.2, 0.8]          [0.1, 0.9]

        OUTPUTS:
        encoded: [0.2, 0.1]
        dim_tuple: ((2,2), (2,2))

        Args:
            A (np.ndarray): transition matrix for hidden states
            B (np.ndarray): matrix for symbol emmissions (observations)

        Raises:
            ValueError: Input matrix is not square (both must be)
            ValueError: One of the input matrices is not symmetric

        Returns:
            Tuple[np.ndarray, Tuple]: encoded param vector,
                input matrix dimensions
        """
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
        """Encodes input (non-symmetric, or not-necessarily symmetric) matrices
        for transitions (A) and observations (B) into a 1-d parameter vector
        theta = (p_1, p_2, ...) and also returns a tuple containing the
        dimensions of the input arrays

        Note that because the input matrices are both stochastic matrices, the
        columns are normalized and so the diagonal entries are not independent
        parameters, so they do not get included in the encoded vector

        EXAMPLE

        INPUTS:
        A = [0.8, 0.3]      B = [0.9, 0.15]
            [0.2, 0.7]          [0.1, 0.85]

        OUTPUTS:
        encoded: [0.3, 0.2, 0.15, 0.1]
        dim_tuple: ((2,2), (2,2))

        Args:
            A (np.ndarray): transition matrix for hidden states
            B (np.ndarray): matrix for symbol emmissions (observations)

        Returns:
            Tuple[np.ndarray, Tuple]: encoded param vector,
                input matrix dimensions
        """
        encoded = np.zeros(mul(*A.shape) + mul(*B.shape) - A.shape[0] - B.shape[0])
        dim_tuple = (A.shape, B.shape)
        # Compress the diagonal entries out of A and B
        A_compressed = np.triu(A, k=1)[:, 1:] + np.tril(A, k=-1)[:, :-1]
        B_compressed = np.triu(B, k=1)[:, 1:] + np.tril(B, k=-1)[:, :-1]
        # Encode the off-diagonals into a vector
        encoded[: mul(*A.shape) - A.shape[0]] = np.ravel(A_compressed)
        encoded[mul(*A.shape) - A.shape[0]:] = np.ravel(B_compressed)
        return encoded, dim_tuple

    @staticmethod
    def _extract_parameters_symmetric(
        param_arr: Union[np.ndarray, Tuple], A_dim: Tuple, B_dim: Tuple
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Code to decode/extract (symmetric )HMM model parameters from an
        input parameter array that has been encoded using
        `_encode_parameters_symmetric` (i.e. this is the inverse operation)

        EXAMPLE

        INPUTS:
        param_arr = [0.2, 0.3],  A_dim = (2, 2),  B_dim = (2, 2)

        OUTPUT:
        A = [0.8, 0.2]      B = [0.7, 0.3]
            [0.2, 0.8]          [0.3, 0.2]

        Args:
            param_arr (Union[np.ndarray, Tuple]): encoded parameter array
            A_dim (Tuple): dimensions of transition matrix
            B_dim (Tuple): dimenstions of observation matrix

        Returns:
            Tuple[np.ndarray, np.ndarray]: transition matrix (A),
                observation matrix (B)
        """

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

    @staticmethod
    def _extract_parameters(
        param_arr: Union[np.ndarray, Tuple], A_dim: Tuple, B_dim: Tuple,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Code to decode/extract (non-symmetric or not-necessarily symmetric)
        HMM model parameters from an input parameter array that has been
        encoded using `_encode_parameters` (i.e. this is the inverse operation)

        EXAMPLE

        INPUTS:
        param_arr = [0.2, 0.3, 0.1, 0.15],  A_dim = (2, 2),  B_dim = (2, 2)

        OUTPUT:
        A = [0.7, 0.2]      B = [0.85, 0.1]
            [0.3, 0.8]          [0.15, 0.9]

        Args:
            param_arr (Union[np.ndarray, Tuple]): encoded parameter array
            A_dim (Tuple): dimensions of transition matrix
            B_dim (Tuple): dimenstions of observation matrix

        Returns:
            Tuple[np.ndarray, np.ndarray]: transition matrix (A),
                observation matrix (B)
        """

        # If this is passed in as a tuple, cast to numpy array
        param_arr = np.array(param_arr)

        # Take the dimension to be the 'true' dimension, less the diagonal terms
        A_size = A_dim[0] * A_dim[1] - A_dim[0]

        # Pull out A, and B specific parameters
        trans_mat = param_arr[:A_size]
        obs_mat = param_arr[A_size:]

        trans_mat = trans_mat.reshape(A_dim[0], A_dim[1] - 1)
        obs_mat = obs_mat.reshape(B_dim[0], B_dim[1] - 1)

        # Now reconstruct the trans matrix diagonal elements: first the
        # following line will add a diagonal of zeros, note this assumes that
        # the matrix is condensed along axis 1
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
        """Calcualte negative log-likelihood value for a particular set of
        parameters (which produce a specific set of predictions and a B matrix)

        Args:
            predictions (np.ndarray): Sequence of unnormalized preiction
                vectors from the bayesian_filter calculation
            obs_ts (np.ndarray): sequence of discrete-state observations
            B (np.ndarray): observation matrix

        Returns:
            float: negative log-likelihood value
        """
        likelihood = 0
        for i, obs in enumerate(obs_ts):
            inner = predictions[i, :] @ B[obs, :]
            if inner <= 0:
                print("OOPS!")
            likelihood -= np.log(inner)
            if likelihood == np.nan:
                print('Hmmm...')
        return likelihood

    @staticmethod
    def calc_likelihood(
        param_arr: Iterable, dim: Tuple, obs_ts: Iterable,
        symmetric: Optional[bool] = False
    ) -> float:
        """Logic to calculate the likelihood values for a given parameter array

        Args:
            param_arr (Iterable): encoded parameter vector
            dim (Tuple): tuple of A and B matrix dimensions (dim_tuple from
                encoding output)
            obs_ts (Iterable): Time-series sequence of (integer) state
                observations
            symmetric (Optional[bool], optional): whether the model
                (A and B matrices) are assumed to be symmetric.
                Defaults to False.

        Returns:
            float: negative log-likeihood of parameter vector, given obs_ts
        """
        A_dim, B_dim = dim
        # Extract parameters
        if symmetric:
            A, B = LikelihoodOptimizer._extract_parameters_symmetric(param_arr, A_dim, B_dim)
        else:
            A, B = LikelihoodOptimizer._extract_parameters(param_arr, A_dim, B_dim)
        # Generate predictions vector
        _, pred = bayesian.forward_algo(obs_ts, A, B)
        # Return likelihood value
        return LikelihoodOptimizer._likelihood(pred, np.array(obs_ts), B)


class CompleteLikelihoodOptimizer(BaseOptimizer):
    def __init__(self):
        pass

    @abstractmethod
    def optimize(self):
        pass


# Method for runnning tests on abstract class TestLikelihoodOptimizer
class TestLikelihoodOptimizer(LikelihoodOptimizer):
    def optimize(self):
        pass

