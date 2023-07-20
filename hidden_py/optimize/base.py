from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Optional, Iterable, Tuple, Union, Iterator, Dict
from operator import mul, eq
from functools import reduce
from itertools import islice, chain
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

    def _build_optimization_constraints(
        self, dim_tuple: Tuple, symmetric: bool
    ) -> Tuple[Dict]:
        if symmetric and any(d > 3 for d in chain(*dim_tuple)):
            raise NotImplementedError(
                'Local Likelihood optimization not currently supported for '
                'symmetric constained matrices with dim > 3'
            )

        if symmetric:
            return (
                # Transition matrix constraints
                {"type": "ineq", "fun": lambda x: 1 - (x[0] + x[1])},
                {"type": "ineq", "fun": lambda x: 1 - (x[0] + x[2])},
                # Observation matrix constraints
                {"type": "ineq", "fun": lambda x: 1 - (x[3] + x[4])},
                {"type": "ineq", "fun": lambda x: 1 - (x[3] + x[5])}
            )

        if not all(reduce(eq, d) for d in dim_tuple):
            raise NotImplementedError(
                "Constraint building is currently only supported for square "
                "transition and observation matrices.."
            )

        n_const_trans = dim_tuple[0][0]
        n_const_obs = dim_tuple[1][0]
        const = []

        for i in range(0, n_const_trans ** 2, n_const_trans):
            const.append({"type": "ineq", "fun": lambda x: 1 - sum(x[i: i + n_const_trans])})

        for i in range(0, n_const_obs ** 2, n_const_obs):
            const.append({"type": "ineq", "fun": lambda x: 1 - sum(x[i: i + n_const_obs])})

        return tuple(const)
 
    @abstractmethod
    def optimize(self):
        pass


class LikelihoodOptimizer(BaseOptimizer):

    @staticmethod
    def _encode_parameters_symmetric(
        trans_mat: np.ndarray, obs_mat: np.ndarray
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
        if trans_mat.shape[0] != trans_mat.shape[1] or obs_mat.shape[0] != obs_mat.shape[1]:
            raise ValueError("Input matrix not square...")

        if not np.all(trans_mat == trans_mat.T) or not np.all(obs_mat == obs_mat.T):
            raise ValueError(
                'Input matrix `trans_mat` or `obs_mat` is not symmetric...'
            )

        dim_tuple = (trans_mat.shape, obs_mat.shape)
        trans_entries = (mul(*trans_mat.shape) - trans_mat.shape[0]) // 2
        obs_entries = (mul(*obs_mat.shape) - obs_mat.shape[0]) // 2
        encoded = np.zeros(trans_entries + obs_entries)

        encoded[:trans_entries] = trans_mat[np.triu_indices(trans_mat.shape[0], k=1)]
        encoded[trans_entries:] = obs_mat[np.triu_indices(obs_mat.shape[0], k=1)]

        return encoded, dim_tuple

    @staticmethod
    def _encode_parameters(
        trans_mat: np.ndarray, obs_mat: np.ndarray
    ) -> Tuple[np.ndarray, Tuple]:
        """Encodes input (non-symmetric, or not-necessarily symmetric) matrices
        for transitions (A) and observations (B) into a 1-d parameter vector
        theta = (p_1, p_2, ...) and also returns a tuple containing the
        dimensions of the input arrays

        Note that because the input matrices are both stochastic matrices, the
        columns are normalized and so the diagonal entries are not independent
        parameters, so they do not get included in the encoded vector. The
        compressed output is stored in a column-major format, as we want to
        impose constraints on column-wise sums for systems with dimesion > 2
        during the optimization

        EXAMPLE

        INPUTS:
        A_1 = [0.8, 0.3]      B_1 = [0.9, 0.15]
              [0.2, 0.7]            [0.1, 0.85]

        A_2

        OUTPUTS:
        encoded: [0.3, 0.2, 0.15, 0.1]
        dim_tuple: ((2,2), (2,2))

        Args:
            trans_mat (np.ndarray): transition matrix for hidden states
            obs_mat (np.ndarray): matrix for symbol emmissions (observations)

        Returns:
            Tuple[np.ndarray, Tuple]: encoded param vector,
                input matrix dimensions
        """
        encoded = np.zeros(mul(*trans_mat.shape) + mul(*obs_mat.shape) - trans_mat.shape[0] - obs_mat.shape[0])
        dim_tuple = (trans_mat.shape, obs_mat.shape)

        trans_flat = np.ravel(trans_mat, order='F')
        obs_flat = np.ravel(obs_mat, order='F')
        trans_compressed = np.delete(trans_flat, slice(0, len(trans_flat), trans_mat.shape[0] + 1))
        obs_compressed = np.delete(obs_flat, slice(0, len(obs_flat), obs_mat.shape[0] + 1))

        encoded[: mul(*trans_mat.shape) - trans_mat.shape[0]] = trans_compressed
        encoded[mul(*trans_mat.shape) - trans_mat.shape[0]:] = obs_compressed

        return encoded, dim_tuple

    @staticmethod
    def _extract_parameters(
        param_arr: Union[np.ndarray, Tuple], trans_mat_dim: Tuple,
        obs_mat_dim: Tuple
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
        trans_comp = param_arr[:mul(*trans_mat_dim) - trans_mat_dim[0]]
        obs_comp = param_arr[mul(*trans_mat_dim) - trans_mat_dim[0]:]

        trans_comp = trans_comp.reshape(trans_mat_dim[1], trans_mat_dim[0] - 1).T
        obs_comp = obs_comp.reshape(obs_mat_dim[1], obs_mat_dim[0] - 1).T

        # Upper and lower triangular components
        trans_up = np.vstack((np.triu(trans_comp, k=1), np.zeros(trans_mat_dim[0])))
        trans_dn = np.vstack((np.zeros(trans_mat_dim[0]), np.tril(trans_comp)))

        obs_up = np.vstack((np.triu(obs_comp, k=1), np.zeros(obs_mat_dim[0])))
        obs_dn = np.vstack((np.zeros(obs_mat_dim[0]), np.tril(obs_comp)))
 
        trans_matrix = trans_up + trans_dn
        obs_matrix = obs_up + obs_dn

        trans_matrix += np.diag(1 - trans_matrix.sum(axis=0))
        obs_matrix += np.diag(1 - obs_matrix.sum(axis=0))

        return trans_matrix, obs_matrix

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
    @numba.jit(nopython=True)
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
            likelihood -= np.log(inner)
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


if __name__ == "__main__":
    # Testing the encoding / decoding logic

    test_matrix = np.array([
        [0.80, 0.10, 0.20],
        [0.15, 0.70, 0.10],
        [0.05, 0.20, 0.70]
    ])

    test_matrix_2 = np.array([
        [0.80, 0.10, 0.20],
        [0.05, 0.60, 0.30],
        [0.15, 0.30, 0.50]
    ])

    test = TestLikelihoodOptimizer()
    encoded, dim_tuple = test._encode_parameters(test_matrix, test_matrix)
    decoded = test._extract_parameters(encoded, *dim_tuple)

    encoded_alt, dim_tuple = test._encode_parameters_alt(test_matrix, test_matrix_2)
    decoded_alt = test._extract_parameters_alt(encoded_alt, *dim_tuple)

    print("--DONE--")