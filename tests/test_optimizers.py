# Testing suite for inferrence routines
import operator
import pytest
import numpy as np
from hidden.dynamics import HMM
from hidden.optimize import base, optimization
from hidden.filters import bayesian


A_test_2 = np.array([[0.7, 0.2], [0.3, 0.8]])
A_test_2_sym = np.array([[0.7, 0.3], [0.3, 0.7]])

B_test_2 = np.array([[0.9, 0.0], [0.1, 1.0]])
B_test_2_sym = np.array([[0.9, 0.1], [0.1, 0.9]])

A_test_3 = np.array([
    [0.8, 0.1, 0.2],
    [0.1, 0.7, 0.2],
    [0.1, 0.2, 0.6]
])

A_test_3_sym = np.array([
    [0.7, 0.1, 0.2],
    [0.1, 0.7, 0.2],
    [0.2, 0.2, 0.6]
])

B_test_3 = np.array([
    [1.0, 0.1, 0.4],
    [0.0, 0.7, 0.3],
    [0.0, 0.2, 0.3]
])

B_test_3_sym = np.array([
    [0.7, 0.1, 0.2],
    [0.1, 0.6, 0.3],
    [0.2, 0.3, 0.5]
])

test_2_enc = np.array([0.2, 0.3, 0.0, 0.1])
test_2_enc_sym = np.array([0.3, 0.1])

test_3_enc = np.array(
    [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.4, 0.0, 0.3, 0.0, 0.2]
)
test_3_enc_sym = np.array([0.1, 0.2, 0.2, 0.1, 0.2, 0.3])

test_data = [[A_test_2, B_test_2], [A_test_3, B_test_3]]
test_data_sym = [[A_test_2_sym, B_test_2_sym], [A_test_3_sym, B_test_3_sym]]

test_data_comp = [
    [A_test_2, B_test_2, test_2_enc],
    [A_test_3, B_test_3, test_3_enc]
]

test_data_comp_sym = [
    [A_test_2_sym, B_test_2_sym, test_2_enc_sym],
    [A_test_3_sym, B_test_3_sym, test_3_enc_sym]
]


# IO tests
@pytest.mark.parametrize(['A_matrix', 'B_matrix'], test_data)
def test_encoding_dimensions(A_matrix, B_matrix):
    # Arrange
    opt = base.TestLikelihoodOptimizer()

    # Act
    _, dim = opt._encode_parameters(A_matrix, B_matrix)

    # Assert
    assert dim[0] == A_matrix.shape
    assert dim[1] == B_matrix.shape


@pytest.mark.parametrize(['A_matrix', 'B_matrix'], test_data)
def test_encoded_length(A_matrix, B_matrix):
    # Arrange
    opt = base.TestLikelihoodOptimizer()

    # Act
    encoded, _ = opt._encode_parameters(A_matrix, B_matrix)

    # Assert
    assert (
        len(encoded) == (
            operator.mul(*A_matrix.shape)
            + operator.mul(*B_matrix.shape)
            - A_matrix.shape[1]
            - B_matrix.shape[1]
        )
    )


@pytest.mark.parametrize(['A_matrix', "B_matrix"], test_data_sym)
def test_encoded_length_symmetric(A_matrix, B_matrix):
    # Arrange
    opt = base.TestLikelihoodOptimizer()

    # Act
    encoded, _ = opt._encode_parameters_symmetric(A_matrix, B_matrix)

    # Assert
    assert (
        len(encoded) == (
            (operator.mul(*A_matrix.shape) - A_matrix.shape[0]) // 2
            + (operator.mul(*B_matrix.shape) - B_matrix.shape[0]) // 2
        )
    )


@pytest.mark.parametrize(['A_matrix', 'B_matrix'], test_data_sym)
def test_non_square_matrix_in_symmetric_encoding_raises(A_matrix, B_matrix):
    # Arrange
    opt = base.TestLikelihoodOptimizer()

    # Act/Assert
    with pytest.raises(ValueError):
        _ = opt._encode_parameters_symmetric(A_matrix[:-1, :], B_matrix)
        _ = opt._encode_parameters_symmetric(A_matrix, B_matrix[1:, :])


@pytest.mark.parametrize(['A_matrix', 'B_matrix'], test_data)
def test_non_symmetric_matrix_in_symmtric_encoding_raises(A_matrix, B_matrix):
    # Arrange
    opt = base.TestLikelihoodOptimizer()

    # Act/Assert
    with pytest.raises(ValueError):
        _ = opt._encode_parameters_symmetric(A_matrix, B_matrix)


# test matrix encoding and parameter extraction
@pytest.mark.parametrize(['A_matrix', 'B_matrix', 'compressed'], test_data_comp)
def test_matrix_encoding(A_matrix, B_matrix, compressed):
    # Arrange
    opt = base.TestLikelihoodOptimizer()

    # Act
    encoded, dim = opt._encode_parameters(A_matrix, B_matrix)

    # Assert
    assert all(encoded == compressed)
    assert dim[0] == A_matrix.shape
    assert dim[1] == B_matrix.shape


@pytest.mark.parametrize(["A_matrix", "B_matrix", "compressed"], test_data_comp_sym)
def test_matrix_encoding_symmetric(A_matrix, B_matrix, compressed):
    # Arrange
    opt = base.TestLikelihoodOptimizer()

    # Act
    encoded, dim = opt._encode_parameters_symmetric(A_matrix, B_matrix)

    # Assert
    assert all(encoded == compressed)
    assert dim[0] == A_matrix.shape
    assert dim[1] == B_matrix.shape


# Test matrix decoding as well
@pytest.mark.parametrize(['A_matrix', 'B_matrix', 'compressed'], test_data_comp)
def test_matrix_decoding(A_matrix, B_matrix, compressed):
    # Arrange
    opt = base.TestLikelihoodOptimizer()

    # Act
    A_decode, B_decode = opt._extract_parameters(compressed, A_matrix.shape, B_matrix.shape)

    # Assert
    assert np.all(np.isclose(A_decode, A_matrix))
    assert np.all(np.isclose(B_decode, B_matrix))


@pytest.mark.parametrize(['A_matrix', 'B_matrix', 'compressed'], test_data_comp_sym)
def test_matrix_decoding_symmetric(A_matrix, B_matrix, compressed):
    # Arrange
    opt = base.TestLikelihoodOptimizer()

    # Act
    A_decode, B_decode = opt._extract_parameters_symmetric(compressed, A_matrix.shape, B_matrix.shape)

    # Assert
    assert np.all(np.isclose(A_decode, A_matrix))
    assert np.all(np.isclose(B_decode, B_matrix))


# TEST FOR EM ALGORITHM
@pytest.mark.parametrize(['A_matrix', 'B_Matrix'], test_data)
def test_xi_matrix_shape(A_matrix, B_matrix):
    # Arrange
    n_steps = 100
    hmm = HMM(*A_matrix.shape)
    hmm.run_dynamics(n_steps)
    obs_ts = hmm.get_obs_ts()
    opt = optimization.EMOptimizer()

    # Act
    alpha_mat = bayesian.alpha_prob(obs_ts, hmm.A, hmm.B, norm=True)
    beta_mat = bayesian.beta_prob(obs_ts, hmm.A, hmm.B, norm=True)
    bayes_mat = bayesian.bayes_estimate(obs_ts, hmm.A, hmm.B)

    sample_xi = opt.xi_matrix(
        hmm.A, hmm.B, obs_ts, alpha_mat, beta_mat, bayes_mat
    )

    # Assert
    assert sample_xi.shape == (hmm.A.shape)


def test_gamma_matrix_size():
    pass


def test_forward_filter_normalization():
    pass


def test_backward_filter_normalization():
    pass


def test_bayes_filter_normalization():
    pass


def test_bw_update_preserves_A_matrix_normalization():
    pass


def test_bw_update_preserves_B_matrix_normalization():
    pass


if __name__ == "__main__":
    pytest.main([__file__])
