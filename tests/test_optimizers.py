# Testing suite for inferrence routines
import os
import operator
import pytest
import numpy as np
from hidden_py.dynamics import HMM
from hidden_py.optimize import base, optimization
from hidden_py.filters import bayesian

TEST_ITERATIONS = 10

# test transition and observation matrices
A_test_2 = np.array([[0.7, 0.2], [0.3, 0.8]])
A_test_2_sym = np.array([[0.7, 0.3], [0.3, 0.7]])

B_test_2 = np.array([[0.9, 0.01], [0.1, 0.99]])
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
    [0.98, 0.1, 0.4],
    [0.01, 0.7, 0.3],
    [0.01, 0.2, 0.3]
])

B_test_3_sym = np.array([
    [0.7, 0.1, 0.2],
    [0.1, 0.6, 0.3],
    [0.2, 0.3, 0.5]
])

# test data - 2-dimensional encoding
test_2_enc = np.array([0.3, 0.2, 0.1, 0.01])
test_2_enc_sym = np.array([0.3, 0.1])

# test data - 3-dimensional encoding
test_3_enc = np.array(
    [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.01, 0.01, 0.1, 0.2, 0.4, 0.3]
)
test_3_enc_sym = np.array([0.1, 0.2, 0.2, 0.1, 0.2, 0.3])

# test data
test_data = [[A_test_2, B_test_2], [A_test_3, B_test_3]]
test_data_sym = [[A_test_2_sym, B_test_2_sym], [A_test_3_sym, B_test_3_sym]]

# compressed test data
test_data_comp = [
    [A_test_2, B_test_2, test_2_enc],
    [A_test_3, B_test_3, test_3_enc]
]

# symmetric test data
test_data_comp_sym = [
    [A_test_2_sym, B_test_2_sym, test_2_enc_sym],
    [A_test_3_sym, B_test_3_sym, test_3_enc_sym]
]


@pytest.fixture
def enable_numba():
    '''Fixture to enable numba'''
    # enable numba
    os.environ["NUMBA_DISABLE_JIT"] = "0"

    yield

    # disable numba
    os.environ["NUMBA_DISABLE_JIT"] = "1"


# IO tests
@pytest.mark.parametrize(['A_matrix', 'B_matrix'], test_data)
def test_encoding_dimensions(A_matrix, B_matrix):
    '''Test that the encoding has the correct dimensions'''
    # Arrange
    opt = base.TestLikelihoodOptimizer()

    # Act
    _, dim = opt._encode_parameters(A_matrix, B_matrix)

    # Assert
    assert dim[0] == A_matrix.shape
    assert dim[1] == B_matrix.shape


@pytest.mark.parametrize(['A_matrix', 'B_matrix'], test_data)
def test_encoded_length(A_matrix, B_matrix):
    '''Test that the encoding has the correct length'''
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
    '''Test that the symmetric encoding has the correct length'''
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
    '''Test that a non-square matrix raises an error in symmetric encoding'''
    # Arrange
    opt = base.TestLikelihoodOptimizer()

    # Act/Assert
    with pytest.raises(ValueError):
        _ = opt._encode_parameters_symmetric(A_matrix[:-1, :], B_matrix)
        _ = opt._encode_parameters_symmetric(A_matrix, B_matrix[1:, :])


@pytest.mark.parametrize(['A_matrix', 'B_matrix'], test_data)
def test_non_symmetric_matrix_in_symmtric_encoding_raises(A_matrix, B_matrix):
    '''Test that a non-symmetric matrix raises an error in symmetric encoding'''
    # Arrange
    opt = base.TestLikelihoodOptimizer()

    # Act/Assert
    with pytest.raises(ValueError):
        _ = opt._encode_parameters_symmetric(A_matrix, B_matrix)


# test matrix encoding and parameter extraction
@pytest.mark.parametrize(['A_matrix', 'B_matrix', 'compressed'], test_data_comp)
def test_matrix_encoding(A_matrix, B_matrix, compressed):
    '''Test that the encoding of the parameters is correct'''
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
    '''Test that the symmetric encoding of the parameters is correct'''
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
    '''Test that the decoding of the compressed parameters is correct'''
    # Arrange
    opt = base.TestLikelihoodOptimizer()

    # Act
    A_decode, B_decode = opt._extract_parameters(compressed, A_matrix.shape, B_matrix.shape)

    # Assert
    assert np.all(np.isclose(A_decode, A_matrix))
    assert np.all(np.isclose(B_decode, B_matrix))


@pytest.mark.parametrize(['A_matrix', 'B_matrix', 'compressed'], test_data_comp_sym)
def test_matrix_decoding_symmetric(A_matrix, B_matrix, compressed):
    '''Test that the symmetric decoding of the compressed parameters is correct'''
    # Arrange
    opt = base.TestLikelihoodOptimizer()

    # Act
    A_decode, B_decode = opt._extract_parameters_symmetric(compressed, A_matrix.shape, B_matrix.shape)

    # Assert
    assert np.all(np.isclose(A_decode, A_matrix))
    assert np.all(np.isclose(B_decode, B_matrix))


@pytest.mark.parametrize('iteration', range(TEST_ITERATIONS))
@pytest.mark.parametrize(['A_matrix', 'B_matrix'], test_data)
def test_xi_matrix_shape(A_matrix, B_matrix, iteration):
    '''Test that the xi matrix has the correct shape'''
    # Arrange
    _ = iteration
    n_steps = 100
    hmm = HMM(A_matrix.shape[0], B_matrix.shape[0])
    hmm.init_uniform_cycle()
    hmm.run_dynamics(n_steps)
    obs_ts = np.array(hmm.obs_ts)
    opt = optimization.EMOptimizer()

    # Act
    alpha_mat = bayesian.alpha_prob(obs_ts, A_matrix, B_matrix, norm=True)
    beta_mat = bayesian.beta_prob(obs_ts, A_matrix, B_matrix, norm=True)
    bayes_mat = bayesian.bayes_estimate(obs_ts, A_matrix, B_matrix)

    sample_xi = opt.xi_matrix(
        obs_ts, A_matrix, B_matrix, alpha_mat, beta_mat, bayes_mat
    )

    # Assert
    assert sample_xi.shape == (hmm.A.shape)


@pytest.mark.parametrize('iteration', range(TEST_ITERATIONS))
@pytest.mark.parametrize(['A_matrix', 'B_matrix'], test_data)
def test_bw_update_preserves_A_matrix_normalization(A_matrix, B_matrix, iteration):
    '''Test that the Baum-Welch update preserves the normalization of the A matrix'''
    # Arrange
    _ = iteration
    n_steps = 10
    hmm = HMM(A_matrix.shape[0], B_matrix.shape[0])
    hmm.init_uniform_cycle()
    hmm.run_dynamics(n_steps)
    obs_ts = np.array(hmm.obs_ts)
    opt = optimization.EMOptimizer()

    # Act
    A_new, B_new = opt.baum_welch_step(A_matrix, B_matrix, obs_ts)
    A_new_2, _ = opt.baum_welch_step(A_new, B_new, obs_ts)

    # Assert
    assert all(np.isclose(A_new.sum(axis=0), np.ones(A_new.shape[1])))
    assert all(np.isclose(A_new_2.sum(axis=0), np.ones(A_new_2.shape[1])))


@pytest.mark.parametrize('iteration', range(TEST_ITERATIONS))
@pytest.mark.parametrize(['A_matrix', 'B_matrix'], test_data)
def test_bw_update_preserves_B_matrix_normalization(A_matrix, B_matrix, iteration):
    '''Test that the Baum-Welch update preserves the normalization of the B matrix'''
    # Arrange
    _ = iteration
    n_steps = 10
    hmm = HMM(A_matrix.shape[0], B_matrix.shape[0])
    hmm.init_uniform_cycle()
    hmm.run_dynamics(n_steps)
    obs_ts = np.array(hmm.obs_ts)
    opt = optimization.EMOptimizer()

    # Act
    A_new, B_new = opt.baum_welch_step(A_matrix, B_matrix, obs_ts)
    _, B_new_2 = opt.baum_welch_step(A_new, B_new, obs_ts)

    # Assert
    assert all(np.isclose(B_new.sum(axis=0), np.ones(B_new.shape[1])))
    assert all(np.isclose(B_new_2.sum(axis=0), np.ones(B_new_2.shape[1])))


def test_calculate_xi():
    '''Test the calculation of the xi matrix'''
    # Define inputs
    trans_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
    obs_ts = np.array([0, 1, 0, 1], dtype=int)
    obs_matrix = np.array([[0.6, 0.4], [0.4, 0.6]])
    beta_norm = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    alpha_norm = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    bayes = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

    # Call the function with the inputs
    result = optimization.EMOptimizer.xi_matrix(
        obs_ts, trans_matrix, obs_matrix, alpha_norm, beta_norm, bayes
    )

    # Define what you expect the output to be
    expected_result = np.array([[0.7, 0.7], [0.8, 0.8]])

    # Assert that the output matches the expected result
    np.testing.assert_array_almost_equal(result, expected_result)


def test_numbafied_optimization_local_likelihood(enable_numba):
    '''Test the numba-compiled local likelihood function'''
    # Define inputs
    obs_ts = np.array([0, 1, 0, 1], dtype=int)
    A_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
    B_matrix = np.array([[0.6, 0.4], [0.4, 0.6]])

    # Call the function with the inputs
    opt = optimization.LocalLikelihoodOptimizer()
    _ = opt.optimize(obs_ts, A_matrix, B_matrix)

    assert True


def test_numbafied_optimization_em(enable_numba):
    '''Test the numba-compiled local likelihood function'''
    # Define inputs
    obs_ts = np.array([0, 1, 0, 1], dtype=int)
    A_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
    B_matrix = np.array([[0.6, 0.4], [0.4, 0.6]])

    # Call the function with the inputs
    opt = optimization.EMOptimizer()
    _ = opt.optimize(obs_ts, A_matrix, B_matrix)

    assert True


if __name__ == "__main__":
    # pytest.main([__file__])
    # for i, (A, B) in enumerate(test_data):
    #     test_bw_update_preserves_A_matrix_normalization(A, B, i)
    pass
