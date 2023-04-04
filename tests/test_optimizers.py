# Testing suite for inferrence routines
import numpy as np
import pytest

from hidden import infer
from hidden import dynamics
from hidden.optimize import optimization

A_test_2 = np.array([
    [0.7, 0.2],
    [0.3, 0.8]
])

B_test_2 = np.array([
    [0.9, 0.0],
    [0.1, 1.0]
])

A_test_3 = np.array([
    [0.8, 0.1, 0.2],
    [0.1, 0.7, 0.2],
    [0.1, 0.2, 0.6]
])

B_test_3 = np.array([
    [1.0, 0.1, 0.4],
    [0.0, 0.7, 0.3],
    [0.0, 0.2, 0.3]
])

test_2_enc = np.array([
    2, 2, 2, 2, 0.2, 0.3, 0.0, 0.1
])

test_3_enc = np.array([
    3, 3, 3, 3, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.4, 0.0, 0.3, 0.0, 0.2
])

# test matrix encoding and parameter extraction
@pytest.mark.parametrize(
    ['A_matrix', 'B_matrix', 'compressed'],
    [
        [A_test_2, B_test_2, test_2_enc],
        [A_test_3, B_test_3, test_3_enc]
    ]
)
def test_matrix_encoding(A_matrix, B_matrix, compressed):
    # Arrange
    opt = optimization.LikelihoodOptimizer()

    # Act
    encoded = opt._encode_parameters(A_matrix, B_matrix)

    # Assert
    assert all(encoded == compressed)


# Test matrix decoding as well
@pytest.mark.parametrize(
    ['A_matrix', 'B_matrix', 'compressed'],
    [
        [A_test_2, B_test_2, test_2_enc],
        [A_test_3, B_test_3, test_3_enc]
    ]
)
def test_matrix_decoding(A_matrix, B_matrix, compressed):
    # Arrange
    opt = optimization.LikelihoodOptimizer()

    # Act
    A_decode, B_decode = opt._extract_parameters(compressed)

    # Assert
    assert np.all(np.isclose(A_decode, A_matrix))
    assert np.all(np.isclose(B_decode, B_matrix))


if __name__ == "__main__":
    pytest.main([__file__])
