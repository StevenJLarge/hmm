# test suite for dynamics routines
import numpy as np
import pytest
from hidden import dynamics

dimension_tests = [(2, 2), (1, 2), (2, 1), (2, 4), (3, 1)]

@pytest.mark.parametrize('dimensions', dimension_tests)
def test_default_initialization(dimensions):
    # Arrancge
    sys_dim = dimensions[0]
    obs_dim = dimensions[1]

    # Act
    hmm = dynamics.HMM(sys_dim, obs_dim)

    # Assert
    assert hmm.A.shape == (sys_dim, sys_dim)
    assert hmm.B.shape == (obs_dim, sys_dim)
