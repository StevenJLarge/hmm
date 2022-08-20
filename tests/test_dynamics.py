# test suite for dynamics routines
import numpy as np
import pytest
from hidden import dynamics

dimension_tests = [(2, 2), (1, 2), (2, 1), (2, 4), (3, 1)]

A_ident = np.eye(2)
B_ident = np.eye(2)


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


@pytest.mark.parametrize('state_value', [0, 1])
def test_set_state_behaviour(state_value):
    # Arrange
    hmm = dynamics.HMM(2, 2)

    # Act
    hmm._set_state(state_value)
    state = hmm.current_state
    desired_state = [0, 0]
    desired_state[state_value] = 1

    # Assert
    assert (state == desired_state).all()


@pytest.mark.parametrize('obs_value', [0, 1])
def test_set_obs_behaviour(obs_value):
    # Arrange
    hmm = dynamics.HMM(2, 2)

    # Act
    hmm._set_obs(obs_value)
    obs = hmm.current_obs
    desired_obs = [0, 0]
    desired_obs[obs_value] = 1

    # Assert
    assert (obs == desired_obs).all()

@pytest.mark.parametrize('invalid_state', [3, -1])
def test_invalid_state_assignment_raises(invalid_state):
    # Arrange
    hmm = dynamics.HMM(2, 2)

    # Act / Assert
    with pytest.raises(IndexError):
        hmm._set_state(invalid_state)


@pytest.mark.parametrize('invalid_obs', [3, -1])
def test_invalid_obs_assignment_raises(invalid_obs):
    # Arrange
    hmm = dynamics.HMM(2, 2)

    # Act / Assert
    with pytest.raises(IndexError):
        hmm._set_obs(invalid_obs)


def test_dynamics_default_behaviour():
    pass


def test_uniform_cycle_initialization():
    pass


def test_invalid_transiton_matrix_raises():
    pass


def test_invalid_observation_matrix_raises():
    pass

