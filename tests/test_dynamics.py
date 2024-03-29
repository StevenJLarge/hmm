# test suite for dynamics routines
import numpy as np
import pytest
from hidden_py import dynamics

dimension_tests = [(2, 2), (1, 2), (2, 1), (2, 4), (3, 1)]

A_ident = np.eye(2)
B_ident = np.eye(2)

def uni_cycle_2d(x):
    return np.array([
        [1 - x, x],
        [x, 1 - x]
    ])

def uni_cycle_3d(x):
    return np.array([
        [1 - 2 * x, x, x],
        [x, 1 - 2 * x, x],
        [x, x, 1 - 2 * x]
    ])


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
    """
    Test the behavior of the _set_obs method in the HMM class.

    Parameters:
    obs_value (int): The value of the observation to be set.

    Returns:
    None
    """
    
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
    """
    Test that assigning an invalid state raises an IndexError.

    Args:
        invalid_state: The invalid state to be assigned.

    Raises:
        IndexError: If the state assignment is invalid.
    """
    # Arrange
    hmm = dynamics.HMM(2, 2)

    # Act / Assert
    with pytest.raises(IndexError):
        hmm._set_state(invalid_state)


@pytest.mark.parametrize('invalid_obs', [3, -1])
def test_invalid_obs_assignment_raises(invalid_obs):
    """
    Test that assigning invalid observations raises an IndexError.

    Args:
        invalid_obs: Invalid observations to be assigned.

    Raises:
        IndexError: If the observations are invalid.

    """
    # Arrange
    hmm = dynamics.HMM(2, 2)

    # Act / Assert
    with pytest.raises(IndexError):
        hmm._set_obs(invalid_obs)


@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('trans_rate', [0.1, 0.2, 0.3])
def test_uniform_cycle_initialization(dim, trans_rate):
    """
    Test the initialization of an HMM with a uniform cycle transition matrix.

    Args:
        dim (int): The dimension of the HMM.
        trans_rate (float): The transition rate for the uniform cycle.

    Returns:
        None
    """

    # Arrange
    hmm = dynamics.HMM(dim, dim)

    if dim == 2:
        target_matrix = uni_cycle_2d(trans_rate)
    else:
        target_matrix = uni_cycle_3d(trans_rate)

    # Act
    hmm.init_uniform_cycle(trans_rate, trans_rate)

    # Assert
    assert (hmm.A == target_matrix).all()
    assert (hmm.B == target_matrix).all()


def test_invalid_cycle_initialization_raises():
    """
    Test case to verify that initializing the cycle probabilities with invalid
    values raises NotImplementedError.
    """
    # Arrange
    hmm = dynamics.HMM(2, 3)

    # Act / Assert
    with pytest.raises(NotImplementedError):
        hmm.init_uniform_cycle(0.1, 0.1)


def test_invalid_transiton_matrix_raises():
    """
    Test that an invalid transition matrix raises a ValueError.
    """

    # Arrange
    hmm = dynamics.HMM(2, 2)
    hmm.init_uniform_cycle()

    # Act
    hmm.A = 2 * np.eye(2)

    # Assert
    with pytest.raises(ValueError):
        hmm._validate_dynamics_matrices()


def test_invalid_observation_matrix_raises():
    """
    Test that an invalid observation matrix raises a ValueError.
    """

    # Arrange
    hmm = dynamics.HMM(2, 2)
    hmm.init_uniform_cycle()

    # Act
    hmm.B = 2 * np.eye(2)

    # Assert
    with pytest.raises(ValueError):
        hmm._validate_dynamics_matrices()


def test_dynamics_default_behaviour():
    """
    Test the default behavior of the dynamics module in the HMM class.

    This test function sets up an HMM object with 2 hidden states and 2 observable states.
    It then runs the dynamics for a specified number of steps and checks the behavior of the HMM.

    Returns:
        None
    """

    # Arrange
    hmm = dynamics.HMM(2, 2)
    hmm.A = 0.5 * np.ones((2, 2))
    hmm.B = np.eye(2)
    n_steps = 5000

    # Act
    hmm.run_dynamics(n_steps, init_state=0)
    state_ts = hmm.state_ts
    obs_ts = hmm.obs_ts

    emp_dist = np.sum(state_ts) / n_steps
    emp_obs = np.sum(obs_ts) / n_steps

    # Assert
    assert len(hmm.state_tracker) == n_steps
    assert len(hmm.obs_tracker) == n_steps

    # assert (state_ts == obs_ts).all()
    assert emp_dist == emp_obs
    # assert np.isclose(emp_dist, 0.5, atol=1e-2)

if __name__ == '__main__':
    pytest.main([__file__])
