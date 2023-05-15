# Testing suite for inferrence routines
import numpy as np
import pytest

from hidden import infer
from hidden import dynamics

# Global configureations
A_test = np.array([
    [0.8, 0.2],
    [0.2, 0.8]
])
B_test = np.array([
    [0.9, 0.1],
    [0.1, 0.9]
])

sample_pred = [
    [0, 0, 0, 0],
    [0, 1, 0, 1],
    [1, 1, 1, 1]
]
sample_state = [
    [1, 1, 1, 1],
    [0, 1, 0, 1],
    [0, 1, 0, 1]
]

sample_err = [1, 0, 0.5]
sample_disc = [2, 0, 1]


@pytest.fixture
def default_hmm():
    hmm = dynamics.HMM(2, 2)
    hmm.A = np.eye(2)
    hmm.B = np.eye(2)
    return hmm


@pytest.fixture
def test_hmm():
    hmm = dynamics.HMM(2, 2)
    hmm.A = A_test
    hmm.B = B_test
    return hmm


# Initialization behaviour
def test_default_constructor_behaviour():
    # Arrange / Act
    BayesInfer = infer.MarkovInfer(2, 2)

    # Assert
    assert BayesInfer.n_sys == 2
    assert BayesInfer.n_obs == 2

    assert BayesInfer.forward_tracker is None
    assert BayesInfer.backward_tracker is None
    assert BayesInfer.bayes_smooth is None
    assert BayesInfer.predictions is None
    assert BayesInfer.predictions_back is None
    assert BayesInfer.alpha_tracker is None
    assert BayesInfer.beta_tracker is None


def test_forward_algorithm_tracking(test_hmm):
    # Arrange
    n_steps = 10
    BayesInfer = infer.MarkovInfer(2, 2)

    # Act
    test_hmm.run_dynamics(10)
    obs_ts = test_hmm.get_obs_ts()
    BayesInfer.forward_algo(obs_ts, test_hmm.A, test_hmm.B)

    # Assert
    assert len(BayesInfer.forward_tracker) == n_steps
    assert len(BayesInfer.prediction_tracker) == n_steps


def test_backward_algorithm_tracking(test_hmm):
    # Arrange
    n_steps = 10
    BayesInfer = infer.MarkovInfer(2, 2)

    # Act
    test_hmm.run_dynamics(10)
    obs_ts = test_hmm.get_obs_ts()
    BayesInfer.backward_algo(obs_ts, test_hmm.A, test_hmm.B)

    # Assert
    assert len(BayesInfer.backward_tracker) == n_steps
    assert len(BayesInfer.predictions_back) == n_steps


def test_smoothing_algorithm_tracking(test_hmm):
    # Arrange
    n_steps = 10
    BayesInfer = infer.MarkovInfer(2, 2)

    # Act
    test_hmm.run_dynamics(n_steps)
    obs_ts = test_hmm.get_obs_ts()
    BayesInfer.bayesian_smooth(obs_ts, test_hmm.A, test_hmm.B)

    # Assert
    assert len(BayesInfer.bayes_smooth) == len(obs_ts)


@pytest.mark.parametrize(
    'sample_data',
    [(i, j, k) for i, j, k in zip(sample_pred, sample_state, sample_disc)]
)
def test_discord_calculation(sample_data):
    # Arrange
    pred_ts = sample_data[0]
    state_ts = sample_data[1]
    discord = sample_data[2]
    BayesInfer = infer.MarkovInfer(2, 2)

    # Act
    discord_ = BayesInfer.discord(state_ts, pred_ts)

    # Assert
    assert discord_ == discord


@pytest.mark.parametrize(
    'sample_data',
    [(i, j, k) for i, j, k in zip(sample_pred, sample_state, sample_err)]
)
def test_error_rate_calculation(sample_data):
    # Arrange
    pred_ts = sample_data[0]
    state_ts = sample_data[1]
    err = sample_data[2]
    BayesInfer = infer.MarkovInfer(2, 2)

    # Act
    error_rate = BayesInfer.error_rate(pred_ts, state_ts)

    # Assert
    assert error_rate == err


# Optimizations


if __name__ == "__main__":
    pytest.main([__file__])
