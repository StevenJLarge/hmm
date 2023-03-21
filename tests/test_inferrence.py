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

    assert BayesInfer.bayes_filter is None
    assert BayesInfer.backward_filter is None
    assert BayesInfer.bayes_smoother is None


def test_bayesian_filter_equations(test_hmm):
    # Arrange
    BayesInfer = infer.MarkovInfer(2, 2)
    bayes_init = np.array([0.8, 0.2])
    obs = 1

    # Act
    # Prediction step
    pred = test_hmm.A @ bayes_init
    # Observation step
    bayes_filter = test_hmm.B[:, obs] * pred
    bayes_filter = bayes_filter / np.sum(bayes_filter)

    # Forward filtering within the infer module
    BayesInfer.bayes_filter = bayes_init
    BayesInfer.bayesian_filter(obs, test_hmm.A, test_hmm.B, False)

    # Assert
    assert (BayesInfer.bayes_filter == bayes_filter).all()


def test_bayesian_backward_filter(test_hmm):
    # Arrange
    BayesInfer = infer.MarkovInfer(2, 2)
    bayes_init = np.array([0.8, 0.2])
    obs = 1

    # Act
    # Backward algorithm (note the transpose here)
    pred = test_hmm.A.T @ bayes_init
    bayes_backward = test_hmm.B[:, obs] * pred
    bayes_backward = bayes_backward / np.sum(bayes_backward)

    # Backward filtering in the infer module
    BayesInfer.back_filter = bayes_init
    BayesInfer.bayesian_back(obs, test_hmm.A, test_hmm.B, False)

    # Assert
    assert (BayesInfer.back_filter == bayes_backward).all()


def test_bayesian_smoothing_filter():
    # TODO
    pass


def test_prediction_tracker_initialization_raises_for_invalid_direction():
    # Arrange
    BayesInfer = infer.MarkovInfer(2, 2)

    # Act / Assert
    with pytest.raises(ValueError):
        BayesInfer._initialize_pred_tracker(mode="INVALID")


def test_forward_algorithm_tracking(test_hmm):
    # Arrange
    n_steps = 10
    BayesInfer = infer.MarkovInfer(2, 2)

    # Act
    test_hmm.run_dynamics(10)
    obs_ts = test_hmm.get_obs_ts()
    BayesInfer.forward_algo(obs_ts, test_hmm.A, test_hmm.B, prediction_tracker=True)

    # Assert
    assert len(BayesInfer.forward_tracker) == n_steps
    assert len(BayesInfer.predictions) == n_steps


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


def test_smoothing_algorithm_tracking(test_hmm):
    # Arrange
    n_steps = 10
    BayesInfer = infer.MarkovInfer(2, 2)

    # Act
    test_hmm.run_dynamics(n_steps)
    obs_ts = test_hmm.get_obs_ts()
    BayesInfer.forward_algo(obs_ts, test_hmm.A, test_hmm.B)
    BayesInfer.backward_algo(obs_ts, test_hmm.A, test_hmm.B)
    BayesInfer.bayesian_smooth(test_hmm.A)

    # Assert
    assert len(BayesInfer.bayes_smoother) == len(obs_ts)


def test_smoothing_on_uninitialized_forward_tracker_raises(test_hmm):
    # Arrange
    n_steps = 10
    BayesInfer = infer.MarkovInfer(2, 2)

    # Act
    test_hmm.run_dynamics(n_steps)
    obs_ts = test_hmm.get_obs_ts()
    BayesInfer.backward_algo(obs_ts, test_hmm.A, test_hmm.B)
    # NOTE I dont like this behaviour...
    BayesInfer._initialize_bayes_tracker()

    # Asset
    with pytest.raises(ValueError):
        BayesInfer.bayesian_smooth(test_hmm.A)


def test_smoothing_on_uninitialized_backward_tracker_raises(test_hmm):
    # Arrange
    n_steps = 10
    BayesInfer = infer.MarkovInfer(2, 2)

    # Act
    test_hmm.run_dynamics(n_steps)
    obs_ts = test_hmm.get_obs_ts()
    BayesInfer.forward_algo(obs_ts, test_hmm.A, test_hmm.B)

    # Asset
    with pytest.raises(ValueError):
        BayesInfer.bayesian_smooth(test_hmm.A)


def test_discord_calculation(default_hmm):
    # Arrange
    BayesInfer = infer.MarkovInfer(2, 2)
    n_steps = 10

    # Act
    default_hmm.run_dynamics(n_steps)
    obs_ts = default_hmm.get_obs_ts()
    BayesInfer.forward_algo(obs_ts, default_hmm.A, default_hmm.B)
    prediction = obs_ts
    discord = BayesInfer.discord(obs_ts, prediction)

    # Assert
    assert discord == 0


@pytest.mark.parametrize('sample_data', [(i, j, k) for i, j, k in zip(sample_pred, sample_state, sample_err)])
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


# Likelihood and related

# Optimizations


if __name__ == "__main__":
    pytest.main([__file__])
