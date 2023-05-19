import numpy as np
import pytest
import itertools

from hidden import dynamics
from hidden.filters import bayesian

# Test suite for filters routines

# Global configureations
A_test_2 = np.array([
    [0.8, 0.3],
    [0.2, 0.7]
])

A_test_3 = np.array([
    [0.8, 0.2, 0.0],
    [0.1, 0.7, 0.1],
    [0.1, 0.1, 0.9]
])

B_test_2 = np.array([
    [0.9, 0.05],
    [0.1, 0.95]
])

B_test_3 = np.array([
    [0.85, 0.05, 0.10],
    [0.10, 0.90, 0.10],
    [0.05, 0.05, 0.80]
])


@pytest.fixture
def test_hmm_2():
    hmm = dynamics.HMM(2, 2)
    hmm.A = A_test_2
    hmm.B = B_test_2
    return hmm


@pytest.fixture
def test_hmm_3():
    hmm = dynamics.HMM(3, 3)
    hmm.A = A_test_3
    hmm.B = B_test_3
    return hmm


def sample_hmm(n_dim):
    hmm = dynamics.HMM(n_dim, n_dim)
    hmm.init_uniform_cycle(0.15, 0.1)
    return hmm


filter_functions = [
    bayesian.bayes_estimate,
    bayesian.forward_algo,
    bayesian.backward_algo,
    bayesian.alpha_prob,
    bayesian.beta_prob
]


filter_test_data = [
    (np.array([0.80, 0.20]), 2),
    (np.array([0.50, 0.50]), 2),
    (np.array([0.01, 0.99]), 2),
    (np.array([0.15, 0.85]), 2),
    (np.array([0.70, 0.20, 0.10]), 3),
    (np.array([0.33, 0.33, 0.34]), 3),
    (np.array([0.01, 0.98, 0.01]), 3),
    (np.array([0.15, 0.75, 0.10]), 3),
]

sample_obs_seq_2 = [0, 1, 1, 1, 0, 0, 0, 1]
sample_obs_seq_3 = [2, 1, 1, 2, 1, 0, 0, 0]


@pytest.mark.parametrize(['filter_func', 'N'], itertools.product(filter_functions, [2, 3]))
def test_return_from_estimate_is_correct_shape_2d(filter_func, N):
    # Arrange
    n_steps = 10
    pred = None

    test_hmm = sample_hmm(N)
 
    # Act
    test_hmm.run_dynamics(n_steps)
    obs_ts = np.array(test_hmm.get_obs_ts())
    res = filter_func(obs_ts, test_hmm.A, test_hmm.B)

    if isinstance(res, tuple):
        pred = res[1]
        res = res[0]

    # Assert
    assert res.shape == (n_steps, test_hmm.n_sys)
    if pred is not None:
        assert pred.shape == (n_steps, test_hmm.n_sys)


@pytest.mark.parametrize(['fwd_init', 'N'], filter_test_data)
def test_forward_filter_equations(fwd_init, N):
    # Arrange
    test_model = sample_hmm(N)
    obs = 1

    # Act
    # Prediction step
    fwd_pred = test_model.A @ fwd_init
    # Observation step
    fwd_filter = test_model.B[:, obs] * fwd_pred
    fwd_filter = fwd_filter / np.sum(fwd_filter)

    # Perform forward filter inside the bayesian module
    res = bayesian._forward_filter(obs, test_model.A, test_model.B, fwd_init)
    fwd_filter_prod = res[0]
    fwd_pred_prod = res[1]

    # Assert
    assert all(np.isclose(fwd_filter, fwd_filter_prod))
    assert all(np.isclose(fwd_pred, fwd_pred_prod))


@pytest.mark.parametrize(['back_init', 'N'], filter_test_data)
def test_backward_filter_equations(back_init, N):
    # Arrange
    test_model = sample_hmm(N)
    obs = 1

    # Act
    # Backwards prediction
    back_pred = test_model.A.T @ back_init
    # Observation step
    back_filter = test_model.B[obs, :] * back_pred
    back_filter = back_filter / np.sum(back_filter)

    # Perform backward filter using the module
    res = bayesian._forward_filter(obs, test_model.A.T, test_model.B.T, back_init)
    back_filter_prod = res[0]
    back_pred_prod = res[1]

    # Assert
    assert all(np.isclose(back_filter, back_filter_prod))
    assert all(np.isclose(back_pred, back_pred_prod))


@pytest.mark.parametrize(['bayes_init', 'N'], filter_test_data)
def test_bayesian_smoothing_equation(bayes_init, N):
    # Arange
    test_model = sample_hmm(N)
    obs = 1

    # Act
    # NOTE need actual simulated data for this
    fwd_tracker, _ = bayesian.forward_algo(obs, test_model.A, test_model.B)

    pass


def test_alpha_calculation():
    # Arrange
    test_model = sample_hmm()
    
    pass


def test_beta_calculation():
    pass

# # MOVE TO FILTERS
# def test_bayesian_filter_equations(test_hmm):
#     # Arrange
#     BayesInfer = infer.MarkovInfer(2, 2)
#     bayes_init = np.array([0.8, 0.2])
#     obs = 1

#     # Act
#     # Prediction step
#     pred = test_hmm.A @ bayes_init
#     # Observation step
#     bayes_filter = test_hmm.B[:, obs] * pred
#     bayes_filter = bayes_filter / np.sum(bayes_filter)

#     # Forward filtering within the infer module
#     BayesInfer.bayes_filter = bayes_init
#     BayesInfer.bayesian_filter(obs, test_hmm.A, test_hmm.B, False)

#     # Assert
#     assert (BayesInfer.bayes_filter == bayes_filter).all()


# # MOVE TO FILTERS
# def test_bayesian_backward_filter(test_hmm):
#     # Arrange
#     BayesInfer = infer.MarkovInfer(2, 2)
#     bayes_init = np.array([0.8, 0.2])
#     obs = 1

#     # Act
#     # Backward algorithm (note the transpose here)
#     pred = test_hmm.A.T @ bayes_init
#     bayes_backward = test_hmm.B[:, obs] * pred
#     bayes_backward = bayes_backward / np.sum(bayes_backward)

#     # Backward filtering in the infer module
#     BayesInfer.back_filter = bayes_init
#     BayesInfer.bayesian_back(obs, test_hmm.A, test_hmm.B, False)

#     # Assert
#     assert (BayesInfer.back_filter == bayes_backward).all()


# def test_bayesian_smoothing_filter():
#     # TODO
#     pass


# def test_prediction_tracker_initialization_raises_for_invalid_direction():
#     # Arrange
#     BayesInfer = infer.MarkovInfer(2, 2)

#     # Act / Assert
#     with pytest.raises(ValueError):
#         BayesInfer._initialize_pred_tracker(mode="INVALID")


if __name__ == "__main__":
    pytest.main([__file__])