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
    hmm.A = np.eye(n_dim)
    hmm.B = np.eye(n_dim)
    return hmm


filter_functions = [
    bayesian.bayes_estimate,
    bayesian.forward_algo,
    bayesian.backward_algo,
    bayesian.alpha_prob,
    bayesian.beta_prob
]


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