import numpy as np
import pytest

from hidden.filters import bayesian

# Test suite for filters routines

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

