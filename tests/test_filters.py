import numpy as np
import pytest
import itertools

from hidden_py import dynamics
from hidden_py.filters import bayesian


# Test suite for filters routines
TEST_ITERATIONS = 3

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

sample_A_2 = np.array([
    [0.7, 0.3],
    [0.3, 0.7]
])
sample_B_2 = np.array([
    [0.9, 0.1],
    [0.1, 0.9]
])

sample_A_3 = np.array([
    [0.6, 0.2, 0.2],
    [0.2, 0.6, 0.2],
    [0.2, 0.2, 0.6]
])
sample_B_3 = np.array([
    [0.8, 0.1, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.1, 0.8]
])


sample_obs_seq_2 = np.array([1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1])
sample_obs_seq_3 = np.array([1, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])

bayes_2_ini = np.array([0.05934857, 0.94065143])
bayes_2_fin = np.array([0.16365312, 0.83634688])
alpha_2_ini = np.array([0.1, 0.9])
alpha_2_fin = np.array([1.16607475e-05, 5.95920813e-05])
beta_2_ini = np.array([4.22875327e-05, 7.44711950e-05])
beta_2_fin = np.array([1.0, 1.0])

bayes_3_ini = np.array([0.22251862, 0.69124149, 0.08623989])
bayes_3_fin = np.array([0.04392142, 0.04392197, 0.91215661])
alpha_3_ini = np.array([0.1, 0.8, 0.1])
alpha_3_fin = np.array([2.12415158e-07, 2.12417828e-07, 4.41142152e-06])
beta_3_ini = np.array([1.07615669e-05, 4.17877471e-06, 4.17078047e-06])
beta_3_fin = np.array([1.0, 1.0, 1.0])

bayes_test_data = (
    [sample_A_2, sample_B_2, sample_obs_seq_2, bayes_2_ini, bayes_2_fin],
    [sample_A_3, sample_B_3, sample_obs_seq_3, bayes_3_ini, bayes_3_fin]
)

alpha_test_data = (
    [sample_A_2, sample_B_2, sample_obs_seq_2, alpha_2_ini, alpha_2_fin],
    [sample_A_3, sample_B_3, sample_obs_seq_3, alpha_3_ini, alpha_3_fin]
)

beta_test_data = (
    [sample_A_2, sample_B_2, sample_obs_seq_2, beta_2_ini, beta_2_fin],
    [sample_A_3, sample_B_3, sample_obs_seq_3, beta_3_ini, beta_3_fin]
)

norm_test_samples = [
    [A_test_2, B_test_2],
    [A_test_3, B_test_3]
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


# @pytest.mark.repeat(5)
@pytest.mark.parametrize(['N'], [[2], [3]])
def test_forward_algo_stays_normalized(N):
    # Arange
    test_model = sample_hmm(N)
    test_model.init_uniform_cycle()
    n_steps = 15

    # Act
    test_model.run_dynamics(n_steps)
    obs = test_model.get_obs_ts()
    fwd_tracker, _ = bayesian.forward_algo(obs, test_model.A, test_model.B)

    # Assert
    assert all(np.isclose(np.ones(n_steps), np.sum(fwd_tracker, axis=1)))


# @pytest.mark.repeat(5)
@pytest.mark.parametrize(['N'], [[2], [3]])
def test_backward_algo_stays_normalized(N):
    # Arange
    test_model = sample_hmm(N)
    test_model.init_uniform_cycle()
    n_steps = 15

    # Act
    test_model.run_dynamics(n_steps)
    obs = test_model.get_obs_ts()
    back_tracker, _ = bayesian.backward_algo(np.array(obs), test_model.A, test_model.B)

    # Assert
    assert all(np.isclose(np.ones(n_steps), np.sum(back_tracker, axis=1)))


# @pytest.mark.repeat(10)
@pytest.mark.parametrize(['N', 'exec_num'], list(itertools.product([2, 3], range(10))))
def test_bayesian_algo_stays_normalized(N, exec_num):
    # Arange
    test_model = sample_hmm(N)
    test_model.init_uniform_cycle()
    n_steps = 15

    # Act
    test_model.run_dynamics(n_steps)
    obs = test_model.get_obs_ts()
    bayes_tracker = bayesian.bayes_estimate(np.array(obs), test_model.A, test_model.B)

    # Assert
    assert all(np.isclose(np.ones(n_steps), np.sum(bayes_tracker, axis=1)))


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
    res = bayesian._forward_filter(np.array(obs), test_model.A, test_model.B, fwd_init)
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
    res = bayesian._forward_filter(np.array(obs), test_model.A.T, test_model.B.T, back_init)
    back_filter_prod = res[0]
    back_pred_prod = res[1]

    # Assert
    assert all(np.isclose(back_filter, back_filter_prod))
    assert all(np.isclose(back_pred, back_pred_prod))


@pytest.mark.parametrize(['A', 'B', 'obs', 'bayes_i', 'bayes_f'], bayes_test_data)
def test_bayesian_smoothing_equation(A, B, obs, bayes_i, bayes_f):
    # Act
    bayes_tracker = bayesian.bayes_estimate(obs, A, B)

    # Assert
    assert all(np.isclose(bayes_tracker[0, :], bayes_i))
    assert all(np.isclose(bayes_tracker[-1, :], bayes_f))


@pytest.mark.parametrize(['A', 'B', 'obs', 'alpha_i', 'alpha_f'], alpha_test_data)
def test_alpha_calculation(A, B, obs, alpha_i, alpha_f):
    # Act
    alpha_tracker = bayesian.alpha_prob(obs, A, B)

    # Assert
    assert all(np.isclose(alpha_tracker[0, :], alpha_i))
    assert all(np.isclose(alpha_tracker[-1, :], alpha_f))


@pytest.mark.parametrize(['A', 'B', 'obs', 'beta_i', 'beta_f'], beta_test_data)
def test_beta_calculation(A, B, obs, beta_i, beta_f):
    # Act
    beta_tracker = bayesian.beta_prob(obs, A, B)

    # Assert
    assert all(np.isclose(beta_tracker[0, :], beta_i))
    assert all(np.isclose(beta_tracker[-1, :], beta_f))


@ pytest.mark.parametrize(['A', 'B', 'obs', 'alpha_i', "alpha_f"], alpha_test_data)
def test_alpha_is_state_normalized_when_keyword_is_passed(A, B, obs, alpha_i, alpha_f):
    # Act
    alpha_norm = bayesian.alpha_prob(obs, A, B, norm=True)

    # Assert
    assert all(np.isclose(alpha_norm.sum(axis=1), 1.0))


@ pytest.mark.parametrize(['A', 'B', 'obs', 'beta_i', "beta_f"], beta_test_data)
def test_beta_is_state_normalized_when_keyword_is_passed(A, B, obs, beta_i, beta_f):
    # Act
    beta_norm = bayesian.beta_prob(obs, A, B, norm=True)

    # Assert
    assert all(np.isclose(beta_norm.sum(axis=1), 1.0))


@ pytest.mark.parametrize(['A', 'B', 'obs', 'alpha_i', "alpha_f"], alpha_test_data)
def test_alpha_normalized_result(A, B, obs, alpha_i, alpha_f):
    # Act
    alpha_norm = bayesian.alpha_prob(obs, A, B, norm=True)

    # Assert
    assert all(np.isclose(alpha_norm[0, :], alpha_i / sum(alpha_i)))
    assert all(np.isclose(alpha_norm[-1, :], alpha_f / sum(alpha_f)))


@ pytest.mark.parametrize(['A', 'B', 'obs', 'beta_i', "beta_f"], beta_test_data)
def test_beta_normalized_result(A, B, obs, beta_i, beta_f):
    # Act
    beta_norm = bayesian.beta_prob(obs, A, B, norm=True)

    # Assert
    assert all(np.isclose(beta_norm[0, :], beta_i / sum(beta_i)))
    assert all(np.isclose(beta_norm[-1, :], beta_f / sum(beta_f)))


@pytest.mark.parametrize('iteration', range(TEST_ITERATIONS))
@pytest.mark.parametrize(['A_matrix', 'B_matrix'], norm_test_samples)
def test_forward_filter_normalization(A_matrix, B_matrix, iteration):
    # Arrange
    n_steps = 100
    _ = iteration
    hmm = dynamics.HMM(A_matrix.shape[0], B_matrix.shape[0])
    hmm.init_uniform_cycle()
    hmm.run_dynamics(n_steps)
    obs_ts = np.array(hmm.get_obs_ts())

    # Act, second arg is prediction tracker
    fwd_tracker, _ = bayesian.forward_algo(obs_ts, A_matrix, B_matrix)

    print(fwd_tracker)
    # Assert
    assert np.all(np.isclose(fwd_tracker.sum(axis=1), np.ones(fwd_tracker.shape[0])))


@pytest.mark.parametrize('iteration', range(TEST_ITERATIONS))
@pytest.mark.parametrize(['A_matrix', "B_matrix"], norm_test_samples)
def test_backward_filter_normalization(A_matrix, B_matrix, iteration):
    # Arrange
    _ = iteration
    n_steps = 100
    hmm = dynamics.HMM(A_matrix.shape[0], B_matrix.shape[0])
    hmm.init_uniform_cycle()
    hmm.run_dynamics(n_steps)
    obs_ts = np.array(hmm.get_obs_ts())

    # Act
    back_tracker, _ = bayesian.backward_algo(obs_ts, A_matrix, B_matrix)

    # Assert
    assert np.all(np.isclose(back_tracker.sum(axis=1), np.ones(back_tracker.shape[0])))


@pytest.mark.parametrize('iteration', range(TEST_ITERATIONS))
@pytest.mark.parametrize(['A_matrix', "B_matrix"], norm_test_samples)
def test_bayes_filter_normalization(A_matrix, B_matrix, iteration):
    # Arrange
    _ = iteration
    n_steps = 100
    hmm = dynamics.HMM(*A_matrix.shape)
    hmm.init_uniform_cycle()
    hmm.run_dynamics(n_steps)
    obs_ts = np.array(hmm.get_obs_ts())

    # Act
    bayes_tracker = bayesian.bayes_estimate(obs_ts, A_matrix, B_matrix)

    # Assert
    assert np.all(np.isclose(bayes_tracker.sum(axis=1), np.ones(bayes_tracker.shape[0])))


if __name__ == "__main__":
    pytest.main([__file__])
