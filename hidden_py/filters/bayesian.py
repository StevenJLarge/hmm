# Bayesian filters -- Fowrward, backward, bayesian
from typing import Iterable, Tuple, Optional
import numpy as np
import numba


# @numba.jit(nopython=True)
def bayes_estimate(
    obs_ts: np.ndarray, trans_matrix: np.ndarray, obs_matrix: np.ndarray
) -> np.ndarray:
    """Implementation of Bayesian 'smoothing' algoithm. This calcualted the
    probability distribution p(x | Y^T), conditioned on both past and future
    informaton

    Args:
        obs_ts (np.ndarray): sequence of observations
        trans_matrix (np.ndarray): hidden state transition rate matrix
        obs_matrix (np.ndarray): observation/emmision probability matrix

    Returns:
        np.ndarray: array with each row representing the inferred probability
        distribution over individual states
    """
    fwd_tracker, pred = forward_algo(obs_ts, trans_matrix, obs_matrix)

    bayes_smooth = np.zeros((len(obs_ts), trans_matrix.shape[1]), dtype=float)
    bayes_smooth[-1, :] = fwd_tracker[-1, :]
    ratio = np.zeros(trans_matrix.shape)

    # Iterate backwards through the forward tracker from N-1 -> 1
    for i in range(fwd_tracker.shape[0] - 1, 0, -1):
        # Ratio of previous bayesian estimates to forward predictions, shaped
        # to match trans_matrix shape
        ratio[:, :] = (bayes_smooth[i, :] / pred[i, :]).reshape(-1, 1)
        # summation term
        summand = np.sum(trans_matrix * ratio, axis=0)
        # Smoothed bayesian estimate
        bayes_smooth[i - 1, :] = fwd_tracker[i - 1, :] * summand

    return bayes_smooth


# @numba.jit(nopython=True)
def forward_algo(
    observations: Iterable, trans_matrix: np.ndarray, obs_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs the forward bayesian filter calculations on an input set of
    bservations (`observations`) based on input transition (`trans_matrix`)
    and observation (`obs_matrix`) matrices.

    Args:
        observations (Iterable): Integer sequence of observed states
        trans_matrix (np.ndarray): Matrix of transition probabilities
        obs_matrix (np.ndarray): Matrix of observation probabilities

    Returns:
        Tuple[np.ndarray, np.ndarray]: Bayesian filtered state estimates,
            Bayesian state predictions (middle-state of recursive Bayesian
            update equations)
    """
    # initialize trackers so that single-observation slices will be
    # contiguous in memory
    fwd_track = np.zeros((len(observations), trans_matrix.shape[0]))
    pred_track = np.zeros((len(observations), trans_matrix.shape[0]))
    fwd_est = np.ones(trans_matrix.shape[0]) / trans_matrix.shape[0]

    for i, obs in enumerate(observations):
        fwd_est, pred = _forward_filter(obs, trans_matrix, obs_matrix, fwd_est)
        fwd_track[i, :] = fwd_est
        pred_track[i, :] = pred

    return fwd_track, pred_track


# @numba.jit(nopython=True)
def backward_algo(
    observations: np.ndarray, trans_matrix: np.ndarray, obs_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Backwards algorithm, calcualtes the probability P(x | Y^[t: T]), that is,
    the probability distribution over the current system states, given all of
    the future observations that will occur.

    Args:
        observations (np.ndarray): forward-time sequence of opservations
        trans_matrix (np.ndarray): hidden state transition matrix
        obs_matrix (np.ndarray): observation/emmision probability matrix

    Returns:
        Tuple[np.ndarray, np.ndarray]: matrix with each row corresponding to
        the inferred probability over states of the hidden system, conditioned
        on all future observations, as well as a matrix of predictions of
        probabilities
    """
    back_track = np.zeros((len(observations), trans_matrix.shape[0]))
    pred_track = np.zeros((len(observations), trans_matrix.shape[0]))
    back = np.ones(trans_matrix.shape[0]) / trans_matrix.shape[0]

    for i, obs in enumerate(observations[::-1]):
        back, pred = _forward_filter(
            obs, trans_matrix.T, obs_matrix.T, back
        )
        back_track[i, :] = back
        pred_track[i, :] = pred

    return np.flipud(back_track), np.flipud(pred_track)


# @numba.jit(nopython=True)
def _forward_filter(
    obs: int, trans_matrix: np.ndarray, obs_matrix: np.ndarray,
    fwd_est: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Implementation of single-step of the Bayesian filter equations, used
    to prtoduce a recursive/running estimate of the hidden state
    probability, conditioned on the entire previous history of observations

    Args:
        obs (int): observation at time t
        A (np.ndarray): transition probability matrix
        B (np.ndarray): observation probability matrix
        bayes_ (np.ndarray): Bayesian filtered estimate

    Returns:
        Tuple[np.ndarray, np.ndarray]: bayesian filter estimate, prediction
    """
    fwd_est = trans_matrix @ fwd_est
    pred = fwd_est.copy()
    fwd_est = obs_matrix[obs, :] * fwd_est
    fwd_est /= np.sum(fwd_est)
    return fwd_est, pred


# @numba.jit(nopython=True)
def _backward_filter(
    obs: int, trans_matrix: np.ndarray, obs_matrix: np.ndarray,
    back_est: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Implementation of single-step of the Bayesian backward filter equations,
    used to produce a recursive/running estimate of the hidden state
    probability, conditioned on the entire future history of observations

    Args:
        obs (int): observation at time t
        A (np.ndarray): transition probability matrix
        B (np.ndarray): observation probability matrix
        back_est_ (np.ndarray): Bayesian filtered backwards estimate

    Returns:
        Tuple[np.ndarray, np.ndarray]: bayesian filter estimate, prediction
    """
    back_est = trans_matrix.T @ back_est
    pred = back_est.copy()
    back_est = obs_matrix[:, obs] * back_est
    back_est /= np.sum(back_est)
    return back_est, pred


# @numba.jit(nopython=True)
def alpha_prob(
    obs_ts: np.ndarray, trans_matrix: np.ndarray, obs_matrix: np.ndarray,
    norm: Optional[bool] = False
) -> np.ndarray:
    """routine to calculate the alpha function P(y^t | x_t ) for all obserations

    Args:
        obs_ts (np.ndarray): sequence of observations
        trans_matrix (np.ndarray): transition rate matrix
        obs_matrix (np.ndarray): observation/emission probability matrix

    Returns:
        np.ndarray: Matrix with each column corresponding to the alpha value
        for each state, and each row representing the estimate at each point
        in time
    """
    alpha_tracker = np.zeros((len(obs_ts), trans_matrix.shape[0]))
    alpha = obs_matrix[:, obs_ts[0]].copy()
    alpha_tracker[0, :] = alpha
    for i, obs in enumerate(obs_ts[1:]):
        alpha = (trans_matrix @ alpha) * obs_matrix[obs, :]
        alpha_tracker[i + 1, :] = alpha
    if norm:
        alpha_tracker = (
            alpha_tracker / np.repeat(
                alpha_tracker
                .sum(axis=1)
                .reshape(-1, 1),
                alpha_tracker.shape[1]
            ).reshape(alpha_tracker.shape)
        )

    return alpha_tracker


# @numba.jit(nopython=True)
def beta_prob(
    obs_ts: np.ndarray, trans_matrix: np.ndarray, obs_matrix: np.ndarray,
    norm: Optional[bool] = False
) -> np.ndarray:
    """routine to calculate the beta function P(y^[t: T] | x_t ) for all obserations

    Args:
        obs_ts (np.ndarray): sequence of observations
        trans_matrix (np.ndarray): transition rate matrix
        obs_matrix (np.ndarray): observation/emission probability matrix

    Returns:
        np.ndarray: Matrix with each column corresponding to the beta value
        for each state, and each row representing the estimate at each point
        in time
    """
    beta_tracker = np.zeros((len(obs_ts), trans_matrix.shape[0]))
    beta = np.ones(trans_matrix.shape[0])
    beta_tracker[0, :] += beta
    for i, obs in enumerate(obs_ts[-1:0:-1]):
        beta = trans_matrix.T @ (beta * obs_matrix[:, obs])
        beta_tracker[i + 1, :] = beta
    if norm:
        beta_tracker = (
            beta_tracker / np.repeat(
                beta_tracker
                .sum(axis=1)
                .reshape(-1, 1),
                beta_tracker.shape[1]
            ).reshape(beta_tracker.shape)
        )
    return beta_tracker[::-1]


if __name__ == "__main__":
    # test out the routines to make sure they all match
    from hidden_py.infer import MarkovInfer
    from hidden_py.dynamics import HMM
    import time

    hmm = HMM(2, 2)
    hmm.init_uniform_cycle()

    hmm.run_dynamics(5)
    obs_ts, state_ts = hmm.get_obs_ts(), hmm.get_state_ts()

    A_perturb = np.array([
        [-0.05, 0.04],
        [0.05, -0.04]
    ])

    A_sample = hmm.A + A_perturb
    B_sample = hmm.B + A_perturb

    analyzer = MarkovInfer(2, 2)

    start_1 = time.time()
    est_bayes_1 = bayes_estimate(np.array(obs_ts), A_sample, B_sample)
    end_1 = time.time()

    start_2 = time.time()
    est_bayes_1 = bayes_estimate(np.array(obs_ts), A_sample, B_sample)
    end_2 = time.time()

    print("----- Timer results -----")
    print(f"First iterations  : {end_1 - start_1}")
    print(f"Second iterations : {end_2 - start_2}")

    print("\n\t-- DONE --\n")
