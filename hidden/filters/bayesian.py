# Bayesian filters -- Fowrward, backward, bayesian
from typing import Iterable, Tuple
import numpy as np
import numba


# Currently I cant numba-optimize the vectorized version of this function (the
# inner list comprehension), as the numpy repeat funtion apparently only works
# within numba if it is acting on contiguous memory, and for some reason the
# _bayes vector is non-contiguous... Currently this will just throw a numba
# performance warning, but the runtime is significatly faster as is, than using
# the vectorized code with no numba.
@numba.jit(nopython=True)
def bayes_estimate(
    obs_ts: np.ndarray, trans_matrix: np.ndarray, obs_matrix: np.ndarray
) -> np.ndarray:
    # For this we dont make use of the predictions
    fwd_tracker, _ = forward_algo(obs_ts, trans_matrix, obs_matrix)
    N = trans_matrix.shape[0]
    bayes_smooth = np.zeros_like(fwd_tracker)
    bayes_smooth[-1, :] = fwd_tracker[-1, :]
    # This makes the `_trans_matrix` contiguous in memory, which is most
    # efficient for numba, especially '@' apparently...
    _trans_matrix = trans_matrix.T.copy()
    for i, (filt, _bayes) in enumerate(
        zip(np.ascontiguousarray(fwd_tracker[-2::-1, :]),
            np.ascontiguousarray(bayes_smooth[:0:-1, :])
    )):
        pred = _trans_matrix @ filt
        summand = np.array(
            [np.sum(_bayes * trans_matrix[:, j] / pred) for j in range(N)]
        )
        bayes_smooth[-(i + 2), :] = filt * summand

    return bayes_smooth


@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
def backward_algo(
    observations: np.ndarray, trans_matrix: np.ndarray, obs_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    back_track = np.zeros((len(observations), trans_matrix.shape[0]))
    pred_track = np.zeros_like(back_track)
    back = np.ones(trans_matrix.shape[0]) / trans_matrix.shape[0]

    for i, obs in enumerate(observations[::-1]):
        back, pred = _backward_filter(
            obs, trans_matrix.T, obs_matrix.T, back
        )
        back_track[i, :] = back
        pred_track[i, :] = pred

    return np.flipud(back_track), np.flipud(pred_track)


@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
def _backward_filter(
    obs: int, trans_matrix: np.ndarray, obs_matrix: np.ndarray,
    back_est: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    back_est = trans_matrix.T @ back_est
    pred = back_est.copy()
    back_est = obs_matrix[:, obs] * back_est
    back_est /= np.sum(back_est)
    return back_est, pred


@numba.jit(nopython=True)
def alpha_prob(
    obs_ts: np.ndarray, trans_matrix: np.ndarray, obs_matrix: np.ndarray
) -> np.ndarray:
    alpha_tracker = np.zeros((len(obs_ts), trans_matrix.shape[0]))
    alpha = obs_matrix[:, obs_ts[0]].copy()
    alpha_tracker[0, :] = alpha
    for i, obs in enumerate(obs_ts[1:]):
        alpha = (trans_matrix @ alpha) * obs_matrix[:, obs]
        alpha_tracker[i + 1, :] = alpha
    return alpha_tracker


@numba.jit(nopython=True)
def beta_prob(
    obs_ts: np.ndarray, trans_matrix: np.ndarray, obs_matrix: np.ndarray
) -> np.ndarray:
    beta_tracker = np.zeros((len(obs_ts), trans_matrix.shape[0]))
    beta = np.ones(2)
    beta_tracker[0, :] += beta
    for i, obs in enumerate(obs_ts[-1:0:-1]):
        beta = trans_matrix.T @ (beta * obs_matrix[:, obs])
        beta_tracker[i + 1, :] = beta
    return beta_tracker[::-1]


if __name__ == "__main__":
    # test out the routines to make sure they all match
    from hidden.infer import MarkovInfer
    from hidden.dynamics import HMM
    import time

    hmm = HMM(2, 2)
    hmm.init_uniform_cycle()

    hmm.run_dynamics(500)
    obs_ts, state_ts = hmm.get_obs_ts(), hmm.get_state_ts()

    analyzer = MarkovInfer(2, 2)

    # Current implementations
    analyzer.forward_algo(obs_ts, hmm.A, hmm.B, prediction_tracker=True)
    analyzer.backward_algo(obs_ts, hmm.A, hmm.B, prediction_tracker=True)
    analyzer.bayesian_smooth(hmm.A)
    analyzer.alpha(hmm.A, hmm.B, obs_ts)
    analyzer.beta(hmm.A, hmm.B, obs_ts)

    fwd_tracker = analyzer.forward_tracker
    bck_tracker = analyzer.backward_tracker
    bayes_tracker = analyzer.bayes_smoother
    alpha = analyzer.alpha_tracker
    beta = analyzer.beta_tracker

    # Filter file implementations
    start_1 = time.time()
    fwd_tracker_alt, _ = forward_algo(np.array(obs_ts), hmm.A, hmm.B)
    bck_tracker_alt, _ = backward_algo(np.array(obs_ts), hmm.A, hmm.B)
    bayes_tracker_alt = bayes_estimate(np.array(obs_ts), hmm.A, hmm.B)
    alpha_alt = alpha_prob(np.array(obs_ts), hmm.A, hmm.B)
    beta_alt = beta_prob(np.array(obs_ts), hmm.A, hmm.B)
    end_1 = time.time()

    # Second repetition is much faster

    start_2 = time.time()
    fwd_tracker_alt, _ = forward_algo(np.array(obs_ts), hmm.A, hmm.B)
    bck_tracker_alt, _ = backward_algo(np.array(obs_ts), hmm.A, hmm.B)
    bayes_tracker_alt = bayes_estimate(np.array(obs_ts), hmm.A, hmm.B)
    alpha_alt = alpha_prob(np.array(obs_ts), hmm.A, hmm.B)
    beta_alt = beta_prob(np.array(obs_ts), hmm.A, hmm.B)
    end_2 = time.time()

    print("----- Timer results -----")
    print(f"First iterations  : {end_1 - start_1}")
    print(f"Second iterations : {end_2 - start_2}")

    print("\n\t-- DONE --\n")
