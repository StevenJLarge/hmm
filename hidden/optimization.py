# Routines for running optimizations in likelihood calcualtions
import os
from typing import Iterable, Tuple, Optional
import numpy as np
import numba
import scipy.optimize
from hidden.infer import LikelihoodOptResult


# @numba.jit(npython=True)
def _forward_algo(
    obs_ts: np.ndarray, A: np.ndarray, B: np.ndarray
) -> np.ndarray:
    predictions = np.zeros((2, len(obs_ts)))
    bayes_filter = np.array([0.5, 0.5])

    for i, obs in enumerate(obs_ts):
        bayes_filter = np.matmul(A, bayes_filter)
        predictions[:, i] = bayes_filter
        bayes_filter = B[:, obs] * bayes_filter
        bayes_filter /= np.sum(bayes_filter)

    return predictions

def _parse_hmm_parameters(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = theta[0] * np.ones((2, 2))
    B = theta[1] * np.ones((2, 2))

    A[0, 0], A[1, 1] = 1 - theta[0], 1 - theta[0]
    B[0, 0], B[1, 1] = 1 - theta[1], 1 - theta[1]

    return A, B


# @numba.jit(nopython=True)
def calc_likelihood_optimizer(
    param_arr: np.ndarray, obs_ts: np.ndarray
) -> float:

    A, B = _parse_hmm_parameters(param_arr)
    predictions = _forward_algo(obs_ts, A, B)

    likelihood = 0
    for i, obs in enumerate(obs_ts):
        # NOTE Error in shape mismatches here
        inner = predictions[:, i].reshape(1, 2) @ B[:, obs]
        likelihood -= np.log(inner)
    return likelihood


def optimize_likelihood_local(
    param_init: np.ndarray, obs_ts: np.ndarray, method: str
):
    bnds = np.ndarray([0, 1] * len(param_init))
    res = scipy.optimize.minimize(
        calc_likelihood_optimizer, param_init,
        args=(obs_ts), method=method, bounds=bnds
    )

    return LikelihoodOptResult(res, 'local', method=method)


def optimize_likelihood_global(
    param_init: np.ndarray, obs_ts: np.ndarray,
    sampling_method: Optional[str] = 'sobol'
):

    bnds = np.ndarray([0, 1] * len(param_init))
    res = scipy.optimize.shgo(
        calc_likelihood_optimizer,
        bounds=bnds,
        args=(obs_ts),
        sampling_method=sampling_method
    )

    return LikelihoodOptResult(res, "global", sampling_method=sampling_method)


if __name__ == "__main__":
    import datetime
    from hidden.dynamics import HMM

    a = 0.3
    b = 0.1

    A = np.array([[1 - a, a],[a, 1 - a]])
    B = np.array([[1 - b, b],[b, 1 - b]])

    hmm = HMM(2, 2)
    hmm.A = A
    hmm.B = B
    hmm.run_dynamics(1000)

    obs_ts = np.array(hmm.get_obs_ts())
    param_init = np.array([0.4, 0.4])
    _ = _forward_algo(obs_ts, A, B)

    print("Local likelihood...")
    start = datetime.datetime.now()
    res_loc = optimize_likelihood_local(param_init, obs_ts, method="SLSQP")
    end = datetime.datetime.now()
    print(f"Local time : {end - start}\n")

    print("Global likelihood...")
    start = datetime.datetime.now()
    res_glob = optimize_likelihood_global(param_init, obs_ts)
    end = datetime.datetime.now()
    print(f"Global time : {end - start}\n")
