# Routines for running optimizations in likelihood calcualtions
import os
from typing import Iterable, Tuple
import numpy as np
import numba
import scipy.optimize


def _forward_algo(
    obs_ts: np.ndarray, A: np.ndarray, B: np.ndarray
) -> np.ndarray:
    pass


def _parse_hmm_parameters(param_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pass


@numba.jit(nopython=True)
def calc_likelihood_optimizer(
    param_arr: np.ndarray, obs_ts: np.ndarray
) -> float:

    A, B = _parse_hmm_parameters(param_arr)
    predictions = _forward_algo(obs_ts, A, B, prediction_tracker=True)

    likelihood = 0
    for bayes, obs in zip(predictions, obs_ts):
        inner = bayes @ B[:, obs]
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


