# File to contain the class definitions and routines for inferring the
# properties of the HMM
from typing import Iterable, Optional, Dict
import numpy as np
from pandas import DataFrame, Series
from hidden_py.optimize.registry import OPTIMIZER_REGISTRY
from hidden_py.optimize.base import OptClass
from hidden_py.optimize.results import OptimizationResult
from hidden_py.filters import bayesian


class MarkovInfer:
    # Type hints for instance variables
    forward_tracker: Iterable
    backward_tracker: Iterable
    predictions: Iterable
    predictions_back: Iterable
    bayes_smooth: Iterable
    alpha_tracker: Iterable
    beta_tracker: Iterable
    n_sys: int
    n_obs: int
 
    def __init__(self, dim_sys: int, dim_obs: int) -> None:
        """Constructor for MarkovInfer class, this generally acts as an
        interface/wrapper for the optimization and filtering routines

        Args:
            dim_sys (int): Dimension of the number of states in a hidden system
            dim_obs (int): Number of possible observations
        """
        # Tracker lists for forward and backward estimates
        self.forward_tracker = None
        self.backward_tracker = None
        self.predictions = None
        self.predictions_back = None
        self.bayes_smooth = None
        self.alpha_tracker = None
        self.beta_tracker = None

        # Dimension of target system and observation vector
        self.n_sys = dim_sys
        self.n_obs = dim_obs

    @staticmethod
    def _validate_input(obs_ts: Iterable) -> np.ndarray:
        """Routine to validate input for observation timeseries. This will cast
        lists, and pandas Series/DataFrme to a 1-d numpy array, for use in the
        lower-level (numba-optimized) filter routines

        Args:
            obs_ts (Iterable): timeseries of observations

        Raises:
            ValueError: Observations (numpy) cannot be interpreted as 1D
            ValueError: Observations (pandas) cannot be interpreted as 1D
            NotImplementedError: Input data type is not supported

        Returns:
            np.ndarray: npmy-converted observation
        """
        # We want this to support input lists as well as pandas Series
        if isinstance(obs_ts, np.ndarray):
            if 1 in obs_ts.shape or len(obs_ts.shape) == 1:
                return obs_ts.flatten()
            else:
                raise ValueError("Input observations must be 1-D...")

        if isinstance(obs_ts, list):
            return np.array(obs_ts)
        if isinstance(obs_ts, (Series, DataFrame)):
            if 1 in obs_ts.shape or len(obs_ts.shape) == 1:
                return obs_ts.to_numpy().flatten()
            else:
                raise ValueError("Input observations must be 1-D...")

        else:
            raise NotImplementedError(
                "observation timeseries must be list or pandas Series/DataFrame"
            )

    def forward_algo(
        self,
        observations: Iterable[int],
        trans_matrix: np.ndarray,
        obs_matrix: np.ndarray,
    ) -> None:
        """Wrapper routine to implement the forward filter algorithm, and write
        results to internal forward_tracker and prediction_tracker variables

        Args:
            observations (Iterable[int]): observation timeseries
            trans_matrix (np.ndarray): transition matrix
            obs_matrix (np.ndarray): observation matrix
        """
        observations = self._validate_input(observations)
        self.forward_tracker, self.prediction_tracker = bayesian.forward_algo(
            observations, trans_matrix, obs_matrix
        )

    def backward_algo(
        self,
        observations: Iterable[int],
        trans_matrix: np.ndarray,
        obs_matrix: np.ndarray,
    ) -> None:
        """Wrapper routine to implement the backward filter algorithm, and write
        results to internal backward_tracker and prediction_back variables

        Args:
            observations (Iterable[int]): observation timeseries
            trans_matrix (np.ndarray): transition matrix
            obs_matrix (np.ndarray): observation matrix
        """
        observations = self._validate_input(observations)
        self.backward_tracker, self.predictions_back = bayesian.backward_algo(
            observations, trans_matrix, obs_matrix
        )

    def alpha(
        self, observations: np.ndarray, trans_matrix: np.ndarray,
        obs_matrix: np.ndarray, norm: Optional[bool] = False
    ) -> None:
        """Wrapper routine to interface with alpha-calcualtion routine. Sets to
        internal alpha_tracker instance variable

        Args:
            observations (np.ndarray): timeseries of observations
            trans_matrix (np.ndarray): transition matrix
            obs_matrix (np.ndarray): observation matrix
            norm (Optional[bool], optional): whether or not there we want to
                use normalized (across states) at each point in time.
                Defaults to False.
        """
        observations = self._validate_input(observations)
        self.alpha_tracker = bayesian.alpha_prob(
            observations, trans_matrix, obs_matrix, norm=norm
        )

    def beta(
        self, observations: np.ndarray, trans_matrix: np.ndarray,
        obs_matrix: np.ndarray, norm: Optional[bool] = False
    ) -> None:
        """Wrapper routine to interface with beta-calcualtion routine. Sets to
        internal beta_tracker instance variable

        Args:
            observations (np.ndarray): timeseries of observations
            trans_matrix (np.ndarray): transition matrix
            obs_matrix (np.ndarray): observation matrix
            norm (Optional[bool], optional): whether or not there we want to
                use normalized (across states) at each point in time.
                Defaults to False.
        """
        observations = self._validate_input(observations)
        self.beta_tracker = bayesian.beta_prob(
            observations, trans_matrix, obs_matrix, norm=norm
        )

    def bayesian_smooth(
        self, observations: np.ndarray, trans_matrix: np.ndarray,
        obs_matrix: np.ndarray
    ) -> None:
        """Wrapper routine to interface with bayesian smoothing routine. Sets
        to internal bayes_smooth instance variable
        
        Args:
            observations (np.ndarray): timeseries of observations
            trans_matrix (np.ndarray): transition matrix
            obs_matrix (np.ndarray): observation matrix
        """
        observations = self._validate_input(observations)
        self.bayes_smooth = bayesian.bayes_estimate(
            observations, trans_matrix, obs_matrix
        )

    def discord(self, est_1: Iterable, est_2: Iterable) -> float:
        """Calculates the discord order parameter for an HMM based on the
        disagreement between two different estimates. This is the typical
        `discord` measure for HMMs when one of the estimates is `naive` (equal
        to observation)

        Args:
            est_1 (Iterable): first estimate of hidden state
            est_2 (Iterable): second estimate of hidden state

        Returns:
            float: Discord order parameter between observations
        """
        error = [1 if f == o else -1 for f, o in zip(est_2, est_1)]
        return 1 - np.mean(error)

    def error_rate(self, pred_ts: Iterable, state_ts: Iterable) -> float:
        """Calculates the error rate of predictions (pred_ts) as compared to
        a known sequence of hidden states (state_ts)

        Args:
            pred_ts (Iterable): state prediction
            state_ts (Iterable): time series of hidden states

        Returns:
            float: error rate of predictions
        """
        return 1 - np.mean([p == s for p, s in zip(pred_ts, state_ts)])

    def optimize(
        self, observations: Iterable, trans_init: np.ndarray,
        obs_init: np.ndarray, symmetric: Optional[bool] = False,
        opt_type: Optional[OptClass] = OptClass.Local,
        algo_opts: Optional[Dict] = {}
    ) -> OptimizationResult:
        """Main entrypoint for optimizing an internal model

        Args:
            observations (Iterable): Time series of observations
            trans_init (np.ndarray): Initial guess at transition rate matrix
            obs_init (np.ndarray): Initial guess at observation matrix
            symmetric (Optional[bool], optional): Flag as to whether or not the
                model is assumed to be symmetric. Only has an impact if Local
                or Global likelihood optimizer is used. Defaults to False.
            opt_type (Optional[OptClass], optional): Optimizer class, must be
                one of the classes contained in the MODEL_REGISTRY.
                Defaults to OptClass.Local.
            algo_opts (Optional[Dict], optional): Additional configuration
                options for the optimization algorithm. Defaults to {}.

        Raises:
            ValueError: If the provided optimization class is not registerd, or
                valid.

        Returns:
            OptimizationResult: Result of optimization
        """
        if not isinstance(opt_type, OptClass):
            raise ValueError(
                'Invalid `opt_class`, must be a member of OptClass enum...'
            )
        observations = self._validate_input(observations)
        # For the global optimizer, dim_tuple, but no initial guesses
        # TODO -- Add verbose option to suppress output
        optimizer = OPTIMIZER_REGISTRY[opt_type](**algo_opts)
        if (opt_type is OptClass.Global):
            print("Running global partial-data likelihood optimization...")
            dim_tuple = (trans_init.shape, obs_init.shape)
            return optimizer.optimize(observations, dim_tuple, symmetric)

        # For EM opt, there is no option to input a symmetric constraint
        elif (opt_type is OptClass.ExpMax):
            print("Running Baum-Welch (EM) optimization...")
            return optimizer.optimize(observations, trans_init, obs_init)

        print("Running local partial-data likelihood optimization...")
        return optimizer.optimize(observations, trans_init, obs_init, symmetric)


if __name__ == "__main__":
    import hidden_py as hp
    import pandas as pd
    from pathlib import Path
    import pickle

    analyzer = MarkovInfer(3, 3)

    A3_init = np.array([
        [0.70, 0.2,0.75],
        [0.20, 0.7,0.15],
        [0.10,0.1,0.1]
    ])

    B3_init = np.array([
        [0.85, 0.10,0.10],
        [0.05, 0.80,0.20],
        [0.10,0.10,0.70]
    ])

    proj_dir = Path(__file__).parents[1]
    read_dir = proj_dir / "data" / "triple_barrier_dt_21_nsig_0.5"
    with open(read_dir / "triple_barrier.pkl", "rb") as f:
        res = pickle.load(f)

    data_ind = res._agg_results['dev_equity']._results["fixed_time"]['canada_equity'].indicator.dropna().to_numpy() + 1

    res_ca_eq = {}
    WIN_SIZE = 252

    for i, idx in enumerate(range(0, len(data_ind[:-WIN_SIZE]), 63)):
        print(f"idx = {idx}")
        if i % 10 == 0:
            print(f"Working on idx {i} (of {len(data_ind[:-WIN_SIZE:63])})")
        _res = analyzer.optimize(data_ind[idx: idx + WIN_SIZE], A3_init, B3_init, opt_type=hp.OptClass.ExpMax)
        res_ca_eq[str(idx)] = _res

    # opt_res = analyzer.optimize(data_ind[:252], A3_init, B3_init, opt_type=OptClass.ExpMax)

    # read_dir = Path(__file__).parents[1] / "data"
    # df_spx = pd.read_csv(read_dir / "df_spx.csv", index_col=0)

    # res = analyzer.optimize(
    #     df_spx.loc[:252].to_numpy() + 1, A3_init, B3_init,
    #     opt_type=hp.OptClass.ExpMax
    # )

    print("--DONE--")
