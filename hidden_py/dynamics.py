import numpy as np
from typing import Iterable, Optional, List


class HMM:
    def __init__(self, dim_sys: int, dim_obs: int) -> None:
        """
        Initialize the dynamics model.

        Args:
            dim_sys (int): The dimensionality of the system state.
            dim_obs (int): The dimensionality of the observed state.

        Returns:
            None
        """
        self.n_sys = dim_sys
        self.n_obs = dim_obs

        self.A = np.zeros((self.n_sys, self.n_sys))
        self.B = np.zeros((self.n_obs, self.n_sys))

        self.state_tracker = None
        self.obs_tracker = None

    def _initialize_tracking(self) -> None:
        """
        Initializes the state and observation trackers.
        """
        self.state_tracker = []
        self.obs_tracker = []

    def initialize_dynamics(self, A: np.ndarray, B: np.ndarray) -> None:
        """
        Initializes the dynamics of the hidden Markov model.

        Args:
            A (np.ndarray): The transition matrix of the HMM.
            B (np.ndarray): The emission/observation matrix of the HMM.

        Returns:
            None
        """
        self.A = A
        self.B = B

    def init_uniform_cycle(
        self, trans_rate: Optional[float] = 0.3,
        error_rate: Optional[float] = 0.1
    ) -> None:
        """
        Initializes the hidden Markov model with a uniform cycle structure.

        Args:
            trans_rate (float, optional): Transition rate between states. Defaults to 0.3.
            error_rate (float, optional): Error rate for observations. Defaults to 0.1.

        Raises:
            NotImplementedError: Raised when the number of system states is not
                equal to the number of observation states.

        Returns:
            None
        """
        if self.n_sys != self.n_obs:
            raise NotImplementedError(
                "Currently, cycle default matrices can only be instantiated "
                "when `n_sys` = `n_obs`"
            )
        self.A = self.init_cycle_matrix(self.n_sys, trans_rate)
        self.B = self.init_cycle_matrix(self.n_obs, error_rate)

    def _init_cycle_matrix(self, size: int, rate: float) -> np.ndarray:
        """
        Helper method to initialize a matrix with a uniform cycle structure.

        Args:
            size (int): The size of the matrix.
            rate (float): The rate to be used in the matrix.

        Returns:
            np.ndarray: The initialized matrix.
        """
        matrix = np.zeros((size, size))

        for i in range(size):
            for j in range(i + 1, size):
                matrix[i, j] = (lambda x: rate if x == 1 else 0)(np.abs(i - j))
                matrix[j, i] = matrix[i, j]
            if i == 0:
                matrix[size - 1, i] = rate
            if i == size - 1:
                matrix[0, i] = rate

            matrix[i, i] = 1 - np.sum(matrix[:, i])

        return matrix

    def _validate_dynamics_matrices(self) -> None:
        """
        Validates the dynamics matrices A and B.

        Raises:
            ValueError: If the transition matrix A or the observation matrix B is invalid.
        """
        A_condition = self.A.sum(axis=0)
        B_condition = self.B.sum(axis=0)

        if not (A_condition == 1).all():
            raise ValueError(
                f"Invalid transition matrix A: {self.A}"
            )
        if not (B_condition == 1).all():
            raise ValueError(
                f"Invalid observation matrix B: {self.B}"
            )

    # Dynamics routines
    def run_dynamics(
        self, n_steps: Optional[int] = 100, init_state: Optional[int] = None
    ) -> None:
        """
        Runs the dynamics of the hidden Markov model for a specified number of steps.

        Args:
            n_steps (Optional[int]): The number of steps to run the dynamics for. Default is 100.
            init_state (Optional[int]): The initial state of the system. If None, a random state is chosen.
                Default is None.

        Returns:
            None
        """

        self._validate_dynamics_matrices()

        if init_state is None:
            self._set_state(np.random.randint(0, self.n_sys))
        else:
            self._set_state(init_state)

        self._set_obs(np.argmax(self.state))

        if self.state_tracker is None or self.obs_tracker is None:
            self._initialize_tracking()

        for _ in range(n_steps):
            self.state_tracker.append(self.state)
            self.obs_tracker.append(self.obs)
            self.step_dynamics()

    def step_dynamics(self) -> None:
        """
        Perform a step of the kinetic Monte Carlo dynamics.

        This method updates the system state and observation based on transition probabilities.

        Returns:
            None
        """
        w_sys = np.random.uniform()
        w_obs = np.random.uniform()

        trans_prob = self.A[:, np.argmax(self.state)]
        new_sys = np.argmax(w_sys < np.cumsum(trans_prob))
        self._set_state(new_sys)

        obs_prob = self.B[:, np.argmax(self.state)]
        new_obs = np.argmax(w_obs < np.cumsum(obs_prob))
        self._set_obs(new_obs)

    def _set_state(self, new_state: int) -> None:
        """
        Set the state of the system to the specified new_state.

        Args:
            new_state (int): The new state to set.

        Raises:
            IndexError: If the new_state is out of bounds for the system dimension.
        """
        if new_state < 0 or new_state > self.n_sys:
            raise IndexError(
                f"New state `{new_state}` out of bounds for system of "
                f"dimension `{self.n_sys}`"
            )
        self.state = np.zeros(self.n_sys)
        self.state[new_state] = 1

    def _set_obs(self, new_obs: int) -> None:
        """
        Set the observation value for the hidden system.

        Args:
            new_obs (int): The new observation value.

        Raises:
            IndexError: If the new observation is out of bounds for the system.

        Returns:
            None
        """
        if new_obs < 0 or new_obs > self.n_obs:
            raise IndexError(
                f"New observation `{new_obs}` out of bounds for system with "
                f"dimension `{self.n_obs}`"
            )
        self.obs = np.zeros(self.n_obs)
        self.obs[new_obs] = 1

    @property
    def state_ts(self) -> List[int]:
        """
        Returns a list of the most probable states at each time step.

        Returns:
            List[int]: A list of integers representing the most probable states at each time step.
        """
        return [np.argmax(s) for s in self.state_tracker]

    @property
    def obs_ts(self) -> List[int]:
        """
        Returns a list of the most likely observations at each time step.

        Returns:
            List[int]: A list of integers representing the most likely observations at each time step.
        """
        return [np.argmax(o) for o in self.obs_tracker]

    @property
    def current_state(self) -> np.array:
        """
        Returns the current state of the dynamics object.

        Returns:
            np.array: The current state.
        """
        return self.state

    @property
    def current_obs(self) -> np.array:
        """
        Returns the current observation.

        Returns:
            np.array: The current observation.
        """
        return self.obs

    @property
    def steady_state(self) -> np.array:
        """
        Calculates the steady state vector of the dynamics matrix.

        Returns:
            np.array: The steady state vector.
        """
        e_vals, e_vecs = np.linalg.eig(self.A)
        ss_idx = np.argmin(np.abs(1 - e_vals))
        return e_vecs[:, ss_idx] / np.sum(e_vecs[:, ss_idx])


def plot_traj(state: Iterable, observation: Iterable) -> None:
    """
    Plots the trajectory of the state and observation over time.

    Args:
        state (Iterable): The state values over time.
        observation (Iterable): The observation values over time.

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='darkgrid')
    Pal = sns.color_palette('hls', 2)

    _, ax = plt.subplots(1, 1, figsize=(6, 3.5))

    ax.plot(state, 'o', markersize=7, color=Pal[0], label='State')
    ax.plot(observation, 'o', markersize=4, color=Pal[1], label='Observation')
    ax.set_xlabel(r"Time", fontsize=15)
    ax.set_ylabel(r"State", fontsize=15)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":

    # Setup
    obj = HMM(3, 3)
    obj.init_uniform_cycle(0.1)

    # Generate synthetic dynamics:
    obj.run_dynamics(n_steps=25)
    state = obj.state_ts
    obs = obj.obs_ts

    plot_traj(state, obs)

