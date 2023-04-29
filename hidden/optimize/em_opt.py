


# Code dumpt from initial attempts at EM integration

    # Inferrence routines
def expectation(
    self, obs_ts: Iterable, A_est: np.ndarray, B_est: np.ndarray
) -> ExpectationResult:
    # Prediction trackers need to be true for likelihood calcualtion
    self.forward_algo(obs_ts, A_est, B_est, prediction_tracker=True)
    self.backward_algo(obs_ts, A_est, B_est, prediction_tracker=True)
    self.bayesian_smooth(A_est)

    return ExpectationResult(
        self.bayes_smoother, self.forward_tracker, self.backward_tracker,
        A_est, B_est
    )

def maximization(
    self, exp_result: ExpectationResult, obs_ts: Iterable
) -> MaximizationResult:
    # Calc xi term from expectation result
    xi = self._calc_xi_term(exp_result, obs_ts)
    # Update matrices
    return self._update_matrices(exp_result, xi, obs_ts)

def _calc_xi_term(
    self, exp: ExpectationResult, obs_ts: np.ndarray
) -> np.ndarray:
    xi = np.zeros((exp.dim, exp.dim, len(exp.gamma) - 1))
    normalization = np.zeros(xi.shape[2])
    for i in range(exp.dim):
        for j in range(exp.dim):
            xi[i, j, :] = exp.alpha_k(j)[:-1] * exp.A[i, j] * exp.beta_k(i)[1:] * exp.B[i, obs_ts[1:]]
            normalization += xi[i, j, :]

    return xi / normalization

def _update_matrices(
    self, exp: ExpectationResult, xi: np.ndarray, obs_ts: Iterable
) -> MaximizationResult:
    A_new = np.zeros_like(exp.A)
    B_new = np.zeros_like(exp.B)

    for i in range(exp.dim):
        for j in range(exp.dim):
            A_new[i, j] = np.sum(xi[i, j, :]) / np.sum(exp.gamma_k(i)[:-1])
            B_new[i, j] = np.sum(exp.gamma_k(i)[np.array(obs_ts) == j]) / np.sum(exp.gamma_k(i))

    # normalize results to enforce probability conservation
    for col in range(A_new.shape[1]):
        A_new[:, col] = A_new[:, col] / np.sum(A_new[:, col])
        B_new[:, col] = B_new[:, col] / np.sum(B_new[:, col])

    # return MaximizationResult(A_new, B_new, exp.A[:, :], exp.B[:, :])

# ANCHOR READY FOR TESTING
def baum_welch(
    self, param_init: Iterable, obs_ts: Iterable,
    maxiter: Optional[int] = 100, tolerance: Optional[float] = 1e-8
) -> Iterable:
    # Iterate through steps of self.expectation, self.maximization
    opt_param = param_init
    A_est, B_est = self._extract_hmm_parameters(opt_param)
    param_tracker = []

    # TODO Add in tolerance checks based on parameter updates (we have no measure fo CDLL, right?)
    for iteration in range(maxiter):
        exp = self.expectation(obs_ts, A_est, B_est)
        maxim = self.maximization(exp, obs_ts)
        A_est, B_est = maxim.A, maxim.B
        param_tracker.append({
            'iteration': iteration,
            'A_est': A_est[:, :],
            'B_est': B_est[:, :]
        })

    # return BaumWelchOptimizationResult(maxim, param_tracker, iteration)

def _update_hidden_estimate(self, A_est: np.ndarray, trans_rate: float):
    A_est = np.diag([trans_rate] * self.n_sys)
    A_est += np.fliplr(np.diag([1 - trans_rate] * self.n_sys))
    return A_est


def likelihood_alpha_beta(self):
    if self.alpha_tracker is None:
        raise ValueError(
            "Must run alpha(...) before calculating liklihood..."
        )

    if self.beta_tracker is None:
        raise ValueError(
            "Must run beta(...) before calculating likelihood..."
        )

    self.likelihood_tracker_ab = []
    for a, b in zip(self.alpha_tracker, self.beta_tracker):
        self.likelihood_tracker_ab.append(np.sum(a * b))


def bayesian_smooth_alpha_beta(self):
    if self.beta_tracker is None or self.alpha_tracker is None or self.likelihood_tracker_ab is None:
        raise ValueError(
            "Must run alpha(...), beta(...), and "
            "likelihood_alpha_beta(...) before calcualting smoothed "
            "estimate..."
        )
    self.bayes_alpha_beta = []
    for a, b, l in zip(self.alpha_tracker, self.beta_tracker, self.likelihood_tracker_ab):
        self.bayes_alpha_beta.append(a * b / l)

