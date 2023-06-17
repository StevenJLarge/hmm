# **Hidden**

#### A Hidden Markov Model package for python

---

### Installation

To install this package simply run the command:

`pip install hidden`

<br />

---

## Overview

This package contains logic for inferring, simulating, and fitting Hidden Markov Models with discrete states and obserfvations. In contrast to more common HMM packages that deal primarily with _mixture models_, output symbols are continuous, and drawn from a distriubtion of values that is somehow conditional on the hidden state.

Here, we have thusfar considered primarily scenarios where the observation value is an integer, and one of the possible hidden states. There are three major use-cases for this codebase: dynamics/simulation, system identification/parameter fitting, and signal processing. In all cases, these functionalities are outlined in several tutorial notebooks in the `notebooks/tutorials` location of the [github repository](https://github.com/StevenJLarge/hmm)

<br />

### Dynamics/Simulation

The `hidden.dynamics` submodule contains the code necessary to simulate the hidden state and observation dynamics as specified by a state transition matrix $A$ (with elements that quantify the rate of transitions between states) and an observation matrix $B$, with elements that quantify the probability of observing a given output symbol, given the current hidden state.

For instance, the code necessary to initialize a hidden Markov model, run the dyanamics, and extract the observation and state time-series

```
from hidden import dynamics

# 2 hidden states, 2 possible observation values
hmm = dynamics.HMM(2, 2)

# Helper routine to initialize A and B matrices
hmm.init_uniform_cycle()

# Run dynamics for 100 time steps
hmm.run_dynamics(100)

# Pull out the observations, and true (simulated) hidden state values
observations = hmm.get_obs_ts()
hidden_states = hmm.get_state_ts()
```

<br />

---

### System Identification

The `infer` submodule contains the code that wraps lower-level functionality (primarily in the `filters/bayesian.py` and `optimize/optimization.py` files) for both signal processing and system identification/parameter fitting.

There are two separate means of performing system identification: Local/Global partial likelihood optimization, and complete-data likelihood optimization. While more comprehensive details are contained in the github notebooks, broadly speaking partial data likelihood optimization performs relatively standard optimizations on the likelihood function $\mathcal{L}(\theta | Y)$ which considers only the observations as the _data_. Effectively, these optimizers wrap the `scipy.opt.minimize` functions by encoding and decoding the $A$ and $B$ matrices into a parameter vector (ensuring that their column-normalization is preserved) and calculating the negative log-likelihood of a particular parameter vector. In practice, given a set of observations, we can initialize an `analyzer` and run either local (using, by default, the `scipy.opt.minimize` function with the `L-BFGS-B` algorithm) or global (using the `scipy` SHGO algorithm) as:

```
from hidden import infer

# Input the dimensions of the HMM and observations
analyzer = infer.MarkovInfer(2, 2)

# Initial estimates of the A and B matrices
A_est = np.array([
    [0.8, 0.1],
    [0.2, 0.9]
])

B_est = np.array([
    [0.9, 0.05],
    [0.1, 0.95]
])

# Run local partial-data likelihood optimization (default behaviour), the symmetric keyword can be used to specify whether or not the A and B matrices are assumed to be symmetric
opt_local = analyzer.optimize(observations, A_est, B_est, symmetric=False)

# And the partial-data global likelihood optimization, the A and B initial matrices are not used in the optimizer, aside from providing a way of specifying the dimension of the parameter vectors
opt_global = analyzer.optimize(observations, A_est, B_est, symmetric=False, opt_type=OptClass.Global)

```

Now, for the complete-data likeihood optimization, the interface is very similar, but behind the scenes the code will implement an implementation of the Baum-Welch reparameterization algorithm (an instance of an Expectation-Maximization algorithm) to find the optimal parameter values. In practice, this can be done as:

```
from hidden import infer
from hidden.optimize.base import OptClass

analyzer = infer.MarkovInfer(2, 2)

res_bw = analyzer.optimize(observations, A_est, B_est, opt_type=OptClass.ExpMax)

```

In all cases, there is also an option to add algorithm options to customize the specifics of the optinization. Most relevant is for the expectation-maximization where you can specify a maximum number of iterations for the algorithm, as well as a threshold on the size of parameter changes (quantified by the matrix norm of pre- and post-update $A$ and $B$ matrices). This can be accessed through the `algo_opts` dictionary. For instance, if we wanted to change the maximum iterations in the BW algorithm to 1000 and set the termination threshold at `1e-10`, we would perform the previous call as

```
options = {
    'maxiter': 1000,
    "threshold": 1e-10
}

res_bw = analyzer.optimize(observations, A_est, B_est, opt_type=OptClass.ExpMax, algo_opts=options)
```

In essence, this set of tools allows you to infer the best model given a set of observed data. Under the hood, many of the tools from the signal processing module are used, but the `analyzer.optimize(...)` function calls largely hide that complexity.

<br />

---

### Signal Processing

<br />

---

## Roadmap
