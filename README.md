# SSM: Bayesian learning and inference for state space models

[![Tests](https://github.com/lindermanlab/ssm-jax-refactor/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/lindermanlab/ssm-jax-refactor/actions/workflows/python-package-conda.yml)


Refactoring the SSM code base to use Jax.

# Example

A quick demonstration of some of the most basic elements of SSM. Check out the [example notebooks](https://github.com/lindermanlab/ssm-jax-refactor/tree/main/notebooks) for more!

```python
from ssm.hmm import GaussianHMM
import jax.random as jr

# create a true HMM model
hmm = GaussianHMM(num_states=5, num_emission_dims=10, seed=jr.PRNGKey(0))
states, data = hmm.sample(key=jr.PRNGKey(1), num_steps=500, num_samples=5)

# create a test HMM model
test_hmm = GaussianHMM(num_states=5, num_emission_dims=10, seed=jr.PRNGKey(32))

# fit it to our sampled data
log_probs, fitted_model, posteriors = test_hmm.fit(data, method="em")
```

# Installation for Development

```bash
# use your favorite venv system
conda env create -n ssm_jax python=3.9
conda activate ssm_jax

# in repo root directory...
pip install -r requirements.txt
```

# Project Structure
```bash
.
├── docs                      # [documentation]
├── notebooks                 # [example jupyter notebooks]
├── ssm                       # [main code repository]
│   ├── hmm                       # hmm   models
│   ├── arhmm                     # arhmm models
│   ├── lds                       # lds   models
│   ├── slds                      # slds  models
│   ├── inference                 # inference code
│   ├── distributions             # distributions (generally, extensions of tfp distributions)
└── tests                     # [tests]
    └── timing_comparisons        # benchmarking code (including comparisons to SSM_v0)
 ```

# Documentation

[Click here for documentation](https://web.stanford.edu/~schlager/ssm_jax/)
