# SSM: Bayesian learning and inference for state space models

Refactoring the SSM code base to use Jax.

[![Tests](https://github.com/lindermanlab/ssm-jax-refactor/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/lindermanlab/ssm-jax-refactor/actions/workflows/python-package-conda.yml)

# Project Structure
```bash
.
├── docs                      # [documentation]
├── notebooks                 # [example jupyter notebooks]
├── ssm                       # [main code repository]
│   ├── hmm                       # hmm models
│   ├── arhmm                     # arhm models
│   ├── lds                       # lds models
│   ├── slds                      # slds models
│   ├── inference                 # inference code
│   ├── distributions             # distributions (generally, extensions of tfp distributions)
└── tests                     # [tests]
    └── timing_comparisons        # benchmarking code (including comparisons to SSM_v0)
 ```

# Installation for Development

```bash
# use your favorite venv system
conda env create -n ssm_jax python=3.9
conda activate ssm_jax

# in repo root directory...
pip install -r requirements.txt
```

# Documentation

[Click here for documentation](https://web.stanford.edu/~schlager/ssm_jax/)
