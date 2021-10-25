#!/bin/zsh
source ~/.zshrc

# run the jax tests 
conda activate ssmjax
pytest -s -v --benchmark-autosave --ignore ssm_v0_benchmark_tests

# run the ssm-v0 tests
conda activate ssmold
cd ssm_v0_benchmark_tests
pytest -s -v --benchmark-autosave