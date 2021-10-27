#!/bin/zsh
# Helper script to launch specific test workflows
# across both SSM-JAX and SSM-V0 (old) codebases

# Usage:
# zsh run_tests.sh <test_match_keyword>
# 
# Example:
# zsh run_tests.sh "arhmm"

source ~/.zshrc

# run the jax tests 
function run_jax_tests () {
    local PREFIX=$1
    conda activate ssmjax
    pytest -s -v --benchmark-save $PREFIX --ignore ssm_v0_benchmark_tests -k $PREFIX
}

function run_old_tests () {
    local PREFIX=$1
    cd ssm_v0_benchmark_tests
    pytest -s -v --benchmark-save $PREFIX -k $PREFIX
    cd ..
}

run_jax_tests $1
run_old_tests $1