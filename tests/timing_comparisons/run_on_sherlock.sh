#!/bin/bash
#
#SBATCH --job-name=ssmjax
#
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G

source ~/.bashrc
conda activate ssmjax
cd /home/users/schlager/lindermanlab/ssm-jax-refactor/tests/timing_comparisons

echo "Using python ..."
which python

echo "\n\n ===== Beginning Test ======"

pytest --ignore ssm_v0_benchmark_tests -s -v -rw --benchmark-autosave

echo "======= End Test ======= "

echo "Complete!"
