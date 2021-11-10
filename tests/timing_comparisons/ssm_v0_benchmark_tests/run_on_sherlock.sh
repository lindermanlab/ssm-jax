#!/bin/bash
#
#SBATCH --job-name=ssmold
#
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

source ~/.bashrc
conda activate ssmold
cd /home/users/schlager/lindermanlab/ssm-jax-refactor/tests/timing_comparisons/ssm_v0_benchmark_tests

echo "Using python ..."
which python

echo "\n\n ===== Beginning Test ======"

pytest -s -v -rw --benchmark-autosave

echo "======= End Test ======= "

echo "Complete!"
