#!/bin/bash
#
#SBATCH --job-name=ssmjax
#
#SBATCH --time=180:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output="logs/ssmold.%A.log"

source ~/.bashrc
conda activate ssmjax
cd /home/users/schlager/lindermanlab/ssm-jax-refactor/tests/timing_comparisons

echo "Using python ..."
which python

echo "\n\n ===== Beginning Test ======"

python run_tests.py --mode "ssm_jax" --name "test_num_trials"

echo "======= End Test ======= "

echo "Complete!"
