#!/bin/bash
#
#SBATCH --job-name=ssmjax
#
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

source ~/.bashrc
conda activate ssmjax
cd /home/users/schlager/lindermanlab/ssm-jax-refactor/tests/timing_comparisons

echo "Using python ..."
which python

echo "\n\n ===== Beginning Test ======"
echo $TEST_NAME

python run_tests.py --mode "ssm_jax" --name $TEST_NAME

echo "======= End Test ======= "

echo "Complete!"
