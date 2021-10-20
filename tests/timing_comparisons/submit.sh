#!/bin/bash
# Submit both ssm-old and ssm-jax test jobs

# ===Tests===
# test_laplace_em_num_trials
# test_lds_em_num_trials
# test_hmm_em_num_trials
# test_hmm_em_num_timesteps

launch_test () {
    TEST_NAME=$1
    echo "Launching jobs for $TEST_NAME"
    sbatch --job-name ssmold.$TEST_NAME --export=TEST_NAME=$TEST_NAME --output="logs/ssmold.$TEST_NAME.%A.log" test_old.sh 
    sbatch --job-name ssmjax.$TEST_NAME --export=TEST_NAME=$TEST_NAME --output="logs/ssmold.$TEST_NAME.%A.log" test_jax.sh 
}

launch_test "test_laplace_em_num_trials"
launch_test "test_lds_em_num_trials"
launch_test "test_lds_em_num_timesteps"
launch_test "test_hmm_em_num_trials"
launch_test "test_hmm_em_num_timesteps"
