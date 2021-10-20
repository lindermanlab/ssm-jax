#!/bin/bash
# Submit both ssm-old and ssm-jax test jobs

sbatch test_old.sh
sbatch test_jax.sh
