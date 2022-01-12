#!/bin/bash
#SBATCH --job-name=default
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:59:59
#SBATCH --output=./Reports/fivo_%A_%a.out
#SBATCH --error=./Reports/fivo_%A_%a.err
#SBATCH --array=0-4

module load python/3.9.0
module load texlive

echo ${SLURM_ARRAY_TASK_ID}
echo ${GLOB_TAG}-${EXP_TAG}
echo ${TEMPER}
which python3.9

pip list

python3.9 _test_fivo.py --model ${MODEL} --seed ${SLURM_ARRAY_TASK_ID} --PLOT 0 --use-sgr ${USE_SGR} --proposal-structure ${PROPOSAL_STRUCTURE} --proposal-type ${PROPOSAL_TYPE} --tilt-structure ${TILT_STRUCTURE} --tilt-type ${TILT_TYPE} --log-group ${GLOB_TAG}-${EXP_TAG} --temper "${TEMPER}" --latent-dim ${LATENT_DIM} --emissions-dim ${EMISSIONS_DIM}
