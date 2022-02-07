#!/bin/bash
#SBATCH --job-name=default
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=07:59:59
#SBATCH --output=./Reports/fivo_%A_%a.out
#SBATCH --error=./Reports/fivo_%A_%a.err
#SBATCH --array=0-9%2

module load python/3.9.0
module load texlive

echo ${SLURM_ARRAY_TASK_ID}
echo ${GLOB_TAG}-${EXP_TAG}
echo ${TEMPER}
which python3.9

pip list

echo "GLOB_TAG:   " $GLOB_TAG
echo "EXP_TAG:    " $EXP_TAG
echo "MODEL:      " $MODEL
echo "USE SGR:    " $USE_SGR
echo "PROP STRUC: " $PROPOSAL_STRUCTURE
echo "PROP TYPE:  " $PROPOSAL_TYPE
echo "TILT STRUC: " $TILT_STRUCTURE
echo "TILT TYPE:  " $TILT_TYPE
echo "TEMPER:     " $TEMPER
echo "LATENT DIM: " $LATENT_DIM
echo "EM DIM:     " $EMISSIONS_DIM
echo "USE VI:     " $USE_VI
echo "N PARTICLE: " $N_PART
echo "ENC STRUCT: " $ENCODER_STRUCT
echo "resamp_crit:" $RESAMP_CRIT

python3.9 _test_fivo.py --model ${MODEL} --seed ${SLURM_ARRAY_TASK_ID} --PLOT 0 --use-sgr ${USE_SGR} --proposal-structure ${PROPOSAL_STRUCTURE} --proposal-type ${PROPOSAL_TYPE} --tilt-structure ${TILT_STRUCTURE} --tilt-type ${TILT_TYPE} --log-group ${GLOB_TAG}-${EXP_TAG} --temper "${TEMPER}" --latent-dim ${LATENT_DIM} --emissions-dim ${EMISSIONS_DIM} --vi-use-tilt-gradient ${USE_VI} --num-particles ${N_PART} --encoder-structure ${ENCODER_STRUCT} --resampling-criterion ${RESAMP_CRIT}
