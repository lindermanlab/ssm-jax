#!/bin/bash
#SBATCH --job-name=default
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:59:59
#SBATCH --output=./Reports/fivo_%A_%a.out
#SBATCH --error=./Reports/fivo_%A_%a.err
#SBATCH --array=0-7%4

module load python/3.9.0
module load texlive

echo ${SLURM_ARRAY_TASK_ID}
echo ${glob_tag}-${exp_tag}
echo ${temper}
which python3.9

pip list

echo "\n"
echo "HYPERPARAMETERS:"
echo "	GLOB_TAG:   " $glob_tag
echo "	EXP_TAG:    " $exp_tag
echo "	MODEL:      " $model
echo "	USE SGR:    " $use_sgr
echo "	PROP STRUC: " $proposal_structure
echo "	PROP TYPE:  " $proposal_type
echo "	TILT STRUC: " $tilt_structure
echo "	TILT TYPE:  " $tilt_type
echo "	TEMPER:     " $temper
echo "	LATENT DIM: " $latent_dim
echo "	EM DIM:     " $emissions_dim
echo "	USE VI:     " $use_vi
echo "	N PARTICLE: " $n_part
echo "	ENC STRUCT: " $enc_struct
echo "	Tr resamp c:" $train_resamp_crit
echo "	Ev resamp c:" $eval_resamp_crit


python3.9 _test_fivo.py --model ${model} --seed ${SLURM_ARRAY_TASK_ID} --PLOT 0 --use-sgr ${use_sgr} --proposal-structure ${proposal_structure} --proposal-type ${proposal_type} --tilt-structure ${tilt_structure} --tilt-type ${tilt_type} --log-group ${glob_tag}-${exp_tag} --temper "${temper}" --latent-dim ${latent_dim} --emissions-dim ${emissions_dim} --vi-use-tilt-gradient ${use_vi} --num-particles ${n_part} --encoder-structure ${enc_struct} --train-resampling-criterion ${train_resamp_crit} --eval-resampling-criterion ${eval_resamp_crit} --dataset ${dataset}
