#!/bin/bash
#SBATCH --job-name=default
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:59:59
#SBATCH --output=./Reports/fivo_%A_%a.out
#SBATCH --error=./Reports/fivo_%A_%a.err
#SBATCH --array=0-0%5

module load python/3.9.0
module load texlive

echo ${SLURM_ARRAY_TASK_ID}
echo ${glob_tag}-${exp_tag}
echo ${temper}
which python3.9

pip list

echo "\n"
echo "HYPERPARAMETERS:"
echo "\tGLOB_TAG:   " $glob_tag
echo "\tEXP_TAG:    " $exp_tag
echo "\tMODEL:      " $model
echo "\tUSE SGR:    " $use_sgr
echo "\tPROP STRUC: " $proposal_structure
echo "\tPROP TYPE:  " $proposal_type
echo "\tTILT STRUC: " $tilt_structure
echo "\tTILT TYPE:  " $tilt_type
echo "\tTEMPER:     " $temper
echo "\tLATENT DIM: " $latent_dim
echo "\tEM DIM:     " $emissions_dim
echo "\tUSE VI:     " $use_vi
echo "\tN PARTICLE: " $n_part
echo "\tENC STRUCT: " $enc_struct
echo "\tTr resamp c:" $train_resamp_crit
echo "\tEv resamp c:" $eval_resamp_crit


python3.9 _test_fivo.py --model ${model} --seed ${SLURM_ARRAY_TASK_ID} --PLOT 0 --use-sgr ${use_sgr} --proposal-structure ${proposal_structure} --proposal-type ${proposal_type} --tilt-structure ${tilt_structure} --tilt-type ${tilt_type} --log-group ${glob_tag}---${exp_tag} --temper "${temper}" --latent-dim ${latent_dim} --emissions-dim ${emissions_dim} --vi-use-tilt-gradient ${use_vi} --num-particles ${n_part} --encoder-structure ${enc_struct} --train-resampling-criterion ${train_resamp_crit} --eval-resampling-criterion ${eval_resamp_crit} --dataset ${dataset}
