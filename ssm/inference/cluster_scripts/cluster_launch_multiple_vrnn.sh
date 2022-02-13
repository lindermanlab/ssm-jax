#!/bin/bash
shopt -s expand_aliases

export dataset='jsb'
export latent_dim=32

export n_part_global=4

# These are reasonably constant for VRNN experiments
export glob_tag="VRNN-${dataset}-${n_part_global}-v3-0-0"
export model='VRNN'
export emissions_dim=1

# launch_cmd () { sbatch -p gpu -G 1 -C GPU_MEM:16GB -t 4:00:00 -J ${glob_tag} --export=GLOB_TAG=$glob_tag,EXP_TAG=$exp_tag,MODEL=${model},USE_SGR=${use_sgr},PROPOSAL_STRUCTURE=${proposal_structure},PROPOSAL_TYPE=${proposal_type},TILT_STRUCTURE=${tilt_structure},TILT_TYPE=${tilt_type},TEMPER=${temper},LATENT_DIM=${latent_dim},EMISSIONS_DIM=${emissions_dim},USE_VI=${use_vi},N_PART=${n_part},ENCODER_STRUCT=${enc_struct},RESAMP_CRIT=${resamp_crit},DATASET={dataset} cluster_scripts/_cluster_launch_multiple.sh ; }

launch_cmd () { sbatch --cpus-per-task=2 --mem=20GB -t 9:59:00 -J ${glob_tag}-${exp_tag} --export=ALL cluster_scripts/_cluster_launch_multiple.sh ; }

# launch_cmd () { sbatch --cpus-per-task=1 -t 0:59:00 -J ${glob_tag}-${exp_tag} --export=ALL cluster_scripts/_cluster_launch_multiple.sh ; }

# BPF-SGR
export exp_tag='a---bpf-sgr'
export use_sgr=0
export proposal_type='NONE'
export proposal_structure='NONE'
export tilt_type='NONE'
export tilt_structure='NONE'
export enc_struct='NONE'
export temper=0.0
export use_vi=0
export n_part=$n_part_global
export train_resamp_crit='ess_criterion'
export eval_resamp_crit='ess_criterion'
launch_cmd

# ELBO
export exp_tag='b---elbo'
export use_sgr=0
export proposal_type='VRNN_FILTERING'
export proposal_structure='VRNN_FILTERING_RESQ'
export tilt_type='NONE'
export tilt_structure='NONE'
export enc_struct='NONE'
export temper=0.0
export use_vi=0
export n_part=1  				# Single particle is used in ELBO
export train_resamp_crit='never_resample'	# No resampling in ELBO
export eval_resamp_crit='never_resample'	# No resampling in ELBO
launch_cmd

# IWAE
export exp_tag='c---iwae'
export use_sgr=0
export proposal_type='VRNN_FILTERING'
export proposal_structure='VRNN_FILTERING_RESQ'
export tilt_type='NONE'
export tilt_structure='NONE'
export enc_struct='NONE'
export temper=0.0
export use_vi=0
export n_part=$n_part_global
export train_resamp_crit='never_resample'	# No resampling in IWAE
export eval_resamp_crit='never_resample'	# No resampling in IWAE
launch_cmd


# FIVO-FILTER
export exp_tag='d---fivo'
export use_sgr=0
export proposal_type='VRNN_FILTERING'
export proposal_structure='VRNN_FILTERING_RESQ'
export tilt_type='NONE'
export tilt_structure='NONE'
export enc_struct='NONE'
export temper=0.0
export use_vi=0
export n_part=$n_part_global
export train_resamp_crit='ess_criterion'
export eval_resamp_crit='ess_criterion'
launch_cmd

# FIVO-SMOOTH
export exp_tag='e---fivo+smooth'
export use_sgr=0
export proposal_type='VRNN_SMOOTHING'
export proposal_structure='VRNN_SMOOTHING_RESQ'
export tilt_type='NONE'
export tilt_structure='NONE'
export enc_struct='BIRNN'
export temper=0.0
export use_vi=0
export n_part=$n_part_global
export train_resamp_crit='ess_criterion'
export eval_resamp_crit='ess_criterion'
launch_cmd


# FIVO-AUX-SGR-WINDOW
export exp_tag='f---fivo-aux-sgr-window'
export use_sgr=1
export proposal_type='VRNN_FILTERING'
export proposal_structure='VRNN_FILTERING_RESQ'
export tilt_type='SINGLE_WINDOW'
export tilt_structure='DIRECT'
export enc_struct='NONE'
export temper=0.0
export use_vi=0
export n_part=$n_part_global
export train_resamp_crit='ess_criterion'
export eval_resamp_crit='ess_criterion'
launch_cmd

# FIVO-AUX-SGR-ENCODED
export exp_tag='g---fivo-aux-sgr-encoded'
export use_sgr=1
export proposal_type='VRNN_SMOOTHING'
export proposal_structure='VRNN_SMOOTHING_RESQ'
export tilt_type='ENCODED'
export tilt_structure='DIRECT'
export enc_struct='BIRNN'
export temper=0.0
export use_vi=0
export n_part=$n_part_global
export train_resamp_crit='ess_criterion'
export eval_resamp_crit='ess_criterion'
launch_cmd


# FIVO-AUX-SGR-WINDOW-TEMPER
export exp_tag='h---fivo-aux-sgr-window-temper'
export use_sgr=1
export proposal_type='VRNN_FILTERING'
export proposal_structure='VRNN_FILTERING_RESQ'
export tilt_type='SINGLE_WINDOW'
export tilt_structure='DIRECT'
export enc_struct='NONE'
export temper=1.0
export use_vi=0
export n_part=$n_part_global
export train_resamp_crit='ess_criterion'
export eval_resamp_crit='ess_criterion'
launch_cmd

# FIVO-AUX-SGR-ENCODED-TEMPER
export exp_tag='i---fivo-aux-sgr-encoded-temper'
export use_sgr=1
export proposal_type='VRNN_SMOOTHING'
export proposal_structure='VRNN_SMOOTHING_RESQ'
export tilt_type='ENCODED'
export tilt_structure='DIRECT'
export enc_struct='BIRNN'
export temper=1.0
export use_vi=0
export n_part=$n_part_global
export train_resamp_crit='ess_criterion'
export eval_resamp_crit='ess_criterion'
launch_cmd

