#!/bin/bash
shopt -s expand_aliases

export glob_tag='GDM-v5-0-0'
export model='GDM'
export temper=0.0
export proposal_type='PERSTEP'
export tilt_type='DIRECT'
export latent_dim=1
export emissions_dim=1
export GLOBAL_n_part=4
export enc_struct='NONE'
export GLOBAL_eval_resamp_crit='always_resample'
export GLOBAL_train_resamp_crit='always_resample'
export dataset='default'

launch_cmd () { sbatch -J ${glob_tag} --export=ALL cluster_scripts/_cluster_launch_multiple.sh ; }

# # BPF-SGR
export exp_tag='a---bpf-sgr'
export use_sgr=1
export use_vi=0
export proposal_structure='BOOTSTRAP'
export tilt_structure='NONE'
export train_resamp_crit=${GLOBAL_train_resamp_crit}
export eval_resamp_crit=${GLOBAL_eval_resamp_crit}
export n_part=${GLOBAL_n_part}
launch_cmd

# # ELBO
export exp_tag='b---elbo'
export use_sgr=0
export use_vi=0
export proposal_structure='DIRECT'
export tilt_structure='NONE'
export train_resamp_crit='never_resample'
export eval_resamp_crit='never_resample'
export n_part=1
launch_cmd

# # IWAE
export exp_tag='c---iwae'
export use_sgr=0
export use_vi=0
export proposal_structure='DIRECT'
export tilt_structure='NONE'
export train_resamp_crit='never_resample'
export eval_resamp_crit='never_resample'
export n_part=${GLOBAL_n_part}
launch_cmd


# FIVO
export exp_tag='d---fivo'
export use_sgr=0
export use_vi=0
export proposal_structure='DIRECT'
export tilt_structure='NONE'
export train_resamp_crit=${GLOBAL_train_resamp_crit}
export eval_resamp_crit=${GLOBAL_eval_resamp_crit}
export n_part=${GLOBAL_n_part}
launch_cmd

# FIVO-SGR
export exp_tag='e---fivo-sgr'
export use_sgr=1
export use_vi=0
export proposal_structure='DIRECT'
export tilt_structure='NONE'
export train_resamp_crit=${GLOBAL_train_resamp_crit}
export eval_resamp_crit=${GLOBAL_eval_resamp_crit}
export n_part=${GLOBAL_n_part}
launch_cmd


# FIVO-AUX
export exp_tag='f---fivo-aux'
export use_sgr=0
export use_vi=0
export proposal_structure='DIRECT'
export tilt_structure='DIRECT'
export train_resamp_crit=${GLOBAL_train_resamp_crit}
export eval_resamp_crit=${GLOBAL_eval_resamp_crit}
export n_part=${GLOBAL_n_part}
launch_cmd

# FIVO-AUX-SGR
export exp_tag='g---fivo-aux-sgr'
export use_sgr=1
export use_vi=0
export proposal_structure='DIRECT'
export tilt_structure='DIRECT'
export train_resamp_crit=${GLOBAL_train_resamp_crit}
export eval_resamp_crit=${GLOBAL_eval_resamp_crit}
export n_part=${GLOBAL_n_part}
launch_cmd

# FIVO-AUX-VI
export exp_tag='h---fivo-aux-vi'
export use_sgr=0
export use_vi=1
export proposal_structure='DIRECT'
export tilt_structure='DIRECT'
export train_resamp_crit=${GLOBAL_train_resamp_crit}
export eval_resamp_crit=${GLOBAL_eval_resamp_crit}
export n_part=${GLOBAL_n_part}
launch_cmd

# FIVO-AUX-VI-SGR
export exp_tag='i---fivo-aux-vi-sgr'
export use_sgr=1
export use_vi=1
export proposal_structure='DIRECT'
export tilt_structure='DIRECT'
export train_resamp_crit=${GLOBAL_train_resamp_crit}
export eval_resamp_crit=${GLOBAL_eval_resamp_crit}
export n_part=${GLOBAL_n_part}
launch_cmd

