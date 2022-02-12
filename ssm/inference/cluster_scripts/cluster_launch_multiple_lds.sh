#!/bin/bash
shopt -s expand_aliases

export glob_tag='LDS-v8-0-0-shortwindow'
export model='LDS'
export proposal_type='SINGLE_WINDOW'
export tilt_type='SINGLE_WINDOW'
export latent_dim=1
export emissions_dim=1
export GLOBAL_num_part=8
export enc_struct='NONE'

launch_cmd () { sbatch -J ${glob_tag} --export=ALL ./cluster_scripts/_cluster_launch_multiple.sh ; }

# BPF-SGR
export exp_tag='a---bpf-sgr'
export use_sgr=1
export proposal_structure='BOOTSTRAP'
export tilt_structure='NONE'
export temper=0.0
export use_vi=0
export n_part=$GLOBAL_num_part
export train_resamp_crit='ess_criterion'
export eval_resamp_crit='ess_criterion'
launch_cmd

# ELBO
export exp_tag='b---bpf-sgr'
export use_sgr=1
export proposal_structure='BOOTSTRAP'
export tilt_structure='NONE'
export temper=0.0
export use_vi=0
export n_part=1
export train_resamp_crit='never_resample'
export eval_resamp_crit='never_resample'
launch_cmd

# IWAE
export exp_tag='c---iwae'
export use_sgr=1
export proposal_structure='BOOTSTRAP'
export tilt_structure='NONE'
export temper=0.0
export use_vi=0
export n_part=$GLOBAL_num_part
export train_resamp_crit='never_resample'
export eval_resamp_crit='never_resample'
launch_cmd


# FIVO
export exp_tag='d---fivo'
export use_sgr=0
export proposal_structure='RESQ'
export tilt_structure='NONE'
export temper=0.0
export use_vi=0
export n_part=$GLOBAL_num_part
export train_resamp_crit='ess_criterion'
export eval_resamp_crit='ess_criterion'
launch_cmd

# FIVO-SGR
export exp_tag='e---fivo-sgr'
export use_sgr=1
export proposal_structure='RESQ'
export tilt_structure='NONE'
export temper=0.0
export use_vi=0
export n_part=$GLOBAL_num_part
export train_resamp_crit='ess_criterion'
export eval_resamp_crit='ess_criterion'
launch_cmd

# FIVO-AUX
export exp_tag='f---fivo-aux'
export use_sgr=0
export proposal_structure='RESQ'
export tilt_structure='DIRECT'
export temper=0.0
export use_vi=0
export n_part=$GLOBAL_num_part
export train_resamp_crit='ess_criterion'
export eval_resamp_crit='ess_criterion'
launch_cmd


# FIVO-AUX-SGR
export exp_tag='g---fivo-aux-sgr'
export use_sgr=1
export proposal_structure='RESQ'
export tilt_structure='DIRECT'
export temper=0.0
export use_vi=0
export n_part=$GLOBAL_num_part
export train_resamp_crit='ess_criterion'
export eval_resamp_crit='ess_criterion'
launch_cmd

# FIVO-AUX-SGR-TEMPERED
export exp_tag='h---fivo-aux-sgr-tempered'
export use_sgr=1
export proposal_structure='RESQ'
export tilt_structure='DIRECT'
export temper=1.0
export use_vi=0
export n_part=$GLOBAL_num_part
export train_resamp_crit='ess_criterion'
export eval_resamp_crit='ess_criterion'
launch_cmd


# FIVO-AUX-VI
export exp_tag='i---fivo-aux-vi'
export use_sgr=0
export proposal_structure='RESQ'
export tilt_structure='DIRECT'
export temper=0.0
export use_vi=1
export n_part=$GLOBAL_num_part
export train_resamp_crit='ess_criterion'
export eval_resamp_crit='ess_criterion'
launch_cmd

# FIVO-AUX-VI-SGR
export exp_tag='j---fivo-aux-vi-sgr'
export use_sgr=1
export proposal_structure='RESQ'
export tilt_structure='DIRECT'
export temper=0.0
export use_vi=1
export n_part=$GLOBAL_num_part
export train_resamp_crit='ess_criterion'
export eval_resamp_crit='ess_criterion'
launch_cmd


# FIVO-AUX-VI-TEMPERED
export exp_tag='k---fivo-aux-vi-tempered'
export use_sgr=0
export proposal_structure='RESQ'
export tilt_structure='DIRECT'
export temper=1.0
export use_vi=1
export n_part=$GLOBAL_num_part
export train_resamp_crit='ess_criterion'
export eval_resamp_crit='ess_criterion'
launch_cmd

# FIVO-AUX-VI-SGR-TEMPERED
export exp_tag='l---fivo-aux-vi-sgr-tempered'
export use_sgr=1
export proposal_structure='RESQ'
export tilt_structure='DIRECT'
export temper=1.0
export use_vi=1
export n_part=$GLOBAL_num_part
export train_resamp_crit='ess_criterion'
launch_cmd




