#!/bin/bash
shopt -s expand_aliases

export glob_tag='GDM-v4-0-0'
export model='GDM'
export temper=0.0
export proposal_type='PERSTEP'
export tilt_type='DIRECT'
export latent_dim=1
export emissions_dim=1
export n_part=4
export enc_struct='NONE'
export train_resamp_crit='always_resample'
export eval_resamp_crit='always_resample'

launch_cmd () { sbatch -J ${glob_tag} --export=ALL cluster_scripts/_cluster_launch_multiple.sh ; }

# # BPF-SGR
export exp_tag='bpf-sgr'
export use_sgr=1
export use_vi=0
export proposal_structure='BOOTSTRAP'
export tilt_structure='NONE'
launch_cmd

# FIVO
export exp_tag='fivo'
export use_sgr=0
export use_vi=0
export proposal_structure='DIRECT'
export tilt_structure='NONE'
launch_cmd

# FIVO-SGR
export exp_tag='fivo-sgr'
export use_sgr=1
export use_vi=0
export proposal_structure='DIRECT'
export tilt_structure='NONE'
launch_cmd

# FIVO-AUX
export exp_tag='fivo-aux'
export use_sgr=0
export use_vi=0
export proposal_structure='DIRECT'
export tilt_structure='DIRECT'
launch_cmd

# FIVO-AUX-SGR
export exp_tag='fivo-aux-sgr'
export use_sgr=1
export use_vi=0
export proposal_structure='DIRECT'
export tilt_structure='DIRECT'
launch_cmd

# FIVO-AUX-VI
export exp_tag='fivo-aux-vi'
export use_sgr=0
export use_vi=1
export proposal_structure='DIRECT'
export tilt_structure='DIRECT'
launch_cmd

# FIVO-AUX-VI-SGR
export exp_tag='fivo-aux-vi-sgr'
export use_sgr=1
export use_vi=1
export proposal_structure='DIRECT'
export tilt_structure='DIRECT'
launch_cmd

