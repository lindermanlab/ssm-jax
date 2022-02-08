#!/bin/bash
shopt -s expand_aliases

glob_tag='GDM-v2-0-0'
model='GDM'
temper=0.0
proposal_type='PERSTEP'
tilt_type='DIRECT'
latent_dim=1
emissions_dim=1
num_particles=4
encoder_struct='NONE'
resamp_crit='always_resample'

launch_cmd () { sbatch -J ${glob_tag} --export=GLOB_TAG=$glob_tag,EXP_TAG=$exp_tag,MODEL=${model},USE_SGR=${use_sgr},PROPOSAL_STRUCTURE=${proposal_structure},PROPOSAL_TYPE=${proposal_type},TILT_STRUCTURE=${tilt_structure},TILT_TYPE=${tilt_type},TEMPER=${temper},LATENT_DIM=${latent_dim},EMISSIONS_DIM=${emissions_dim},USE_VI=${use_vi},N_PART=${num_particles},ENCODER_STRUCT=${encoder_struct},RESAMP_CRIT=${resamp_crit} cluster_scripts/_cluster_launch_multiple.sh ; }

# # BPF-SGR
exp_tag='bpf-sgr'
use_sgr=1
use_vi=0
proposal_structure='BOOTSTRAP'
tilt_structure='NONE'
launch_cmd

# FIVO
exp_tag='fivo'
use_sgr=0
use_vi=0
proposal_structure='DIRECT'
tilt_structure='NONE'
launch_cmd

# FIVO-SGR
exp_tag='fivo-sgr'
use_sgr=1
use_vi=0
proposal_structure='DIRECT'
tilt_structure='NONE'
launch_cmd

# FIVO-AUX
exp_tag='fivo-aux'
use_sgr=0
use_vi=0
proposal_structure='DIRECT'
tilt_structure='DIRECT'
launch_cmd

# FIVO-AUX-SGR
exp_tag='fivo-aux-sgr'
use_sgr=1
use_vi=0
proposal_structure='DIRECT'
tilt_structure='DIRECT'
launch_cmd

# FIVO-AUX-VI
exp_tag='fivo-aux-vi'
use_sgr=0
use_vi=1
proposal_structure='DIRECT'
tilt_structure='DIRECT'
launch_cmd

# FIVO-AUX-VI-SGR
exp_tag='fivo-aux-vi-sgr'
use_sgr=1
use_vi=1
proposal_structure='DIRECT'
tilt_structure='DIRECT'
launch_cmd

