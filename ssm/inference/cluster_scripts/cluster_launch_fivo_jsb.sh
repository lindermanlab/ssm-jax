#!/bin/bash
shopt -s expand_aliases

glob_tag='FIVO-benchmark-v0-0-0'
model='VRNN'
proposal_type='VRNN_FILTERING'i
proposal_structure='VRNN_FILTERING_RESQ'
tilt_type='NONE'
tilt_structure='NONE'
latent_dim=1
emissions_dim=1
use_sgr=0
temper=0.0
use_vi=0

launch_cmd () { sbatch -p gpu -G 1 -C GPU_MEM:16GB -t 4:00:00 -J ${glob_tag} --export=GLOB_TAG=$glob_tag,EXP_TAG=$exp_tag,MODEL=${model},USE_SGR=${use_sgr},PROPOSAL_STRUCTURE=${proposal_structure},PROPOSAL_TYPE=${proposal_type},TILT_STRUCTURE=${tilt_structure},TILT_TYPE=${tilt_type},TEMPER=${temper},LATENT_DIM=${latent_dim},EMISSIONS_DIM=${emissions_dim},USE_VI=${use_vi},N_PART=${n_part} cluster_scripts/_cluster_launch_multiple.sh ; }

# n4
exp_tag='n4'
n_part=4
launch_cmd

# BPF-SGR
exp_tag='n8'
n_part=8
launch_cmd

# BPF-SGR
exp_tag='n16'
n_part=16
launch_cmd

