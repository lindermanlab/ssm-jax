#!/bin/bash
shopt -s expand_aliases

glob_tag='VRNN-JSB-v0-0-0'
model='VRNN'
latent_dim=32
emissions_dim=1
use_sgr=0
temper=0.0
use_vi=0
n_part=4
resamp_crit='ess_criterion'

launch_cmd () { sbatch -p gpu -G 1 -C GPU_MEM:16GB -t 4:00:00 -J ${glob_tag} --export=GLOB_TAG=$glob_tag,EXP_TAG=$exp_tag,MODEL=${model},USE_SGR=${use_sgr},PROPOSAL_STRUCTURE=${proposal_structure},PROPOSAL_TYPE=${proposal_type},TILT_STRUCTURE=${tilt_structure},TILT_TYPE=${tilt_type},TEMPER=${temper},LATENT_DIM=${latent_dim},EMISSIONS_DIM=${emissions_dim},USE_VI=${use_vi},N_PART=${n_part},ENCODER_STRUCT=${enc_struct},RESAMP_CRIT=${resamp_crit} cluster_scripts/_cluster_launch_multiple.sh ; }

# BPF-SGR
exp_tag='BPF-SGR'
use_sgr=0
proposal_type='NONE'
proposal_structure='NONE'
tilt_type='NONE'
tilt_structure='NONE'
enc_struct='NONE'
launch_cmd


# FIVO-FILTER
exp_tag='FIVO'
use_sgr=0
proposal_type='VRNN_FILTERING'
proposal_structure='VRNN_FILTERING_RESQ'
tilt_type='NONE'
tilt_structure='NONE'
enc_struct='NONE'
launch_cmd

# FIVO-FILTER
exp_tag='FIVO+smooth'
use_sgr=0
proposal_type='VRNN_SMOOTHING'
proposal_structure='VRNN_SMOOTHING_RESQ'
tilt_type='NONE'
tilt_structure='NONE'
enc_struct='BIRNN'
launch_cmd


# FIVO-AUX-SGR-WINDOW
exp_tag='FIVO-AUX-SGR-WINDOW'
use_sgr=1
proposal_type='VRNN_SMOOTHING'
proposal_structure='VRNN_SMOOTHING_RESQ'
tilt_type='SINGLE_WINDOW'
tilt_structure='DIRECT'
enc_struct='BIRNN'
launch_cmd

# FIVO-AUX-SGR-ENCODED
exp_tag='FIVO-AUX-SGR-ENCODED'
use_sgr=1
proposal_type='VRNN_SMOOTHING'
proposal_structure='VRNN_SMOOTHING_RESQ'
tilt_type='ENCODED'
tilt_structure='DIRECT'
enc_struct='BIRNN'
launch_cmd


