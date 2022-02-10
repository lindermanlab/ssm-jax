#!/bin/bash
shopt -s expand_aliases

dataset='jsb'
latent_dim=32

n_part=4

# These are reasonably constant for VRNN experiments
glob_tag="VRNN-${dataset}-${n_part}-v1-0-5"
model='VRNN'
emissions_dim=1
temper=0.0
use_vi=0
resamp_crit='ess_criterion'

# launch_cmd () { sbatch -p gpu -G 1 -C GPU_MEM:16GB -t 4:00:00 -J ${glob_tag} --export=GLOB_TAG=$glob_tag,EXP_TAG=$exp_tag,MODEL=${model},USE_SGR=${use_sgr},PROPOSAL_STRUCTURE=${proposal_structure},PROPOSAL_TYPE=${proposal_type},TILT_STRUCTURE=${tilt_structure},TILT_TYPE=${tilt_type},TEMPER=${temper},LATENT_DIM=${latent_dim},EMISSIONS_DIM=${emissions_dim},USE_VI=${use_vi},N_PART=${n_part},ENCODER_STRUCT=${enc_struct},RESAMP_CRIT=${resamp_crit},DATASET={dataset} cluster_scripts/_cluster_launch_multiple.sh ; }

launch_cmd () { sbatch --cpus-per-task=4 --mem=20GB -t 9:59:00 -J ${glob_tag} --export=GLOB_TAG=$glob_tag,EXP_TAG=$exp_tag,MODEL=${model},USE_SGR=${use_sgr},PROPOSAL_STRUCTURE=${proposal_structure},PROPOSAL_TYPE=${proposal_type},TILT_STRUCTURE=${tilt_structure},TILT_TYPE=${tilt_type},TEMPER=${temper},LATENT_DIM=${latent_dim},EMISSIONS_DIM=${emissions_dim},USE_VI=${use_vi},N_PART=${n_part},ENCODER_STRUCT=${enc_struct},RESAMP_CRIT=${resamp_crit},DATASET=${dataset} cluster_scripts/_cluster_launch_multiple.sh ; }


# BPF-SGR
exp_tag='bpf-sgr'
use_sgr=0
proposal_type='NONE'
proposal_structure='NONE'
tilt_type='NONE'
tilt_structure='NONE'
enc_struct='NONE'
launch_cmd


# FIVO-FILTER
exp_tag='fivo'
use_sgr=0
proposal_type='VRNN_FILTERING'
proposal_structure='VRNN_FILTERING_RESQ'
tilt_type='NONE'
tilt_structure='NONE'
enc_struct='NONE'
launch_cmd

# FIVO-FILTER
exp_tag='fivo+smooth'
use_sgr=0
proposal_type='VRNN_SMOOTHING'
proposal_structure='VRNN_SMOOTHING_RESQ'
tilt_type='NONE'
tilt_structure='NONE'
enc_struct='BIRNN'
launch_cmd


# FIVO-AUX-SGR-WINDOW
exp_tag='fivo-aux-sgr-window'
use_sgr=1
proposal_type='VRNN_FILTERING'
proposal_structure='VRNN_FILTERING_RESQ'
tilt_type='SINGLE_WINDOW'
tilt_structure='DIRECT'
enc_struct='NONE'
launch_cmd

# FIVO-AUX-SGR-ENCODED
exp_tag='fivo-aux-sgr-encoded'
use_sgr=1
proposal_type='VRNN_SMOOTHING'
proposal_structure='VRNN_SMOOTHING_RESQ'
tilt_type='ENCODED'
tilt_structure='DIRECT'
enc_struct='BIRNN'
launch_cmd


# FIVO-AUX-SGR-WINDOW-TEMPER
exp_tag='fivo-aux-sgr-window-temper'
use_sgr=1
proposal_type='VRNN_FILTERING'
proposal_structure='VRNN_FILTERING_RESQ'
tilt_type='SINGLE_WINDOW'
tilt_structure='DIRECT'
enc_struct='NONE'
temper=1.0
launch_cmd

# FIVO-AUX-SGR-ENCODED-TEMPER
exp_tag='fivo-aux-sgr-encodedtempered'
use_sgr=1
proposal_type='VRNN_SMOOTHING'
proposal_structure='VRNN_SMOOTHING_RESQ'
tilt_type='SINGLE_WINDOW'
tilt_structure='DIRECT'
enc_struct='NONE'
temper=1.0
launch_cmd

