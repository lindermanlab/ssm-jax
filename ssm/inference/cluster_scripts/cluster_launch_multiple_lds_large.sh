#!/bin/bash

glob_tag='LDS-large-v1-0-0'
model='LDS'
proposal_type='PERSTEP'
tilt_type='SINGLEWINDOW'
latent_dim=5
emissions_dim=10

# # BPF-SGR
exp_tag='bpf-sgr'
use_sgr=1
proposal_structure='BOOTSTRAP'
tilt_structure='NONE'
temper=0.0
sbatch --export=GLOB_TAG=$glob_tag,EXP_TAG=$exp_tag,MODEL=${model},USE_SGR=${use_sgr},PROPOSAL_STRUCTURE=${proposal_structure},PROPOSAL_TYPE=${proposal_type},TILT_STRUCTURE=${tilt_structure},TILT_TYPE=${tilt_type},TEMPER=${temper},LATENT_DIM=${latent_dim},EMISSIONS_DIM=${emissions_dim} cluster_scripts/_cluster_launch_multiple.sh

# FIVO
exp_tag='fivo'
use_sgr=0
proposal_structure='RESQ'
tilt_structure='NONE'
temper=0.0
sbatch --export=GLOB_TAG=$glob_tag,EXP_TAG=$exp_tag,MODEL=${model},USE_SGR=${use_sgr},PROPOSAL_STRUCTURE=${proposal_structure},PROPOSAL_TYPE=${proposal_type},TILT_STRUCTURE=${tilt_structure},TILT_TYPE=${tilt_type},TEMPER=${temper},LATENT_DIM=${latent_dim},EMISSIONS_DIM=${emissions_dim} cluster_scripts/_cluster_launch_multiple.sh

# FIVO-AUX-SGR
exp_tag='fivo-aux-sgr'
use_sgr=1
proposal_structure='RESQ'
tilt_structure='DIRECT'
temper=0.0
sbatch --export=GLOB_TAG=$glob_tag,EXP_TAG=$exp_tag,MODEL=${model},USE_SGR=${use_sgr},PROPOSAL_STRUCTURE=${proposal_structure},PROPOSAL_TYPE=${proposal_type},TILT_STRUCTURE=${tilt_structure},TILT_TYPE=${tilt_type},TEMPER=${temper},LATENT_DIM=${latent_dim},EMISSIONS_DIM=${emissions_dim} cluster_scripts/_cluster_launch_multiple.sh

# FIVO-AUX-SGR-TEMPERED
exp_tag='fivo-aux-sgr-tempered'
use_sgr=1
proposal_structure='RESQ'
tilt_structure='DIRECT'
temper=4.0
sbatch --export=GLOB_TAG=$glob_tag,EXP_TAG=$exp_tag,MODEL=${model},USE_SGR=${use_sgr},PROPOSAL_STRUCTURE=${proposal_structure},PROPOSAL_TYPE=${proposal_type},TILT_STRUCTURE=${tilt_structure},TILT_TYPE=${tilt_type},TEMPER=${temper},LATENT_DIM=${latent_dim},EMISSIONS_DIM=${emissions_dim} cluster_scripts/_cluster_launch_multiple.sh
