#!/bin/bash
shopt -s expand_aliases

glob_tag='SVM-v1-0-0'
model='SVM'
proposal_type='SINGLE_WINDOW'
tilt_type='SINGLE_WINDOW'
latent_dim=1
emissions_dim=1

launch_cmd () { sbatch -J ${glob_tag} --export=GLOB_TAG=$glob_tag,EXP_TAG=$exp_tag,MODEL=${model},USE_SGR=${use_sgr},PROPOSAL_STRUCTURE=${proposal_structure},PROPOSAL_TYPE=${proposal_type},TILT_STRUCTURE=${tilt_structure},TILT_TYPE=${tilt_type},TEMPER=${temper},LATENT_DIM=${latent_dim},EMISSIONS_DIM=${emissions_dim},USE_VI=${use_vi} cluster_scripts/_cluster_launch_multiple.sh ; }


# BPF-SGR
exp_tag='bpf-sgr'
use_sgr=1
proposal_structure='BOOTSTRAP'
tilt_structure='NONE'
temper=0.0
use_vi=0
launch_cmd

# FIVO
exp_tag='fivo'
use_sgr=0
proposal_structure='RESQ'
tilt_structure='NONE'
temper=0.0
use_vi=0
launch_cmd

# FIVO-SGR
exp_tag='fivo-sgr'
use_sgr=1
proposal_structure='RESQ'
tilt_structure='NONE'
temper=0.0
use_vi=0
launch_cmd

# FIVO-AUX-SGR
exp_tag='fivo-aux-sgr'
use_sgr=1
proposal_structure='RESQ'
tilt_structure='DIRECT'
temper=0.0
use_vi=0
launch_cmd

# FIVO-AUX-SGR-TEMPERED
exp_tag='fivo-aux-sgr-tempered'
use_sgr=1
proposal_structure='RESQ'
tilt_structure='DIRECT'
temper=1.0
use_vi=0
launch_cmd

# FIVO-AUX-VI
exp_tag='fivo-aux-vi'
use_sgr=0
proposal_structure='RESQ'
tilt_structure='DIRECT'
temper=0.0
use_vi=1
launch_cmd

# FIVO-AUX-VI-SGR
exp_tag='fivo-aux-vi-sgr'
use_sgr=1
proposal_structure='RESQ'
tilt_structure='DIRECT'
temper=0.0
use_vi=1
launch_cmd

# FIVO-AUX-VI-TEMPERED
exp_tag='fivo-aux-vi-tempered'
use_sgr=0
proposal_structure='RESQ'
tilt_structure='DIRECT'
temper=1.0
use_vi=1
launch_cmd

# FIVO-AUX-VI-SGR-TEMPERED
exp_tag='fivo-aux-vi-sgr-tempered'
use_sgr=1
proposal_structure='RESQ'
tilt_structure='DIRECT'
temper=1.0
use_vi=1
launch_cmd




