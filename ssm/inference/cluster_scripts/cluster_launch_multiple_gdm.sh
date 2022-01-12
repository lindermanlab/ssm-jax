#!/bin/bash

glob_tag='GDM-cr-v5-0-0'
model='LDS'

# # BPF-SGR
exp_tag='bpf-sgr'
use_sgr=1
proposal_structure='BOOTSTRAP'
tilt_structure='NONE'
temper=0.0
sbatch --export=GLOB_TAG=$glob_tag,EXP_TAG=$exp_tag,MODEL=${model},USE_SGR=${use_sgr},PROPOSAL_STRUCTURE=${proposal_structure},TILT_STRUCTURE=${tilt_structure},TEMPER=${temper} cluster_scripts/_cluster_launch_multiple.sh

# FIVO
exp_tag='fivo'
use_sgr=0
proposal_structure='DIRECT'
tilt_structure='NONE'
temper=0.0
sbatch --export=GLOB_TAG=$glob_tag,EXP_TAG=$exp_tag,MODEL=${model},USE_SGR=${use_sgr},PROPOSAL_STRUCTURE=${proposal_structure},TILT_STRUCTURE=${tilt_structure},TEMPER=${temper} cluster_scripts/_cluster_launch_multiple.sh

# FIVO-SGR
exp_tag='fivo'
use_sgr=1
proposal_structure='DIRECT'
tilt_structure='NONE'
temper=0.0
sbatch --export=GLOB_TAG=$glob_tag,EXP_TAG=$exp_tag,MODEL=${model},USE_SGR=${use_sgr},PROPOSAL_STRUCTURE=${proposal_structure},TILT_STRUCTURE=${tilt_structure},TEMPER=${temper} cluster_scripts/_cluster_launch_multiple.sh

# FIVO-AUX
exp_tag='fivo-aux'
use_sgr=0
proposal_structure='DIRECT'
tilt_structure='DIRECT'
temper=0.0
sbatch --export=GLOB_TAG=$glob_tag,EXP_TAG=$exp_tag,MODEL=${model},USE_SGR=${use_sgr},PROPOSAL_STRUCTURE=${proposal_structure},TILT_STRUCTURE=${tilt_structure},TEMPER=${temper} cluster_scripts/_cluster_launch_multiple.sh

# FIVO-AUX-SGR
exp_tag='fivo-aux-sgr'
use_sgr=1
proposal_structure='DIRECT'
tilt_structure='DIRECT'
temper=0.0
sbatch --export=GLOB_TAG=$glob_tag,EXP_TAG=$exp_tag,MODEL=${model},USE_SGR=${use_sgr},PROPOSAL_STRUCTURE=${proposal_structure},TILT_STRUCTURE=${tilt_structure},TEMPER=${temper} cluster_scripts/_cluster_launch_multiple.sh

