#!/bin/bash

# Job name:
#SBATCH --job-name=EQCSP
#SBATCH --output 03_pick_%A_%a.log

# Project:
#SBATCH --account=XXXXXX

#SBATCH --ntasks-per-node=1

#####   ARRAY JOB TASK       ########
#SBATCH --array=0-11         # index starts at zero, so 0-11 for 12 tasks!!

#####   CPU NODES, NORMAL:   ########
#SBATCH --partition=normal
#SBATCH --cpus-per-task=40   # 40 or 52
#SBATCH --mem=178G           # picking in 2018/2019 takes at most 120 GB for 40 cores, so normal node should suffice
#SBATCH --time=03-00:00:00   # takes about 30 hours per month

## Set up job environment:
# set -o errexit             # Exit the script on any error
# set -o nounset             # Treat any unset variables as an error

module --quiet purge         # Reset the modules to the system default

#######################################################################

# load the Anaconda3
ml load Miniconda3/4.9.2
ml load FFTW/3.3.9-intel-2021a
ml load CUDA/11.3.1

# Set the ${PS1} (needed in the source of the Anaconda environment)
# export PS1=\$

# Source the conda environment setup
# The variable ${EBROOTANACONDA3} or ${EBROOTMINICONDA3}
# So use one of the following lines
# comes with the module load command
# source ${EBROOTANACONDA3}/etc/profile.d/conda.sh
source ${EBROOTMINICONDA3}/etc/profile.d/conda.sh

# Deactivate any spill-over environment from the login node
conda deactivate &>/dev/null

# Activate the environment by using the full path (not name)
# to the environment. The full path is listed if you do
# conda info --envs at the command prompt.
conda activate my_environment   # use full path to environment

# Explicitly set env- variable for python (numpy?)
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=/localscratch/$SLURM_JOB_ID # $LOCALSCRATCH

# Execute Python code
cd path_to_folder
python 03_pick_events.py
