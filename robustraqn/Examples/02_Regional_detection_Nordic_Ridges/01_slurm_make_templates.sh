#!/bin/bash

# Job name:
#SBATCH --job-name=EQCST
#SBATCH --output 01_templates_%A_%a.out

# Project:
#SBATCH --account=XXXXXX

# Wall time limit:
#SBATCH --time=00-02:00:00

## Limiting factor for template creation is most likely the memory when loading in data
## for one full day on each core. For this problem, use at most 16 workers when 362 GB
## memory available. Request 32 cores, but limit in python script to the right number of
## workers.
#SBATCH --cpus-per-task=32


# worker with full day of data can take up to 15.5 GB memory
#SBATCH --mem=362G  # can run max. 20-22 cores per node with 362 GB

#####   ARRAY JOB TASK       ########
#SBATCH --array=0-7          # 8-10  nodes is an ok limit; otherwise too much reading in parallel

##SBATCH --partition=normal
#SBATCH --partition=bigmem

# export SLURM_MEM=178G
export SLURM_MEM=362G
# export SLURM_MEM=1600G     #  for nodes with a lot of memory


## Set up job environment:
# set -o errexit  # Exit the script on any error
# set -o nounset  # Treat any unset variables as an error
# Problem:  bash: GDAL_DATA: unbound variable

module --quiet purge  # Reset the modules to the system default

#######################################################################

# load the Anaconda3
ml load Miniconda3/4.9.2
ml load FFTW/3.3.9-intel-2021a
ml load CUDA/11.3.1
#ml load Arm-Forge/21.1

# Set the ${PS1} (needed in the source of the Anaconda environment)
# export PS1=\$

# Source the conda environment setup
# The variable ${EBROOTANACONDA3} or ${EBROOTMINICONDA3}
source ${EBROOTMINICONDA3}/etc/profile.d/conda.sh

# Deactivate any spill-over environment from the login node
conda deactivate &>/dev/null

# Activate the environment by using the full path (not name)
# to the environment. The full path is listed if you do
# conda info --envs at the command prompt.
conda activate /path/to/my/environment

# Explicitly set env- variable for python (numpy?)
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
export TMPDIR=/localscratch/$SLURM_JOB_ID # $LOCALSCRATCH

# Execute Python code
cd my_folder
/usr/bin/time -v python 01_make_templates.py
