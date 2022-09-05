#!/bin/bash

# Job name:
#SBATCH --job-name=EQCS
#SBATCH --output 02_detect_%A_%a.out

# Project:
#SBATCH --account=XXXXXX

# Probably not necessary to define number of nodes for the array tasks - job should figure this out itself
#SBATCH --ntasks-per-node=1

#####   ARRAY JOB TASK       ########
#SBATCH --array=0-144 	     # each array job task takes one month of data

#####   CPU NODES, NORMAL:   ########
##SBATCH --partition=normal
##SBATCH --cpus-per-task=40  # 40 or 52
##SBATCH --mem=178G           #178G
##SBATCH --time=01-00:00:00

#####   GPU NODES:           ########
#SBATCH --partition=accel
#SBATCH --cpus-per-task=24
#SBATCH --mem=364G               # at least 100 GB
#SBATCH --gpus=4                 # 2500 templates per run use at least 14.3/16.3 GB GPU RAM
#SBATCH --time=03-00:00:00       # Single month on 4 GPUs, 15000 templates - ca. 55 hrs / 2.2 days

#SBATCH --gres=localscratch:200G
##### Comparing CPU to GPU nodes, running big detection problem on GPUs is 2.5 times quicker and 4.2 times more cost efficient (Sigma2 - Saga cluster)

## Set up job environment:
# set -o errexit  # Exit the script on any error
# set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default

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
source ${EBROOTMINICONDA3}/etc/profile.d/conda.sh

# Deactivate any spill-over environment from the login node
conda deactivate &>/dev/null

# Activate the environment by using the full path (not name)
# to the environment. The full path is listed if you do
# conda info --envs at the command prompt.
conda activate my_environment

# Explicitly set env- variable for python (numpy?)
export NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
if [[ -v SLURM_JOB_ID ]]; then
  export TMPDIR=/localscratch/$SLURM_JOB_ID/TMP
  mkdir $TMPDIR
fi
export TMP=$TMPDIR
export TEMP=$TMPDIR


# Execute Python code
cd /cluster/projects/nn9861k/Runs/Ridge/Detection
/usr/bin/time -v python 02_detect_events.py
