#!/bin/bash
#SBATCH --job-name=noise_gen     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=zatkins@princeton.edu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

### set environment ###
module use --append /home/mfh2/shared/modulefiles
module load so_stack
module load openmpi/gcc/2.0.2/64

# we'll manage our own pixell version in conda
module unload pixell

# to be able to use enlib -- TODO: upgrade to module
export PYTHONPATH="${PYTHONPATH}:/home/zatkins/repos/other"
export ENLIB_COMP=hippo

# activate conda
conda activate cmb_princeton

qid="s18_03"
downgrade=4
lmax=5000
notes="mnms"

python -u /home/zatkins/repos/ACTCollaboration/mnms/scripts/noise_gen_wav.py \
--qid ${qid} --lmax ${lmax} --downgrade ${downgrade} --notes ${notes} --mask-version v1 --mask-name BN_bottomcut
