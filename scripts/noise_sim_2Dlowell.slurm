#!/bin/bash
#SBATCH --job-name=noise_sim         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=zatkins@princeton.edu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module use --append /home/mfh2/shared/modulefiles
module load so_stack
module load openmpi/gcc/2.0.2/64
conda activate cmb_princeton

srun -u python -u /home/zatkins/repos/ACTCollaboration/S18_S19_prelims/scripts/noise_sim_2Dlowell.py --nsims 2 --split 0 1 --qid s18_04 --downgrade 4 --notes newcode4 #--no-bin-apod --mask-version v1 --mask-name BN_bottomcut
