#!/bin/bash
#SBATCH --job-name=noise_gen     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=zatkins@princeton.edu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module use --append /home/mfh2/shared/modulefiles
module load so_stack
module load openmpi/gcc/2.0.2/64
conda activate cmb_princeton

qid="s18_03"
downgrade=4
notes="newcode4"
width_deg=4
height_deg=4
delta_ell_smooth=400

srun -u python -u /home/zatkins/repos/ACTCollaboration/mnms/scripts/noise_gen.py \
--qid ${qid} --downgrade ${downgrade} --width-deg ${width_deg} --height-deg ${height_deg} --delta-ell-smooth ${delta_ell_smooth} --notes ${notes} \
--no-bin-apod --mask-version v1 --mask-name BN_bottomcut