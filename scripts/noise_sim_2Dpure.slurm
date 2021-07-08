#!/bin/bash
#SBATCH --job-name=noisesim   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=zatkins@princeton.edu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module use --append /home/mfh2/shared/modulefiles
module load so_stack
module load openmpi/gcc/2.0.2/64
conda activate cmb_princeton

qid="s18_04"
downgrade=4
notes="newcode4_filter_pure"
width_deg=4
height_deg=4
delta_ell_smooth=400
ell_large_small_split=0
ell_taper_width=0

srun -u python -u /home/zatkins/repos/ACTCollaboration/mnms/scripts/noise_sim_2Dpure.py --maps-start 1 --maps-end 2 --qid ${qid} --downgrade ${downgrade} --width-deg ${width_deg} --height-deg ${height_deg} --delta-ell-smooth ${delta_ell_smooth} --ell-large-small-split ${ell_large_small_split} --ell-taper-width ${ell_taper_width} --notes ${notes} --no-bin-apod --mask-version v1 --mask-name BN_bottomcut 
python -u /home/zatkins/repos/ACTCollaboration/mnms/scripts/make_coadd_from_splits.py --maps-start 1 --maps-end 2 --qid ${qid} --downgrade ${downgrade} --width-deg ${width_deg} --height-deg ${height_deg} --delta-ell-smooth ${delta_ell_smooth} --ell-large-small-split ${ell_large_small_split} --ell-taper-width ${ell_taper_width} --notes ${notes} --no-bin-apod --mask-version v1 --mask-name BN_bottomcut
