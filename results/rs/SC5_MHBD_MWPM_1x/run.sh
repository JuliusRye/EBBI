#!/bin/bash
#SBATCH --job-name="rs_SC5_MHBD_MWPM_1x"
#SBATCH --partition=qist-fast
#SBATCH --ntasks=1
#SBATCH --mem=2G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julius.rye.boennelykke@nbi.ku.dk
#SBATCH -o results/rs/SC5_MHBD_MWPM_1x/logs/%x_%a.out
#SBATCH -e results/rs/SC5_MHBD_MWPM_1x/logs/%x_%a.err
#SBATCH --time=10-00:00:00
#SBATCH --array=1-25%25

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" results/rs/SC5_MHBD_MWPM_1x/params.csv)

IFS=',' read -r seed N S <<< "$LINE"

srun python ./src/methods/rs.py --seed $seed -N $N -S $S --save_path results/rs/SC5_MHBD_MWPM_1x/results/sweep_${SLURM_ARRAY_TASK_ID}.csv -E 1 -d 5 -r 5 --noise_model MHBD_noise -p 0.001 -eta 500 -c 0.1 -v
