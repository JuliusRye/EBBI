#!/bin/bash
#SBATCH --job-name="rs_SC3_HBD_MWPM_2_5x"
#SBATCH --partition=qist-fast
#SBATCH --ntasks=1
#SBATCH --mem=2G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julius.rye.boennelykke@nbi.ku.dk
#SBATCH -o results/rs/SC3_HBD_MWPM_2_5x/logs/%x_%a.out
#SBATCH -e results/rs/SC3_HBD_MWPM_2_5x/logs/%x_%a.err
#SBATCH --time=50:00:00
#SBATCH --array=1-25%25

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" results/rs/SC3_HBD_MWPM_2_5x/params.csv)

IFS=',' read -r seed N S <<< "$LINE"

srun python ./src/methods/rs.py --seed $seed -N $N -S $S --save_path results/rs/SC3_HBD_MWPM_2_5x/results/sweep_${SLURM_ARRAY_TASK_ID}.csv -E 1 -d 3 -r 3 --noise_model HBD_noise -p 0.0025 -eta 500 -c 0.1 -v
