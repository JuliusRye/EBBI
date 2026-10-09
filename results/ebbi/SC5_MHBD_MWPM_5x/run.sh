#!/bin/bash
#SBATCH --job-name="ebbi_SC5_MHBD_MWPM_5x"
#SBATCH --partition=qist-fast
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julius.rye.boennelykke@nbi.ku.dk
#SBATCH -o results/ebbi/SC5_MHBD_MWPM_5x/logs/%x_%a.out
#SBATCH -e results/ebbi/SC5_MHBD_MWPM_5x/logs/%x_%a.err
#SBATCH --time=5-00:00:00
#SBATCH --array=1-25%25

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" results/ebbi/SC5_MHBD_MWPM_5x/params.csv)

IFS=',' read -r seed N Ni M P i w S <<< "$LINE"

srun uv run --no-sync ebbi --seed $seed -N $N -Ni $Ni -M $M -P $P -i $i -w $w -S $S --save_path results/ebbi/SC5_MHBD_MWPM_5x/results/sweep_${SLURM_ARRAY_TASK_ID}.csv -E 1 -d 5 -r 5 --noise_model MHBD_noise -p 0.005 -eta 500 -c 0.1 -v
