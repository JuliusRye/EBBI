#!/bin/bash
#SBATCH --job-name="SC7_MHBD_MWPM_eta_sweep"
#SBATCH --partition=qist-fat
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julius.rye.boennelykke@nbi.ku.dk
#SBATCH -o results/thresshold/SC7_MHBD_MWPM_eta_sweep/logs/%x_%a.out
#SBATCH -e results/thresshold/SC7_MHBD_MWPM_eta_sweep/logs/%x_%a.err
#SBATCH --time=10:00:00
#SBATCH --array=1-200%26

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" results/thresshold/SC7_MHBD_MWPM_eta_sweep/params.csv)

IFS=',' read -r seed p eta <<< "$LINE"

srun uv run --no-sync ebbi-pymatching --seed $seed -p $p -eta $eta --save_path results/thresshold/SC7_MHBD_MWPM_eta_sweep/results/sweep_${SLURM_ARRAY_TASK_ID}.csv --noise_model MHBD_noise -s 100_000_000 -d 7 -r 7 -cds 0000000000000000000000000000000000000000000000000 0303030303030303030303030303030303030303030303030 0333333003030003030300030300030303000303003333330 -v
