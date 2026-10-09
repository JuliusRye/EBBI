#!/bin/bash
#SBATCH --job-name="SC9_MHBD_MWPM_eta_sweep"
#SBATCH --partition=qist-fat
#SBATCH --ntasks=1
#SBATCH --mem=6GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julius.rye.boennelykke@nbi.ku.dk
#SBATCH -o results/thresshold/SC9_MHBD_MWPM_eta_sweep/logs/%x_%a.out
#SBATCH -e results/thresshold/SC9_MHBD_MWPM_eta_sweep/logs/%x_%a.err
#SBATCH --time=10:00:00
#SBATCH --array=1-2000%50

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" results/thresshold/SC9_MHBD_MWPM_eta_sweep/params.csv)

IFS=',' read -r seed p eta <<< "$LINE"

srun uv run --no-sync ebbi-pymatching --seed $seed -p $p -eta $eta --save_path results/thresshold/SC9_MHBD_MWPM_eta_sweep/results/sweep_${SLURM_ARRAY_TASK_ID}.csv --noise_model MHBD_noise -s 100_000_000 -d 9 -r 9 -cds 000000000000000000000000000000000000000000000000000000000000000000000000000000000 030303030303030303030303030303030303030303030303030303030303030303030303030303030 033333333003030300030303030003030300030303030003030300030303030003030300333333330 -v
