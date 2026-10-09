#!/bin/bash
#SBATCH --job-name="dec_SC5_HBD_MWPM"
#SBATCH --partition=qist-fat
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julius.rye.boennelykke@nbi.ku.dk
#SBATCH -o results/brute_force/dec_SC5_HBD_MWPM/logs/%x_%a.out
#SBATCH -e results/brute_force/dec_SC5_HBD_MWPM/logs/%x_%a.err
#SBATCH --time=5:00:00
#SBATCH --array=1-4%4

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" results/brute_force/dec_SC5_HBD_MWPM/params.csv)

IFS=',' read -r seed cds <<< "$LINE"

srun python ./src/candidate/pymatching_candidate.py --seed $seed -cds $cds --save_path results/brute_force/dec_SC5_HBD_MWPM/results/sweep_${SLURM_ARRAY_TASK_ID}.csv -d 5 -r 5 --noise_model HBD_noise -p 0.001 -eta 500 -s 100_000_000 -v
