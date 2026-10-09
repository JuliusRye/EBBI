#!/bin/bash
#SBATCH --job-name="dec_SC3_HBD_BPOSD"
#SBATCH --partition=qist-fat
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julius.rye.boennelykke@nbi.ku.dk
#SBATCH -o results/brute_force/dec_SC3_HBD_BPOSD/logs/%x_%a.out
#SBATCH -e results/brute_force/dec_SC3_HBD_BPOSD/logs/%x_%a.err
#SBATCH --time=12:00:00
#SBATCH --array=1-4%4

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" results/brute_force/dec_SC3_HBD_BPOSD/params.csv)

IFS=',' read -r seed cds <<< "$LINE"

srun python ./src/candidate/bposd_candidate.py --seed $seed -cds $cds --save_path results/brute_force/dec_SC3_HBD_BPOSD/results/sweep_${SLURM_ARRAY_TASK_ID}.csv -d 3 -r 3 --noise_model HBD_noise -p 0.001 -eta 500 -s 100_000_000 --max_shot_size 10_000 --bposd_osd_method osd_cs --bposd_osd_order 60 -v
