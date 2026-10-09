import os, sys
import pandas as pd
from zipfile import ZipFile
from pathlib import Path

paths = sys.argv[1:]

if "-h" in paths or "--help" in paths or len(paths) == 0:
    print("Usage: python compress.py <path_to_job_folder_1> <path_to_job_folder_2> ...")
    print("This script will compress the results of the EBBI/RS method runs in the specified job folders. For each job folder, it will create a results.csv file that combines all the individual csv result files, a checkpoints.csv file that combines all the individual checkpoint csv files, and a raw.zip file that contains all the original result and log files (and deletes them from the folder). The results.csv and checkpoints.csv files will have an additional column called source_file that indicates the original file each row came from.")
    sys.exit(0)

for job_folder_path in paths:
    if not os.path.isdir(job_folder_path):
        print(f"Error: {job_folder_path} is not a valid directory.")
        continue
    print(f"Compressing results in {job_folder_path}...")

    results_folder = os.path.join(job_folder_path, 'results')
    logs_folder = os.path.join(job_folder_path, 'logs')
    csv_results_save_path = os.path.join(job_folder_path, 'results.csv')
    csv_checkpoints_save_path = os.path.join(job_folder_path, 'checkpoints.csv')
    zip_save_path = os.path.join(job_folder_path, 'raw.zip')

    # Create combined csv file for results
    csv_results_files = [os.path.join(results_folder, csv_file) for csv_file in os.listdir(results_folder) if not csv_file.endswith("checkpoints.csv") and csv_file.endswith(".csv")]
    dfs = []
    for csv_file in csv_results_files:
        try:
            df = pd.read_csv(csv_file, dtype={"clifford_deformation": str}).assign(source_file=csv_file)
            dfs.append(df)
        except Exception as e: # Empty files or files with wrong format will be skipped
            continue
    if len(dfs) > 0:
        pd.concat(dfs, ignore_index=True).to_csv(csv_results_save_path, index=False)
        print(f"    Combined {len(dfs)} csv result files into results.csv")
    else:
        print("    No valid csv results files found to combine.")

    # Create combined csv file for checkpoints
    csv_checkpoints_files = [os.path.join(results_folder, csv_file) for csv_file in os.listdir(results_folder) if csv_file.endswith("checkpoints.csv")]
    dfs = []
    for csv_file in csv_checkpoints_files:
        try:
            df = pd.read_csv(csv_file, dtype={"clifford_deformation": str}).assign(source_file=csv_file)
            dfs.append(df)
        except Exception as e: # Empty files or files with wrong format will be skipped
            continue
    if len(dfs) > 0:
        pd.concat(dfs, ignore_index=True).to_csv(csv_checkpoints_save_path, index=False)
        print(f"    Combined {len(dfs)} csv checkpoint files into checkpoints.csv")
    else:
        print("    No valid csv checkpoints files found to combine.")

    # Create zip file
    results_folder = Path(results_folder)
    logs_folder = Path(logs_folder)
    with ZipFile(zip_save_path, "w") as zipf:
        for file in results_folder.rglob("*"):
            if file.is_file():
                zipf.write(file, arcname=file.relative_to(job_folder_path))
                file.unlink()
        os.rmdir(results_folder)
        for file in logs_folder.rglob("*"):
            if file.is_file():
                zipf.write(file, arcname=file.relative_to(job_folder_path))
                file.unlink()
        os.rmdir(logs_folder)
    print(f"    Compressed original csv result and log files into raw.zip and deleted the original files.")
