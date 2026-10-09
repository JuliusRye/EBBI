# Distance-3 surface codes with MHBD noise and PyMatching decoder

The multiplier in the folder name indicates how much $p$ has been increased during the search stage of the method.

> **Note:** the `run.sh` files inside the job folders below were generated *before* the repo
> was packaged as `EBBI`, so they still invoke the old flat layout (`python ./src/...`). They
> are kept unchanged as provenance for the results they produced; newly generated jobs use the
> installed console scripts instead.

## SC3_MHBD_MWPM_1x

```shell
uv run ebbi-job-rs --save_folder_name SC3_MHBD_MWPM_1x -s 1_000_000 -S 100_000_000_000 -R 25 --rs_args="-E 1 -d 3 -r 3 --noise_model MHBD_noise -p 0.001 -eta 500 -c 0.1" --job_memory 2G --time_limit_per_job 50:00:00 --job_name="rs_SC3_MHBD_MWPM_1x" --auto_run_slurm --partition qist-fast
```

## SC3_MHBD_MWPM_2_5x

```shell
uv run ebbi-job-rs --save_folder_name SC3_MHBD_MWPM_2_5x -s 400_000 -S 100_000_000_000 -R 25 --rs_args="-E 1 -d 3 -r 3 --noise_model MHBD_noise -p 0.0025 -eta 500 -c 0.1" --job_memory 2G --time_limit_per_job 50:00:00 --job_name="rs_SC3_MHBD_MWPM_2_5x" --auto_run_slurm --partition qist-fast
```

## SC3_MHBD_MWPM_5x

```shell
uv run ebbi-job-rs --save_folder_name SC3_MHBD_MWPM_5x -s 200_000 -S 100_000_000_000 -R 25 --rs_args="-E 1 -d 3 -r 3 --noise_model MHBD_noise -p 0.005 -eta 500 -c 0.1" --job_memory 2G --time_limit_per_job 50:00:00 --job_name="rs_SC3_MHBD_MWPM_5x" --auto_run_slurm --partition qist-fast --parallel_jobs 5
```

-----------------------------------------------------------------------------------

# Distance-5 surface codes with MHBD noise and PyMatching decoder

The multiplier in the folder name indicates how much $p$ has been increased during the search stage of the method.

## SC5_MHBD_MWPM_1x

```shell
uv run ebbi-job-rs --save_folder_name SC5_MHBD_MWPM_1x -s 10_000_000 -S 100_000_000_000 -R 25 --rs_args="-E 1 -d 5 -r 5 --noise_model MHBD_noise -p 0.001 -eta 500 -c 0.1" --job_memory 2G --time_limit_per_job 10-00:00:00 --job_name="rs_SC5_MHBD_MWPM_1x" --auto_run_slurm --partition qist-fast
```

## SC5_MHBD_MWPM_2_5x

```shell
uv run ebbi-job-rs --save_folder_name SC5_MHBD_MWPM_2_5x -s 4_000_000 -S 100_000_000_000 -R 25 --rs_args="-E 1 -d 5 -r 5 --noise_model MHBD_noise -p 0.0025 -eta 500 -c 0.1" --job_memory 2G --time_limit_per_job 10-00:00:00 --job_name="rs_SC5_MHBD_MWPM_2_5x" --auto_run_slurm --partition qist-fast
```

## SC5_MHBD_MWPM_5x

```shell
uv run ebbi-job-rs --save_folder_name SC5_MHBD_MWPM_5x -s 2_000_000 -S 100_000_000_000 -R 25 --rs_args="-E 1 -d 5 -r 5 --noise_model MHBD_noise -p 0.005 -eta 500 -c 0.1" --job_memory 2G --time_limit_per_job 10-00:00:00 --job_name="rs_SC5_MHBD_MWPM_5x" --auto_run_slurm --partition qist-fast
```
-----------------------------------------------------------------------------------

# Distance-3 surface codes with HBD noise and PyMatching decoder

The multiplier in the folder name indicates how much $p$ has been increased during the search stage of the method.

## SC3_HBD_MWPM_1x

```shell
uv run ebbi-job-rs --save_folder_name SC3_HBD_MWPM_1x -s 1_000_000 -S 100_000_000_000 -R 25 --rs_args="-E 1 -d 3 -r 3 --noise_model HBD_noise -p 0.001 -eta 500 -c 0.1" --job_memory 2G --time_limit_per_job 50:00:00 --job_name="rs_SC3_HBD_MWPM_1x" --auto_run_slurm --partition qist-fast
```

## SC3_HBD_MWPM_2_5x

```shell
uv run ebbi-job-rs --save_folder_name SC3_HBD_MWPM_2_5x -s 400_000 -S 100_000_000_000 -R 25 --rs_args="-E 1 -d 3 -r 3 --noise_model HBD_noise -p 0.0025 -eta 500 -c 0.1" --job_memory 2G --time_limit_per_job 50:00:00 --job_name="rs_SC3_HBD_MWPM_2_5x" --auto_run_slurm --partition qist-fast
```

## SC3_HBD_MWPM_5x

```shell
uv run ebbi-job-rs --save_folder_name SC3_HBD_MWPM_5x -s 200_000 -S 100_000_000_000 -R 25 --rs_args="-E 1 -d 3 -r 3 --noise_model HBD_noise -p 0.005 -eta 500 -c 0.1" --job_memory 2G --time_limit_per_job 50:00:00 --job_name="rs_SC3_HBD_MWPM_5x" --auto_run_slurm --partition qist-fast --parallel_jobs 5
```

-----------------------------------------------------------------------------------

# Distance-5 surface codes with HBD noise and PyMatching decoder

The multiplier in the folder name indicates how much $p$ has been increased during the search stage of the method.

## SC5_HBD_MWPM_1x

```shell
uv run ebbi-job-rs --save_folder_name SC5_HBD_MWPM_1x -s 10_000_000 -S 100_000_000_000 -R 25 --rs_args="-E 1 -d 5 -r 5 --noise_model HBD_noise -p 0.001 -eta 500 -c 0.1" --job_memory 2G --time_limit_per_job 10-00:00:00 --job_name="rs_SC5_HBD_MWPM_1x" --auto_run_slurm --partition qist-fast
```

## SC5_HBD_MWPM_2_5x

```shell
uv run ebbi-job-rs --save_folder_name SC5_HBD_MWPM_2_5x -s 4_000_000 -S 100_000_000_000 -R 25 --rs_args="-E 1 -d 5 -r 5 --noise_model HBD_noise -p 0.0025 -eta 500 -c 0.1" --job_memory 2G --time_limit_per_job 10-00:00:00 --job_name="rs_SC5_HBD_MWPM_2_5x" --auto_run_slurm --partition qist-fast
```

## SC5_HBD_MWPM_5x

```shell
uv run ebbi-job-rs --save_folder_name SC5_HBD_MWPM_5x -s 2_000_000 -S 100_000_000_000 -R 25 --rs_args="-E 1 -d 5 -r 5 --noise_model HBD_noise -p 0.005 -eta 500 -c 0.1" --job_memory 2G --time_limit_per_job 10-00:00:00 --job_name="rs_SC5_HBD_MWPM_5x" --auto_run_slurm --partition qist-fast
```
