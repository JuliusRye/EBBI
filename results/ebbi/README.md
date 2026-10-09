# Distance-3 surface codes with MHBD noise and PyMatching decoder

The multiplier in the folder name indicates how much $p$ has been increased during the search stage of the method.

## SC3_MHBD_MWPM_1x

```shell
uv run ebbi-job --save_folder_name SC3_MHBD_MWPM_1x -N 10 -S 10_000_000_000 -R 25 -w 500 --ebbi_args="-E 1 -d 3 -r 3 --noise_model MHBD_noise -p 0.001 -eta 500 -c 0.1" --job_memory 10G --time_limit_per_job 2:00:00 --job_name="ebbi_SC3_MHBD_MWPM_1x" --auto_run_slurm --partition qist-fast
```

## SC3_MHBD_MWPM_2_5x

```shell
uv run ebbi-job --save_folder_name SC3_MHBD_MWPM_2_5x -N 10 -S 10_000_000_000 -R 25 -w 500 --ebbi_args="-E 1 -d 3 -r 3 --noise_model MHBD_noise -p 0.0025 -eta 500 -c 0.1" --job_memory 10G --time_limit_per_job 2:00:00 --job_name="ebbi_SC3_MHBD_MWPM_2_5x" --auto_run_slurm --partition qist-fast
```

## SC3_MHBD_MWPM_5x

```shell
uv run ebbi-job --save_folder_name SC3_MHBD_MWPM_5x -N 10 -S 10_000_000_000 -R 25 -w 500 --ebbi_args="-E 1 -d 3 -r 3 --noise_model MHBD_noise -p 0.005 -eta 500 -c 0.1" --job_memory 10G --time_limit_per_job 2:00:00 --job_name="ebbi_SC3_MHBD_MWPM_5x" --auto_run_slurm --partition qist-fast
```

-----------------------------------------------------------------------------------

# Distance-5 surface codes with MHBD noise and PyMatching decoder using W=500 instead of 50

The multiplier in the folder name indicates how much $p$ has been increased during the search stage of the method.

## SC5_MHBD_MWPM_1x

```shell
uv run ebbi-job --save_folder_name SC5_MHBD_MWPM_1x -N 10 -S 100_000_000_000 -R 25 -w 500 --ebbi_args="-E 1 -d 5 -r 5 --noise_model MHBD_noise -p 0.001 -eta 500 -c 0.1" --job_memory 20G --time_limit_per_job 5-00:00:00 --job_name="ebbi_SC5_MHBD_MWPM_1x" --auto_run_slurm --partition qist-fast --parallel_jobs 25
```

## SC5_MHBD_MWPM_2_5x

```shell
uv run ebbi-job --save_folder_name SC5_MHBD_MWPM_2_5x -N 10 -S 100_000_000_000 -R 25 -w 500 --ebbi_args="-E 1 -d 5 -r 5 --noise_model MHBD_noise -p 0.0025 -eta 500 -c 0.1" --job_memory 20G --time_limit_per_job 5-00:00:00 --job_name="ebbi_SC5_MHBD_MWPM_2_5x" --auto_run_slurm --partition qist-fast --parallel_jobs 25
```

## SC5_MHBD_MWPM_5x

```shell
uv run ebbi-job --save_folder_name SC5_MHBD_MWPM_5x -N 10 -S 100_000_000_000 -R 25 -w 500 --ebbi_args="-E 1 -d 5 -r 5 --noise_model MHBD_noise -p 0.005 -eta 500 -c 0.1" --job_memory 20G --time_limit_per_job 5-00:00:00 --job_name="ebbi_SC5_MHBD_MWPM_5x" --auto_run_slurm --partition qist-fast --parallel_jobs 25
```

-----------------------------------------------------------------------------------

# Distance-3 surface codes with HBD noise and PyMatching decoder

The multiplier in the folder name indicates how much $p$ has been increased during the search stage of the method.

## SC3_HBD_MWPM_1x

```shell
uv run ebbi-job --save_folder_name SC3_HBD_MWPM_1x -N 10 -S 10_000_000_000 -R 25 -w 500 --ebbi_args="-E 1 -d 3 -r 3 --noise_model HBD_noise -p 0.001 -eta 500 -c 0.1" --job_memory 10G --time_limit_per_job 2:00:00 --job_name="ebbi_SC3_HBD_MWPM_1x" --auto_run_slurm --partition qist-fast
```

## SC3_HBD_MWPM_2_5x

```shell
uv run ebbi-job --save_folder_name SC3_HBD_MWPM_2_5x -N 10 -S 10_000_000_000 -R 25 -w 500 --ebbi_args="-E 1 -d 3 -r 3 --noise_model HBD_noise -p 0.0025 -eta 500 -c 0.1" --job_memory 10G --time_limit_per_job 2:00:00 --job_name="ebbi_SC3_HBD_MWPM_2_5x" --auto_run_slurm --partition qist-fast
```

## SC3_HBD_MWPM_5x

```shell
uv run ebbi-job --save_folder_name SC3_HBD_MWPM_5x -N 10 -S 10_000_000_000 -R 25 -w 500 --ebbi_args="-E 1 -d 3 -r 3 --noise_model HBD_noise -p 0.005 -eta 500 -c 0.1" --job_memory 10G --time_limit_per_job 2:00:00 --job_name="ebbi_SC3_HBD_MWPM_5x" --auto_run_slurm --partition qist-fast
```

-----------------------------------------------------------------------------------

# Distance-5 surface codes with HBD noise and PyMatching decoder using W=500 instead of 50

The multiplier in the folder name indicates how much $p$ has been increased during the search stage of the method.

## SC5_HBD_MWPM_1x

```shell
uv run ebbi-job --save_folder_name SC5_HBD_MWPM_1x -N 10 -S 100_000_000_000 -R 25 -w 500 --ebbi_args="-E 1 -d 5 -r 5 --noise_model HBD_noise -p 0.001 -eta 500 -c 0.1" --job_memory 20G --time_limit_per_job 5-00:00:00 --job_name="ebbi_SC5_HBD_MWPM_1x" --auto_run_slurm --partition qist-fast --parallel_jobs 25
```

## SC5_HBD_MWPM_2_5x

```shell
uv run ebbi-job --save_folder_name SC5_HBD_MWPM_2_5x -N 10 -S 100_000_000_000 -R 25 -w 500 --ebbi_args="-E 1 -d 5 -r 5 --noise_model HBD_noise -p 0.0025 -eta 500 -c 0.1" --job_memory 20G --time_limit_per_job 5-00:00:00 --job_name="ebbi_SC5_HBD_MWPM_2_5x" --auto_run_slurm --partition qist-fast --parallel_jobs 25
```

## SC5_HBD_MWPM_5x

```shell
uv run ebbi-job --save_folder_name SC5_HBD_MWPM_5x -N 10 -S 100_000_000_000 -R 25 -w 500 --ebbi_args="-E 1 -d 5 -r 5 --noise_model HBD_noise -p 0.005 -eta 500 -c 0.1" --job_memory 20G --time_limit_per_job 5-00:00:00 --job_name="ebbi_SC5_HBD_MWPM_5x" --auto_run_slurm --partition qist-fast --parallel_jobs 25
```
