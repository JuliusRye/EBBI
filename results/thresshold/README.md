# MHBD $p$ sweep

## SC3_MHBD_MWPM_p_sweep

```Shell
uv run ebbi-job-threshold --save_folder_name SC3_MHBD_MWPM_p_sweep --job_memory 2GB --job_name SC3_MHBD_MWPM_p_sweep --time_limit_per_job 1:00:00 --parallel_jobs 20 --repeats 1 -p 0.022 0.015 0.01 0.0068 0.0046 0.0031 0.0022 0.0015 0.001 0.00068 0.00046 0.00031 0.00022 0.00015 0.0001 -eta 500 --candidate_args="--noise_model MHBD_noise -s 100_000_000 -d 3 -r 3 -cds 000000000 030303030 033000330" --auto_run --partition qist-fat
```

## SC5_MHBD_MWPM_p_sweep

```Shell
uv run ebbi-job-threshold --save_folder_name SC5_MHBD_MWPM_p_sweep --job_memory 3GB --job_name SC5_MHBD_MWPM_p_sweep --time_limit_per_job 10:00:00 --parallel_jobs 26 --repeats 1 -p 0.022 0.015 0.01 0.0068 0.0046 0.0031 0.0022 0.0015 0.001 0.00068 0.00046 0.00031 0.00022 0.00015 0.0001 -eta 500 --candidate_args="--noise_model MHBD_noise -s 100_000_000 -d 5 -r 5 -cds 0000000000000000000000000 0303030303030303030303030 0333300300030300030033330" --auto_run --partition qist-fat
```

## SC7_MHBD_MWPM_p_sweep

```Shell
uv run ebbi-job-threshold --save_folder_name SC7_MHBD_MWPM_p_sweep --job_memory 4GB --job_name SC7_MHBD_MWPM_p_sweep --time_limit_per_job 20:00:00 --parallel_jobs 26 --repeats 1 -p 0.022 0.015 0.01 0.0068 0.0046 0.0031 0.0022 0.0015 0.001 0.00068 0.00046 0.00031 0.00022 0.00015 0.0001 -eta 500 --candidate_args="--noise_model MHBD_noise -s 100_000_000 -d 7 -r 7 -cds 0000000000000000000000000000000000000000000000000 0303030303030303030303030303030303030303030303030 0333333003030003030300030300030303000303003333330" --auto_run --partition qist-fat
```

## SC9_MHBD_MWPM_p_sweep

```Shell
uv run ebbi-job-threshold --save_folder_name SC9_MHBD_MWPM_p_sweep --job_memory 6GB --job_name SC9_MHBD_MWPM_p_sweep --time_limit_per_job 50:00:00 --parallel_jobs 50 --repeats 1 -p 0.022 0.015 0.01 0.0068 0.0046 0.0031 0.0022 0.0015 0.001 0.00068 0.00046 0.00031 0.00022 0.00015 0.0001 -eta 500 --candidate_args="--noise_model MHBD_noise -s 100_000_000 -d 9 -r 9 -cds 000000000000000000000000000000000000000000000000000000000000000000000000000000000 030303030303030303030303030303030303030303030303030303030303030303030303030303030 033333333003030300030303030003030300030303030003030300030303030003030300333333330" --auto_run --partition qist-fat
```

# MHBD $\eta$ sweep

## SC3_MHBD_MWPM_eta_sweep

```Shell
uv run ebbi-job-threshold --save_folder_name SC3_MHBD_MWPM_eta_sweep --job_memory 2GB --job_name SC3_MHBD_MWPM_eta_sweep --time_limit_per_job 1:00:00 --parallel_jobs 20 --repeats 1 -p 0.001 -eta 0.5 1 2 5 10 20 50 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000 --candidate_args="--noise_model MHBD_noise -s 100_000_000 -d 3 -r 3 -cds 000000000 030303030 033000330" --auto_run --partition qist-fat
```

## SC5_MHBD_MWPM_eta_sweep

```Shell
uv run ebbi-job-threshold --save_folder_name SC5_MHBD_MWPM_eta_sweep --job_memory 3GB --job_name SC5_MHBD_MWPM_eta_sweep --time_limit_per_job 5:00:00 --parallel_jobs 26 --repeats 1 -p 0.001 -eta 0.5 1 2 5 10 20 50 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000 --candidate_args="--noise_model MHBD_noise -s 100_000_000 -d 5 -r 5 -cds 0000000000000000000000000 0303030303030303030303030 0333300300030300030033330" --auto_run --partition qist-fat
```

## SC7_MHBD_MWPM_eta_sweep

```Shell
uv run ebbi-job-threshold --save_folder_name SC7_MHBD_MWPM_eta_sweep --job_memory 4GB --job_name SC7_MHBD_MWPM_eta_sweep --time_limit_per_job 10:00:00 --parallel_jobs 26 --repeats 10 -p 0.001 -eta 0.5 1 2 5 10 20 50 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000 --candidate_args="--noise_model MHBD_noise -s 100_000_000 -d 7 -r 7 -cds 0000000000000000000000000000000000000000000000000 0303030303030303030303030303030303030303030303030 0333333003030003030300030300030303000303003333330" --auto_run --partition qist-fat
```

## SC9_MHBD_MWPM_eta_sweep

```Shell
uv run ebbi-job-threshold --save_folder_name SC9_MHBD_MWPM_eta_sweep --job_memory 6GB --job_name SC9_MHBD_MWPM_eta_sweep --time_limit_per_job 10:00:00 --parallel_jobs 50 --repeats 100 -p 0.001 -eta 0.5 1 2 5 10 20 50 100 200 500 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000 --candidate_args="--noise_model MHBD_noise -s 100_000_000 -d 9 -r 9 -cds 000000000000000000000000000000000000000000000000000000000000000000000000000000000 030303030303030303030303030303030303030303030303030303030303030303030303030303030 033333333003030300030303030003030300030303030003030300030303030003030300333333330" --auto_run --partition qist-fat
```
