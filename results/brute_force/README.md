
# Brute force search

Test all distance-3 surface code Clifford deformations.

## all_SC3_MHBD_MWPM_1x

```shell
uv run ebbi-job-brute-force --save_folder_name all_SC3_MHBD_MWPM_1x --cds_per_job 100 --parallel_jobs 25 --candidate_type PyMatching --candidate_args="-d 3 -r 3 --noise_model MHBD_noise -p 0.001 -eta 500 -s 100_000_000" --job_memory 2G --time_limit_per_job 10:00:00 --job_name all_SC3_MHBD_MWPM_1x --auto_run_slurm
```

## all_SC3_MHBD_MWPM_2_5x

```shell
uv run ebbi-job-brute-force --save_folder_name all_SC3_MHBD_MWPM_2_5x --cds_per_job 100 --parallel_jobs 25 --candidate_type PyMatching --candidate_args="-d 3 -r 3 --noise_model MHBD_noise -p 0.0025 -eta 500 -s 100_000_000" --job_memory 2G --time_limit_per_job 10:00:00 --job_name all_SC3_MHBD_MWPM_2_5x --auto_run_slurm
```

## all_SC3_MHBD_MWPM_5x

```shell
uv run ebbi-job-brute-force --save_folder_name all_SC3_MHBD_MWPM_5x --cds_per_job 100 --parallel_jobs 25 --candidate_type PyMatching --candidate_args="-d 3 -r 3 --noise_model MHBD_noise -p 0.005 -eta 500 -s 100_000_000" --job_memory 2G --time_limit_per_job 10:00:00 --job_name all_SC3_MHBD_MWPM_5x --auto_run_slurm
```

## all_SC3_HBD_MWPM_1x

```shell
uv run ebbi-job-brute-force --save_folder_name all_SC3_HBD_MWPM_1x --cds_per_job 100 --parallel_jobs 25 --candidate_type PyMatching --candidate_args="-d 3 -r 3 --noise_model HBD_noise -p 0.001 -eta 500 -s 100_000_000" --job_memory 2G --time_limit_per_job 10:00:00 --job_name all_SC3_HBD_MWPM_1x --auto_run_slurm
```

## all_SC3_HBD_MWPM_2_5x

```shell
uv run ebbi-job-brute-force --save_folder_name all_SC3_HBD_MWPM_2_5x --cds_per_job 100 --parallel_jobs 25 --candidate_type PyMatching --candidate_args="-d 3 -r 3 --noise_model HBD_noise -p 0.0025 -eta 500 -s 100_000_000" --job_memory 2G --time_limit_per_job 10:00:00 --job_name all_SC3_HBD_MWPM_2_5x --auto_run_slurm
```

## all_SC3_HBD_MWPM_5x

```shell
uv run ebbi-job-brute-force --save_folder_name all_SC3_HBD_MWPM_5x --cds_per_job 100 --parallel_jobs 25 --candidate_type PyMatching --candidate_args="-d 3 -r 3 --noise_model HBD_noise -p 0.005 -eta 500 -s 100_000_000" --job_memory 2G --time_limit_per_job 10:00:00 --job_name all_SC3_HBD_MWPM_5x --auto_run_slurm
```

-----------------------------------------------------------------------------------

## all_SC3_MHBD_BPOSD_1x

```shell
uv run ebbi-job-brute-force --save_folder_name all_SC3_MHBD_BPOSD_1x --cds_per_job 10 --parallel_jobs 25 --candidate_type BPOSD --candidate_args="-d 3 -r 3 --noise_model MHBD_noise -p 0.001 -eta 500 -s 10_000_000" --job_memory 2G --time_limit_per_job 10:00:00 --job_name all_SC3_MHBD_BPOSD_1x --auto_run_slurm
```

## all_SC3_HBD_BPOSD_1x

```shell
uv run ebbi-job-brute-force --save_folder_name all_SC3_HBD_BPOSD_1x --cds_per_job 10 --parallel_jobs 25 --candidate_type BPOSD --candidate_args="-d 3 -r 3 --noise_model HBD_noise -p 0.001 -eta 500 -s 10_000_000" --job_memory 2G --time_limit_per_job 10:00:00 --job_name all_SC3_HBD_BPOSD_1x --auto_run_slurm
```

-----------------------------------------------------------------------------------

# Extended

Clifford-deformations that generalize to large code distances

## ext_SC3_MHBD

```shell
uv run ebbi-job-brute-force --save_folder_name ext_SC3_MHBD --from_file results/brute_force/ext_SC3.txt --cds_per_job 1 --parallel_jobs 10 --candidate_args="-d 3 -r 3 --noise_model MHBD_noise -p 0.001 -eta 500 -s 1_000_000_000" --job_memory 2G --time_limit_per_job 10:00:00 --job_name ext_SC3_MHBD --auto_run_slurm
```

## ext_SC5_MHBD

```shell
uv run ebbi-job-brute-force --save_folder_name ext_SC5_MHBD --from_file results/brute_force/ext_SC5.txt --cds_per_job 1 --parallel_jobs 10 --candidate_args="-d 5 -r 5 --noise_model MHBD_noise -p 0.001 -eta 500 -s 1_000_000_000" --job_memory 3G --time_limit_per_job 10:00:00 --job_name ext_SC5_MHBD --auto_run_slurm
```

## ext_SC7_MHBD

```shell
uv run ebbi-job-brute-force --save_folder_name ext_SC7_MHBD --from_file results/brute_force/ext_SC7.txt --cds_per_job 1 --parallel_jobs 10 --candidate_args="-d 7 -r 7 --noise_model MHBD_noise -p 0.001 -eta 500 -s 1_000_000_000" --job_memory 4G --time_limit_per_job 10:00:00 --job_name ext_SC7_MHBD --auto_run_slurm
```

## ext_SC9_MHBD

```shell
uv run ebbi-job-brute-force --save_folder_name ext_SC9_MHBD --from_file results/brute_force/ext_SC9.txt --cds_per_job 1 --parallel_jobs 10 --candidate_args="-d 9 -r 9 --noise_model MHBD_noise -p 0.001 -eta 500 -s 1_000_000_000" --job_memory 6G --time_limit_per_job 10:00:00 --job_name ext_SC9_MHBD --auto_run_slurm
```

## ext_SC11_MHBD

```shell
uv run ebbi-job-brute-force --save_folder_name ext_SC11_MHBD --from_file results/brute_force/ext_SC11.txt --cds_per_job 1 --parallel_jobs 100 --candidate_args="-d 11 -r 11 --noise_model MHBD_noise -p 0.001 -eta 500 -s 1_000_000_000" --job_memory 10G --time_limit_per_job 25:00:00 --job_name ext_SC11_MHBD --auto_run_slurm
```

-----------------------------------------------------------------------------------

# Precise extimate for MHBD noise

## used_SC3_MHBD

```shell
uv run ebbi-job-brute-force --save_folder_name used_SC3_MHBD --from_file results/brute_force/used_SC3_MHBD.txt --cds_per_job 10 --parallel_jobs 50 --candidate_args="-d 3 -r 3 --noise_model MHBD_noise -p 0.001 -eta 500 -s 100_000_000" --job_memory 2G --time_limit_per_job 10:00:00 --job_name used_SC3_MHBD --auto_run_slurm
```

## used_SC5_MHBD

```shell
uv run ebbi-job-brute-force --save_folder_name used_SC5_MHBD --from_file results/brute_force/used_SC5_MHBD.txt --cds_per_job 10 --parallel_jobs 50 --candidate_args="-d 5 -r 5 --noise_model MHBD_noise -p 0.001 -eta 500 -s 100_000_000" --job_memory 4G --time_limit_per_job 20:00:00 --job_name used_SC5_MHBD --auto_run_slurm
```

-----------------------------------------------------------------------------------

# Precise extimate for HBD noise

## used_SC3_HBD

```shell
uv run ebbi-job-brute-force --save_folder_name used_SC3_HBD --from_file results/brute_force/used_SC3_HBD.txt --cds_per_job 10 --parallel_jobs 50 --candidate_args="-d 3 -r 3 --noise_model HBD_noise -p 0.001 -eta 500 -s 100_000_000" --job_memory 2G --time_limit_per_job 10:00:00 --job_name used_SC3_HBD --auto_run_slurm
```

## used_SC5_HBD

```shell
uv run ebbi-job-brute-force --save_folder_name used_SC5_HBD --from_file results/brute_force/used_SC5_HBD.txt --cds_per_job 10 --parallel_jobs 50 --candidate_args="-d 5 -r 5 --noise_model HBD_noise -p 0.001 -eta 500 -s 100_000_000" --job_memory 4G --time_limit_per_job 20:00:00 --job_name used_SC5_HBD --auto_run_slurm
```

-----------------------------------------------------------------------------------

# Decoder comparison for MHBD noise

The four well known Clifford-deformations listed in `decoder_d3.txt` and `decoder_d5.txt` re-run
with every decoder in `src/EBBI/candidate/`, to check how much of the logical error rate reported by
PyMatching is an artifact of the decoder (DEM decomposition, graphlike-error requirement) rather
than a property of the deformed code itself.

| Array task | d=3 | d=5 | Name |
| --- | --- | --- | --- |
| 1 | `000000000` | `0000000000000000000000000` | CSS |
| 2 | `030303030` | `0303030303030303030303030` | XZZX |
| 3 | `222222222` | `2222222222222222222222222` | XY |
| 4 | `033000330` | `0333300300030300030033330` | B-XZZX |

All the jobs below use `--cds_per_job 1`, so each Clifford-deformation is its own array task and the
four of them run in parallel. NOTE: a candidate only writes its row once every one of its shots is
done, so an array task that hits its time limit loses that Clifford-deformation entirely - the shot
counts and time limits are picked to leave a wide margin, but if a task does time out, lower `-s`
(or, for BP+OSD and Tesseract, the accuracy dial) rather than raising the wall clock indefinitely.

Everything runs on `qist-fat` except the CudaQ tensor-network decoder, which needs a GPU and so goes
to `qist-gpu` with `--gpus_per_job 1`. That decoder is distance-3 only: contracting the distance-5
tensor network is far too expensive to be worth running.

The operating point is `p = 0.001`, `eta = 500`, i.e. the same one as the `used_SC*` runs above, so
the results are directly comparable to them. The HBD counterparts of every job below are in the next
section; everything said here applies to those as well.

## dec_SC3_MHBD_MWPM

```shell
uv run ebbi-job-brute-force --save_folder_name dec_SC3_MHBD_MWPM --from_file results/brute_force/decoder_d3.txt --cds_per_job 1 --parallel_jobs 4 --candidate_type PyMatching --candidate_args="-d 3 -r 3 --noise_model MHBD_noise -p 0.001 -eta 500 -s 100_000_000" --job_memory 2G --time_limit_per_job 1:00:00 --job_name dec_SC3_MHBD_MWPM --partition qist-fat --auto_run_slurm
```

## dec_SC5_MHBD_MWPM

```shell
uv run ebbi-job-brute-force --save_folder_name dec_SC5_MHBD_MWPM --from_file results/brute_force/decoder_d5.txt --cds_per_job 1 --parallel_jobs 4 --candidate_type PyMatching --candidate_args="-d 5 -r 5 --noise_model MHBD_noise -p 0.001 -eta 500 -s 100_000_000" --job_memory 4G --time_limit_per_job 5:00:00 --job_name dec_SC5_MHBD_MWPM --partition qist-fat --auto_run_slurm
```

## dec_SC3_MHBD_BM

```shell
uv run ebbi-job-brute-force --save_folder_name dec_SC3_MHBD_BM --from_file results/brute_force/decoder_d3.txt --cds_per_job 1 --parallel_jobs 4 --candidate_type BeliefMatching --candidate_args="-d 3 -r 3 --noise_model MHBD_noise -p 0.001 -eta 500 -s 100_000_000 --max_shot_size 100_000 --bm_max_bp_iters 20" --job_memory 2G --time_limit_per_job 12:00:00 --job_name dec_SC3_MHBD_BM --partition qist-fat --auto_run_slurm
```

## dec_SC5_MHBD_BM

```shell
uv run ebbi-job-brute-force --save_folder_name dec_SC5_MHBD_BM --from_file results/brute_force/decoder_d5.txt --cds_per_job 1 --parallel_jobs 4 --candidate_type BeliefMatching --candidate_args="-d 5 -r 5 --noise_model MHBD_noise -p 0.001 -eta 500 -s 100_000_000 --max_shot_size 100_000 --bm_max_bp_iters 20" --job_memory 4G --time_limit_per_job 2-00:00:00 --job_name dec_SC5_MHBD_BM --partition qist-fat --auto_run_slurm
```

## dec_SC3_MHBD_BPOSD

```shell
uv run ebbi-job-brute-force --save_folder_name dec_SC3_MHBD_BPOSD --from_file results/brute_force/decoder_d3.txt --cds_per_job 1 --parallel_jobs 4 --candidate_type BPOSD --candidate_args="-d 3 -r 3 --noise_model MHBD_noise -p 0.001 -eta 500 -s 100_000_000 --max_shot_size 10_000 --bposd_osd_method osd_cs --bposd_osd_order 60" --job_memory 4G --time_limit_per_job 12:00:00 --job_name dec_SC3_MHBD_BPOSD --partition qist-fat --auto_run_slurm
```

## dec_SC5_MHBD_BPOSD

```shell
uv run ebbi-job-brute-force --save_folder_name dec_SC5_MHBD_BPOSD --from_file results/brute_force/decoder_d5.txt --cds_per_job 1 --parallel_jobs 4 --candidate_type BPOSD --candidate_args="-d 5 -r 5 --noise_model MHBD_noise -p 0.001 -eta 500 -s 100_000_000 --max_shot_size 10_000 --bposd_osd_method osd_cs --bposd_osd_order 60" --job_memory 8G --time_limit_per_job 5-00:00:00 --job_name dec_SC5_MHBD_BPOSD --partition qist-fat --auto_run_slurm
```

## dec_SC3_MHBD_TESS

```shell
uv run ebbi-job-brute-force --save_folder_name dec_SC3_MHBD_TESS --from_file results/brute_force/decoder_d3.txt --cds_per_job 1 --parallel_jobs 4 --candidate_type Tesseract --candidate_args="-d 3 -r 3 --noise_model MHBD_noise -p 0.001 -eta 500 -s 100_000_000 --max_shot_size 10_000 --tesseract_det_beam 5 --tesseract_pqlimit 200_000" --job_memory 4G --time_limit_per_job 12:00:00 --job_name dec_SC3_MHBD_TESS --partition qist-fat --auto_run_slurm
```

## dec_SC5_MHBD_TESS

```shell
uv run ebbi-job-brute-force --save_folder_name dec_SC5_MHBD_TESS --from_file results/brute_force/decoder_d5.txt --cds_per_job 1 --parallel_jobs 4 --candidate_type Tesseract --candidate_args="-d 5 -r 5 --noise_model MHBD_noise -p 0.001 -eta 500 -s 100_000_000 --max_shot_size 10_000 --tesseract_det_beam 5 --tesseract_pqlimit 200_000" --job_memory 8G --time_limit_per_job 2-00:00:00 --job_name dec_SC5_MHBD_TESS --partition qist-fat --auto_run_slurm
```

## dec_SC3_MHBD_TN

Tensor-network decoder on the GPU partition. There is deliberately no distance-5 counterpart.

```shell
uv run ebbi-job-brute-force --save_folder_name dec_SC3_MHBD_TN --from_file results/brute_force/decoder_d3.txt --cds_per_job 1 --parallel_jobs 1 --candidate_type CudaQTensor --candidate_args="-d 3 -r 3 --noise_model MHBD_noise -p 0.001 -eta 500 -s 100_000_000 --max_shot_size 10_000" --job_memory 8G --time_limit_per_job 2-00:00:00 --job_name dec_SC3_MHBD_TN --partition qist-gpu --gpus_per_job 1 --auto_run_slurm
```

-----------------------------------------------------------------------------------

# Decoder comparison for HBD noise

The same nine jobs as in the section above, with `--noise_model HBD_noise` instead of
`--noise_model MHBD_noise`. See that section for the Clifford-deformation to array-task mapping, the
partition choice and the note on time limits.

## dec_SC3_HBD_MWPM

```shell
uv run ebbi-job-brute-force --save_folder_name dec_SC3_HBD_MWPM --from_file results/brute_force/decoder_d3.txt --cds_per_job 1 --parallel_jobs 4 --candidate_type PyMatching --candidate_args="-d 3 -r 3 --noise_model HBD_noise -p 0.001 -eta 500 -s 100_000_000" --job_memory 2G --time_limit_per_job 1:00:00 --job_name dec_SC3_HBD_MWPM --partition qist-fat --auto_run_slurm
```

## dec_SC5_HBD_MWPM

```shell
uv run ebbi-job-brute-force --save_folder_name dec_SC5_HBD_MWPM --from_file results/brute_force/decoder_d5.txt --cds_per_job 1 --parallel_jobs 4 --candidate_type PyMatching --candidate_args="-d 5 -r 5 --noise_model HBD_noise -p 0.001 -eta 500 -s 100_000_000" --job_memory 4G --time_limit_per_job 5:00:00 --job_name dec_SC5_HBD_MWPM --partition qist-fat --auto_run_slurm
```

## dec_SC3_HBD_BM

```shell
uv run ebbi-job-brute-force --save_folder_name dec_SC3_HBD_BM --from_file results/brute_force/decoder_d3.txt --cds_per_job 1 --parallel_jobs 4 --candidate_type BeliefMatching --candidate_args="-d 3 -r 3 --noise_model HBD_noise -p 0.001 -eta 500 -s 100_000_000 --max_shot_size 100_000 --bm_max_bp_iters 20" --job_memory 2G --time_limit_per_job 12:00:00 --job_name dec_SC3_HBD_BM --partition qist-fat --auto_run_slurm
```

## dec_SC5_HBD_BM

```shell
uv run ebbi-job-brute-force --save_folder_name dec_SC5_HBD_BM --from_file results/brute_force/decoder_d5.txt --cds_per_job 1 --parallel_jobs 4 --candidate_type BeliefMatching --candidate_args="-d 5 -r 5 --noise_model HBD_noise -p 0.001 -eta 500 -s 100_000_000 --max_shot_size 100_000 --bm_max_bp_iters 20" --job_memory 4G --time_limit_per_job 2-00:00:00 --job_name dec_SC5_HBD_BM --partition qist-fat --auto_run_slurm
```

## dec_SC3_HBD_BPOSD

```shell
uv run ebbi-job-brute-force --save_folder_name dec_SC3_HBD_BPOSD --from_file results/brute_force/decoder_d3.txt --cds_per_job 1 --parallel_jobs 4 --candidate_type BPOSD --candidate_args="-d 3 -r 3 --noise_model HBD_noise -p 0.001 -eta 500 -s 100_000_000 --max_shot_size 10_000 --bposd_osd_method osd_cs --bposd_osd_order 60" --job_memory 4G --time_limit_per_job 12:00:00 --job_name dec_SC3_HBD_BPOSD --partition qist-fat --auto_run_slurm
```

## dec_SC5_HBD_BPOSD

```shell
uv run ebbi-job-brute-force --save_folder_name dec_SC5_HBD_BPOSD --from_file results/brute_force/decoder_d5.txt --cds_per_job 1 --parallel_jobs 4 --candidate_type BPOSD --candidate_args="-d 5 -r 5 --noise_model HBD_noise -p 0.001 -eta 500 -s 100_000_000 --max_shot_size 10_000 --bposd_osd_method osd_cs --bposd_osd_order 60" --job_memory 8G --time_limit_per_job 5-00:00:00 --job_name dec_SC5_HBD_BPOSD --partition qist-fat --auto_run_slurm
```

## dec_SC3_HBD_TESS

```shell
uv run ebbi-job-brute-force --save_folder_name dec_SC3_HBD_TESS --from_file results/brute_force/decoder_d3.txt --cds_per_job 1 --parallel_jobs 4 --candidate_type Tesseract --candidate_args="-d 3 -r 3 --noise_model HBD_noise -p 0.001 -eta 500 -s 100_000_000 --max_shot_size 10_000 --tesseract_det_beam 5 --tesseract_pqlimit 200_000" --job_memory 4G --time_limit_per_job 12:00:00 --job_name dec_SC3_HBD_TESS --partition qist-fat --auto_run_slurm
```

## dec_SC5_HBD_TESS

```shell
uv run ebbi-job-brute-force --save_folder_name dec_SC5_HBD_TESS --from_file results/brute_force/decoder_d5.txt --cds_per_job 1 --parallel_jobs 4 --candidate_type Tesseract --candidate_args="-d 5 -r 5 --noise_model HBD_noise -p 0.001 -eta 500 -s 100_000_000 --max_shot_size 10_000 --tesseract_det_beam 5 --tesseract_pqlimit 200_000" --job_memory 8G --time_limit_per_job 2-00:00:00 --job_name dec_SC5_HBD_TESS --partition qist-fat --auto_run_slurm
```

## dec_SC3_HBD_TN

Tensor-network decoder on the GPU partition. There is deliberately no distance-5 counterpart.

```shell
uv run ebbi-job-brute-force --save_folder_name dec_SC3_HBD_TN --from_file results/brute_force/decoder_d3.txt --cds_per_job 1 --parallel_jobs 1 --candidate_type CudaQTensor --candidate_args="-d 3 -r 3 --noise_model HBD_noise -p 0.001 -eta 500 -s 100_000_000 --max_shot_size 10_000" --job_memory 8G --time_limit_per_job 2-00:00:00 --job_name dec_SC3_HBD_TN --partition qist-gpu --gpus_per_job 1 --auto_run_slurm
```
