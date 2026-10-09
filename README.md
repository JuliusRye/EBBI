# EBBI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23239737.svg)](https://doi.org/10.5281/zenodo.23239737)

Evolution-Based Best-arm Identification (EBBI) for discovering Clifford-deformed surface codes.

The repository is an installable Python package (`EBBI`, living in [src/EBBI/](src/EBBI/)) plus a
set of console scripts.

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/). From the repository root:

```shell
uv sync
```

This creates `.venv/` and installs the package in editable mode together with all dependencies.

Two dependencies are marked Linux-only (`cudaq-qec` and `tesseract-decoder`) and are skipped on
Windows as those packages are not supported on windows.

All results come from runs on a linux cluster.

## Console scripts

Run any of these with `uv run <command> --help` for the full argument list.

Commands are intended to be run from the repo root to work properly.

### Search methods

| Command | Description |
|---|---|
| `ebbi` | Run the Evolution-Based Best-arm Identification search. |
| `ebbi-rs` | Run the random search baseline. |

### Single-decoder LER estimation

Estimate logical error rates for explicitly given Clifford-deformations:

| Command | Decoder |
|---|---|
| `ebbi-pymatching` | PyMatching (MWPM), optionally correlated MWPM |
| `ebbi-beliefmatching` | Belief propagation + MWPM |
| `ebbi-bposd` | Belief propagation + ordered-statistics decoding |
| `ebbi-tesseract` | Tesseract (Linux only) |
| `ebbi-cudaqtensor` | CUDA-Q tensor-network decoder (Linux only) |

### SLURM job generators

Each of these writes a job folder under `results/<kind>/<--save_folder_name>/` containing
`run.sh`, `params.csv`, `config.json`, `logs/` and `results/`. Use `--jobs_root` to write
somewhere else.

| Command | Output root |
|---|---|
| `ebbi-job` | `results/ebbi` |
| `ebbi-job-rs` | `results/rs` |
| `ebbi-job-brute-force` | `results/brute_force` |
| `ebbi-job-threshold` | `results/thresshold` |

## Example: a local run

```shell
uv run ebbi-pymatching -cds 000000000 -s 10000 -d 3 -r 3 --save_path results/pymatching_d3.csv -v
```

```shell
uv run ebbi --seed 42 -N 4 -S 20000 -E 1 -d 3 -r 3 --noise_model HBD_noise -p 0.001 -eta 500 --save_path results/ebbi_d3.csv -v
```

## Cluster workflow

Job submission happens from a Linux login node:

1. Sync the environment once, on the login node:

   ```shell
   uv sync
   ```

2. Generate a job folder (see [results/ebbi/README.md](results/ebbi/README.md),
   [results/rs/README.md](results/rs/README.md),
   [results/brute_force/README.md](results/brute_force/README.md) and
   [results/thresshold/README.md](results/thresshold/README.md) for the exact invocations used
   for the published runs):

   ```shell
   uv run ebbi-job --save_folder_name SC3_HBD_MWPM_1x -N 10 -S 10_000_000_000 -R 25 --ebbi_args="-E 1 -d 3 -r 3 --noise_model HBD_noise -p 0.001 -eta 500 -c 0.1" --job_memory 10G --time_limit_per_job 2:00:00 --job_name ebbi_SC3_HBD_MWPM_1x
   ```

3. Submit it:

   ```shell
   sbatch results/ebbi/SC3_HBD_MWPM_1x/run.sh
   ```

   (All four generators accept `--auto_run_slurm` to submit the job for you right after generating it.)

   The cluster-specific settings are opt-in: pass `--partition <name>` to choose a SLURM partition
   and `--mail_user <address>` (with `--mail_type`, default `ALL`) for e-mail notifications. When
   they are left out, no `--partition`/`--mail-*` lines are written and the cluster defaults apply.
   The `run.sh` files under `results/` were generated for our cluster and are kept as is.

The generated `run.sh` calls the console scripts through `uv run --no-sync`, so the environment
must already be built by step 1 — compute nodes never resolve or download anything.

## Repository layout

| Path | Contents |
|---|---|
| [src/EBBI/qecc/](src/EBBI/qecc/) | Stim circuits for Clifford-deformed QEC codes and the noise models. |
| [src/EBBI/candidate/](src/EBBI/candidate/) | One candidate class per decoder; sampling and LER estimation. |
| [src/EBBI/methods/](src/EBBI/methods/) | The EBBI and Random Search algorithms. |
| [src/EBBI/cli/](src/EBBI/cli/) | Console-script entry points, shared argument parsing and the SLURM job generators. |
| [results/](results/) | Generated job folders and the collected result CSVs. |
| [figures/](figures/) | Plotting notebooks and the produced figures. |

## Paper figures and tables

The notebooks in [figures/](figures/) are run from inside that folder; each reads the collected
`results.csv`/`checkpoints.csv` files under [results/](results/) and writes its panels to
`figures/outputs/<notebook>/`. The multi-panel figures in
[figures/outputs/final/](figures/outputs/final/) were assembled from these panels in a separate
LaTeX document that is not part of this repository.

| Paper | Final figure | Notebook | Panels / output | Data |
|---|---|---|---|---|
| Fig. 1 | `final/EBBI.pdf` | [EBBI_figure_graphs.ipynb](figures/outputs/EBBI_fig_comp/EBBI_figure_graphs.ipynb), [Clifford-deformation_clooud.ipynb](figures/outputs/EBBI_fig_comp/Clifford-deformation_clooud.ipynb) | `outputs/EBBI_fig_comp/` | Illustrative only |
| Fig. 2 | `final/results.pdf` | [comparizon.ipynb](figures/comparizon.ipynb) | `outputs/comparizon/SC{3,5}_{HBD,MHBD}_MWPM.pdf`, `legend.pdf` | `results/ebbi/*`, `results/rs/*`; reference $P_L$ from `brute_force/all_SC3_{HBD,MHBD}` (d=3) and `brute_force/used_SC5_{HBD,MHBD}` (d=5) |
| Fig. 3 | `final/generalization.pdf` | [generalization.ipynb](figures/generalization.ipynb) | `outputs/generalization/generalization.pdf` (panel c) | `brute_force/ext_SC{3,5,7,9,11}_MHBD` |
| Fig. 4 | `final/circuits.pdf` | — | — | — |
| Fig. 5, 6 | — | [all_SC3.ipynb](figures/all_SC3.ipynb) | `outputs/all_SC3/{HBD,MHBD}_distribution.pdf` | `brute_force/all_SC3_{HBD,MHBD}_MWPM_1x` |
| Fig. 7, 8 | — | [all_SC3.ipynb](figures/all_SC3.ipynb) | `outputs/all_SC3/{HBD,MHBD}_p_search_correlation.pdf` | `brute_force/all_SC3_{HBD,MHBD}_MWPM_{1x,2_5x,5x}` |
| Fig. 9 | — | [all_SC3.ipynb](figures/all_SC3.ipynb) | `outputs/all_SC3/neighborhood_correlation.pdf` | `brute_force/all_SC3_{HBD,MHBD}_MWPM_1x` |
| Fig. 10 | — | [all_SC3.ipynb](figures/all_SC3.ipynb) | `outputs/all_SC3/decoder_correlation.pdf` | `brute_force/all_SC3_{HBD,MHBD}_{MWPM,BPOSD}_1x` |
| Fig. 11 | `final/results_ci.pdf` | [comparizon.ipynb](figures/comparizon.ipynb) | `outputs/comparizon/SC{3,5}_{HBD,MHBD}_MWPM_p_0_{1,25,5}.pdf` | `results/ebbi/*`, `results/rs/*`; reference $P_L$ from `brute_force/used_SC{3,5}_{HBD,MHBD}` |
| Fig. 12 | — | [tables.ipynb](figures/tables.ipynb) | `outputs/tables/decoder_ranking.pdf` | `brute_force/dec_*` |
| Table II | — | — | — | Shots per CD from `results/rs/*/config.json` |
| Table III | — | — | — | PyMatching rows of Table V, rounded |
| Table IV | — | [comparizon.ipynb](figures/comparizon.ipynb) | `outputs/comparizon/ebbi_convergence_table.tex` | `results/ebbi/*`, `results/rs/*` |
| Table V | — | [tables.ipynb](figures/tables.ipynb) | `outputs/tables/decoder_table.tex` | `brute_force/dec_*` |
| Table VI | — | [tables.ipynb](figures/tables.ipynb) | `outputs/tables/runtime_table.tex` | `brute_force/dec_*` |

Paths in the *Data* column are relative to [results/](results/). The remaining notebook outputs
(e.g. [noise_model_sweep.ipynb](figures/noise_model_sweep.ipynb), reading `results/thresshold/`,
and `final/comparizon.pdf`) are not used in the paper.

# Naming convention

| Paper naming | code base naming |
| ------------ | ---------------- |
| HBD | HBD |
| HBD-SI1000 | MHBD |
| random search | rs |
| distance-3 surface code | SC3 |
| distance-5 surface code | SC5 |

## Citation

If you use this software, please cite it via its Zenodo record:
[doi:10.5281/zenodo.23239737](https://doi.org/10.5281/zenodo.23239737). Full citation metadata
is in [CITATION.cff](CITATION.cff).

## License

Licensed under the [Apache License, Version 2.0](LICENSE). Copyright (c) 2026 University of
Copenhagen.
