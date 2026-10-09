"""Command-line entry point: run the Evolution Based Best-arm Identification framework."""

import argparse
from time import perf_counter

import jax.numpy as jnp

from EBBI.candidate.str_cd_convertion import cd_to_str
from EBBI.cli import arg_manager
from EBBI.cli._common import (
    build_candidate_class,
    build_qecc,
    derive_seeds,
    print_args,
    resolve_save_path,
    time_print,
)
from EBBI.methods.ebbi import EvolutionBasedBestArmIdentification


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Arguments for the Evolution Based Best-arm Identification (EBBi) framework.',
        formatter_class=lambda prog: argparse.MetavarTypeHelpFormatter(prog, max_help_position=60)
    )
    arg_manager.qec_args(parser)
    arg_manager.general_args(parser)
    arg_manager.method_args(parser)
    arg_manager.special_candidate_args(parser)
    # EBBI specific args
    parser.add_argument('-N', '--candidates', type=int, required=True, help='[Required] Number of simultaneous candidates for EBBI.')
    parser.add_argument('-Ni', '--candidates_initially', type=int, default=None, help='(default=N) Number of simultaneous candidates for EBBI.')
    parser.add_argument('-M', '--mutation_probability', type=float, default=None, help='(default=0.43/N) Probability of changing Clifford-deformation on a data qubit.')
    parser.add_argument('-P', '--prune_threshold', type=float, default=0.05, help='(default=0.05) Threshold for pruning candidates based on their posterior distributions.')
    parser.add_argument('-i', '--improvement_factor', type=float, default=1.5, help='(default=1.5) How much the variance of the LER of a candidate should be improved each time it is selected for more sampling.')
    parser.add_argument('-w', '--warmup', type=int, default=500, help='(default=500) The minimum number of errors a candidate must observe during warmup to ensure that its posterior distribution is localized enough to avoid immediately being pruned.')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.use_corr_mwpm:
        assert args.candidate_class == 'PyMatching', "The argument --use_corr_mwpm is only applicable when --candidate_class is set to PyMatching. Please set it accordingly."

    print_args(args)
    save_path = resolve_save_path(args)
    qecc_x, qecc_z, noise_type = build_qecc(args)
    num_data_qubits = qecc_x.num_data_qubits
    num_logical_qubits = qecc_x.num_logical_qubits

    if args.verbose:
        print("Finished\nInitializing decoder...", flush=True, end=' ')
    candidate_class, candidate_kwargs = build_candidate_class(args, qecc_x, qecc_z)

    if args.verbose:
        print("Finished", flush=True)
        print('\n' + 50*'-' + '\n')
        print("Running EBBI:")

    seeds = derive_seeds(args, args.experiments)

    with open(save_path, "w") as f:
        # Header
        f.write("seed,clifford_deformation,shots,method,num_candidates,mutation_rate,prune_threshold,QECC,n,k,d,noise-model,p,eta\n")
        t_start = perf_counter()
        for i in range(args.experiments):
            t_elapsed = perf_counter() - t_start
            if args.verbose:
                print(f"    {i}/{args.experiments}: time used: {time_print(t_elapsed)}, remaining time: {time_print(t_elapsed / i * (args.experiments - i)) if i > 0 else "N/A"}", flush=True)
            if args.checkpoint_exponent_base is None:
                ebbi = EvolutionBasedBestArmIdentification(
                    seed=seeds[i], # Use a different seed for each run
                    num_data_qubits=qecc_x.num_data_qubits,
                    active_deformation=jnp.array(args.active_deformations),
                    candidate_class=candidate_class,
                    candidate_kwargs=candidate_kwargs,
                    max_shot_size=args.max_shot_size,
                    total_shots=args.total_shots,
                    num_simultaneous_candidates=args.candidates,
                    num_initial_candidates=args.candidates_initially if args.candidates_initially is not None else args.candidates,
                    improvement_factor=args.improvement_factor,
                    mutation_probability=args.mutation_probability if args.mutation_probability is not None else 0.43/args.candidates,
                    prune_threshold=args.prune_threshold,
                    candidate_min_errors_for_warmup=args.warmup,
                    rounds=args.rounds,
                    noise_type=noise_type,
                    verbose=args.method_verbose
                )
            else:
                ebbi = EvolutionBasedBestArmIdentification(
                    seed=seeds[i], # Use a different seed for each run
                    num_data_qubits=qecc_x.num_data_qubits,
                    active_deformation=jnp.array(args.active_deformations),
                    candidate_class=candidate_class,
                    candidate_kwargs=candidate_kwargs,
                    max_shot_size=args.max_shot_size,
                    total_shots=args.total_shots,
                    num_simultaneous_candidates=args.candidates,
                    num_initial_candidates=args.candidates_initially if args.candidates_initially is not None else args.candidates,
                    improvement_factor=args.improvement_factor,
                    mutation_probability=args.mutation_probability if args.mutation_probability is not None else 0.43/args.candidates,
                    prune_threshold=args.prune_threshold,
                    candidate_min_errors_for_warmup=args.warmup,
                    rounds=args.rounds,
                    noise_type=noise_type,
                    checkpoint_exponent_base=args.checkpoint_exponent_base,
                    check_point_path=f"{save_path[:-4]}_exp_{i}_checkpoints.csv",
                    verbose=args.method_verbose,
                    force_override=args.force_override
                )
            found_clifford_deformation = cd_to_str(ebbi.run_method().clifford_deformation)
            f.write(f"{seeds[i]},{found_clifford_deformation},{args.total_shots},EBBI,{args.candidates},{args.mutation_probability},{args.prune_threshold},{args.qec},{num_data_qubits},{num_logical_qubits},{args.distance},{args.noise_model},{args.error_probability},{args.error_bias}\n")
            f.flush() # Ensure that results are written to the file after each experiment
        t_elapsed = perf_counter() - t_start
        if args.verbose:
            print(f"    {i+1}/{args.experiments}: time used: {time_print(t_elapsed)}, remaining time: {time_print(0)}", flush=True)


if __name__ == "__main__":
    main()
