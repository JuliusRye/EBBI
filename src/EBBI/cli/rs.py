"""Command-line entry point: run the Random Search (RS) baseline method."""

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
from EBBI.methods.rs import RandomSearch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Arguments for the Random Search (RS) method.',
        formatter_class=lambda prog: argparse.MetavarTypeHelpFormatter(prog, max_help_position=60)
    )
    arg_manager.qec_args(parser)
    arg_manager.general_args(parser)
    arg_manager.method_args(parser)
    arg_manager.special_candidate_args(parser)
    # RS specific args
    parser.add_argument('-N', '--candidates', type=int, default=None, help='[Required or --shots_per_candidate] Number of candidates to probe during the RS method.')
    parser.add_argument('-s', '--shots_per_candidate', type=int, default=None, help='[Required or --candidates] Number of shots to take per candidate during the RS method.')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.use_corr_mwpm:
        assert args.candidate_class == 'PyMatching', "The argument --use_corr_mwpm is only applicable when --candidate_class is set to PyMatching. Please set it accordingly."

    print_args(args)

    # Validate that either candidates or shots_per_candidate is provided, but not both
    if args.candidates is None and args.shots_per_candidate is None:
        raise ValueError("You must provide either --candidates or --shots_per_candidate.")
    if args.candidates is not None and args.shots_per_candidate is not None:
        raise ValueError("You cannot provide both --candidates and --shots_per_candidate. Please provide only one of these arguments.")

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
        print("Running RS:")

    seeds = derive_seeds(args, args.experiments)

    with open(save_path, "w") as f:
        # Header
        f.write("seed,clifford_deformation,shots,method,candidates,QECC,n,k,d,noise-model,p,eta\n")
        t_start = perf_counter()
        for i in range(args.experiments):
            t_elapsed = perf_counter() - t_start
            if args.verbose:
                print(f"    {i}/{args.experiments}: time used: {time_print(t_elapsed)}, remaining time: {time_print(t_elapsed / i * (args.experiments - i)) if i > 0 else "N/A"}", flush=True)
            if args.checkpoint_exponent_base is None:
                rs = RandomSearch(
                    seed=seeds[i], # Use a different seed for each run
                    num_data_qubits=qecc_x.num_data_qubits,
                    active_deformation=jnp.array(args.active_deformations),
                    candidate_class=candidate_class,
                    candidate_kwargs=candidate_kwargs,
                    max_shot_size=args.max_shot_size,
                    total_shots=args.total_shots,
                    num_candidates_to_test=args.candidates if args.candidates is not None else args.total_shots // args.shots_per_candidate,
                    rounds=args.rounds,
                    noise_type=noise_type,
                    verbose=args.method_verbose
                )
            else:
                rs = RandomSearch(
                    seed=seeds[i], # Use a different seed for each run
                    num_data_qubits=qecc_x.num_data_qubits,
                    active_deformation=jnp.array(args.active_deformations),
                    candidate_class=candidate_class,
                    candidate_kwargs=candidate_kwargs,
                    max_shot_size=args.max_shot_size,
                    total_shots=args.total_shots,
                    num_candidates_to_test=args.candidates if args.candidates is not None else args.total_shots // args.shots_per_candidate,
                    rounds=args.rounds,
                    noise_type=noise_type,
                    checkpoint_exponent_base=args.checkpoint_exponent_base,
                    check_point_path=f"{save_path[:-4]}_exp_{i}_checkpoints.csv",
                    verbose=args.method_verbose,
                    force_override=args.force_override
                )
            found_clifford_deformation = cd_to_str(rs.run_method().clifford_deformation)
            f.write(f"{seeds[i]},{found_clifford_deformation},{args.total_shots},RS,{args.candidates},{args.qec},{num_data_qubits},{num_logical_qubits},{args.distance},{args.noise_model},{args.error_probability},{args.error_bias}\n")
            f.flush() # Ensure that results are written to the file after each experiment
        t_elapsed = perf_counter() - t_start
        if args.verbose:
            print(f"    {i+1}/{args.experiments}: time used: {time_print(t_elapsed)}, remaining time: {time_print(0)}", flush=True)


if __name__ == "__main__":
    main()
