"""Command-line entry point: estimate LERs for given Clifford-deformations with BP+OSD."""

import argparse
from time import perf_counter

import EBBI.candidate.str_cd_convertion as strcd
from EBBI.candidate.bposd_candidate import BPOSDCandidate
from EBBI.cli import arg_manager
from EBBI.cli._common import (
    build_qecc,
    derive_seeds,
    print_args,
    resolve_save_path,
    time_print,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Estimate the logical error rate of a Clifford-deformation using BP+OSD (belief propagation with ordered-statistics decoding post-processing).',
        formatter_class=lambda prog: argparse.MetavarTypeHelpFormatter(prog, max_help_position=60)
    )
    arg_manager.qec_args(parser)
    arg_manager.candidate_args(parser)
    arg_manager.general_args(parser)
    arg_manager.special_candidate_args(parser, include=['bposd_max_bp_iters', 'bposd_bp_method', 'bposd_osd_order', 'bposd_osd_method'])
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print_args(args)
    save_path = resolve_save_path(args)

    qecc_x, qecc_z, noise_type = build_qecc(args)
    num_data_qubits = qecc_x.num_data_qubits
    num_logical_qubits = qecc_x.num_logical_qubits

    if args.verbose:
        print("Finished", flush=True)
        print('\n' + 50*'-' + '\n')
        print("Estimating LERs:")

    seeds = derive_seeds(args, len(args.clifford_deformations))

    with open(save_path, 'w') as f:
        # Header (the first 17 columns are byte-identical to the other candidates; the
        # BP+OSD specific settings are appended at the end)
        f.write("seed,clifford_deformation,alpha,beta,x-err,y-err,z-err,no-err,shots,method,QECC,n,k,d,noise-model,p,eta,max_bp_iters,bp_method,osd_order,osd_method\n")
        t_start = perf_counter()
        for i, cd_str in enumerate(args.clifford_deformations):
            t_elapsed = perf_counter() - t_start
            clifford_deformation = strcd.str_to_cd(cd_str)
            if args.verbose:
                print(f"    {i}/{len(args.clifford_deformations)}: time used: {time_print(t_elapsed)}, remaining time: {time_print(t_elapsed / i * (len(args.clifford_deformations) - i)) if i > 0 else "N/A"}", flush=True)
            candidate = BPOSDCandidate(
                clifford_deformation, seed=seeds[i], qecc_x=qecc_x, qecc_z=qecc_z,
                rounds=args.rounds, noise_type=noise_type,
                max_bp_iters=args.bposd_max_bp_iters, bp_method=args.bposd_bp_method,
                osd_order=args.bposd_osd_order, osd_method=args.bposd_osd_method,
            )
            candidate.sample(args.shots, max_shot_size=args.max_shot_size)
            if args.verbose:
                ler = candidate.expected_ler
                unc = candidate.variance ** 0.5
                print(f"        CD = {clifford_deformation} : LER = {ler:.2E} ± {unc:.2E}", flush=True)
            cd_str = strcd.cd_to_str(clifford_deformation)
            f.write(f"{seeds[i]},{cd_str},{candidate.alpha},{candidate.beta},{candidate.logical_x},{candidate.logical_y},{candidate.logical_z},{candidate.logical_I},{args.shots},bposd,{args.qec},{num_data_qubits},{num_logical_qubits},{args.distance},{args.noise_model},{args.error_probability},{args.error_bias},{args.bposd_max_bp_iters},{args.bposd_bp_method},{args.bposd_osd_order},{args.bposd_osd_method}\n")
        t_elapsed = perf_counter() - t_start
        if args.verbose:
            print(f"    {i+1}/{len(args.clifford_deformations)}: time used: {time_print(t_elapsed)}, remaining time: {time_print(0)}", flush=True)


if __name__ == "__main__":
    main()
