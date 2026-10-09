"""Helpers shared by the EBBI command-line entry points."""

import os

from jax import random


def time_print(duration_sec: float) -> str:
    hours = duration_sec // 3600
    minutes = (duration_sec % 3600) // 60
    seconds = duration_sec % 60
    return f"{hours:2.0f}h {minutes:2.0f}m {seconds:2.0f}s"


def print_args(args) -> None:
    """Echo the parsed arguments when running verbosely."""
    if args.verbose:
        print("Parsed arguments:")
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")
        print('\n' + 50*'-' + '\n')


def resolve_save_path(args) -> str:
    """Append .csv if missing and refuse to clobber an existing file."""
    save_path = args.save_path if args.save_path.endswith('.csv') else args.save_path + '.csv'
    if os.path.exists(save_path) and not args.force_override:
        raise ValueError(
            f"File {save_path} already exists. Please choose another name or delete the "
            "existing file to avoid overwriting previous results."
        )
    return save_path


def build_qecc(args):
    """Build the noise model and the X/Z basis code pair. Returns (qecc_x, qecc_z, noise_type)."""
    import EBBI.qecc.noise_models as noise_models

    if args.verbose:
        print("Initializing noise-model...", flush=True, end=' ')

    noise_model, noise_type = noise_models.get_noise_model(
        args.noise_model, args.error_probability, args.error_bias
    )

    if args.verbose:
        print("Finisehd\nInitializing QEC code...", flush=True, end=' ')

    match args.qec:
        case 'SurfaceCode':
            from EBBI.qecc.surface_code import SurfaceCode
            qecc_x = SurfaceCode(distance=args.distance, basis="X", noise_model=noise_model)
            qecc_z = SurfaceCode(distance=args.distance, basis="Z", noise_model=noise_model)
        case _:
            raise ValueError(
                f"Unknown quantum error-correcting code: {args.qec}, options=[SurfaceCode]"
            )
    return qecc_x, qecc_z, noise_type


def derive_seeds(args, n: int):
    """One seed per experiment / per Clifford-deformation, from --seed or --seeds."""
    if args.seeds is None:
        return random.randint(random.key(args.seed), (n,), 0, 1E31-1)
    assert len(args.seeds) == n, (
        f"The number of seeds provided in --seeds ({len(args.seeds)}) must match the number "
        f"of runs ({n})."
    )
    return args.seeds


def build_candidate_class(args, qecc_x, qecc_z):
    """Resolve --candidate_class to (class, kwargs). Imports lazily, one decoder per branch."""
    match args.candidate_class:
        case 'PyMatching':
            from EBBI.candidate.pymatching_candidate import PyMatchingCandidate
            return PyMatchingCandidate, {
                "qecc_x": qecc_x,
                "qecc_z": qecc_z,
                "use_corr_mwpm": args.use_corr_mwpm,
            }
        case 'BPOSD':
            from EBBI.candidate.bposd_candidate import BPOSDCandidate
            return BPOSDCandidate, {
                "qecc_x": qecc_x,
                "qecc_z": qecc_z,
                "max_bp_iters": args.bposd_max_bp_iters,
                "bp_method": args.bposd_bp_method,
                "osd_order": args.bposd_osd_order,
                "osd_method": args.bposd_osd_method,
            }
        case 'BeliefMatching':
            from EBBI.candidate.beliefmatching_candidate import BeliefMatchingCandidate
            return BeliefMatchingCandidate, {
                "qecc_x": qecc_x,
                "qecc_z": qecc_z,
                "max_bp_iters": args.bm_max_bp_iters,
                "bp_method": args.bm_bp_method,
            }
        case 'Tesseract':
            from EBBI.candidate.tesseract_candidate import TesseractCandidate
            return TesseractCandidate, {
                "qecc_x": qecc_x,
                "qecc_z": qecc_z,
                "det_beam": args.tesseract_det_beam,
                "beam_climbing": args.tesseract_beam_climbing,
                "pqlimit": args.tesseract_pqlimit,
                "det_penalty": args.tesseract_det_penalty,
            }
        case 'CudaQTensor':
            from EBBI.candidate.cudaqtensor_candidate import CudaQTNCandidate
            return CudaQTNCandidate, {
                "qecc_x": qecc_x,
                "qecc_z": qecc_z,
            }
        case 'Datasheet':
            from EBBI.candidate.datasheet_candidate import DatasheetCandidate
            if args.datasheet_csv_path is None:
                raise ValueError(
                    "The argument --datasheet_csv_path is required when --candidate_class is set "
                    "to Datasheet. Please point it at a CSV file holding the pre-computed logical "
                    "outcome counts of every Clifford-deformation the method can reach."
                )
            return DatasheetCandidate, {
                "csv_file_path": args.datasheet_csv_path,
            }
        case _:
            raise ValueError(
                f"Unknown candidate class: {args.candidate_class}, "
                "options=[PyMatching, BPOSD, BeliefMatching, Tesseract, CudaQTensor, Datasheet]"
            )
