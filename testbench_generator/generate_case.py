from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]


EXPORT_SPECS = {
    "A_testbench.txt": {"kind": "hex", "bits": 16, "format": "signed integer, 16-bit hex"},
    "B_testbench.txt": {"kind": "hex", "bits": 16, "format": "Q8, 16-bit hex"},
    "C_testbench.txt": {"kind": "hex", "bits": 16, "format": "Q8, 16-bit hex"},
    "D_testbench.txt": {"kind": "hex", "bits": 16, "format": "Q8, 16-bit hex"},
    "delta_testbench.txt": {"kind": "hex", "bits": 16, "format": "Q8, 16-bit hex"},
    "u_shape_int.txt": {"kind": "hex", "bits": 16, "format": "Q8, 16-bit hex"},
    "y_q16_answer.txt": {"kind": "hex", "bits": 32, "format": "Q16, 32-bit hex"},
    "y_q16_float_answer.txt": {
        "kind": "float",
        "format": "dequantized Python integer model (Q16 / 65536)",
    },
    "y_origin_answer.txt": {
        "kind": "float",
        "format": "original float selective_scan reference",
    },
}

HARDWARE_INPUT_ORDER = (
    "A_testbench.txt",
    "B_testbench.txt",
    "delta_testbench.txt",
    "u_shape_int.txt",
    "C_testbench.txt",
    "D_testbench.txt",
)
MERGED_INPUT_FILENAME = "test_in_0.txt"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def dt_rank_value(value: str) -> int | str:
    if value.lower() == "auto":
        return "auto"
    return positive_int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one random selective_scan hardware test case."
    )
    parser.add_argument("--L", "--seq-len", dest="seq_len", type=positive_int, default=4)
    parser.add_argument("--N", "--d-state", dest="d_state", type=positive_int, default=16)
    parser.add_argument("--d-model", type=positive_int, default=16)
    parser.add_argument("--expand", type=positive_int, default=2)
    parser.add_argument("--n-layer", type=positive_int, default=1)
    parser.add_argument("--vocab-size", type=positive_int, default=256)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--dt-rank", type=dt_rank_value, default="auto")
    parser.add_argument("--d-conv", type=positive_int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case-name")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SCRIPT_DIR / "generated_cases",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files in an existing case with the same name.",
    )
    args = parser.parse_args()

    if args.n_layer != 1:
        parser.error("--n-layer must be 1 because one case represents one selective_scan core.")
    if args.case_name:
        invalid_chars = '<>:"/\\|?*'
        if (
            args.case_name in {".", ".."}
            or any(char in args.case_name for char in invalid_chars)
        ):
            parser.error("--case-name must be a single valid folder name.")

    return args


def expected_line_counts(args: argparse.Namespace, d_inner: int) -> dict[str, int]:
    return {
        "A_testbench.txt": d_inner * args.d_state,
        "B_testbench.txt": args.batch_size * args.seq_len * args.d_state,
        "C_testbench.txt": args.batch_size * args.seq_len * args.d_state,
        "D_testbench.txt": d_inner,
        "delta_testbench.txt": args.batch_size * args.seq_len * d_inner,
        "u_shape_int.txt": args.batch_size * args.seq_len * d_inner,
        "y_q16_answer.txt": args.batch_size * args.seq_len * d_inner,
        "y_q16_float_answer.txt": args.batch_size * args.seq_len * d_inner,
        "y_origin_answer.txt": args.batch_size * args.seq_len * d_inner,
    }


def prepare_case_directory(case_dir: Path, overwrite: bool) -> None:
    managed_files = [*EXPORT_SPECS, MERGED_INPUT_FILENAME, "experiment_config.json"]
    existing = [case_dir / name for name in managed_files if (case_dir / name).exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Case already contains generated files ({names}). Use --overwrite to replace them."
        )

    case_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in existing:
            path.unlink()


def load_model_module(temp_dir: Path):
    source_path = PROJECT_ROOT / "model.py"
    if not source_path.is_file():
        raise FileNotFoundError(f"Cannot find model.py at {source_path}")

    isolated_model_path = temp_dir / "model.py"
    shutil.copy2(source_path, isolated_model_path)
    module_name = "_mamba_selective_scan_case_model"
    spec = importlib.util.spec_from_file_location(module_name, isolated_model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load model module from {isolated_model_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module, module_name


def verify_exports(
    case_dir: Path,
    expected_counts: dict[str, int],
) -> dict[str, dict[str, int | str]]:
    results: dict[str, dict[str, int | str]] = {}

    for filename, spec in EXPORT_SPECS.items():
        path = case_dir / filename
        if not path.is_file():
            raise RuntimeError(f"Missing export: {path}")

        lines = [line.strip() for line in path.read_text(encoding="ascii").splitlines()]
        expected_count = expected_counts[filename]
        if len(lines) != expected_count:
            raise RuntimeError(
                f"{filename}: expected {expected_count} lines, found {len(lines)}"
            )

        if spec["kind"] == "hex":
            hex_digits = int(spec["bits"]) // 4
            pattern = re.compile(rf"[0-9A-F]{{{hex_digits}}}")
            invalid_line = next((line for line in lines if not pattern.fullmatch(line)), None)
            if invalid_line is not None:
                raise RuntimeError(f"{filename}: invalid hex value {invalid_line!r}")
        else:
            try:
                values = [float(line) for line in lines]
            except ValueError as error:
                raise RuntimeError(f"{filename}: invalid float value") from error
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(f"{filename}: contains NaN or infinity")

        results[filename] = {
            "line_count": len(lines),
            "format": str(spec["format"]),
        }

    return results


def merge_hardware_inputs(case_dir: Path) -> dict[str, object]:
    """Merge the six 16-bit input files in the order expected by the RTL."""
    output_path = case_dir / MERGED_INPUT_FILENAME
    sections = []
    next_start_line = 1

    with output_path.open("w", encoding="ascii", newline="\n") as output_file:
        for filename in HARDWARE_INPUT_ORDER:
            source_path = case_dir / filename
            if not source_path.is_file():
                raise RuntimeError(f"Cannot merge missing input file: {source_path}")

            values = source_path.read_text(encoding="ascii").splitlines()
            for value in values:
                output_file.write(f"{value}\n")

            line_count = len(values)
            end_line = next_start_line + line_count - 1
            sections.append(
                {
                    "source": filename,
                    "start_line": next_start_line,
                    "end_line": end_line,
                    "line_count": line_count,
                }
            )
            next_start_line = end_line + 1

    return {
        "filename": MERGED_INPUT_FILENAME,
        "order": list(HARDWARE_INPUT_ORDER),
        "line_count": next_start_line - 1,
        "sections": sections,
    }


def main() -> None:
    args = parse_args()
    d_inner = args.d_model * args.expand
    case_name = args.case_name or (
        f"case_L{args.seq_len}_D{args.d_model}_N{args.d_state}"
        f"_E{args.expand}_B{args.batch_size}_seed{args.seed}"
    )
    case_dir = (args.output_root.expanduser().resolve() / case_name).resolve()
    prepare_case_directory(case_dir, args.overwrite)

    old_cwd = Path.cwd()
    raw_dir_path = SCRIPT_DIR / "_raw_work" / uuid.uuid4().hex
    raw_dir_path.mkdir(parents=True)
    module_name = None
    try:
        mamba_module, module_name = load_model_module(raw_dir_path)
        torch.manual_seed(args.seed)
        model_args = mamba_module.ModelArgs(
            d_model=args.d_model,
            n_layer=args.n_layer,
            vocab_size=args.vocab_size,
            d_state=args.d_state,
            expand=args.expand,
            dt_rank=args.dt_rank,
            d_conv=args.d_conv,
        )
        model = mamba_module.Mamba(model_args)
        model.eval()

        input_ids = torch.randint(
            low=0,
            high=model_args.vocab_size,
            size=(args.batch_size, args.seq_len),
            dtype=torch.long,
        )
        with torch.inference_mode():
            logits = model(input_ids)

        for filename in EXPORT_SPECS:
            source = raw_dir_path / filename
            if not source.is_file():
                raise RuntimeError(f"model.py did not generate {filename}")
            shutil.copy2(source, case_dir / filename)
    finally:
        os.chdir(old_cwd)
        if module_name is not None:
            sys.modules.pop(module_name, None)
        shutil.rmtree(raw_dir_path, ignore_errors=True)
        try:
            raw_dir_path.parent.rmdir()
        except OSError:
            pass

    export_results = verify_exports(
        case_dir,
        expected_line_counts(args, d_inner),
    )
    merged_input = merge_hardware_inputs(case_dir)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "parameters": {
            "batch_size": args.batch_size,
            "L": args.seq_len,
            "d_model": args.d_model,
            "d_state_N": args.d_state,
            "expand": args.expand,
            "n_layer": args.n_layer,
            "vocab_size_requested": args.vocab_size,
            "vocab_size_padded": model_args.vocab_size,
            "dt_rank_requested": args.dt_rank,
            "dt_rank_resolved": model_args.dt_rank,
            "d_conv": args.d_conv,
        },
        "derived": {
            "d_inner": d_inner,
            "input_ids_shape": list(input_ids.shape),
            "logits_shape": list(logits.shape),
        },
        "input_ids": input_ids.tolist(),
        "exports": export_results,
        "merged_hardware_input": merged_input,
        "notes": [
            "N is the same parameter as d_state.",
            "d_inner equals d_model multiplied by expand.",
            "Random model weights and token IDs are reproducible with the same seed.",
            "A and D retain the initialization defined by the current model.py.",
            "y_q16_float_answer.txt is the dequantized Python integer approximation.",
            "y_origin_answer.txt is the original float reference for hardware error analysis.",
        ],
    }
    config_path = case_dir / "experiment_config.json"
    config_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Generated case: {case_dir}")
    print(
        f"Parameters: L={args.seq_len}, d_model={args.d_model}, "
        f"N={args.d_state}, d_inner={d_inner}, seed={args.seed}"
    )
    for filename, result in export_results.items():
        print(f"  {filename}: {result['line_count']} lines")
    print(
        f"  {merged_input['filename']}: {merged_input['line_count']} lines "
        "(A, B, delta, u, C, D)"
    )
    print(f"  {config_path.name}: parameters and random input IDs")


if __name__ == "__main__":
    main()
