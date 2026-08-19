from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
Q16_SCALE = 65536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a 32-bit Q16 RTL output with a float selective_scan answer."
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        required=True,
        help="Generated case directory containing experiment_config.json.",
    )
    parser.add_argument(
        "--rtl",
        type=Path,
        help="ModelSim test_out_0.txt. Defaults to CASE_DIR/test_out_0.txt.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Float reference. Defaults to CASE_DIR/y_origin_answer.txt.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=SCRIPT_DIR / "comparison_summary.csv",
        help="Cross-case summary table.",
    )
    parser.add_argument(
        "--no-details",
        action="store_true",
        help="Do not write per-value error_details.csv in the case directory.",
    )
    return parser.parse_args()


def read_q16_hex(path: Path) -> tuple[list[int], list[float]]:
    integers = []
    values = []
    for line_number, raw_line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        text = raw_line.strip()
        if not text:
            continue
        if len(text) > 8:
            raise ValueError(f"{path.name} line {line_number}: expected at most 8 hex digits")
        try:
            unsigned = int(text, 16)
        except ValueError as error:
            raise ValueError(f"{path.name} line {line_number}: invalid hex value {text!r}") from error
        if unsigned > 0xFFFFFFFF:
            raise ValueError(f"{path.name} line {line_number}: value exceeds 32 bits")
        signed = unsigned - (1 << 32) if unsigned & 0x80000000 else unsigned
        integers.append(signed)
        values.append(signed / Q16_SCALE)
    if not values:
        raise ValueError(f"{path} contains no Q16 values")
    return integers, values


def read_float_values(path: Path) -> list[float]:
    values = []
    for line_number, raw_line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        text = raw_line.strip()
        if not text:
            continue
        try:
            value = float(text)
        except ValueError as error:
            raise ValueError(f"{path.name} line {line_number}: invalid float value {text!r}") from error
        if not math.isfinite(value):
            raise ValueError(f"{path.name} line {line_number}: value is not finite")
        values.append(value)
    if not values:
        raise ValueError(f"{path} contains no float values")
    return values


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def round_half_away_from_zero(value: float) -> int:
    magnitude = math.floor(abs(value) + 0.5)
    return -magnitude if value < 0 else magnitude


def load_parameters(case_dir: Path) -> dict[str, object]:
    config_path = case_dir / "experiment_config.json"
    if not config_path.is_file():
        return {}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parameters = dict(config.get("parameters", {}))
    parameters.update(config.get("derived", {}))
    parameters["seed"] = config.get("seed", "")
    return parameters


def calculate_metrics(
    rtl_ints: list[int],
    rtl_values: list[float],
    reference_values: list[float],
) -> tuple[dict[str, float | int], list[dict[str, float | int | str]]]:
    if len(rtl_values) != len(reference_values):
        raise ValueError(
            f"Length mismatch: RTL has {len(rtl_values)} values, "
            f"reference has {len(reference_values)} values"
        )

    signed_errors = [rtl - reference for rtl, reference in zip(rtl_values, reference_values)]
    absolute_errors = [abs(error) for error in signed_errors]
    squared_errors = [error * error for error in signed_errors]
    absolute_reference_sum = sum(abs(value) for value in reference_values)
    rounded_reference_q16 = [
        round_half_away_from_zero(value * Q16_SCALE) for value in reference_values
    ]
    integer_errors = [rtl - reference for rtl, reference in zip(rtl_ints, rounded_reference_q16)]

    count = len(rtl_values)
    mae = sum(absolute_errors) / count
    mse = sum(squared_errors) / count
    metrics: dict[str, float | int] = {
        "sample_count": count,
        "mae": mae,
        "rmse": math.sqrt(mse),
        "median_absolute_error": percentile(absolute_errors, 50),
        "p95_absolute_error": percentile(absolute_errors, 95),
        "max_absolute_error": max(absolute_errors),
        "mean_signed_error_bias": sum(signed_errors) / count,
        "relative_mae_percent": (
            sum(absolute_errors) / absolute_reference_sum * 100
            if absolute_reference_sum != 0
            else math.nan
        ),
        "mae_lsb": mae * Q16_SCALE,
        "max_absolute_error_lsb": max(absolute_errors) * Q16_SCALE,
        "rounded_q16_exact_count": sum(error == 0 for error in integer_errors),
        "rounded_q16_exact_percent": sum(error == 0 for error in integer_errors) / count * 100,
        "rounded_q16_within_1_lsb_percent": sum(abs(error) <= 1 for error in integer_errors) / count * 100,
    }

    details = []
    for index, (rtl_int, rtl, reference, error, abs_error, integer_error) in enumerate(
        zip(
            rtl_ints,
            rtl_values,
            reference_values,
            signed_errors,
            absolute_errors,
            integer_errors,
        )
    ):
        details.append(
            {
                "index": index,
                "rtl_hex": f"{rtl_int & 0xFFFFFFFF:08X}",
                "rtl_q16_integer": rtl_int,
                "rtl_float": rtl,
                "reference_float": reference,
                "signed_error": error,
                "absolute_error": abs_error,
                "absolute_error_lsb": abs_error * Q16_SCALE,
                "error_vs_rounded_reference_lsb": integer_error,
            }
        )
    return metrics, details


def write_details(path: Path, details: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)


def update_summary(path: Path, row: dict[str, object]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    if path.is_file():
        with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
            existing_rows = list(csv.DictReader(csv_file))

    key = (str(row["case_name"]), str(row["reference_file"]))
    existing_rows = [
        old
        for old in existing_rows
        if (old.get("case_name"), old.get("reference_file")) != key
    ]
    all_rows = [*existing_rows, row]
    fieldnames = list(row)
    for old in existing_rows:
        for field in old:
            if field not in fieldnames:
                fieldnames.append(field)

    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)


def main() -> None:
    args = parse_args()
    case_dir = args.case_dir.expanduser().resolve()
    rtl_path = (args.rtl or case_dir / "test_out_0.txt").expanduser().resolve()
    reference_path = (
        args.reference or case_dir / "y_origin_answer.txt"
    ).expanduser().resolve()

    if not rtl_path.is_file():
        raise FileNotFoundError(f"Cannot find RTL output: {rtl_path}")
    if not reference_path.is_file():
        raise FileNotFoundError(f"Cannot find float reference: {reference_path}")

    rtl_ints, rtl_values = read_q16_hex(rtl_path)
    reference_values = read_float_values(reference_path)
    metrics, details = calculate_metrics(rtl_ints, rtl_values, reference_values)
    parameters = load_parameters(case_dir)

    report = {
        "case_name": case_dir.name,
        "rtl_file": str(rtl_path),
        "reference_file": str(reference_path),
        "q16_scale": Q16_SCALE,
        "parameters": parameters,
        "metrics": metrics,
    }
    report_path = case_dir / "error_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    if not args.no_details:
        write_details(case_dir / "error_details.csv", details)

    summary_row: dict[str, object] = {
        "case_name": case_dir.name,
        "reference_file": reference_path.name,
        "rtl_file": str(rtl_path),
        "L": parameters.get("L", ""),
        "d_model": parameters.get("d_model", ""),
        "d_state_N": parameters.get("d_state_N", ""),
        "expand": parameters.get("expand", ""),
        "d_inner": parameters.get("d_inner", ""),
        "seed": parameters.get("seed", ""),
        **metrics,
    }
    update_summary(args.summary_csv, summary_row)

    print(f"Case: {case_dir.name}")
    print(f"RTL: {rtl_path}")
    print(f"Reference: {reference_path}")
    print(f"Samples: {metrics['sample_count']}")
    print(f"MAE: {metrics['mae']:.9f} ({metrics['mae_lsb']:.3f} LSB)")
    print(f"RMSE: {metrics['rmse']:.9f}")
    print(
        f"Max absolute error: {metrics['max_absolute_error']:.9f} "
        f"({metrics['max_absolute_error_lsb']:.3f} LSB)"
    )
    print(f"Relative MAE: {metrics['relative_mae_percent']:.6f}%")
    print(
        "Exact after rounding float reference to Q16: "
        f"{metrics['rounded_q16_exact_count']}/{metrics['sample_count']} "
        f"({metrics['rounded_q16_exact_percent']:.3f}%)"
    )
    print(f"Report: {report_path}")
    if not args.no_details:
        print(f"Details: {case_dir / 'error_details.csv'}")
    print(f"Summary table: {args.summary_csv.expanduser().resolve()}")


if __name__ == "__main__":
    main()
