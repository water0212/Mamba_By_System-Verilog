from __future__ import annotations

import argparse

import torch

from model import (
    Q16_SCALE,
    REPO_ROOT,
    TEST_OUT_PATH,
    create_scan_inputs,
    error_stats,
    hardware_approx_float_x,
    hardware_integer_x,
    load_decimal_states,
    load_hex_states,
    load_test_out,
    original_float_x,
    quantize_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare original float x, Python hardware approximation, and RTL x"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def location(flat_index: int) -> tuple[int, int, int]:
    length_index = flat_index // (32 * 16)
    within_state = flat_index % (32 * 16)
    d_index = within_state // 16
    n_index = within_state % 16
    return length_index, d_index, n_index


def print_error_row(
    label: str,
    candidate: torch.Tensor,
    reference: torch.Tensor,
    state_index: int | None = None,
) -> None:
    stats = error_stats(candidate, reference)
    length_index, d_index, n_index = location(stats.max_flat_index)
    if state_index is not None:
        length_index = state_index
    print(
        f"{label:<25} "
        f"MAE={stats.mae:10.6f} LSB  "
        f"RMSE={stats.rmse:10.6f} LSB  "
        f"MAX={stats.max_abs:10.6f} LSB  "
        f"RelL2={stats.relative_l2_percent:8.4f}%  "
        f"max@(l={length_index},d={d_index},n={n_index})"
    )


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available")

    device = torch.device(args.device)
    inputs = create_scan_inputs(device)
    quantized = quantize_inputs(inputs)

    original_q16 = original_float_x(inputs) * Q16_SCALE
    approximate_q16 = hardware_approx_float_x(quantized)
    integer_reference_q16 = hardware_integer_x(quantized)
    rtl_q16 = load_test_out(TEST_OUT_PATH).to(device)

    saved_original = load_decimal_states("x_origin_answer", REPO_ROOT).to(device)
    saved_approximate_hex = load_hex_states("x", REPO_ROOT).to(device)
    saved_origin_max = float((original_q16.to(torch.float64) - saved_original).abs().max())
    saved_approx_exact = int(
        (torch.trunc(approximate_q16).to(torch.int64) == saved_approximate_hex).sum()
    )
    rtl_exact = int((integer_reference_q16 == rtl_q16).sum().item())

    print("Numerical validation for x_0..x_3")
    print("  original    : float SSM, displayed in Q16 code units")
    print("  approximation: quantized inputs + approximate exp + float recurrence")
    print("  RTL         : ModelSim test_out_0, signed Q16 integers")
    print(f"  saved origin regeneration max difference: {saved_origin_max:.9f} LSB")
    print(f"  truncated approximation vs x_0..x_3 hex : {saved_approx_exact}/2048 exact")
    print(f"  integer reference vs RTL                 : {rtl_exact}/2048 bit-exact")
    print()

    print("Per-state error (all values are Q16 LSB)")
    for length_index in range(4):
        print(f"x_{length_index}")
        print_error_row(
            "  Approx vs Original",
            approximate_q16[:, length_index : length_index + 1],
            original_q16[:, length_index : length_index + 1],
            length_index,
        )
        print_error_row(
            "  RTL vs Original",
            rtl_q16[:, length_index : length_index + 1],
            original_q16[:, length_index : length_index + 1],
            length_index,
        )
        print_error_row(
            "  RTL vs Approx",
            rtl_q16[:, length_index : length_index + 1],
            approximate_q16[:, length_index : length_index + 1],
            length_index,
        )
    print()

    print("Overall error")
    print_error_row("Approx vs Original", approximate_q16, original_q16)
    print_error_row("RTL vs Original", rtl_q16, original_q16)
    print_error_row("RTL vs Approx", rtl_q16, approximate_q16)

    rtl_vs_approx_abs = (rtl_q16.to(torch.float64) - approximate_q16).abs()
    within_half = int((rtl_vs_approx_abs <= 0.5 + 1e-9).sum().item())
    within_one = int((rtl_vs_approx_abs <= 1.0 + 1e-9).sum().item())
    total_error = error_stats(rtl_q16, original_q16)
    implementation_error = error_stats(rtl_q16, approximate_q16)
    exported_approx_error = error_stats(rtl_q16, saved_approximate_hex)
    exported_approx_exact = int((rtl_q16 == saved_approximate_hex).sum().item())

    print()
    print("Presentation-ready summary")
    print(f"  RTL vs original Relative L2 error : {total_error.relative_l2_percent:.4f}%")
    print(f"  RTL vs original MAE               : {total_error.mae / Q16_SCALE:.10f}")
    print(f"  RTL vs original maximum error     : {total_error.max_abs / Q16_SCALE:.10f}")
    print(f"  RTL vs approximation <= 0.5 LSB   : {within_half}/2048")
    print(f"  RTL vs approximation <= 1.0 LSB   : {within_one}/2048")
    print(
        f"  RTL implementation MAE            : "
        f"{implementation_error.mae:.6f} LSB"
    )
    print(f"  RTL vs exported x hex MAE          : {exported_approx_error.mae:.6f} LSB")
    print(f"  RTL vs exported x hex maximum      : {exported_approx_error.max_abs:.6f} LSB")
    print(f"  RTL vs exported x hex exact        : {exported_approx_exact}/2048")


if __name__ == "__main__":
    main()
