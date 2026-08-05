from __future__ import annotations

import argparse

import torch

from model import (
    Q16_SCALE,
    TEST_OUT_PATH,
    benchmark,
    create_scan_inputs,
    load_test_out,
    original_float_x,
    print_timing,
    quantize_and_pack,
)


FPGA_INPUT_LOAD_CYCLES = 832
FPGA_KERNEL_CYCLES = 2054


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate float-software vs FPGA end-to-end x latency"
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--clock-mhz", type=float, default=50.0)
    parser.add_argument(
        "--h2d-us",
        type=float,
        default=0.0,
        help="Measured host-to-FPGA overhead excluding the 832 load cycles",
    )
    parser.add_argument(
        "--d2h-us",
        type=float,
        default=0.0,
        help="Measured non-overlapped FPGA-to-host output overhead",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args = parse_args()
    if (
        args.warmup < 0
        or args.runs <= 0
        or args.clock_mhz <= 0
        or args.h2d_us < 0
        or args.d2h_us < 0
    ):
        raise ValueError("All counts/frequencies must be positive and transfer times nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available")

    device = torch.device(args.device)
    inputs = create_scan_inputs(device)
    hardware_output = load_test_out(TEST_OUT_PATH).to(device)

    original_timing = benchmark(
        lambda: original_float_x(inputs),
        device,
        args.warmup,
        args.runs,
    )
    preprocess_timing = benchmark(
        lambda: quantize_and_pack(inputs),
        device,
        args.warmup,
        args.runs,
    )
    postprocess_timing = benchmark(
        lambda: hardware_output.to(torch.float32) / Q16_SCALE,
        device,
        args.warmup,
        args.runs,
    )

    input_load_ms = FPGA_INPUT_LOAD_CYCLES / (args.clock_mhz * 1000.0)
    kernel_ms = FPGA_KERNEL_CYCLES / (args.clock_mhz * 1000.0)
    h2d_ms = args.h2d_us / 1000.0
    d2h_ms = args.d2h_us / 1000.0
    estimated_fpga_ms = (
        preprocess_timing.median_ms
        + h2d_ms
        + input_load_ms
        + kernel_ms
        + d2h_ms
        + postprocess_timing.median_ms
    )
    speedup = original_timing.median_ms / estimated_fpga_ms

    print("End-to-end x comparison")
    print("  common start : float A/B/delta/u in CPU memory")
    print("  common end   : x_0..x_3 available as floating-point values")
    print("  dimensions   : batch=1, L=4, D=32, N=16")
    print(f"  warm-up/runs : {args.warmup}/{args.runs}")
    print()
    print_timing("Original float software x core", original_timing)
    print()
    print_timing("Hardware preprocessing (quantize + pack 832 words)", preprocess_timing)
    print()
    print_timing("Hardware postprocessing (Q16 -> float)", postprocess_timing)
    print()
    print("FPGA latency components")
    print(f"  input load   : {FPGA_INPUT_LOAD_CYCLES} cycles = {input_load_ms:.6f} ms")
    print(f"  x kernel     : {FPGA_KERNEL_CYCLES} cycles = {kernel_ms:.6f} ms")
    print(f"  H2D overhead : {h2d_ms:.6f} ms")
    print(f"  D2H overhead : {d2h_ms:.6f} ms")
    print(f"  clock        : {args.clock_mhz:.3f} MHz")
    print()
    print(f"Estimated FPGA end-to-end latency: {estimated_fpga_ms:.6f} ms")
    print(f"Estimated end-to-end speedup     : {speedup:.3f}x")
    if args.h2d_us == 0.0 and args.d2h_us == 0.0:
        print("WARNING: transfer overhead defaults to zero; this is not yet a board measurement.")
    print("Text-file I/O is intentionally excluded from both sides.")


if __name__ == "__main__":
    main()
