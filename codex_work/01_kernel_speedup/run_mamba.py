from __future__ import annotations

import argparse

import torch

from model import (
    TEST_OUT_PATH,
    benchmark,
    create_scan_inputs,
    hardware_integer_x,
    load_test_out,
    print_timing,
    quantize_inputs,
)


FPGA_KERNEL_CYCLES = 2054


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the CPU integer-equivalent x core with the FPGA kernel"
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--clock-mhz", type=float, default=50.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.runs <= 0 or args.clock_mhz <= 0:
        raise ValueError("warmup >= 0, runs > 0, and clock-mhz > 0 are required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available")

    device = torch.device(args.device)
    inputs = create_scan_inputs(device)
    quantized = quantize_inputs(inputs)

    reference_output = hardware_integer_x(quantized).cpu()
    model_sim_output = load_test_out(TEST_OUT_PATH)
    exact_count = int((reference_output == model_sim_output).sum().item())

    timing = benchmark(
        lambda: hardware_integer_x(quantized),
        device,
        args.warmup,
        args.runs,
    )
    fpga_kernel_ms = FPGA_KERNEL_CYCLES / (args.clock_mhz * 1000.0)
    speedup = timing.median_ms / fpga_kernel_ms

    print("Kernel-only speed comparison")
    print("  scope        : quantized A/B/delta/u -> Q16 x_0..x_3")
    print("  dimensions   : batch=1, L=4, D=32, N=16")
    print(f"  warm-up      : {args.warmup}")
    print(f"  measured runs: {args.runs}")
    print(f"  CPU device   : {device}")
    print(f"  RTL check    : {exact_count}/2048 bit-exact")
    print()
    print_timing("CPU integer-equivalent core", timing)
    print()
    print("FPGA kernel")
    print(f"  cycles       : {FPGA_KERNEL_CYCLES}")
    print(f"  clock        : {args.clock_mhz:.3f} MHz")
    print(f"  latency      : {fpga_kernel_ms:.6f} ms")
    print()
    print(f"Kernel speedup (CPU median / FPGA latency): {speedup:.3f}x")
    print("Note: this excludes quantization, loading, transfer, and dequantization.")


if __name__ == "__main__":
    main()
