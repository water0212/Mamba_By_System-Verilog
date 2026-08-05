from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
import types
from pathlib import Path

import torch


ORIGINAL_MAMBA_DIR = Path(
    r"C:\Users\water\Desktop\VHDL\MAMBA\MAMBA\mamba-min_python"
)
sys.path.insert(0, str(ORIGINAL_MAMBA_DIR))

from model import Mamba, ModelArgs  # noqa: E402


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def print_summary(name: str, samples_ms: list[float]) -> None:
    print(f"\n{name}")
    print(f"  samples : {len(samples_ms)}")
    print(f"  mean    : {statistics.fmean(samples_ms):.6f} ms")
    print(f"  median  : {statistics.median(samples_ms):.6f} ms")
    print(f"  min     : {min(samples_ms):.6f} ms")
    print(f"  max     : {max(samples_ms):.6f} ms")
    print(f"  stdev   : {statistics.pstdev(samples_ms):.6f} ms")
    print(f"  p95     : {percentile(samples_ms, 0.95):.6f} ms")


def measure_full_forward(
    model: Mamba,
    input_ids: torch.Tensor,
    device: torch.device,
    warmup_runs: int,
    measure_runs: int,
) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup_runs):
            model(input_ids)
        synchronize(device)

        samples_ms = []
        for _ in range(measure_runs):
            synchronize(device)
            start = time.perf_counter()
            model(input_ids)
            synchronize(device)
            samples_ms.append((time.perf_counter() - start) * 1000.0)

    return samples_ms


def install_selective_scan_timers(
    model: Mamba,
    device: torch.device,
    samples_ms: list[float],
) -> None:
    for layer in model.layers:
        block = layer.mixer
        original_function = block.selective_scan.__func__

        def timed_selective_scan(self, *args, _original=original_function, **kwargs):
            synchronize(device)
            start = time.perf_counter()
            result = _original(self, *args, **kwargs)
            synchronize(device)
            samples_ms.append((time.perf_counter() - start) * 1000.0)
            return result

        block.selective_scan = types.MethodType(timed_selective_scan, block)


def measure_selective_scan(
    model: Mamba,
    input_ids: torch.Tensor,
    device: torch.device,
    warmup_runs: int,
    measure_runs: int,
) -> list[float]:
    samples_ms: list[float] = []
    install_selective_scan_timers(model, device, samples_ms)

    with torch.inference_mode():
        for _ in range(warmup_runs):
            model(input_ids)
        synchronize(device)
        samples_ms.clear()

        for _ in range(measure_runs):
            model(input_ids)
        synchronize(device)

    expected_samples = measure_runs * len(model.layers)
    if len(samples_ms) != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} selective_scan samples, "
            f"but collected {len(samples_ms)}"
        )

    return samples_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the minimal Mamba model")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.runs <= 0:
        raise ValueError("--warmup must be >= 0 and --runs must be > 0")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available")

    # Disable the per-call prints in the original model profiler.
    os.environ["MAMBA_PROFILE"] = "0"

    device = torch.device(args.device)
    torch.manual_seed(0)

    model_args = ModelArgs(
        d_model=16,
        n_layer=1,
        vocab_size=256,
        d_state=16,
        expand=2,
    )
    model = Mamba(model_args).to(device)
    model.eval()

    # This is the integer form of the current [1, 2, 2.1, 0.7] input.
    input_ids = torch.tensor([[1, 2, 2, 0]], dtype=torch.long, device=device)

    print("Mamba timing benchmark")
    print(f"  device  : {device}")
    print(f"  shape   : batch=1, seq_len=4, d_inner=32, d_state=16")
    print(f"  warm-up : {args.warmup} runs")
    print(f"  measure : {args.runs} runs")

    forward_samples = measure_full_forward(
        model, input_ids, device, args.warmup, args.runs
    )
    scan_samples = measure_selective_scan(
        model, input_ids, device, args.warmup, args.runs
    )

    print_summary("Full model forward", forward_samples)
    print_summary("Selective scan", scan_samples)


if __name__ == "__main__":
    main()
