from __future__ import annotations

import importlib.util
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_MODEL_PATH = (
    REPO_ROOT.parent / "MAMBA" / "MAMBA" / "mamba-min_python" / "model.py"
)
HARDWARE_MODEL_PATH = REPO_ROOT / "model.py"
TEST_OUT_PATH = REPO_ROOT / "MAMBA" / "DE0_CV" / "simulation" / "tb" / "test_out_0.txt"

Q8_SCALE = 1 << 8
Q16_SCALE = 1 << 16


@dataclass(frozen=True)
class ScanInputs:
    u: torch.Tensor
    delta: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor


@dataclass(frozen=True)
class QuantizedInputs:
    u_q8: torch.Tensor
    delta_q8: torch.Tensor
    a_q0: torch.Tensor
    b_q8: torch.Tensor


@dataclass(frozen=True)
class TimingStats:
    samples_ms: list[float]
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float
    p95_ms: float


@dataclass(frozen=True)
class ErrorStats:
    count: int
    mae: float
    rmse: float
    bias: float
    max_abs: float
    relative_l2_percent: float
    weighted_ma_percent: float
    max_flat_index: int


def _load_original_model_module():
    module_name = "mamba_original_for_benchmark"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, ORIGINAL_MODEL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load original model from {ORIGINAL_MODEL_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def create_scan_inputs(device: torch.device) -> ScanInputs:
    original = _load_original_model_module()

    torch.manual_seed(0)
    args = original.ModelArgs(
        d_model=16,
        n_layer=1,
        vocab_size=256,
        d_state=16,
        expand=2,
    )
    model = original.Mamba(args).to(device)
    model.eval()
    input_ids = torch.tensor([[1, 2, 2, 0]], dtype=torch.long, device=device)

    with torch.inference_mode():
        hidden = model.embedding(input_ids)
        residual_block = model.layers[0]
        mixer = residual_block.mixer
        hidden = residual_block.norm(hidden)

        projected = mixer.in_proj(hidden)
        u, _ = projected.split([args.d_inner, args.d_inner], dim=-1)
        sequence_length = u.shape[1]
        u = mixer.conv1d(u.transpose(1, 2))[:, :, :sequence_length]
        u = F.silu(u.transpose(1, 2))

        a = -torch.exp(mixer.A_log.float())
        projected_ssm = mixer.x_proj(u)
        delta_rank, b, _ = projected_ssm.split(
            [args.dt_rank, args.d_state, args.d_state], dim=-1
        )
        delta = F.softplus(mixer.dt_proj(delta_rank))

    return ScanInputs(
        u=u.detach(),
        delta=delta.detach(),
        a=a.detach(),
        b=b.detach(),
    )


def original_float_x(inputs: ScanInputs) -> torch.Tensor:
    delta_a = torch.exp(
        torch.einsum("bld,dn->bldn", inputs.delta, inputs.a)
    )
    delta_b_u = torch.einsum(
        "bld,bln,bld->bldn", inputs.delta, inputs.b, inputs.u
    )

    batch, length, d_inner = inputs.u.shape
    d_state = inputs.a.shape[1]
    state = torch.zeros(
        (batch, d_inner, d_state),
        dtype=delta_a.dtype,
        device=delta_a.device,
    )
    states = []
    for index in range(length):
        state = delta_a[:, index] * state + delta_b_u[:, index]
        states.append(state)
    return torch.stack(states, dim=1)


def quantize_inputs(inputs: ScanInputs) -> QuantizedInputs:
    return QuantizedInputs(
        u_q8=torch.round(inputs.u * Q8_SCALE).to(torch.int64),
        delta_q8=torch.round(inputs.delta * Q8_SCALE).to(torch.int64),
        a_q0=torch.round(inputs.a).to(torch.int64),
        b_q8=torch.round(inputs.b * Q8_SCALE).to(torch.int64),
    )


def pack_hardware_input(quantized: QuantizedInputs) -> torch.Tensor:
    # Hardware input order: A, B, delta, u. Each value is transferred as 16 bits.
    return torch.cat(
        (
            quantized.a_q0.reshape(-1),
            quantized.b_q8.reshape(-1),
            quantized.delta_q8.reshape(-1),
            quantized.u_q8.reshape(-1),
        )
    ).to(torch.int16).contiguous()


def quantize_and_pack(inputs: ScanInputs) -> tuple[QuantizedInputs, torch.Tensor]:
    quantized = quantize_inputs(inputs)
    return quantized, pack_hardware_input(quantized)


def round_shift_signed(value: torch.Tensor, shift: int) -> torch.Tensor:
    if shift == 0:
        return value
    half_lsb = 1 << (shift - 1)
    return torch.where(
        value >= 0,
        (value + half_lsb) >> shift,
        -(((-value) + half_lsb) >> shift),
    )


def hardware_exp_q8(delta_a_q8: torch.Tensor) -> torch.Tensor:
    # exp(x) = 2^(x * log2(e)); 369/256 approximates log2(e).
    y_q8 = (delta_a_q8 * 369) >> 8
    integer_part = y_q8 >> 8
    fraction_q8 = y_q8 & 0xFF
    two_to_fraction_q8 = Q8_SCALE + fraction_q8

    right_amount = torch.clamp(-integer_part, min=0, max=62)
    left_amount = torch.clamp(integer_part, min=0, max=23)
    shifted_right = two_to_fraction_q8 >> right_amount
    shifted_right = torch.where(
        (-integer_part) >= 63,
        torch.zeros_like(shifted_right),
        shifted_right,
    )
    shifted_left = two_to_fraction_q8 << left_amount
    shifted_left = torch.where(
        integer_part > 23,
        torch.full_like(shifted_left, (1 << 32) - 1),
        shifted_left,
    )
    return torch.where(integer_part < 0, shifted_right, shifted_left)


def hardware_terms(quantized: QuantizedInputs) -> tuple[torch.Tensor, torch.Tensor]:
    delta_a_q8 = (
        quantized.delta_q8.unsqueeze(-1) * quantized.a_q0.unsqueeze(0).unsqueeze(0)
    )
    exp_q8 = hardware_exp_q8(delta_a_q8)

    delta_b_u_q24 = (
        quantized.delta_q8.unsqueeze(-1)
        * quantized.b_q8.unsqueeze(2)
        * quantized.u_q8.unsqueeze(-1)
    )
    delta_b_u_q16 = round_shift_signed(delta_b_u_q24, 8)
    return exp_q8, delta_b_u_q16


def hardware_integer_x(quantized: QuantizedInputs) -> torch.Tensor:
    exp_q8, delta_b_u_q16 = hardware_terms(quantized)
    batch, length, d_inner = quantized.u_q8.shape
    d_state = quantized.a_q0.shape[1]
    state_q16 = torch.zeros(
        (batch, d_inner, d_state),
        dtype=torch.int64,
        device=quantized.u_q8.device,
    )
    states = []
    for index in range(length):
        feedback_q16 = round_shift_signed(exp_q8[:, index] * state_q16, 8)
        state_q16 = feedback_q16 + delta_b_u_q16[:, index]
        states.append(state_q16)
    return torch.stack(states, dim=1)


def hardware_approx_float_x(quantized: QuantizedInputs) -> torch.Tensor:
    exp_q8, delta_b_u_q16 = hardware_terms(quantized)
    exp_q8 = exp_q8.to(torch.float32)
    delta_b_u_q16 = delta_b_u_q16.to(torch.float32)

    batch, length, d_inner = quantized.u_q8.shape
    d_state = quantized.a_q0.shape[1]
    state_q16 = torch.zeros(
        (batch, d_inner, d_state),
        dtype=torch.float32,
        device=quantized.u_q8.device,
    )
    states = []
    for index in range(length):
        state_q16 = (
            exp_q8[:, index] * state_q16 / Q8_SCALE
            + delta_b_u_q16[:, index]
        )
        states.append(state_q16)
    return torch.stack(states, dim=1)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    function: Callable[[], object],
    device: torch.device,
    warmup_runs: int,
    measure_runs: int,
) -> TimingStats:
    with torch.inference_mode():
        for _ in range(warmup_runs):
            function()
        synchronize(device)

        samples_ms = []
        for _ in range(measure_runs):
            synchronize(device)
            start = time.perf_counter()
            function()
            synchronize(device)
            samples_ms.append((time.perf_counter() - start) * 1000.0)

    ordered = sorted(samples_ms)
    p95_index = round((len(ordered) - 1) * 0.95)
    return TimingStats(
        samples_ms=samples_ms,
        mean_ms=statistics.fmean(samples_ms),
        median_ms=statistics.median(samples_ms),
        min_ms=min(samples_ms),
        max_ms=max(samples_ms),
        stdev_ms=statistics.pstdev(samples_ms),
        p95_ms=ordered[p95_index],
    )


def load_test_out(path: Path = TEST_OUT_PATH) -> torch.Tensor:
    values = []
    for line in path.read_text(encoding="ascii").splitlines():
        text = line.strip()
        if not text:
            continue
        unsigned = int(text, 16)
        values.append(unsigned - (1 << 32) if unsigned >= (1 << 31) else unsigned)
    if len(values) != 4 * 32 * 16:
        raise ValueError(f"Expected 2048 hardware outputs, found {len(values)} in {path}")
    return torch.tensor(values, dtype=torch.int64).reshape(1, 4, 32, 16)


def load_hex_states(prefix: str, directory: Path = REPO_ROOT) -> torch.Tensor:
    states = []
    for index in range(4):
        path = directory / f"{prefix}_{index}.txt"
        values = []
        for line in path.read_text(encoding="ascii").splitlines():
            text = line.strip()
            if not text:
                continue
            unsigned = int(text, 16)
            values.append(unsigned - (1 << 32) if unsigned >= (1 << 31) else unsigned)
        if len(values) != 32 * 16:
            raise ValueError(f"Expected 512 values, found {len(values)} in {path}")
        states.append(torch.tensor(values, dtype=torch.int64).reshape(1, 32, 16))
    return torch.stack(states, dim=1)


def load_decimal_states(prefix: str, directory: Path = REPO_ROOT) -> torch.Tensor:
    states = []
    for index in range(4):
        path = directory / f"{prefix}_{index}.txt"
        values = [
            float(line.strip())
            for line in path.read_text(encoding="ascii").splitlines()
            if line.strip()
        ]
        if len(values) != 32 * 16:
            raise ValueError(f"Expected 512 values, found {len(values)} in {path}")
        states.append(torch.tensor(values, dtype=torch.float64).reshape(1, 32, 16))
    return torch.stack(states, dim=1)


def error_stats(candidate: torch.Tensor, reference: torch.Tensor) -> ErrorStats:
    candidate64 = candidate.detach().cpu().to(torch.float64)
    reference64 = reference.detach().cpu().to(torch.float64)
    difference = candidate64 - reference64
    absolute = difference.abs()
    flat_max_index = int(torch.argmax(absolute).item())
    reference_norm = torch.linalg.vector_norm(reference64)
    relative_l2 = torch.linalg.vector_norm(difference) / reference_norm
    weighted_ma = absolute.sum() / reference64.abs().sum()
    return ErrorStats(
        count=difference.numel(),
        mae=float(absolute.mean().item()),
        rmse=float(torch.sqrt((difference * difference).mean()).item()),
        bias=float(difference.mean().item()),
        max_abs=float(absolute.max().item()),
        relative_l2_percent=float(relative_l2.item() * 100.0),
        weighted_ma_percent=float(weighted_ma.item() * 100.0),
        max_flat_index=flat_max_index,
    )


def print_timing(name: str, stats: TimingStats) -> None:
    print(f"{name}")
    print(f"  mean   : {stats.mean_ms:.6f} ms")
    print(f"  median : {stats.median_ms:.6f} ms")
    print(f"  min    : {stats.min_ms:.6f} ms")
    print(f"  max    : {stats.max_ms:.6f} ms")
    print(f"  stdev  : {stats.stdev_ms:.6f} ms")
    print(f"  p95    : {stats.p95_ms:.6f} ms")
