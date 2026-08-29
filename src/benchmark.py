"""
src/benchmark.py

Section 3.5: Evaluation - Beyond Accuracy.

Measures model size on disk, parameter count, and REAL inference latency (not theoretical
FLOPs) for each variant, on the same fixed evaluation set, in the same runtime (ONNX Runtime)
where relevant, plus a documented power-draw estimate.
"""
import os
import time
import numpy as np
import torch


def model_size_on_disk_kb(path) -> float:
    return round(os.path.getsize(path) / 1024, 2)


def count_onnx_params(onnx_path) -> int:
    import onnx
    model = onnx.load(str(onnx_path))
    total = 0
    for initializer in model.graph.initializer:
        total += int(np.prod(initializer.dims))
    return total


def benchmark_onnx_latency(session, sample_input: np.ndarray, n_warmup=20, n_runs=500):
    """
    Measures real single-image inference latency through the actual ONNX Runtime session.

    METHODOLOGY NOTE (important for reproducibility at this model's scale): a naive
    "time.perf_counter() before and after each individual call, averaged" approach is
    unreliable here because this model's true inference time (tens of microseconds) is
    comparable to the overhead of Python's own timer calls and loop bookkeeping -- early
    testing showed run-to-run latency swings large enough to flip which side of a millisecond
    -level budget a variant landed on, purely from measurement noise, not real performance
    differences. Instead we time a large BLOCK of n_runs sequential inferences with a single
    perf_counter call before and after the whole block, then divide by n_runs -- this
    amortizes per-call Python/timer overhead across many calls rather than re-paying it (and
    re-measuring its jitter) on every single one, giving a far more stable and reproducible
    mean latency estimate. n_warmup runs before the timed block are discarded (session/JIT
    warm-up, memory allocation settling), standard practice for fair latency benchmarking.
    """
    input_name = session.get_inputs()[0].name
    single_input = sample_input[:1].astype(np.float32)  # batch size 1 -- realistic edge inference

    for _ in range(n_warmup):
        session.run(None, {input_name: single_input})

    # Block-timed for stability (see docstring), plus a second pass of per-call timings
    # (fewer repeats) purely to report a std/percentile spread for transparency.
    t0 = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, {input_name: single_input})
    total_elapsed_ms = (time.perf_counter() - t0) * 1000
    mean_ms = total_elapsed_ms / n_runs

    per_call_latencies_ms = []
    for _ in range(50):
        t0 = time.perf_counter()
        session.run(None, {input_name: single_input})
        per_call_latencies_ms.append((time.perf_counter() - t0) * 1000)
    per_call_latencies_ms = np.array(per_call_latencies_ms)

    return {
        "mean_ms": round(float(mean_ms), 5),
        "median_ms": round(float(np.median(per_call_latencies_ms)), 5),
        "p95_ms": round(float(np.percentile(per_call_latencies_ms, 95)), 5),
        "std_ms": round(float(per_call_latencies_ms.std()), 5),
        "fps": round(float(1000 / mean_ms), 1),
        "measurement_method": f"block-timed mean over {n_runs} sequential calls (see docstring); "
                               f"std/median/p95 from a separate 50-call per-call-timed sample",
    }


def estimate_power_draw(latency_ms: float, device_power_watts=2.7) -> dict:
    """
    DOCUMENTED ESTIMATE, not a real hardware measurement (Section 3.5 explicitly allows
    'estimate or measure ... a documented estimate'). We do not have physical edge hardware
    with power monitoring available in this environment, so we estimate energy-per-inference
    using a published-range sustained power draw for a low-power ARM-class edge board under
    inference load (~2.5-3.0W, typical of a Raspberry Pi 4 / Jetson Nano in a comparable idle-
    to-moderate-load inference state; we use 2.7W as a representative midpoint) multiplied by
    the REAL measured latency for each variant. This is directly stated as an approximation:
    actual power draw depends on the specific board, clock speed, and thermal state, and would
    need a real power monitor (e.g. a Jetson's onboard INA3221 sensor) to measure precisely.
    """
    energy_per_inference_mj = device_power_watts * (latency_ms / 1000) * 1000  # millijoules
    inferences_per_joule = 1000 / energy_per_inference_mj if energy_per_inference_mj > 0 else float("inf")
    return {
        "assumed_device_power_watts": device_power_watts,
        "energy_per_inference_mj": round(energy_per_inference_mj, 4),
        "inferences_per_joule": round(inferences_per_joule, 2),
        "note": "Documented estimate based on typical low-power edge-board sustained power "
                "draw, NOT a real hardware measurement -- see docstring.",
    }
