"""Latency for the scorecard. Scores themselves are not recomputed.

Neural rows: CUDA, 3 warmup forwards, median of 10. CPU detectors: one pass
on the saved median_x3 of record 5. DSP file times stay in DENOISE_SCORES.md.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fiber.anomaly.detect import cusum_alarm, spatial_z, temporal_z
from fiber.benchmark.vitrine import by_name


def _median_seconds(fn, repeats: int = 10, warmup: int = 3) -> float:
    import torch

    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    return float(np.median(samples))


def _params(model) -> int:
    return int(sum(item.numel() for item in model.parameters()))


def _neural(device: str) -> list[dict]:
    import torch

    from fiber.models.conv1d_denoise import load_weights
    from fiber.models.conv_ae import load_ae
    from fiber.models.patch_classifier import load_patch_classifier
    from fiber.models.unet2d_denoise import load_unet

    jobs = []

    conv, _dev = load_weights(ROOT / by_name("conv1d_n2self")["weights"], device)
    batch = torch.zeros(32, 1, 2364, device=device)
    jobs.append(("conv1d_n2self", conv, lambda: conv(batch), "32x2364 profiles"))

    unet, _blob, _dev = load_unet(ROOT / by_name("unet2d_n2self")["weights"], device)
    batch_u = torch.zeros(16, 1, 32, 128, device=device)
    jobs.append(("unet2d_n2self", unet, lambda: unet(batch_u), "16 crops 32x128"))

    ae, _blob, _dev = load_ae(ROOT / by_name("conv_ae_bg")["weights"], device)
    batch_a = torch.zeros(64, 1, 8, 64, device=device)
    jobs.append(("conv_ae_bg", ae, lambda: ae(batch_a), "64 crops 8x64"))

    for name, workload in (
        ("patch_classifier", "256 crops 28x40"),
        ("patch_scanner", "256 crops 28x40"),
    ):
        model, _blob, _dev = load_patch_classifier(ROOT / by_name(name)["weights"], device)
        batch_p = torch.zeros(256, 1, 28, 40, device=device)
        jobs.append((name, model, lambda model=model, batch_p=batch_p: model(batch_p), workload))

    rows = []
    for name, model, fn, workload in jobs:
        with torch.inference_mode():
            elapsed = _median_seconds(fn)
        path = ROOT / by_name(name)["weights"]
        rows.append({
            "name": name,
            "device": device,
            "latency_s": round(elapsed, 6),
            "workload": workload,
            "parameters": _params(model),
            "weight_bytes": path.stat().st_size,
        })
        print(f"{name} {device} {elapsed:.6f}s params={rows[-1]['parameters']} bytes={rows[-1]['weight_bytes']}", flush=True)
    return rows


def _cpu_detectors() -> list[dict]:
    clean_path = ROOT / "artifacts" / "denoise" / "5" / "median_x3.npy"
    record = ROOT / "artifacts" / "records" / "5"
    if not clean_path.is_file():
        print("skip cpu detectors, median_x3 of record 5 is missing", flush=True)
        return []
    clean = np.load(clean_path)
    valid = np.load(record / "valid.npy")
    x_m = np.load(record / "x_m.npy")
    time_s = np.load(record / "time_s.npy")
    rows = []

    def once(name: str, fn) -> None:
        started = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - started
        rows.append({
            "name": name,
            "device": "cpu",
            "latency_s": round(elapsed, 3),
            "workload": "record 5 median_x3, one pass",
            "parameters": 0,
            "weight_bytes": 0,
        })
        print(f"{name} cpu {elapsed:.3f}s", flush=True)

    from scripts.benchmark_models import profile_median_z

    once("spatial_z_halo", lambda: spatial_z(clean, valid, x_m))
    once("profile_median_z", lambda: profile_median_z(clean, valid))
    once("temporal_z", lambda: temporal_z(clean, valid, time_s))
    z, background, _scale = spatial_z(clean, valid, x_m)
    residual = np.where(valid, clean.astype(np.float64) - background, np.nan)
    once("cusum", lambda: cusum_alarm(residual, valid, time_s, drift_c=0.8, threshold_c=6.0))
    del z
    return rows


def main() -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, flush=True)
    payload = {
        "device": device,
        "protocol": "neural: median of 10 forwards after 3 warmup, CUDA sync; detectors: one pass on record 5",
        "rows": _neural(device) + _cpu_detectors(),
    }
    dest = ROOT / "artifacts" / "benchmark" / "vitrine_latency.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", dest, flush=True)


if __name__ == "__main__":
    main()
