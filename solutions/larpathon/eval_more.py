"""Run the shipped ensemble on more real CSVs and on labeled synthetic test sequences.

Synthetic labels are a score only. They are not used to train or to set the threshold.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch

from mgts_anomaly import run_pipeline

SYNTH_CSV = Path(r"C:\Users\bobr\haklolarp\artifacts\dts_synth_v1\csv\test")
SYNTH_SEQ = Path(r"C:\Users\bobr\haklolarp\artifacts\dts_synth_v1\sequences.csv")
SYNTH_LABELS = Path(r"C:\Users\bobr\haklolarp\artifacts\dts_synth_v1\labels.csv")
REAL_DIR = Path(r"C:\Users\bobr\haklolarp\MGTS_data")
OUT_DIR = Path(r"C:\Users\bobr\test2_larpathon\reports\more_data")

HEAT_LABELS = {"tracing", "hotspot", "section_heating", "slow_warming"}
COOL_LABELS = {"coldspot", "cooling"}
MAX_TRACES = 400
MAX_SPACE = 2600


def _intervals_overlap(a0, a1, b0, b1) -> bool:
    return a0 <= b1 and b0 <= a1


def _event_box(event: dict, timestamps: list, dists) -> tuple:
    t0, t1 = event["t_range"]
    s0, s1 = event["s_range"]
    time_start = pd.to_datetime(timestamps[t0], utc=True)
    time_end = pd.to_datetime(timestamps[t1], utc=True)
    meter_start = float(min(dists[s0], dists[s1]))
    meter_end = float(max(dists[s0], dists[s1]))
    return time_start, time_end, meter_start, meter_end


def _public_event(event: dict, timestamps: list, dists) -> dict:
    time_start, time_end, meter_start, meter_end = _event_box(event, timestamps, dists)
    return {
        "type": event["type_hypothesis"],
        "duration_min": round(float(event["duration_min"]), 1),
        "spatial_extent_m": round(float(event["spatial_extent_m"]), 2),
        "amplitude_c": round(float(event["amplitude"]), 2),
        "meters": [round(meter_start, 1), round(meter_end, 1)],
        "t_start": str(time_start),
        "t_end": str(time_end),
    }


def score_against_labels(result: dict, rows: pd.DataFrame) -> dict:
    timestamps = result["timestamps"]
    dists = result["dists"]
    detections = [
        _event_box(event, timestamps, dists) for event in result["events"]
    ]
    loaded_start = pd.to_datetime(timestamps[0], utc=True)
    loaded_end = pd.to_datetime(timestamps[-1], utc=True)

    def in_loaded(row) -> bool:
        start = pd.to_datetime(row.t_start, utc=True)
        end = pd.to_datetime(row.t_end, utc=True)
        return _intervals_overlap(start, end, loaded_start, loaded_end)

    events = rows[(rows.kind == "event") & rows.apply(in_loaded, axis=1)]
    nuisances = rows[(rows.kind == "nuisance") & rows.apply(in_loaded, axis=1)]
    hit_ids = []
    by_label: dict[str, list[bool]] = {}
    for row in events.itertuples(index=False):
        start = pd.to_datetime(row.t_start, utc=True)
        end = pd.to_datetime(row.t_end, utc=True)
        meter_start = float(min(row.dist_start_m, row.dist_end_m))
        meter_end = float(max(row.dist_start_m, row.dist_end_m))
        hit = any(
            _intervals_overlap(start, end, det[0], det[1])
            and _intervals_overlap(meter_start, meter_end, det[2], det[3])
            for det in detections
        )
        by_label.setdefault(str(row.label), []).append(hit)
        if hit:
            hit_ids.append(str(row.label))

    false_alarms = 0
    nuisance_only = 0
    for det in detections:
        overlaps_event = False
        overlaps_nuisance = False
        for row in events.itertuples(index=False):
            start = pd.to_datetime(row.t_start, utc=True)
            end = pd.to_datetime(row.t_end, utc=True)
            if _intervals_overlap(det[0], det[1], start, end) and _intervals_overlap(
                det[2], det[3], float(row.dist_start_m), float(row.dist_end_m)
            ):
                overlaps_event = True
                break
        if overlaps_event:
            continue
        for row in nuisances.itertuples(index=False):
            start = pd.to_datetime(row.t_start, utc=True)
            end = pd.to_datetime(row.t_end, utc=True)
            if _intervals_overlap(det[0], det[1], start, end) and _intervals_overlap(
                det[2], det[3], float(row.dist_start_m), float(row.dist_end_m)
            ):
                overlaps_nuisance = True
                break
        if overlaps_nuisance:
            nuisance_only += 1
        else:
            false_alarms += 1

    per_label = {
        label: {"hit": sum(flags), "n": len(flags)} for label, flags in sorted(by_label.items())
    }
    n_events = int(sum(item["n"] for item in per_label.values()))
    n_hit = int(sum(item["hit"] for item in per_label.values()))
    return {
        "events_in_loaded_window": n_events,
        "events_hit": n_hit,
        "by_label": per_label,
        "detections": len(detections),
        "false_alarms": false_alarms,
        "nuisance_only": nuisance_only,
    }


def select_synth() -> list[str]:
    sequences = pd.read_csv(SYNTH_SEQ)
    chosen = sequences[
        (sequences.split == "test")
        & (sequences.has_events == True)
        & (sequences.n_traces <= MAX_TRACES)
        & (sequences.n_x <= MAX_SPACE)
    ]
    return [str(seq_id) for seq_id in chosen.seq_id.tolist()]


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(torch.cuda.get_device_name(0), flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(SYNTH_LABELS)
    report = {"device": device, "epochs": 6, "percentile": 99.2, "synthetic": [], "real": []}

    for seq_id in select_synth():
        csv_path = SYNTH_CSV / f"{seq_id}.csv"
        started = time.perf_counter()
        result = run_pipeline(
            str(csv_path),
            plot_path=str(OUT_DIR / f"{seq_id}.png"),
            seed=0,
            epochs=6,
            device=device,
        )
        score = score_against_labels(result, labels[labels.seq_id == seq_id])
        elapsed = round(time.perf_counter() - started, 1)
        row = {"seq_id": seq_id, "seconds": elapsed, "shape": list(result["matrix"].shape), **score}
        report["synthetic"].append(row)
        print(
            f"{seq_id} hit {score['events_hit']}/{score['events_in_loaded_window']} "
            f"detections={score['detections']} false={score['false_alarms']} {elapsed}s",
            flush=True,
        )

    for name in ("2.csv", "3.csv", "4.csv"):
        started = time.perf_counter()
        result = run_pipeline(
            str(REAL_DIR / name),
            plot_path=str(OUT_DIR / f"real_{name}.png"),
            seed=0,
            epochs=6,
            device=device,
        )
        public = [_public_event(event, result["timestamps"], result["dists"]) for event in result["events"]]
        elapsed = round(time.perf_counter() - started, 1)
        report["real"].append(
            {
                "file": name,
                "seconds": elapsed,
                "shape": list(result["matrix"].shape),
                "n_events": len(public),
                "events": public,
            }
        )
        print(f"{name} events={len(public)} {elapsed}s", flush=True)

    out = OUT_DIR / "eval.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
