"""Command-line entry for the MGTS thermal anomaly ensemble."""

from __future__ import annotations

import argparse
import json
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"

import torch

from mgts_anomaly import (
    SEGMENT_ROWS,
    TEMPORAL_MEDIAN_STEPS,
    combine_detector_and_opencv,
    iter_csvs,
    median_timestamp_step_minutes,
    run_pipeline,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect thermal anomalies on MGTS DTS CSV profiles."
    )
    parser.add_argument("path", help="One CSV file or a directory of CSV files")
    parser.add_argument("--max-rows", type=int, default=None, help="Read only the first N rows")
    parser.add_argument("--seed", type=int, default=0, help="NumPy and PyTorch seed")
    parser.add_argument("--percentile", type=float, default=99.2, help="Score percentile threshold")
    parser.add_argument("--epochs", type=int, default=6, help="Autoencoder epochs")
    parser.add_argument("--plot-path", default=None, help="PNG path when the input is one CSV")
    parser.add_argument("--plot-dir", default=None, help="Directory for one PNG per CSV")
    parser.add_argument(
        "--events-dir",
        default=None,
        help="Directory for one {filename}_events.json per CSV",
    )
    parser.add_argument("--no-ae", action="store_true", help="Statistical branch only")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--temporal-window",
        type=int,
        default=TEMPORAL_MEDIAN_STEPS,
        help="Absolute temporal-median window in time steps (not a fraction of the file)",
    )
    parser.add_argument(
        "--segment-rows",
        type=int,
        default=SEGMENT_ROWS,
        help="Rows in each score-threshold segment",
    )
    parser.add_argument(
        "--use-opencv",
        action="store_true",
        help="Segment the ensemble score with OpenCV. This is already the default.",
    )
    parser.add_argument(
        "--no-opencv",
        action="store_true",
        help="Skip OpenCV segmentation, the extra figures, and the OpenCV JSON.",
    )
    parser.add_argument(
        "--opencv-score-percentile",
        type=float,
        default=95.0,
        help="OpenCV score-mask percentile. Does not change the detector percentile.",
    )
    parser.add_argument(
        "--opencv-contrast-min",
        type=float,
        default=1.2,
        help="OpenCV neighbor-ring contrast minimum in degrees C.",
    )
    args = parser.parse_args(argv)
    # The off switch wins when both flags are passed. --use-opencv stays for older commands.
    args.use_opencv = not args.no_opencv
    return args


def _plot_path_for(csv_path: str, args: argparse.Namespace, multiple: bool) -> str | None:
    from pathlib import Path

    stem = Path(csv_path).stem
    if args.plot_path and args.plot_dir:
        raise SystemExit("pass only one of --plot-path and --plot-dir")
    if args.plot_path:
        if multiple:
            raise SystemExit("--plot-path is for a single CSV; use --plot-dir for a directory")
        return args.plot_path
    if args.plot_dir:
        out_dir = Path(args.plot_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir / f"anomaly_{stem}.png")
    return str(Path.cwd() / f"anomaly_{stem}.png")


def _event_record(event: dict) -> dict:
    start_m = float(event.get("distance_start_m", 0.0))
    end_m = float(event.get("distance_end_m", event.get("spatial_extent_m", 0.0)))
    t0, t1 = event["t_range"]
    s0, s1 = event["s_range"]
    return {
        "meters": [round(start_m, 4), round(end_m, 4)],
        "duration_min": round(float(event["duration_min"]), 4),
        "amplitude": round(float(event["amplitude"]), 4),
        "local_amplitude": round(float(event["local_amplitude"]), 4),
        "confidence": round(float(event["confidence"]), 4),
        "type_hypothesis": str(event["type_hypothesis"]),
        "periodic": bool(event["periodic"]) if "periodic" in event else False,
        "source": "detector",
        "t_range": [int(t0), int(t1)],
        "s_range": [int(s0), int(s1)],
    }


def _write_events_json(events: list, dest: str) -> None:
    from pathlib import Path

    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [_event_record(event) for event in events]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _opencv_record(component: dict, dists, time_step_min: float) -> dict:
    """Meters and minutes for one OpenCV component. Does not threshold the score."""
    s0, s1 = int(component["s_range"][0]), int(component["s_range"][1])
    t0, t1 = int(component["t_range"][0]), int(component["t_range"][1])
    if s0 > s1:
        s0, s1 = s1, s0
    if t0 > t1:
        t0, t1 = t1, t0
    start_m = float(dists[s0])
    end_m = float(dists[s1])
    if start_m > end_m:
        start_m, end_m = end_m, start_m
    area = int(component["n_pixels"]) if "n_pixels" in component else int(component["area"])
    return {
        "meters": [round(start_m, 4), round(end_m, 4)],
        "duration_min": round((t1 - t0 + 1) * float(time_step_min), 4),
        "area": area,
        "t_range": [t0, t1],
        "s_range": [s0, s1],
        "source": "opencv",
        "type_hypothesis": (
            str(component["type_hypothesis"])
            if "type_hypothesis" in component
            else "unknown"
        ),
        "overlaps_detector": bool(component.get("overlaps_detector", False)),
    }


def _write_opencv_json(components: list, dists, time_step_min: float, dest: str) -> None:
    from pathlib import Path

    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [_opencv_record(component, dists, time_step_min) for component in components]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_combined_json(
    events: list,
    components: list,
    dists,
    time_step_min: float,
    dest: str,
) -> None:
    """Write detector rows, relabeled both on a 30% hit, plus unique OpenCV rows."""
    from pathlib import Path

    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for row in combine_detector_and_opencv(events, components):
        if row["source"] == "opencv":
            payload.append(_opencv_record(row, dists, time_step_min))
            continue
        record = _event_record(row)
        record["source"] = row["source"]
        payload.append(record)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _ranges_overlap(left, right) -> bool:
    a0, a1 = float(left[0]), float(left[1])
    b0, b1 = float(right[0]), float(right[1])
    if a0 > a1:
        a0, a1 = a1, a0
    if b0 > b1:
        b0, b1 = b1, b0
    return a0 <= b1 and b0 <= a1


def opencv_matches_event(component: dict, event: dict) -> bool:
    """Rough match: the meter intervals overlap and the time ranges overlap."""
    return _ranges_overlap(component["meters"], event["meters"]) and _ranges_overlap(
        component["t_range"], event["t_range"]
    )


def split_opencv_matches(components: list, events: list) -> tuple[list, list]:
    """Split OpenCV records into those that meet a detector event and the rest."""
    matched: list = []
    only: list = []
    for component in components:
        if any(opencv_matches_event(component, event) for event in events):
            matched.append(component)
        else:
            only.append(component)
    return matched, only


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("cuda unavailable, using cpu", flush=True)
        device = "cpu"
    csv_paths = list(iter_csvs(args.path))
    multiple = len(csv_paths) > 1
    for csv_path in csv_paths:
        plot_path = _plot_path_for(csv_path, args, multiple)
        if plot_path is not None:
            from pathlib import Path

            Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        result = run_pipeline(
            csv_path,
            use_ae=not args.no_ae,
            percentile=args.percentile,
            max_rows=args.max_rows,
            plot_path=plot_path,
            seed=args.seed,
            epochs=args.epochs,
            device=device,
            window_steps=args.temporal_window,
            segment_rows=args.segment_rows,
            use_opencv=args.use_opencv,
            opencv_score_percentile=args.opencv_score_percentile,
            opencv_contrast_min=args.opencv_contrast_min,
        )
        if args.events_dir:
            from pathlib import Path

            dest = str(Path(args.events_dir) / f"{Path(csv_path).name}_events.json")
            _write_events_json(result["events"], dest)
            print(f"wrote {dest}", flush=True)
            if args.use_opencv:
                step = median_timestamp_step_minutes(result["timestamps"])
                if step is None:
                    step = 2.0
                opencv_dest = str(Path(args.events_dir) / f"{Path(csv_path).name}_opencv.json")
                _write_opencv_json(
                    result.get("opencv_events") or [],
                    result["dists"],
                    step,
                    opencv_dest,
                )
                print(f"wrote {opencv_dest}", flush=True)
                combined_dest = str(Path(args.events_dir) / f"{Path(csv_path).name}_combined.json")
                _write_combined_json(
                    result["events"],
                    result.get("opencv_events") or [],
                    result["dists"],
                    step,
                    combined_dest,
                )
                print(f"wrote {combined_dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
