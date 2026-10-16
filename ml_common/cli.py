"""CLI: python -m ml_common.cli predict --model fiber --out events.json

Без --csv берётся синтетическое поле (не MGTS). Потоки ограничены четырьмя.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml_common.protocol import limit_threads

limit_threads()


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _predict(args) -> int:
    from ml_common.service import predict_field
    from ml_common.synthetic import field_from_csv, planted_field

    if args.csv:
        field = field_from_csv(args.csv)
    else:
        field, _truth = planted_field()
    if args.dts_id:
        field["dts_id"] = args.dts_id
    if args.channel_id:
        field["channel_id"] = args.channel_id
    kwargs = {}
    if args.model == "larpathon":
        percentile = args.percentile
        if percentile is None and not args.csv:
            percentile = 95.0
        kwargs = {"use_ae": bool(args.ae), "epochs": args.epochs, "percentile": percentile}
    document = predict_field(args.model, field, **kwargs)
    _write(Path(args.out), document)
    print(
        f"{document['model_key']} zones={len(document['zones'])} "
        f"seconds={document['runtime']['seconds']} -> {args.out}"
    )
    return 0


def _eval(args) -> int:
    from ml_common.eval import run_eval

    names = [item.strip() for item in args.models.split(",") if item.strip()]
    report = run_eval(names, use_ae=bool(args.ae), epochs=args.epochs)
    public = json.loads(json.dumps(report))
    if not args.full:
        for item in public["models"].values():
            item.pop("events", None)
    _write(Path(args.out), public)
    for name, item in report["models"].items():
        if item.get("ok"):
            metrics = item["metrics"]
            hits = ",".join(f"{hit['name']}:{hit['n_zones']}" for hit in metrics["hits"])
            print(f"{name} ok zones={metrics['n_zones']} hits={hits} seconds={metrics['seconds']}")
        else:
            print(f"{name} FAIL {item['error']}")
    return 0 if all(item.get("ok") for item in report["models"].values()) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ml_common.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    predict = sub.add_parser("predict", help="одно поле -> events.json v1")
    predict.add_argument("--model", required=True, choices=["fiber", "ens_gen2", "larpathon"])
    predict.add_argument("--out", required=True, help="путь events.json")
    predict.add_argument("--csv", default=None, help="MGTS CSV; без флага — синтетика")
    predict.add_argument("--synthetic", action="store_true", help="явно синтетическое поле (значение по умолчанию)")
    predict.add_argument("--dts-id", default=None)
    predict.add_argument("--channel-id", default=None)
    predict.add_argument("--ae", action="store_true", help="larpathon: учить автоэнкодер на этом поле")
    predict.add_argument("--epochs", type=int, default=1)
    predict.add_argument("--percentile", type=float, default=None, help="larpathon: процентиль скора; на синтетике по умолчанию 95")
    predict.set_defaults(func=_predict)

    evaluate = sub.add_parser("eval", help="общие метрики на синтетике")
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--models", default="fiber,larpathon,ens_gen2")
    evaluate.add_argument("--ae", action="store_true")
    evaluate.add_argument("--epochs", type=int, default=1)
    evaluate.add_argument("--full", action="store_true", help="вложить сами events.json в отчёт")
    evaluate.set_defaults(func=_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
