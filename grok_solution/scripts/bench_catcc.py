"""GroupKFold CATCC против EventNet и LGBM на бюджетах меток 1/5/10%.

Реальные трассы не читаются. Официальный test не участвует в критерии включения.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dts import events as E, semi_contrastive as C  # noqa: E402


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def released_ceiling(ds):
    """Полные модели models_release/ только как потолок на v8 test. Без дообучения."""
    import joblib
    import torch
    from dts import models

    ck_path = ROOT / "models_release" / "eventnet.pt"
    lg_path = ROOT / "models_release" / "lgbm.joblib"
    if not ck_path.exists() or not lg_path.exists():
        return {"skipped": True}
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    mu = np.asarray(ck["mu"], np.float32)
    sd = np.asarray(ck["sd"], np.float32)
    net = models.EventNet(len(E.CLASSES), len(mu), cin=int(ck.get("cin", 3)))
    net.load_state_dict(ck["state"])
    net.eval()
    obj = joblib.load(lg_path)
    lgbm = obj["model"] if isinstance(obj, dict) else obj
    kind = obj.get("kind", "full") if isinstance(obj, dict) else "full"
    mask = (np.asarray(ds["version"]).astype(str) == "v8") & (np.asarray(ds["split"]).astype(str) == "test")
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return {"skipped": True, "reason": "no v8 test windows"}
    scal = np.asarray(ds["scal"][idx], np.float32)
    img = np.asarray(ds["img"][idx], np.float32)
    ctx = np.asarray(ds["ctx"][idx], np.float32)
    y = np.asarray(ds["y"][idx])
    s = torch.from_numpy(((scal - mu) / sd).astype(np.float32))
    xt = torch.from_numpy(img)
    temperature = float(ck.get("temperature", 1.0))
    with torch.no_grad():
        lo = net(xt, s)
        lo = 0.5 * (lo + net(torch.flip(xt, [3]), s))
        lo = lo / max(temperature, 1e-2)
        p_net = torch.softmax(lo, 1).numpy()
    feat = C.lgbm_feature_matrix(img, scal, ctx)
    if getattr(lgbm, "n_features_in_", feat.shape[1]) != feat.shape[1]:
        return {"skipped": True, "reason": f"lgbm expects {lgbm.n_features_in_} cols, got {feat.shape[1]}", "kind": kind}
    raw = np.asarray(lgbm.predict_proba(feat), np.float64)
    p_lgb = np.zeros_like(p_net)
    for col, cls in enumerate(lgbm.classes_):
        p_lgb[:, int(cls)] = raw[:, col]
    p_lgb = E.temper_proba(p_lgb, float(ck.get("lgbm_temperature", 1.0)))
    weight = float(np.clip(ck.get("cnn_weight", 0.5), 0.0, 1.0))
    blend = weight * p_net + (1.0 - weight) * p_lgb
    return {
        "skipped": False,
        "split": "v8 test",
        "note": "потолок 100% меток v8, не участник сравнения бюджетов и не дообучался здесь",
        "n": int(len(idx)),
        "cnn_weight": weight,
        "eventnet": C.score_probs(y, p_net),
        "lgbm": C.score_probs(y, p_lgb),
        "blend": C.score_probs(y, blend),
    }


def _fmt(block, name):
    return block[name]["macro_f1"]


def _per_class_row(block, name):
    cells = []
    for cls in E.CLASSES:
        cells.append(f"{block[name]['per_class'][cls]:.3f}")
    return cells


def render_report(result) -> str:
    b = result["budgets"]
    lines = [
        "# TZ E — CATCC (полуобучение классов, TS-TCC / CA-TCC)",
        "",
        "Нейросетевой классификатор типа события по окну синтетики. Текущие EventNet и LGBM",
        "переобучены на тех же индексах меток и том же GroupKFold по эпизоду (`seq_id`).",
        "Реальные трассы MGTS не используются ни как метки, ни как неразмеченные окна.",
        "",
        "## Окно",
        "",
        "Бокс совпадает с `events.box_indices` / `crop_and_features` (то же окно, что у EventNet).",
        "Из него берутся два ряда длины 64:",
        "",
        "- средний ряд зоны — медиана `dT_time` и `dT_space` поперёк волокна, плюс доля валидных узлов;",
        "- поперечный профиль — медиана тех же каналов по времени.",
        "",
        "Единицы — σ, как в кэше. В сеть ряд входит делением на фиксированный CLIP=12, без подгонки",
        "масштаба по выборке. Голова видит эмбеддинг и четыре числа амплитуды (медиана, max, min, пик).",
        "",
        "## Самообучение и дообучение",
        "",
        "CA-TCC: 1D-свёртки времени и профиля, кросс-внимание временных токенов к профилю,",
        "временной контраст (контекст одного вида предсказывает будущие шаги другого, NT-Xent)",
        "и контекстный NT-Xent. BatchNorm нет — он стирает величину перегрева.",
        "",
        "Аугментации SSL: маскирование отсчётов и частично перекрывающаяся обрезка с линейным",
        "возвратом на исходную сетку. Масштабирования амплитуды и перестановок нет.",
        "Оба вида одной серии перекрываются не меньше чем на половину более короткого окна.",
        "",
        "Дальше, внутри обучающих эпизодов фолда: голова на замороженном энкодере, короткое",
        "дообучение энкодера на бюджете меток, псевдометки оставшихся окон при уверенности ≥ 0.9,",
        "ещё одна эпохальная дообучение головы на метках и псевдометках (вес псевдометки 0.5).",
        "Точность псевдометок по истинным классам пишется только как диагностика и порог не выбирает.",
        "",
        "Бюджет 1/5/10% — стратифицированная доля событий обучающего фолда, в каждом присутствующем",
        "классе хотя бы один пример. v8 и v10 одного `seq_id` не расходятся по фолдам: это парные",
        "генерации. Официальный test в критерий не входит.",
        "",
        "## Критерий",
        "",
        "Модуль включается (`ENABLED=True`), только если OOF macro-F1 при 5% меток строго выше",
        "и EventNet, и LGBM. Иначе код остаётся доступным, а классификатор этапа 2 не подменяется.",
        "",
        f"Решение этого прогона: **{'включён' if result['enabled'] else 'выключен'}**.",
        "",
        "## OOF GroupKFold (v8+v10, dev)",
        "",
        f"Окон {result['n']['windows']}, dev {result['n']['dev']}, test {result['n']['test']},",
        f"эпизодов dev {result['n']['episodes_dev']}. Время {result['seconds']} с.",
        "Реальные данные: не использовались.",
        "",
        "| бюджет | CATCC | EventNet | LGBM | отказ CATCC | отказ EventNet | отказ LGBM | доля псевдо |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("0.01", "0.05", "0.10"):
        block = b[key]
        pseudo = block["pseudo"].get("frac_pseudo_mean")
        pseudo_s = "—" if pseudo is None else f"{pseudo:.3f}"
        lines.append(
            f"| {float(key):.0%} | {block['catcc']['macro_f1']:.3f} | {block['eventnet']['macro_f1']:.3f} | "
            f"{block['lgbm']['macro_f1']:.3f} | {block['catcc']['reject_rate']:.3f} | "
            f"{block['eventnet']['reject_rate']:.3f} | {block['lgbm']['reject_rate']:.3f} | {pseudo_s} |"
        )
    lines += [
        "",
        "Отказ — доля окон с max softmax < 0.9. Macro-F1 считается по всем окнам, без отказа.",
        "Диагностическая точность псевдометок (не для отбора):",
    ]
    for key in ("0.01", "0.05", "0.10"):
        acc = b[key]["pseudo"].get("pseudo_accuracy_diagnostic_mean")
        acc_s = "—" if acc is None else f"{acc:.3f}"
        lines.append(f"- {float(key):.0%}: {acc_s}, среднее число псевдометок {b[key]['pseudo'].get('n_pseudo_mean', 0):.1f}.")
    lines += [
        "",
        "### F1 по классам при 5%",
        "",
        "| класс | CATCC | EventNet | LGBM |",
        "|---|---:|---:|---:|",
    ]
    for cls in E.CLASSES:
        lines.append(
            f"| {cls} | {b['0.05']['catcc']['per_class'][cls]:.3f} | "
            f"{b['0.05']['eventnet']['per_class'][cls]:.3f} | {b['0.05']['lgbm']['per_class'][cls]:.3f} |"
        )
    lines += ["", "### Тот же OOF отдельно по версии, macro-F1", "", "| бюджет | версия | CATCC | EventNet | LGBM |", "|---:|---|---:|---:|---:|"]
    for key in ("0.01", "0.05", "0.10"):
        for ver, block_v in b[key]["catcc"]["by_version"].items():
            lines.append(
                f"| {float(key):.0%} | {ver} | {block_v['macro_f1']:.3f} | "
                f"{b[key]['eventnet']['by_version'][ver]['macro_f1']:.3f} | "
                f"{b[key]['lgbm']['by_version'][ver]['macro_f1']:.3f} |"
            )
    lines += ["", "По фолдам (macro-F1):"]
    for key in ("0.01", "0.05", "0.10"):
        folds = ", ".join(f"{v:.3f}" for v in b[key]["catcc"]["per_fold"])
        lines.append(f"- CATCC {float(key):.0%}: {folds}")
    if result.get("test"):
        lines += ["", "## Официальный test (отчёт, не критерий)", "", "| бюджет | CATCC | EventNet | LGBM |", "|---:|---:|---:|---:|"]
        for key, block in result["test"].items():
            lines.append(
                f"| {float(key):.0%} | {block['catcc']['macro_f1']:.3f} | {block['eventnet']['macro_f1']:.3f} | {block['lgbm']['macro_f1']:.3f} |"
            )
    ceiling = result.get("released_full_labels") or {}
    if ceiling and not ceiling.get("skipped"):
        lines += [
            "",
            "## Потолок релиза на v8 test",
            "",
            "Это `models_release` при полном обучении, не бюджет. В критерий 5% не входит.",
            "",
            f"| EventNet | LGBM | blend |",
            f"|---:|---:|---:|",
            f"| {ceiling['eventnet']['macro_f1']:.3f} | {ceiling['lgbm']['macro_f1']:.3f} | {ceiling['blend']['macro_f1']:.3f} |",
        ]
    lines += [
        "",
        "## Что принято",
        "",
        result.get("decision_text", ""),
        "",
        "Файлы: `dts/semi_contrastive.py`, `scripts/bench_catcc.py`, `tests/test_catcc.py`, `artifacts/eval/catcc.json`.",
        "",
    ]
    return "\n".join(lines)


def render_round2(round1, probe, enabled: bool, decision: str) -> str:
    """Отчёт обеих попыток. Базовые EventNet/LGBM — цифры первого прогона, без переобучения."""
    r1 = round1["budgets"]
    r2 = probe["budgets"]
    collapse = probe["collapse"]
    lines = [
        "# TZ E — CATCC (полуобучение классов, TS-TCC / CA-TCC)",
        "",
        "Нейросетевой классификатор типа события по окну синтетики. Сравнение с EventNet и LGBM",
        "идёт на одних и тех же индексах меток и одном GroupKFold по эпизоду (`seq_id`).",
        "Реальные трассы MGTS не используются ни как метки, ни как неразмеченные окна.",
        "",
        f"Решение после второй попытки: **{'включён' if enabled else 'выключен'}**.",
        "",
        decision,
        "",
        "## Окно",
        "",
        "Бокс совпадает с `events.box_indices` / `crop_and_features` (то же окно, что у EventNet).",
        "Два ряда длины 64: медиана зоны поперёк волокна (временной ряд) и медиана по времени",
        "(поперечный профиль), каналы `dT_time`, `dT_space`, доля валидных узлов. Вход делится",
        "на фиксированный CLIP=12. К эмбеддингу добавляются четыре числа амплитуды",
        "(медиана, max, min, пик канала `dT_time`).",
        "",
        "## Первая попытка — MLP-голова",
        "",
        "CA-TCC без BatchNorm: 1D-свёртки, кросс-внимание, временной и контекстный NT-Xent.",
        "Аугментации только на SSL: маскирование и перекрывающаяся обрезка, без масштаба амплитуды",
        "и без перестановок. Дальше голова на бюджете 1/5/10% и псевдометки при уверенности ≥ 0.9.",
        "Официальный test в критерий не входил.",
        "",
        "Уверенность MLP нигде не достигла 0.9, поэтому псевдометки не использовались, а доля отказов",
        "равна 1. OOF macro-F1 при 5% ниже и EventNet, и LGBM. Модуль оставлен выключенным.",
        "",
        "| бюджет | CATCC MLP | EventNet | LGBM | отказ MLP | доля псевдо |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("0.01", "0.05", "0.10"):
        block = r1[key]
        pseudo = block["pseudo"].get("frac_pseudo_mean")
        pseudo_s = "—" if pseudo is None else f"{pseudo:.3f}"
        lines.append(
            f"| {float(key):.0%} | {block['catcc']['macro_f1']:.3f} | {block['eventnet']['macro_f1']:.3f} | "
            f"{block['lgbm']['macro_f1']:.3f} | {block['catcc']['reject_rate']:.3f} | {pseudo_s} |"
        )
    lines += [
        "",
        "F1 MLP по классам при 5%:",
        "",
        "| класс | CATCC MLP | EventNet | LGBM |",
        "|---|---:|---:|---:|",
    ]
    for cls in E.CLASSES:
        lines.append(
            f"| {cls} | {r1['0.05']['catcc']['per_class'][cls]:.3f} | "
            f"{r1['0.05']['eventnet']['per_class'][cls]:.3f} | {r1['0.05']['lgbm']['per_class'][cls]:.3f} |"
        )
    lines += [
        "",
        "## Вторая попытка — замороженные SSL-признаки",
        "",
        "Энкодер тот же. Самообучение: AdamW lr 3e-4, betas (0.9, 0.99), weight decay 3e-4,",
        "температура NT-Xent 0.2, потеря `1.0 * временной контраст + 0.7 * контекстный`, batch 64.",
        "Негативы — другие окна батча (другие эпизоды), не сдвиги той же серии. Маски только на SSL.",
        "После SSL энкодер заморожен. На эмбеддинге и четырёх числах амплитуды обучаются две головы:",
        "логистический зонд (признаки стандартизованы по обучающему фолду) и LGBM.",
        "Псевдометки с уверенностью ≥ 0.9 берутся только если их доля среди неразмеченных ≥ 0.2,",
        "вес такой метки 0.3. Иначе голова остаётся только на истинных метках бюджета.",
        "EventNet и LGBM на 59 признаках заново не учились: в таблице их OOF из первой попытки",
        "на том же кэше, том же seed и том же GroupKFold. Официальный test во второй попытке не пересчитывался.",
        "",
        f"Коллапс представлений: средний std эмбеддинга на обучающем фолде {collapse['mean_std']:.4f}.",
        f"Порог коллапса 1e-3, сработал: **{'да' if collapse['collapsed'] else 'нет'}**.",
        "",
        "| фолд | n | dim | mean std | min std | max std |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in collapse["folds"]:
        lines.append(
            f"| {int(row['fold']) + 1} | {row['n']} | {row['dim']} | {row['mean_std']:.4f} | "
            f"{row['min_std']:.4f} | {row['max_std']:.4f} |"
        )
    lines += [
        "",
        f"Окон {round1['n']['windows']}, dev {round1['n']['dev']}. SSL второй попытки {probe['seconds']} с.",
        "Реальные данные: не использовались.",
        "",
        "| бюджет | линейный зонд | эмбеддинг+LGBM | EventNet | LGBM 59 | отказ зонда | отказ emb+LGBM | псевдо зонд | псевдо emb+LGBM |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("0.01", "0.05", "0.10"):
        lin, emb = r2[key]["linear"], r2[key]["embed_lgbm"]
        lines.append(
            f"| {float(key):.0%} | {lin['macro_f1']:.3f} | {emb['macro_f1']:.3f} | "
            f"{r1[key]['eventnet']['macro_f1']:.3f} | {r1[key]['lgbm']['macro_f1']:.3f} | "
            f"{lin['reject_rate']:.3f} | {emb['reject_rate']:.3f} | "
            f"{lin['pseudo'].get('frac_pseudo_mean', 0):.3f} | {emb['pseudo'].get('frac_pseudo_mean', 0):.3f} |"
        )
    lines += [
        "",
        "Отказ — доля окон с max вероятности < 0.9. Macro-F1 считается по всем окнам, без отказа.",
        "Доля псевдо — сколько неразмеченных окон фолда реально вошло в дообучение (0, если порог доли не взят).",
        "",
        "| бюджет | зонд: фолдов с псевдо | доля conf≥0.9 | emb+LGBM: фолдов с псевдо | доля conf≥0.9 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for key in ("0.01", "0.05", "0.10"):
        lin, emb = r2[key]["linear"]["pseudo"], r2[key]["embed_lgbm"]["pseudo"]
        lines.append(
            f"| {float(key):.0%} | {lin.get('used_folds', 0)} | {lin.get('frac_confident_mean', 0):.3f} | "
            f"{emb.get('used_folds', 0)} | {emb.get('frac_confident_mean', 0):.3f} |"
        )
    lines += [
        "",
        "### F1 по классам при 5%",
        "",
        "| класс | линейный зонд | эмбеддинг+LGBM | EventNet | LGBM 59 |",
        "|---|---:|---:|---:|---:|",
    ]
    for cls in E.CLASSES:
        lines.append(
            f"| {cls} | {r2['0.05']['linear']['per_class'][cls]:.3f} | "
            f"{r2['0.05']['embed_lgbm']['per_class'][cls]:.3f} | "
            f"{r1['0.05']['eventnet']['per_class'][cls]:.3f} | {r1['0.05']['lgbm']['per_class'][cls]:.3f} |"
        )
    lines += ["", "По фолдам при 5% (macro-F1):"]
    lines.append("- линейный зонд: " + ", ".join(f"{v:.3f}" for v in r2["0.05"]["linear"]["per_fold"]))
    lines.append("- эмбеддинг+LGBM: " + ", ".join(f"{v:.3f}" for v in r2["0.05"]["embed_lgbm"]["per_fold"]))
    lines.append("- MLP первой попытки: " + ", ".join(f"{v:.3f}" for v in r1["0.05"]["catcc"]["per_fold"]))
    if round1.get("test"):
        lines += [
            "",
            "## Официальный test первой попытки (отчёт, не критерий)",
            "",
            "| бюджет | CATCC MLP | EventNet | LGBM |",
            "|---:|---:|---:|---:|",
        ]
        for key, block in round1["test"].items():
            lines.append(
                f"| {float(key):.0%} | {block['catcc']['macro_f1']:.3f} | "
                f"{block['eventnet']['macro_f1']:.3f} | {block['lgbm']['macro_f1']:.3f} |"
            )
    ceiling = round1.get("released_full_labels") or {}
    if ceiling and not ceiling.get("skipped"):
        lines += [
            "",
            "## Потолок релиза на v8 test",
            "",
            "Это `models_release` при полном обучении, не бюджет. В критерий 5% не входит.",
            "",
            "| EventNet | LGBM | blend |",
            "|---:|---:|---:|",
            f"| {ceiling['eventnet']['macro_f1']:.3f} | {ceiling['lgbm']['macro_f1']:.3f} | {ceiling['blend']['macro_f1']:.3f} |",
        ]
    lines += [
        "",
        "## Критерий",
        "",
        "Модуль включается (`ENABLED=True`), только если лучшая из двух голов второй попытки",
        "при 5% меток строго выше и EventNet, и LGBM первой попытки. Иначе этап 2 не подменяется.",
        "",
        "Файлы: `dts/semi_contrastive.py`, `scripts/bench_catcc.py`, `tests/test_catcc.py`, `artifacts/eval/catcc.json`.",
        "",
    ]
    return "\n".join(lines)


def run_round2(args) -> None:
    """Одна починка: SSL заново, головы на замороженных признаках. Базовые модели не учим."""
    cache = Path(args.cache)
    if not cache.exists():
        raise SystemExit(f"window cache missing: {cache}")
    prior_path = Path(args.out)
    if not prior_path.exists():
        raise SystemExit(f"round-1 json missing: {prior_path}")
    round1 = json.loads(prior_path.read_text())
    print("load", cache, flush=True)
    ds = C.load_windows(cache)
    print("windows", len(ds["y"]), "round2 frozen probes", flush=True)
    probe = C.run_probe_benchmark(
        ds, budgets=(0.01, 0.05, 0.10), n_splits=args.folds, seed=args.seed,
        ssl_epochs=args.ssl_epochs, batch_size=64, width=32, threads=args.threads,
        ssl_lr=3e-4, ssl_wd=3e-4, pseudo_min_frac=0.2, pseudo_weight=0.3, temperature=0.2)
    focus_r1 = round1["budgets"]["0.05"]
    enet5 = float(focus_r1["eventnet"]["macro_f1"])
    lgbm5 = float(focus_r1["lgbm"]["macro_f1"])
    lin5 = float(probe["budgets"]["0.05"]["linear"]["macro_f1"])
    emb5 = float(probe["budgets"]["0.05"]["embed_lgbm"]["macro_f1"])
    best5 = max(lin5, emb5)
    best_name = "линейный зонд" if lin5 >= emb5 else "эмбеддинг+LGBM"
    enabled = C.decision_enabled(best5, enet5, lgbm5)
    if enabled:
        decision = (
            f"Принято для экспериментального включения: при 5% меток {best_name} "
            f"(macro-F1 {best5:.3f}) строго выше EventNet ({enet5:.3f}) и LGBM на 59 признаках ({lgbm5:.3f}). "
            "Цифры базовых моделей взяты из первой попытки на тех же индексах. "
            "`dts.semi_contrastive.ENABLED = True`. В сквозной pipeline этап 2 сам модуль не вызывает."
        )
    else:
        decision = (
            f"Оставлено выключенным. При 5% меток ни линейный зонд ({lin5:.3f}), ни LGBM на SSL-эмбеддинге "
            f"({emb5:.3f}) не выше LGBM на 59 признаках ({lgbm5:.3f}) и EventNet ({enet5:.3f}). "
            f"Средний std эмбеддинга {probe['collapse']['mean_std']:.4f}, "
            f"коллапс {'да' if probe['collapse']['collapsed'] else 'нет'}. "
            "`ENABLED = False`, этап 2 не подменяется."
        )
    merged = dict(round1)
    merged["enabled"] = bool(enabled)
    merged["round2"] = to_jsonable(probe)
    merged["decision_text"] = decision
    merged["criterion_round2"] = (
        "better of frozen linear probe and embedding LGBM, OOF macro-F1 at 5% labels, "
        "strictly above round-1 EventNet and LGBM"
    )
    set_enabled_flag(bool(enabled))
    prior_path.write_text(json.dumps(to_jsonable(merged), ensure_ascii=False, indent=1))
    Path(args.report).write_text(render_round2(round1, probe, enabled, decision))
    b1, b5, b10 = probe["budgets"]["0.01"], probe["budgets"]["0.05"], probe["budgets"]["0.10"]
    winner = b5["linear"] if lin5 >= emb5 else b5["embed_lgbm"]
    pc = ",".join(f"{name}={winner['per_class'][name]:.3f}" for name in E.CLASSES)
    line = (
        f"CATCC2_DONE macro5_linear={lin5:.4f} macro5_emb_lgbm={emb5:.4f} "
        f"lgbm5={lgbm5:.4f} enet5={enet5:.4f} "
        f"mean_std={probe['collapse']['mean_std']:.4f} collapsed={int(probe['collapse']['collapsed'])} "
        f"pseudo5_linear={b5['linear']['pseudo'].get('frac_pseudo_mean', 0):.3f} "
        f"pseudo5_emb={b5['embed_lgbm']['pseudo'].get('frac_pseudo_mean', 0):.3f} "
        f"reject5_linear={b5['linear']['reject_rate']:.3f} "
        f"reject5_emb={b5['embed_lgbm']['reject_rate']:.3f} "
        f"macro1_linear={b1['linear']['macro_f1']:.4f} macro1_emb={b1['embed_lgbm']['macro_f1']:.4f} "
        f"macro10_linear={b10['linear']['macro_f1']:.4f} macro10_emb={b10['embed_lgbm']['macro_f1']:.4f} "
        f"enabled={int(enabled)} n={round1['n']['windows']} sec={probe['seconds']} per_class5={pc}\n"
    )
    with open(args.log, "a", encoding="utf-8") as fh:
        fh.write(line)
    print(line, flush=True)
    print("wrote", prior_path, args.report, flush=True)


def set_enabled_flag(enabled: bool) -> None:
    path = ROOT / "dts" / "semi_contrastive.py"
    text = path.read_text()
    updated, n = re.subn(r"^ENABLED = (True|False)$", f"ENABLED = {bool(enabled)}", text, count=1, flags=re.M)
    if n != 1:
        raise RuntimeError("ENABLED flag not found")
    if updated != text:
        path.write_text(updated)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/catcc_windows_v1.npz")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "eval" / "catcc.json"))
    ap.add_argument("--report", default=str(ROOT / "docs" / "TZ_E_CATCC.md"))
    ap.add_argument("--log", default="/workspace/grok_logs/_events.log")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ssl-epochs", type=int, default=8)
    ap.add_argument("--sup-epochs", type=int, default=8)
    ap.add_argument("--pseudo-epochs", type=int, default=4)
    ap.add_argument("--eventnet-epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--limit-files", type=int, default=0)
    ap.add_argument("--skip-test-fit", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--round", type=int, default=1, choices=(1, 2))
    args = ap.parse_args()
    if args.round == 2:
        run_round2(args)
        return
    t0 = time.time()
    cache = Path(args.cache)
    if cache.exists() and not args.rebuild and args.limit_files <= 0:
        print("load", cache, flush=True)
        ds = C.load_windows(cache)
    else:
        ds = C.collect_synth_windows(
            ROOT, versions=("v8", "v10"), splits=("train", "val", "test"),
            limit_files=args.limit_files or None)
        if args.limit_files <= 0:
            n_v8 = int((ds["version"] == "v8").sum())
            n_v10 = int((ds["version"] == "v10").sum())
            if n_v8 < 2000 or n_v10 < 2000:
                raise SystemExit(f"too few windows v8={n_v8} v10={n_v10}")
            C.save_windows(cache, ds)
            print("saved", cache, flush=True)
    print("windows", len(ds["y"]), "load/collect", round(time.time() - t0, 1), flush=True)
    result = C.run_group_benchmark(
        ds, budgets=(0.01, 0.05, 0.10), n_splits=args.folds, seed=args.seed,
        ssl_epochs=args.ssl_epochs, sup_epochs=args.sup_epochs, pseudo_epochs=args.pseudo_epochs,
        eventnet_epochs=args.eventnet_epochs, batch_size=args.batch_size, width=args.width,
        threads=args.threads, do_test=not args.skip_test_fit, verbose=True)
    try:
        result["released_full_labels"] = released_ceiling(ds)
    except Exception as exc:
        result["released_full_labels"] = {"skipped": True, "reason": str(exc)}
    focus = result["budgets"]["0.05"]
    if result["enabled"]:
        decision = (
            "Принято для экспериментального включения: при 5% меток OOF macro-F1 выше EventNet и LGBM "
            "на тех же индексах. `dts.semi_contrastive.ENABLED = True`. В сквозной pipeline этап 2 "
            "сам модуль не вызывает — флаг только фиксирует, что критерий ТЗ выполнен."
        )
    else:
        decision = (
            "Оставлено выключенным. При 5% меток OOF macro-F1 не выше одновременно EventNet и LGBM "
            f"(CATCC {focus['catcc']['macro_f1']:.3f}, EventNet {focus['eventnet']['macro_f1']:.3f}, "
            f"LGBM {focus['lgbm']['macro_f1']:.3f}). `ENABLED = False`, этап 2 не подменяется."
        )
    result["decision_text"] = decision
    set_enabled_flag(bool(result["enabled"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_jsonable(result), ensure_ascii=False, indent=1))
    Path(args.report).write_text(render_report(result))
    b1, b5, b10 = result["budgets"]["0.01"], result["budgets"]["0.05"], result["budgets"]["0.10"]
    pc = ",".join(f"{name}={b5['catcc']['per_class'][name]:.3f}" for name in E.CLASSES)
    line = (
        f"CATCC_DONE macro5={b5['catcc']['macro_f1']:.4f} enet5={b5['eventnet']['macro_f1']:.4f} "
        f"lgbm5={b5['lgbm']['macro_f1']:.4f} macro1={b1['catcc']['macro_f1']:.4f} "
        f"enet1={b1['eventnet']['macro_f1']:.4f} lgbm1={b1['lgbm']['macro_f1']:.4f} "
        f"macro10={b10['catcc']['macro_f1']:.4f} enet10={b10['eventnet']['macro_f1']:.4f} "
        f"lgbm10={b10['lgbm']['macro_f1']:.4f} enabled={int(result['enabled'])} "
        f"reject5={b5['catcc']['reject_rate']:.3f} pseudo5={b5['pseudo'].get('frac_pseudo_mean', 0):.3f} "
        f"n={result['n']['windows']} sec={result['seconds']} per_class5={pc}\n"
    )
    with open(args.log, "a", encoding="utf-8") as fh:
        fh.write(line)
    print(line, flush=True)
    print("wrote", out, args.report, flush=True)


if __name__ == "__main__":
    main()
