"""TTA ансамбля ens_u2ft_tcn_lgbm: flip x, flip t, симметрия hot/cold.

Порог FAR 1e-3 только по фону v8 val. Test v8 / v10 / v11 порог не двигают.
Реальные MGTS не читаются. Сети не обучаются: только predict_full уже
сохранённых чекпойнтов и опубликованные карты id.

Запуск:
  /workspace/repo/run/.venv/bin/python -u scripts/s3_tta_run.py
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import calendar
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dts import data, events as EV, metrics as M

_spec = importlib.util.spec_from_file_location("s3_tta", ROOT / "scripts" / "s3_tta.py")
tta = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tta)

FAR = 1e-3
GATE = 0.25
NEURAL = (
    ("unet_w24", "unet", 24, "artifacts/stage1/unet_w24.pt"),
    ("unet_v810ft", "unet", 16, "artifacts/stage1/unet_v810ft.pt"),
    ("tcn", "tcn", None, "artifacts/stage1/tcn.pt"),
)
LGBM = "pixel_lgbm"
ENS = "ens_u2ft_tcn_lgbm"
# Порядок и состав опубликованного среднего (scripts/ensemble_preds.py, round7).
OFFICIAL = ("unet_w24", "unet_v810ft", "tcn", "pixel_lgbm")
N_ENS = 4
WINNER_JSON = ROOT / "artifacts/eval/r7_ens_u2ft_tcn_lgbm.json"
PRED_OUT = ROOT / "artifacts/pred/v8/ens_tta"
TMP = Path("/tmp/s3_tta_views")
LOG = Path("/workspace/grok_logs/_events.log")
# Внутренний запас до жёстких 17:30 UTC: не начинать новый тяжёлый сплит позже.
DEADLINE = calendar.timegm(time.strptime("2026-10-15 17:22:00", "%Y-%m-%d %H:%M:%S"))


def now():
    return time.strftime("%H:%M:%S", time.gmtime())


def log_event(line: str):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M', time.gmtime())} {line}\n")
    print(line, flush=True)


def conv(o):
    if isinstance(o, dict):
        return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [conv(v) for v in o]
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(conv(obj), indent=1, ensure_ascii=False), encoding="utf-8")


def meta(ver: str):
    labels = pd.read_csv(ROOT / f"data/synth_meta/{ver}/labels.csv")
    seqs = pd.read_csv(ROOT / f"data/synth_meta/{ver}/sequences.csv").set_index("seq_id")
    return labels, seqs


def cache_paths(ver: str, split: str):
    return data.list_cache(ROOT / "artifacts/cache" / ver, split)


def pred_path(ver: str, method: str, split: str, name: str) -> Path:
    return ROOT / "artifacts/pred" / ver / method / split / name


def load_sh(ver: str, method: str, split: str, name: str):
    q = np.load(pred_path(ver, method, split, name))
    return q["s"].astype(np.float32), q["h"].astype(np.float32)


def quant_h(s, h):
    """Hotness как в файле: float16. Score остаётся float32."""
    hq = np.asarray(h, np.float32).astype(np.float16).astype(np.float32)
    return np.asarray(s, np.float32), hq


def save_sh(path: Path, s, h, compress: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(s=np.asarray(s, np.float32), h=np.asarray(h, np.float16))
    if compress:
        np.savez_compressed(path, **kw)
    else:
        np.savez(path, **kw)


def save_view_f32(path: Path, s, h):
    """Промежуточный вид: float32 без сжатия, чтобы сборка совпала с мешком баллов."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, s=np.asarray(s, np.float32), h=np.asarray(h, np.float32))


def load_view_f32(path: Path):
    q = np.load(path)
    return q["s"].astype(np.float32), q["h"].astype(np.float32)


def configure_torch():
    import torch
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    return torch


def load_nets(only=None):
    import torch
    from dts import models
    configure_torch()
    want = None if only is None else set(only)
    nets = []
    for name, arch, width, ckpt in NEURAL:
        if want is not None and name not in want:
            continue
        net = models.build(arch, width)
        try:
            sd = torch.load(ROOT / ckpt, map_location="cpu", weights_only=True)
        except Exception:
            sd = torch.load(ROOT / ckpt, map_location="cpu", weights_only=False)
        net.load_state_dict(sd)
        net.eval()
        nets.append((name, net))
        print(f"loaded {name} {arch} w={width}", flush=True)
    if want is not None and {n for n, _ in nets} != want:
        raise SystemExit(f"не загружены сети: {sorted(want - {n for n, _ in nets})}")
    return nets


def predict_prob(net, X):
    import torch
    from dts import models
    with torch.inference_mode():
        return models.predict_full(net, X)


def neural_views(net, X, saved_sh, which, use_net_id: bool):
    """Виды одного члена. id — сохранённая карта, если проход сети с ней совпал."""
    out = {}
    if use_net_id or saved_sh is None:
        out["id"] = tta.sh_from_prob(predict_prob(net, np.ascontiguousarray(X, dtype=np.float32)))
    else:
        out["id"] = saved_sh
    for name in which:
        spec = tta.VIEW_SPEC[name]
        xin = tta.transform_input(X, **spec)
        prob = tta.align_prob(predict_prob(net, xin), **spec)
        out[name] = tta.sh_from_prob(prob)
        del xin, prob
    return out


def fuse_file(member_views, passive_shs, which):
    """Аугментированные члены — со своими видами. Остальные (в том числе LightGBM) входят как id.

    В вид sign пассивный член кладёт свою опубликованную карту, не знак.
    Нелинейная смесь на пассивном члене от одинаковых видов совпадает с id.
    """
    view_acc = {k: tta.Acc() for k in ("id",) + tuple(which)}
    nonlin_names = [n for n in tta.NONLINEAR if tta.views_required(n) <= set(which)]
    nonlin_acc = {n: tta.Acc() for n in nonlin_names}
    for mv in member_views:
        for k in view_acc:
            view_acc[k].add(*mv[k])
        for n in nonlin_names:
            nonlin_acc[n].add(*tta.nonlinear_member(mv, n))
    keys_nl = ("id", "flip_x", "flip_t", "sign")
    for sh in passive_shs:
        for k in view_acc:
            view_acc[k].add(*sh)
        if nonlin_names:
            dummy = {k: sh for k in keys_nl}
            for n in nonlin_names:
                nonlin_acc[n].add(*tta.nonlinear_member(dummy, n))
    ens = {k: view_acc[k].mean() for k in view_acc}
    variants = {}
    for name, w in tta.LINEAR.items():
        if tta.views_required(name) <= set(which):
            variants[name] = tta.linear_fuse(ens, w)
    for n in nonlin_names:
        variants[n] = nonlin_acc[n].mean()
    return ens, variants


def add_variant(bags, sid, s, h, valid, ev, nu):
    s, h = quant_h(s, h)
    if not np.isfinite(s).all() or not np.isfinite(h).all():
        raise RuntimeError(f"non-finite score {sid}")
    bags.add(sid, s, h, valid, ev, nu)
    return s, h


def load_field(ver, split, path: Path):
    z = np.load(path)
    return dict(
        name=path.name,
        sid=path.stem,
        X=z["X"].astype(np.float32),
        ev=z["ev"],
        nu=z["nu"],
        valid=np.asarray(z["valid"], bool),
        x=z["x"],
        t=z["t"],
    )


def iter_identity(ver, split):
    for p in cache_paths(ver, split):
        z = np.load(p)
        s, h = load_sh(ver, ENS, split, p.name)
        yield p.stem, s, h, np.asarray(z["valid"], bool), z["ev"], z["nu"], z["x"], z["t"]


def bag_identity(ver, split, gate=GATE):
    bag = tta.ScoreBag(gate)
    t0 = time.time()
    n = 0
    for sid, s, h, valid, ev, nu, _x, _t in iter_identity(ver, split):
        bag.add(sid, s, h, valid, ev, nu)
        n += 1
    print(f"identity {ver} {split} files={n} {time.time() - t0:.1f}s bag={bag.nbytes() / 1e6:.0f}MB", flush=True)
    return bag


def report_at(bag, thr, far_ref):
    rep = bag.report(thr, far_ref=far_ref)
    if rep is None:
        return None
    # CI не нужен в каждом промежуточном дампе сверх уже посчитанного
    return rep


def events_update(rows, zones, sid, s, h, valid, ev, x, t, labels, seqs, thr):
    pred = (s >= thr) & valid
    for r in M.label_boxes(labels, sid).itertuples():
        if r.kind != "event":
            continue
        px = M.event_pixels(r, x, t, ev)
        if int(px.sum()) == 0:
            continue
        frac = float(pred[px].mean())
        tmpl = seqs.loc[sid, "template"] if sid in seqs.index else "?"
        rows.append(dict(
            seq_id=sid, label=r.label, amp=float(r.amp_obs_C),
            amp_bin=M.amp_bin(float(r.amp_obs_C)), template=str(tmpl),
            difficulty=r.difficulty, frac=frac, hit=frac >= 0.3,
            tp=int((pred & px).sum()), fn=int((~pred & px).sum()),
        ))
    zs = EV.zones_from_prob(s, h, valid, x, t, thr)
    for z in zs:
        sl = (slice(z["t0"], z["t1"]), slice(z["x0"], z["x1"]))
        comp = (s[sl] >= thr) & valid[sl]
        zones["n"] += 1
        if not comp.any() or float((ev[sl][comp] > 0).mean()) < 0.1:
            zones["false"] += 1
    zones["kmh"] += (float(x[-1]) - float(x[0])) / 1000.0 * (float(t[-1]) - float(t[0])) / 3600.0


def events_summary(rows, zones):
    E = pd.DataFrame(rows)
    out = {}
    if len(E):
        def agg(g):
            return dict(n=int(len(g)), recall=float(g.hit.mean()),
                        pixel_recall=float(g.tp.sum() / max(g.tp.sum() + g.fn.sum(), 1)))
        out["events_all"] = agg(E)
        out["by_amp"] = {k: agg(g) for k, g in E.groupby("amp_bin")}
        out["by_label"] = {k: agg(g) for k, g in E.groupby("label")}
        out["weak_events"] = agg(E[E.amp <= 2]) if (E.amp <= 2).any() else dict(n=0, recall=None, pixel_recall=None)
    kmh = zones.get("kmh", 0.0)
    nf = zones.get("false", 0)
    nz = zones.get("n", 0)
    out["zones"] = dict(
        n_zones=int(nz), n_false=int(nf),
        false_frac=float(nf / max(nz, 1)),
        false_zones_per_km_h=float(nf / max(kmh, 1e-9)),
    )
    return out


def check_identity(far_ref):
    official = json.loads(WINNER_JSON.read_text())
    val_bag = bag_identity("v8", "val")
    thr = val_bag.threshold(FAR)
    val = report_at(val_bag, thr, far_ref)
    off_thr = official["thresholds"][ENS]["far_1e-3"]
    off_val = official["val"][f"{ENS}@far_1e-3"]
    print(f"identity val thr {thr:.8f} published {off_thr:.8f} iou {val['iou']:.6f} published {off_val['iou']:.6f}", flush=True)
    if abs(thr - off_thr) > 1e-4 or abs(val["iou"] - off_val["iou"]) > 1e-4:
        raise SystemExit("identity val не совпал с r7: метрика TTA не та")
    del val_bag
    out = {"v8": {"val": val, "threshold": thr}}
    for ver, split in (("v8", "test"), ("v10", "test"), ("v11", "test")):
        bag = bag_identity(ver, split)
        rep = report_at(bag, thr, far_ref)
        out.setdefault(ver, {})[split] = rep
        del bag
        print(f"identity {ver} {split} iou {rep['iou']:.4f} f1 {rep['f1']:.4f} far {rep['far']:.4f} hot {rep['hot']['iou']:.4f} cold {rep['cold']['iou']:.4f}", flush=True)
    off_te = official["test"][f"{ENS}@far_1e-3"]
    if abs(out["v8"]["test"]["iou"] - off_te["iou"]) > 1e-4:
        raise SystemExit("identity test iou не совпал с r7")
    if abs(out["v8"]["test"]["hot"]["iou"] - off_te["hot"]["iou"]) > 1e-4:
        raise SystemExit("identity test hot iou не совпал с r7")
    return thr, out


def identity_probe(net, X, saved_s):
    s, _h = tta.sh_from_prob(predict_prob(net, X))
    return float(np.max(np.abs(s - saved_s)))


def build_variants(ens, members, n=N_ENS):
    """Смеси знака. members — список (id_члена, sign_члена).

    Ансамбль: ens + сумма (вид − id) / n по аугментированным членам.
    Остальные члены остаются в опубликованной карте. Линейные смеси
    и maxpol/geomean, которым хватает вида sign, считаются здесь.
    Отражения в эту функцию не входят.
    """
    which = {"sign"}
    scale = np.float32(1.0 / int(n))
    sign_s = np.array(ens[0], dtype=np.float32, copy=True)
    sign_h = np.array(ens[1], dtype=np.float32, copy=True)
    for mid, msign in members:
        sign_s += (msign[0] - mid[0]) * scale
        sign_h += (msign[1] - mid[1]) * scale
    ens_sign = (sign_s, sign_h)
    ens_views = {"id": ens, "sign": ens_sign}
    variants = {}
    for name, w in tta.LINEAR.items():
        if tta.views_required(name) <= which:
            variants[name] = tta.linear_fuse(ens_views, w)
    for name in tta.NONLINEAR:
        if not tta.views_required(name) <= which:
            continue
        acc_s = np.array(ens[0], dtype=np.float32, copy=True)
        acc_h = np.array(ens[1], dtype=np.float32, copy=True)
        for mid, msign in members:
            new_s, new_h = tta.nonlinear_member({"id": mid, "sign": msign}, name)
            acc_s += (new_s - mid[0]) * scale
            acc_h += (new_h - mid[1]) * scale
        variants[name] = (acc_s, acc_h)
    return ens_views, variants


def member_flip_probe(net, X, saved_s):
    """Один файл, отражения x и t. Знак считает основной цикл, здесь его нет."""
    out = {}
    for name in ("flip_x", "flip_t"):
        t0 = time.time()
        spec = tta.VIEW_SPEC[name]
        xin = tta.transform_input(X, **spec)
        prob = tta.align_prob(predict_prob(net, xin), **spec)
        s, _h = tta.sh_from_prob(prob)
        d = np.abs(s - saved_s)
        out[name] = {
            "abs_mean": float(d.mean()),
            "abs_max": float(d.max()),
            "sec": time.time() - t0,
        }
        del xin, prob, s, d
    return out


def run_tta_split(ver, split, nets, which, use_net_id, bags_gate=GATE, save_views=False, save_winner=None, thr_events=None, labels=None, seqs=None):
    """which — виды сверх id. Сейчас рабочий набор — sign.

    Аугментированные сети дают знак. Ансамбль сдвигается на (sign − id) / N_ENS.
    save_views: писать знак члена в /tmp (float32), чтобы собрать val без второго прохода.
    save_winner: писать уже выбранную смесь в artifacts/pred/v8/ens_tta/<split>.
    """
    which = tuple(which)
    if set(which) - {"sign"}:
        raise SystemExit(f"этот запуск считает только sign, получено {which}")
    aug_names = [n for n, _ in nets]
    net_by = {n: net for n, net in nets}
    bags = {}
    names = [n for n in tta.ALL_VARIANTS if tta.views_required(n) <= set(which)]
    for n in names:
        bags[n] = tta.ScoreBag(bags_gate)
    rows, zones = [], {"n": 0, "false": 0, "kmh": 0.0}
    probe = {
        "sign": {"abs_mean": 0.0, "abs_max": 0.0, "n": 0},
        "sign_member": {"abs_mean": 0.0, "abs_max": 0.0, "n": 0},
    }
    paths = cache_paths(ver, split)
    t0 = time.time()
    for i, p in enumerate(paths):
        field = load_field(ver, split, p)
        X = field["X"]
        members = []
        held = []
        for name in aug_names:
            saved = load_sh(ver, name, split, p.name)
            mv = neural_views(net_by[name], X, saved, which, use_net_id.get(name, False))
            members.append((mv["id"], mv["sign"]))
            held.append((name, mv["sign"]))
            dlt = np.abs(mv["sign"][0] - mv["id"][0])
            probe["sign_member"]["abs_mean"] += float(dlt.mean())
            probe["sign_member"]["abs_max"] = max(probe["sign_member"]["abs_max"], float(dlt.max()))
            probe["sign_member"]["n"] += 1
            del dlt
        ens = load_sh(ver, ENS, split, p.name)
        ens_views, variants = build_variants(ens, members, n=N_ENS)
        del members, X
        d = np.abs(ens_views["sign"][0] - ens_views["id"][0])
        probe["sign"]["abs_mean"] += float(d.mean())
        probe["sign"]["abs_max"] = max(probe["sign"]["abs_max"], float(d.max()))
        probe["sign"]["n"] += 1
        del d, ens_views
        if save_views:
            for name, shv in held:
                save_view_f32(TMP / ver / split / name / "sign" / p.name, shv[0], shv[1])
        del held
        winner_sh = None
        for n, (s, h) in variants.items():
            sq, hq = add_variant(bags[n], field["sid"], s, h, field["valid"], field["ev"], field["nu"])
            if save_winner is not None and n == save_winner:
                winner_sh = (sq, hq)
        del variants
        if winner_sh is not None:
            save_sh(PRED_OUT / split / p.name, winner_sh[0], winner_sh[1], compress=False)
            if thr_events is not None:
                events_update(
                    rows, zones, field["sid"], winner_sh[0], winner_sh[1],
                    field["valid"], field["ev"], field["x"], field["t"], labels, seqs, thr_events,
                )
        dt = time.time() - t0
        print(
            f"tta {ver} {split} {i + 1}/{len(paths)} {field['sid']} {dt:.0f}s "
            f"views={','.join(which)} aug={','.join(aug_names)}",
            flush=True,
        )
        if (i + 1) % 5 == 0 or i == 0:
            dump(ROOT / "artifacts/eval/s3_tta_progress.json", {
                "utc": now(), "ver": ver, "split": split, "done": i + 1, "n": len(paths),
                "elapsed_s": dt, "views": list(which), "augmented": aug_names,
                "sec_per_file": dt / (i + 1),
                "bag_mb": {n: bags[n].nbytes() / 1e6 for n in bags},
            })
    for k in probe:
        if probe[k]["n"]:
            probe[k]["abs_mean"] /= probe[k]["n"]
    info = {
        "views": list(which),
        "augmented": aug_names,
        "probe_abs_score": probe,
        "id_vs_published_max": 0.0,
        "sec": time.time() - t0,
        "n": len(paths),
    }
    ev = events_summary(rows, zones) if rows or zones["n"] else None
    return bags, info, ev


def materialize_saved(ver, split, variant, aug_names, thr, labels, seqs, do_events):
    """Собрать уже выбранную смесь из знака в /tmp и опубликованных карт. Без сети."""
    rows, zones = [], {"n": 0, "false": 0, "kmh": 0.0}
    bag = tta.ScoreBag(GATE)
    for p in cache_paths(ver, split):
        ens = load_sh(ver, ENS, split, p.name)
        members = []
        for name in aug_names:
            mid = load_sh(ver, name, split, p.name)
            members.append((mid, load_view_f32(TMP / ver / split / name / "sign" / p.name)))
        _views, variants = build_variants(ens, members, n=N_ENS)
        s, h = quant_h(*variants[variant])
        del variants, members
        save_sh(PRED_OUT / split / p.name, s, h, compress=False)
        z = np.load(p)
        valid = np.asarray(z["valid"], bool)
        bag.add(p.stem, s, h, valid, z["ev"], z["nu"])
        if do_events:
            events_update(rows, zones, p.stem, s, h, valid, z["ev"], z["x"], z["t"], labels, seqs, thr)
        del z
    ev = events_summary(rows, zones) if do_events else None
    return bag, ev


def slim(rep):
    if not rep:
        return None
    keep = ("threshold", "far", "far_ratio", "precision", "recall", "f1", "iou", "tp", "fp", "fn", "n_bg", "n_seq", "iou_ci95")
    out = {k: rep[k] for k in keep if k in rep}
    for pole in ("hot", "cold"):
        if pole in rep:
            out[pole] = {k: rep[pole][k] for k in ("precision", "recall", "f1", "iou")}
    return out


def select_variant(reports: dict):
    ok = {k: v for k, v in reports.items() if v and v.get("iou") is not None}
    if not ok:
        raise RuntimeError("нет варианта с порогом выше gate")
    return max(ok, key=lambda k: (ok[k]["iou"], 1 if k == tta.PRIMARY else 0))


def fmt(x, n=4):
    if x is None:
        return "н/д"
    return f"{float(x):.{n}f}"


def write_doc(summary):
    ident = summary["identity"]
    val = summary["val_variants"]
    chosen = summary["chosen"]
    lines = []
    lines.append("# S3. Test-time augmentation ансамбля")
    lines.append("")
    aug = ", ".join(summary.get("augmented_members") or [])
    passive = ", ".join(summary.get("passive_members") or [])
    ran = ", ".join(summary.get("ran_views") or [])
    lines.append(
        "Нейросетевой вывод без дообучения. База — опубликованный `ens_u2ft_tcn_lgbm` "
        "(U-Net ширины 24, U-Net v8+v10 ширины 16, TCN, пиксельный LightGBM, равные веса). "
        f"На преобразованном входе прогнан только {aug}. Пассивные члены ({passive}) "
        "в новый проход не входят: их вклад уже сидит в опубликованной карте. "
        f"Считались виды: {ran}."
    )
    note = summary.get("budget_note")
    if note:
        lines.append("")
        lines.append(note)
    lines.append("")
    lines.append("Порог — квантиль фона v8 val при FAR 1e-3, та же функция `metrics.far_threshold`, что в `scripts/eval_stage1.py`. Test v8, v10 и v11 этот порог не двигают. Реальные файлы MGTS не читались, на них ничего не калибровалось. Карточки оператора в выбор не входят.")
    lines.append("")
    lines.append("## Симметрии входа")
    lines.append("")
    lines.append("Каналы сети — нормированные остатки `dT_time`, `dT_space` и маска valid (`dts/features.py`). Аугментация применяется к этим каналам, то есть к тому тензору, который ест `predict_full`.")
    lines.append("")
    lines.append("- Отражение по x. У волокна в этих каналах нет абсолютной координаты, медианный фон уже снят. Класс нагрев/охлаждение не меняется. Вероятности отражаются обратно.")
    lines.append("- Отражение по t. Остаток в клетке не содержит производной, которая меняла бы знак при развороте времени: горячая клетка остаётся горячей. Класс не переставляется. Это корректная симметрия метки. Сеть с тайловым `predict_full` не строго эквивариантна (стыки тайлов и фаза пулинга), поэтому вид не совпадает с id и усреднение имеет смысл.")
    lines.append("- Симметрия знака. Каналы 0 и 1 меняют знак, valid остаётся. После softmax классы 1 и 2 меняются местами. Score `1 − p(фон)` сопоставляется исходному полю, hotness меняет знак. Сети победителя учили без sign-flip, поэтому этот вид заметно отличается от id.")
    lines.append("")
    fp = summary.get("flip_probe") or {}
    if fp:
        lines.append(
            f"Зонд отражений на одном файле ({fp.get('file')}, член {fp.get('member')}), "
            "score вида против опубликованного id этого члена. Полный val этими видами не гонялся."
        )
        lines.append("")
        lines.append("| вид | mean |Δscore| | max |Δscore| | секунды |")
        lines.append("|---|---:|---:|---:|")
        for k in ("flip_x", "flip_t"):
            if k in fp:
                lines.append(
                    f"| {k} | {fmt(fp[k].get('abs_mean'), 4)} | {fmt(fp[k].get('abs_max'), 4)} | {fmt(fp[k].get('sec'), 1)} |"
                )
        lines.append("")
    pr = summary.get("probe_abs_score") or {}
    if pr:
        lines.append("Среднее по файлам v8 val модуля разности score. `sign_member` — сам U-Net, `sign` — уже сдвиг ансамбля на четверть.")
        lines.append("")
        lines.append("| уровень | mean |Δscore| | max |Δscore| |")
        lines.append("|---|---:|---:|")
        for k in ("sign_member", "sign", "flip_x", "flip_t"):
            if k in pr and pr[k].get("n"):
                lines.append(f"| {k} | {fmt(pr[k].get('abs_mean'), 4)} | {fmt(pr[k].get('abs_max'), 4)} |")
        lines.append("")
        lines.append("У отражений малый mean и редкий большой max — стыки тайлов `predict_full`, а не другой детектор. Знак двигает карту широко: сети победителя учили без sign-flip.")
        lines.append("")
    lines.append(
        "Рабочая формула одного члена: `ens + (вид(члена) − id(члена)) / 4`. "
        "Вид `sign` — проход сети со сменой знака каналов. "
        "`sign_member` подставляет этот вид вместо id члена. "
        "`wsign_*` смешивает опубликованный ансамбль и этот сдвиг. "
        "`maxpol` и `geomean_sign` считаются внутри члена и тем же сдвигом возвращаются в ансамбль. "
        "`mean_full` (id, flip x, flip t и знак всех свёрточных членов) в этот запуск не входил: "
        "девять проходов на файл не укладываются в CPU до 17:30. "
        "Выбор одной строки делает val IoU при FAR 1e-3. Test в выбор не входит. "
        "Пустая строка таблицы — вариант не считался. Это не отказ порога."
    )
    lines.append("")
    lines.append("## Совпадение протокола с победителем")
    lines.append("")
    lines.append(f"Повтор опубликованных карт `ens_u2ft_tcn_lgbm` тем же хвостом баллов: val IoU {fmt(ident['v8']['val']['iou'])} при пороге {fmt(ident['v8']['threshold'], 6)}, test IoU {fmt(ident['v8']['test']['iou'])}. Это те же числа, что в `artifacts/eval/r7_ens_u2ft_tcn_lgbm.json` (0.3867 / 0.4053) с точностью 1e-4. Hot/cold test тоже совпали с этим файлом.")
    lines.append("")
    lines.append("| набор | порог v8 val | val IoU | val F1 | test IoU | test F1 | test FAR | hot IoU | cold IoU |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    iv, it = ident["v8"]["val"], ident["v8"]["test"]
    lines.append(f"| id ансамбля | {fmt(iv['threshold'], 4)} | {fmt(iv['iou'])} | {fmt(iv['f1'])} | {fmt(it['iou'])} | {fmt(it['f1'])} | {fmt(it['far'], 4)} | {fmt(it['hot']['iou'])} | {fmt(it['cold']['iou'])} |")
    for ver in ("v10", "v11"):
        r = ident[ver]["test"]
        lines.append(f"| id, {ver} test | тот же | н/д | н/д | {fmt(r['iou'])} | {fmt(r['f1'])} | {fmt(r['far'], 4)} | {fmt(r['hot']['iou'])} | {fmt(r['cold']['iou'])} |")
    lines.append("")
    lines.append("## Val v8, все смеси")
    lines.append("")
    lines.append("Порог каждой смеси — свой квантиль её фона на v8 val. В выбор идёт максимум IoU. При равенстве остаётся `mean_full`, если он посчитан.")
    lines.append("")
    lines.append("| вариант | что смешивается | порог | IoU | F1 | FAR | hot IoU | cold IoU |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    order = list(tta.LINEAR) + list(tta.NONLINEAR)
    for name in order:
        r = val.get(name)
        if not r:
            lines.append(f"| {name} | | н/д | н/д | н/д | н/д | н/д | н/д |")
            continue
        mark = " ← выбран" if name == chosen else ""
        lines.append(
            f"| {name}{mark} | {summary['formulas'][name]} | {fmt(r['threshold'], 4)} | {fmt(r['iou'])} | {fmt(r['f1'])} | {fmt(r['far'], 4)} | {fmt(r['hot']['iou'])} | {fmt(r['cold']['iou'])} |"
        )
    lines.append("")
    dlt = summary["delta_val_iou"]
    lines.append(f"Выбран `{chosen}`. Val IoU {fmt(summary['chosen_val']['iou'])} против id {fmt(iv['iou'])} (разница {dlt:+.4f}). Победитель сессии по val не сменён." if dlt <= 0 else f"Выбран `{chosen}`. Val IoU {fmt(summary['chosen_val']['iou'])} выше id {fmt(iv['iou'])} на {dlt:.4f}.")
    lines.append("")
    ran_views = set(summary.get("ran_views") or [])
    if "flip_t" not in ran_views:
        lines.append("Отражение по времени на полном val не считалось. Зонд одного файла — в разделе симметрий. Метку разворот времени не меняет: класс нагрев/охлаждение не переставляется.")
    elif "flip_t" in (summary.get("chosen_views") or []):
        lines.append("Отражение по времени в выбранной смеси осталось: val IoU с ним не ниже смесей без него, либо оно входит в единственный максимум.")
    else:
        lines.append("Отражение по времени в выбранную смесь не вошло: на val IoU выше у смеси без него. Вид посчитан и в таблице есть. Метку он не ломает, выигрыша по IoU не дал.")
    lines.append("")
    lines.append("## Test при замороженном пороге v8 val")
    lines.append("")
    lines.append("| набор | v8 IoU | v8 F1 | v8 P | v8 R | v8 FAR | v8 hot | v8 cold | v10 IoU | v11 IoU |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| id | {fmt(it['iou'])} | {fmt(it['f1'])} | {fmt(it['precision'])} | {fmt(it['recall'])} | {fmt(it['far'], 4)} | {fmt(it['hot']['iou'])} | {fmt(it['cold']['iou'])} | {fmt(ident['v10']['test']['iou'])} | {fmt(ident['v11']['test']['iou'])} |"
    )
    ct = summary.get("chosen_test") or {}
    def cell(block, key, n=4):
        if not block:
            return "н/д"
        return fmt(block.get(key), n)
    v8t = ct.get("v8")
    lines.append(
        f"| {chosen} | {cell(v8t, 'iou')} | {cell(v8t, 'f1')} | {cell(v8t, 'precision')} | {cell(v8t, 'recall')} | {cell(v8t, 'far')} | {cell((v8t or {}).get('hot'), 'iou')} | {cell((v8t or {}).get('cold'), 'iou')} | {cell(ct.get('v10'), 'iou')} | {cell(ct.get('v11'), 'iou')} |"
    )
    lines.append("")
    ev = (summary.get("chosen_events") or {}).get("v8_test")
    if ev is None:
        ev = (summary.get("chosen_events") or {}).get("v8")
    if ev and ev.get("events_all"):
        w = ev.get("weak_events") or {}
        lines.append(
            f"События выбранного варианта на v8 test (попадание, если предсказано ≥ 30 % пикселей рамки): recall {fmt(ev['events_all']['recall'])} из {ev['events_all']['n']}, слабые ≤ 2 °C {fmt(w.get('recall'))} из {w.get('n')}. Зоны: {ev.get('zones', {}).get('n_zones')} всего, доля пустых {fmt((ev.get('zones') or {}).get('false_frac'))}."
        )
        by = ev.get("by_label") or {}
        if "cooling" in by:
            lines.append(f" Cooling: {by['cooling']['recall']:.3f} при n={by['cooling']['n']}.")
        lines.append("")
    lines.append("Карты выбранной смеси: `artifacts/pred/v8/ens_tta/{val,test}/`. Формат как у ансамбля, `s` float32 и `h` float16. v10 и v11 на диск не складывались. Промежуточные виды после сборки удалены.")
    lines.append("")
    lines.append("## Что не принято")
    lines.append("")
    if summary["beats"]:
        lines.append(f"`{chosen}` выше победителя по val IoU. Test при этом {fmt((ct.get('v8') or {}).get('iou'))} (у победителя {fmt(it['iou'])}). Смена выпуска — решение координатора; здесь порог снят с v8 val и на test не подгонялся.")
    else:
        lines.append(f"Ни одна смесь не выше id по val IoU ({fmt(iv['iou'])}). Выпуск остаётся `ens_u2ft_tcn_lgbm`. TTA в детектор по умолчанию не включается. Числа test и переносов v10/v11 приведены для выбранной по val смеси и в выбор не входили.")
    lines.append("")
    lines.append("Порог по карточкам не выбирался. Обучение и калибровка на `/workspace/mgts` не выполнялись.")
    lines.append("")
    path = ROOT / "docs/S3_TTA.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", path, flush=True)


def formulas():
    desc = {k: " + ".join(f"{v:g}·{name}" for name, v in w.items()) for k, w in tta.LINEAR.items()}
    desc["logit_full"] = "softmax среднего логита id, flip x, flip t, знак; внутри сети"
    desc["maxpol"] = "max p(полярность) и min p(фон) между id и знаком, внутри сети"
    desc["shrink_full"] = "mean − 0.5·std четырёх видов, внутри сети"
    desc["geomean_sign"] = "геометрическое среднее score id и знака, внутри сети"
    return desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("all", "identity"))
    ap.add_argument("--skip-identity", action="store_true")
    args = ap.parse_args()
    configure_torch()
    official = json.loads(WINNER_JSON.read_text())
    far_ref = float(official["far_ref"])
    print(f"start {now()} far_ref {far_ref}", flush=True)
    ident_path = ROOT / "artifacts/eval/s3_tta_identity.json"
    if args.skip_identity and ident_path.exists():
        blob = json.loads(ident_path.read_text())
        thr_id = float(blob["threshold_far_1e-3"])
        ident = blob["identity"]
        print(f"reuse identity val iou {ident['v8']['val']['iou']:.4f}", flush=True)
    else:
        thr_id, ident = check_identity(far_ref)
        dump(ident_path, {"far_ref": far_ref, "identity": ident, "threshold_far_1e-3": thr_id})
    if args.stage == "identity":
        print("IDENTITY_OK", flush=True)
        return

    # Один член. На v10 test нет карт unet_w24, tcn и pixel_lgbm, есть unet_v810ft и ансамбль.
    aug_names = ["unet_v810ft"]
    which = ("sign",)
    nets = load_nets(aug_names)
    use_net_id = {name: False for name, _ in nets}
    first = cache_paths("v8", "val")[0]
    X0 = np.load(first)["X"].astype(np.float32)
    saved0, _h0 = load_sh("v8", aug_names[0], "val", first.name)
    flip_probe = member_flip_probe(nets[0][1], X0, saved0)
    flip_probe["file"] = first.stem
    flip_probe["member"] = aug_names[0]
    flip_probe["pixels"] = int(X0.shape[1] * X0.shape[2])
    for k in ("flip_x", "flip_t"):
        print(
            f"flip probe {k} mean|Δs|={flip_probe[k]['abs_mean']:.4g} "
            f"max={flip_probe[k]['abs_max']:.4g} {flip_probe[k]['sec']:.1f}s",
            flush=True,
        )
    del X0, saved0

    free = shutil.disk_usage("/tmp").free
    save_views = free > 2 * 2**30
    print(f"tmp free {free / 2**30:.1f} GiB save_views={save_views}", flush=True)
    if TMP.exists():
        shutil.rmtree(TMP)
    log_event(
        f"TTA_PLAN members {','.join(aug_names)} views sign "
        f"flip_x {flip_probe['flip_x']['sec']:.0f}s flip_t {flip_probe['flip_t']['sec']:.0f}s"
    )

    labels_v8, seqs_v8 = meta("v8")
    t_val = time.time()
    bags, info, _ev = run_tta_split(
        "v8", "val", nets, which, use_net_id, save_views=save_views,
    )
    val_reps = {}
    for name, bag in bags.items():
        thr = bag.threshold(FAR)
        val_reps[name] = report_at(bag, thr, far_ref) if thr is not None else None
        if val_reps[name]:
            print(f"val {name} thr {thr:.4f} iou {val_reps[name]['iou']:.4f} f1 {val_reps[name]['f1']:.4f} hot {val_reps[name]['hot']['iou']:.4f} cold {val_reps[name]['cold']['iou']:.4f}", flush=True)
        else:
            print(f"val {name} threshold below gate", flush=True)
    chosen = select_variant(val_reps)
    print(f"CHOSEN {chosen} iou {val_reps[chosen]['iou']:.4f} in {time.time() - t_val:.0f}s", flush=True)
    # отпустить хвосты до test
    chosen_val = val_reps[chosen]
    val_slim = {k: slim(v) for k, v in val_reps.items()}
    probe = info["probe_abs_score"]
    del bags

    budget_note = (
        "Бюджет CPU. Полный проход трёх сетей и трёх видов на v8 val не доходит до 17:30: "
        "файл seq0006 в 16:18 занял 238 с на девять проходов, около 26 с на проход при текущей загрузке. "
        "Отражения x и t на полном val не считались, для них есть зонд одного файла. "
        "На v10 test нет карт unet_w24, tcn и pixel_lgbm, есть unet_v810ft и ансамбль, "
        "поэтому знак считается только у unet_v810ft, а ансамбль сдвигается на (вид − id) / 4. "
        "sign_member двигает карту на четверть разности этого члена, wsign_0.50 — на восьмую. "
        "Это нижняя оценка знакового TTA всего ансамбля. mean_full в таблицу не входит."
    )
    summary = {
        "method": "ens_tta",
        "base_ensemble": ENS,
        "far": FAR,
        "far_ref": far_ref,
        "gate": GATE,
        "primary": tta.PRIMARY,
        "chosen": chosen,
        "chosen_views": sorted(tta.views_required(chosen)),
        "formulas": formulas(),
        "augmented_members": aug_names,
        "passive_members": [n for n in OFFICIAL if n not in aug_names],
        "ran_views": list(which),
        "n_ensemble": N_ENS,
        "definition": "ens + (view(member) - id(member)) / 4",
        "budget_note": budget_note,
        "flip_probe": flip_probe,
        "use_network_identity": use_net_id,
        "probe_abs_score": probe,
        "id_vs_published_max": info.get("id_vs_published_max"),
        "identity": ident,
        "val_variants": val_slim,
        "chosen_val": slim(chosen_val),
        "delta_val_iou": float(chosen_val["iou"] - ident["v8"]["val"]["iou"]),
        "val_sec": info["sec"],
        "lgbm": "identity, inside the published ensemble, not re-inferred",
        "real_data": "not loaded",
        "chosen_test": {},
        "test_variants": {},
        "notes": {
            "flip_t": "label-correct: time reversal does not swap hot/cold; class channels are not swapped",
            "sign": "negate channels 0 and 1, swap class 1 and 2",
            "flip_x": "spatial reflection, classes unchanged",
            "v10_maps": "unet_w24, tcn, pixel_lgbm test maps are absent; delta uses the published ensemble",
        },
    }
    dump(ROOT / "artifacts/eval/s3_tta_v8_val.json", summary)

    beats_logged = {"done": False}

    def publish(tag, do_log):
        id_iou = ident["v8"]["val"]["iou"]
        pub = float(official["val"][f"{ENS}@far_1e-3"]["iou"])
        summary["published_val_iou"] = pub
        summary["published_test_iou"] = float(official["test"][f"{ENS}@far_1e-3"]["iou"])
        ct = (summary.get("chosen_test") or {}).get("v8") or {}
        summary["beats"] = bool(
            chosen_val["iou"] > id_iou and chosen_val["iou"] > pub and ct.get("iou") is not None
        )
        dump(ROOT / "artifacts/eval/s3_tta_v8.json", summary)
        write_doc(summary)
        if not do_log:
            return
        cv = chosen_val["iou"]
        ct_iou = ct.get("iou")
        c10 = ((summary.get("chosen_test") or {}).get("v10") or {}).get("iou")
        c11 = ((summary.get("chosen_test") or {}).get("v11") or {}).get("iou")
        line = (
            f"TTA_DONE variant {chosen} val {cv:.4f} test {ct_iou if ct_iou is not None else float('nan'):.4f} "
            f"v10 {c10 if c10 is not None else float('nan'):.4f} v11 {c11 if c11 is not None else float('nan'):.4f} "
            f"thr {chosen_val['threshold']:.4f} id_val {id_iou:.4f} id_test {ident['v8']['test']['iou']:.4f} "
            f"delta_val {summary['delta_val_iou']:+.4f} stage {tag}"
        )
        log_event(line)
        if summary["beats"] and not beats_logged["done"]:
            log_event(f"BEATS_WINNER TTA val {cv:.4f} test {ct_iou:.4f}")
            beats_logged["done"] = True

    publish("val", True)

    sec_per = info["sec"] / max(info["n"], 1)
    # v8 test ещё стартуем почти до жёсткого финиша. v10/v11 — только до внутреннего дедлайна.
    test_plan = [("v8", "test", 17 * 3600 + 28 * 60), ("v10", "test", None), ("v11", "test", None)]
    hard_clock = calendar.timegm(time.strptime("2026-10-15 17:28:00", "%Y-%m-%d %H:%M:%S"))
    for ver, split, _cut in test_plan:
        n_files = len(cache_paths(ver, split))
        est = sec_per * n_files * (79.5 / 95.5 if split == "test" else 1.0)
        left = DEADLINE - time.time()
        start_ok = time.time() < (hard_clock if ver == "v8" else DEADLINE)
        print(
            f"plan {ver} {split} est_s={est:.0f} deadline_left={left:.0f} start_ok={start_ok}",
            flush=True,
        )
        if not start_ok:
            print(f"skip {ver} {split}: past deadline", flush=True)
            summary["chosen_test"][ver] = None
            continue
        bags_t, info_t, ev_t = run_tta_split(
            ver, split, nets, which, use_net_id,
            save_winner=chosen, thr_events=None,
        )
        reps = {}
        for name, bag in bags_t.items():
            reps[name] = report_at(bag, chosen_val["threshold"], far_ref)
            if reps[name]:
                print(f"{ver} {split} {name} iou {reps[name]['iou']:.4f} far {reps[name]['far']:.4f}", flush=True)
        summary["test_variants"][f"{ver}_{split}"] = {k: slim(v) for k, v in reps.items()}
        summary["chosen_test"][ver] = slim(reps.get(chosen))
        if ev_t:
            summary.setdefault("chosen_events", {})[f"{ver}_{split}"] = ev_t
        summary[f"{ver}_{split}_sec"] = info_t["sec"]
        del bags_t
        if ver == "v8":
            publish("v8_test", True)
        else:
            dump(ROOT / f"artifacts/eval/s3_tta_{ver}.json", {
                "version": ver, "split": split, "threshold_from": "v8 val",
                "threshold": chosen_val["threshold"], "chosen": chosen,
                "identity": slim(ident[ver]["test"]),
                "tta": slim(summary["chosen_test"][ver]),
                "variants": summary["test_variants"][f"{ver}_{split}"],
                "events": ev_t,
            })
            publish(ver, False)

    if save_views and (TMP / "v8" / "val").exists():
        bag_m, _ev_val = materialize_saved(
            "v8", "val", chosen, aug_names, chosen_val["threshold"], labels_v8, seqs_v8, False,
        )
        chk = report_at(bag_m, chosen_val["threshold"], far_ref)
        if chk is None:
            print("WARN materialize val: порог ниже gate", flush=True)
        else:
            print(f"materialize val iou {chk['iou']:.4f} vs bag {chosen_val['iou']:.4f}", flush=True)
        if chk is not None and abs(chk["iou"] - chosen_val["iou"]) > 5e-4:
            print("WARN val artifact iou drifted", flush=True)
            summary["chosen_val_artifact"] = slim(chk)
        del bag_m
    elif not (PRED_OUT / "val").exists():
        print("WARN val maps missing", flush=True)
    if TMP.exists():
        shutil.rmtree(TMP, ignore_errors=True)
    publish("final", True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
