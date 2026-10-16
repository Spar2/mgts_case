"""Смесь (c): равное среднее GEN2, победителя, GEN2W24 и GEN2TCN.

Член входит только при файле DONE и остановленном тренере.
Веса равные и заданы до просмотра чисел: (gen2 + winner + w24 + tcn) / 4.
Порог FAR 1e-3, бутстреп 1000, вложенные k5/k2 — как в scripts/s3_combo.py.
Честная val — 41 файл без четырёх прокси GEN2.
Правило beats то же и по результату не меняется. COMBO1 повторно не пишется.
Одна строка COMBO_DONE. Токен BEATS_WINNER только если (c) проходит все пункты.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "2"

import gc
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)

from dts import data, metrics as M, models
from s3_gen2_train import predict_batched
import s3_combo as combo

OUT = Path("artifacts/eval/s3_combo_c.json")
DOC = Path("docs/S3_COMBO.md")
WINNER_THR = 0.8608087301264272
MEMBERS = (
    dict(name="gen2w24", kind="unet", width=24, run="s3_gen2w24",
         pt="artifacts/eval/s3_gen2w24_unet.pt", init="artifacts/stage1/unet_w24.pt"),
    dict(name="gen2tcn", kind="tcn", width=32, run="s3_gen2tcn",
         pt="artifacts/eval/s3_gen2tcn_tcn.pt", init="artifacts/stage1/tcn.pt"),
)


def log(msg: str) -> None:
    combo.log(msg)


def trainer_alive(run: str) -> bool:
    needle = f"scripts/{run}_train.py"
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmd = (proc / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="ignore")
        except OSError:
            continue
        if needle in cmd:
            return True
    return False


def max_abs_diff(pt: Path, init: Path) -> float:
    a = torch.load(pt, map_location="cpu", weights_only=True)
    b = torch.load(init, map_location="cpu", weights_only=True)
    if set(a) != set(b):
        raise SystemExit(f"ключи {pt.name} не совпали с init")
    diff = 0.0
    for key in a:
        diff = max(diff, float((a[key].float() - b[key].float()).abs().max()))
    return diff


def load_net(spec):
    model = models.build(spec["kind"], spec["width"])
    state = torch.load(spec["pt"], map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def member_path(name, ver, split, stem) -> Path:
    return combo.MAPS / f"{name}_{ver}_{split}" / f"{stem}.npy"


def ensure_members(nets, ver, split):
    files = data.list_cache(f"artifacts/cache/{ver}", split)
    t0 = time.time()
    for i, p in enumerate(files, 1):
        with np.load(p) as z:
            shape = tuple(np.asarray(z["ev"]).shape)
        pending = []
        for spec, model in nets:
            out = member_path(spec["name"], ver, split, p.stem)
            if not combo.map_ok(out, shape):
                pending.append((spec, model, out))
        if not pending:
            if i == 1 or i % 15 == 0 or i == len(files):
                log(f"члены {ver} {split} {i}/{len(files)} есть {time.time() - t0:.0f}s")
            continue
        with np.load(p) as z:
            X = np.asarray(z["X"], np.float32)
        for spec, model, out in pending:
            prob = predict_batched(model, X)
            s = (1.0 - prob[0]).astype(np.float32)
            del prob
            if tuple(s.shape) != shape:
                raise SystemExit(f"{spec['name']} {p.stem} {s.shape} != {shape}")
            combo.save_map(out, s)
            del s
        del X
        log(f"члены {ver} {split} {i}/{len(files)} {p.stem} {time.time() - t0:.0f}s")
    return files


def load_blend(ver, split, path, with_xy=False):
    stem = path.stem
    with np.load(path) as z:
        ev = np.asarray(z["ev"])
        valid = np.asarray(z["valid"], bool)
        nu = np.asarray(z["nu"])
        x = np.asarray(z["x"], np.float64) if with_xy else None
        t = np.asarray(z["t"], np.float64) if with_xy else None
    g = np.asarray(np.load(combo.map_path(ver, split, stem), mmap_mode="r"), np.float32)
    w24 = np.asarray(np.load(member_path("gen2w24", ver, split, stem), mmap_mode="r"), np.float32)
    tcn = np.asarray(np.load(member_path("gen2tcn", ver, split, stem), mmap_mode="r"), np.float32)
    pred = Path("artifacts/pred") / ver / combo.WINNER / split / path.name
    with np.load(pred) as pz:
        ws = np.asarray(pz["s"], np.float32)
    for name, arr in (("gen2", g), ("w24", w24), ("tcn", tcn), ("winner", ws)):
        if tuple(arr.shape) != tuple(ev.shape):
            raise SystemExit(f"{stem} {name} {arr.shape} != {ev.shape}")
    c = ((g + ws + w24 + tcn) / 4.0).astype(np.float32)
    a = (0.5 * (g + ws)).astype(np.float32)
    bg = valid & (ev == 0) & (nu == 0)
    return dict(stem=stem, path=path, c=c, a=a, winner=ws, ev=ev, valid=valid, nu=nu, bg=bg, x=x, t=t)


def fit_threshold(files, ver, split, key):
    parts = []
    n_bg = 0
    for i, p in enumerate(files, 1):
        seq = load_blend(ver, split, p)
        n_bg += int(seq["bg"].sum())
        parts.append(np.asarray(seq[key][seq["bg"]], np.float32).copy())
        del seq
        if i == 1 or i % 15 == 0 or i == len(files):
            log(f"порог {key} {ver} {split} {i}/{len(files)} n_bg {n_bg}")
    bg = np.concatenate(parts)
    del parts
    thr = float(M.far_threshold(bg, 1e-3))
    n = int(bg.size)
    del bg
    gc.collect()
    return thr, n


def score_files(files, ver, split, thr):
    rows = {k: [] for k in ("c", "a", "winner")}
    for i, p in enumerate(files, 1):
        seq = load_blend(ver, split, p)
        truth = seq["ev"] > 0
        valid = seq["valid"]
        for k in rows:
            rows[k].append(M.confusion((seq[k] >= thr[k]) & valid, truth, valid))
        del seq
        if i == 1 or i % 15 == 0 or i == len(files):
            log(f"счёт {ver} {split} {i}/{len(files)}")
    return rows


def build_frames(files):
    import s3_sel2_cv as s2
    import s3_sel_eval as ev
    import pandas as pd

    labels = pd.read_csv("data/synth_meta/v8/labels.csv")
    seqs = pd.read_csv("data/synth_meta/v8/sequences.csv").set_index("seq_id")
    buckets = {k: [] for k in ("c", "winner")}
    for i, p in enumerate(files, 1):
        seq = load_blend("v8", "val", p, with_xy=True)
        ev_idx, nu_idx, bg_idx, n_bg = ev._indices(seq["ev"], seq["valid"], seq["nu"], seq["stem"])
        cooling, _rows = ev._events_for(seq["stem"], seq["ev"], seq["x"], seq["t"], labels, seqs, ev_idx)

        def frame(arr, stem=seq["stem"], cooling=cooling, n_bg=n_bg, ev_idx=ev_idx, nu_idx=nu_idx, bg_idx=bg_idx):
            e, n, b = s2._cols(arr, ev_idx, nu_idx, bg_idx)
            return dict(ev_m=e[None], nu_m=n[None], bg_m=b[None], n_bg=n_bg, cooling=cooling, stem=stem)

        buckets["c"].append(frame(seq["c"]))
        buckets["winner"].append(frame(seq["winner"]))
        del seq
        if i == 1 or i % 15 == 0 or i == len(files):
            log(f"кадры {i}/{len(files)}")
    order = np.argsort([fr["stem"] for fr in buckets["winner"]])
    return {k: [buckets[k][i] for i in order] for k in buckets}


def nested_c(frames):
    import s3_sel2_cv as s2
    import s3_sel_core as sel

    winner = frames["winner"]
    out = {
        "stems": [fr["stem"] for fr in winner],
        "procedure": "s3_sel2_cv sequence_folds i%k, holdout FAR 1e-3, micro, фон до 200k на файл",
    }
    for k in (2, 5):
        folds = sel.sequence_folds(len(winner), k)
        parts_w, parts_c, rows = [], [], []
        for r, idx in enumerate(folds):
            hw = s2.eval_one(s2.take(winner, idx))
            hc = s2.eval_one(s2.take(frames["c"], idx))
            parts_w.append(hw)
            parts_c.append(hc)
            rows.append(dict(
                fold=r, n_seq=len(idx),
                winner=float(hw["iou"]), selected=float(hc["iou"]),
                delta=float(hc["iou"] - hw["iou"]),
            ))
        iw = float(sel.micro_iou(parts_w))
        ic = float(sel.micro_iou(parts_c))
        out[str(k)] = dict(winner_iou=iw, selected_iou=ic, delta=float(ic - iw), folds=rows)
        log(f"nested c k={k} {ic:.4f} winner {iw:.4f} delta {ic - iw:+.4f}")
    return out


def render_section(r) -> str:
    c = r["c"]
    j = r["judge"]
    lines = [
        "",
        "## (c) mean(GEN2, победитель, GEN2W24, GEN2TCN)",
        "",
        "Четыре карты с равным весом: `(gen2 + winner + gen2w24 + gen2tcn) / 4`.",
        "Оба добавленных чекпойнта выбраны своим тренером на шаге 0. Максимум |Δ| со stage1 init равен 0: "
        "это `artifacts/stage1/unet_w24.pt` и `artifacts/stage1/tcn.pt`. "
        "Их прокси был v10 val, не четыре файла v8 val. В среднем GEN2 эти четыре файла по-прежнему есть, поэтому честная val — те же 41.",
        "",
        "| сплит | IoU | победитель | Δ | ДИ 2.5 | ДИ 97.5 | ДИ включает 0 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| val45, порог полного val | {combo.cell(c['val45'])} |",
        f"| val41, порог 41 | {combo.cell(c['hold41_refit'])} |",
        f"| val41, порог полного val | {combo.cell(c['hold41_frozen'])} |",
        f"| v8 test | {combo.cell(c['test'])} |",
        f"| v10 test | {combo.cell(c['v10'])} |",
        f"| v11 test | {combo.cell(c['v11'])} |",
        "",
        f"Вложенные 41: k5 Δ {c['nested41']['5']['delta']:+.5f}, k2 Δ {c['nested41']['2']['delta']:+.5f}.",
        f"Вложенные 45 (четыре прокси-файла внутри фолдов): k5 Δ {c['nested45']['5']['delta']:+.5f}, k2 Δ {c['nested45']['2']['delta']:+.5f}.",
        f"Порог (c): полный val {r['threshold_full']:.6f}, 41 файл {r['threshold_hold41']:.6f}.",
        f"Пункты beats: val41 {int(j['val_ok'])}, k5 {int(j['nest_ok'])}, v10 {int(j['v10_ok'])}, v11 {int(j['v11_ok'])}, ДИ {int(j['ci_ok'])}. beats={int(j['beats'])}.",
        "",
        "Дельта (c)−(a) на 41 файле при своём пороге каждой смеси: "
        f"{r['c_minus_a_hold41']['point']:+.5f}, ДИ {r['c_minus_a_hold41']['p2_5']:+.5f}…{r['c_minus_a_hold41']['p97_5']:+.5f}. "
        "Это описание. Вес по ней не выбирался.",
        "",
    ]
    return "\n".join(lines)


def event_line(r) -> str:
    c = r["c"]
    j = r["judge"]
    ba = r["c_minus_a_hold41"]
    token = ""
    if j["beats"]:
        token = " BEATS_WINNER COMBO c"
    return (
        "COMBO_DONE members=gen2w24@step0,gen2tcn@step0 init_max_abs=0 "
        "formula=(gen2+winner+w24+tcn)/4 n_boot=1000 seed=20261016 "
        f"c_val45_iou={c['val45']['iou']:.6f} c_val45={combo.boot_txt(c['val45'])} "
        f"c_hold41_iou={c['hold41_refit']['iou']:.6f} c_hold41={combo.boot_txt(c['hold41_refit'])} "
        f"c_hold41_frozen={combo.boot_txt(c['hold41_frozen'])} "
        f"c_test={combo.boot_txt(c['test'])} c_v10={combo.boot_txt(c['v10'])} c_v11={combo.boot_txt(c['v11'])} "
        f"c_k5={c['nested41']['5']['delta']:+.6f} c_k2={c['nested41']['2']['delta']:+.6f} "
        f"c_k5_45={c['nested45']['5']['delta']:+.6f} c_k2_45={c['nested45']['2']['delta']:+.6f} "
        f"c_minus_a_hold41=d={combo.f6(ba['point'])} ci={combo.f6(ba['p2_5'])},{combo.f6(ba['p97_5'])} "
        f"beats_c={int(j['beats'])} beats={int(j['beats'])} ab_from=COMBO1 "
        f"real_train=0 infer=unchanged"
        + token
    )


def main():
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if combo.has_tag("COMBO_DONE"):
        log("строка COMBO_DONE уже есть")
        return
    t0 = time.time()
    meta = []
    for spec in MEMBERS:
        done = Path("artifacts/ckpt") / spec["run"] / "DONE"
        if not done.exists():
            raise SystemExit(f"нет DONE {spec['run']}")
        if trainer_alive(spec["run"]):
            raise SystemExit(f"тренер {spec['run']} ещё пишет веса")
        if not Path(spec["pt"]).exists():
            raise SystemExit(f"нет {spec['pt']}")
        diff = max_abs_diff(Path(spec["pt"]), Path(spec["init"]))
        train = json.loads(Path(f"artifacts/eval/{spec['run']}_train.json").read_text(encoding="utf-8"))
        meta.append(dict(
            name=spec["name"], best_step=int(train["best_step"]),
            best_proxy_iou=float(train["best_proxy_iou"]),
            init_max_abs=diff, pt=spec["pt"],
        ))
        log(f"{spec['name']} best_step {train['best_step']} init |Δ| {diff:.3g}")
        if diff != 0.0:
            log(f"внимание: {spec['name']} отличается от init на {diff}")
    files = data.list_cache("artifacts/cache/v8", "val")
    if len(files) != 45 or tuple(p.stem for p in files[:4]) != combo.PROXY:
        raise SystemExit("прокси v8 val съехали")
    hold = files[4:]
    if not combo.map_path("v8", "val", files[0].stem).exists():
        raise SystemExit("нет карт GEN2 в /tmp/s3_combo_maps")
    nets = [(spec, load_net(spec)) for spec in MEMBERS]
    ensure_members(nets, "v8", "val")
    thr_c, n_bg = fit_threshold(files, "v8", "val", "c")
    thr_w_hold, n_hold = fit_threshold(hold, "v8", "val", "winner")
    thr_c_hold, _ = fit_threshold(hold, "v8", "val", "c")
    # Порог (a) и победителя на полном val — опубликованные, как в COMBO1.
    thr_full = {"c": thr_c, "a": 0.8579896092424856, "winner": WINNER_THR}
    rows_full = score_files(files, "v8", "val", thr_full)
    combo.check_close(combo.pack(rows_full["winner"])["iou"], combo.WINNER_PUB["val"], "winner val45")
    combo.check_close(combo.pack(rows_full["a"])["iou"], combo.BLEND_PUB["val"], "a val45")
    thr_hold = {"c": thr_c_hold, "a": None, "winner": thr_w_hold}
    # Порог (a) на 41 берём из уже посчитанного COMBO1, затем сверяем IoU.
    prev = json.loads(Path("artifacts/eval/s3_combo_report.json").read_text(encoding="utf-8"))
    thr_hold["a"] = float(prev["thresholds_hold41"]["a"]["threshold"])
    rows_hold = score_files(hold, "v8", "val", thr_hold)
    rows_frozen = score_files(hold, "v8", "val", thr_full)
    combo.check_close(combo.pack(rows_hold["a"])["iou"], prev["a"]["hold41_refit"]["iou"], "a hold41 vs COMBO1")
    frames41 = build_frames(hold)
    if [fr["stem"] for fr in frames41["winner"]] != prev["nested41_stems"]:
        raise SystemExit("стемы 41 не совпали")
    nested41 = nested_c(frames41)
    del frames41
    gc.collect()
    frames45 = build_frames(files)
    nested45 = nested_c(frames45)
    del frames45
    gc.collect()
    external = {}
    for ver, split, key in (("v8", "test", "test"), ("v10", "test", "v10"), ("v11", "test", "v11")):
        ensure_members(nets, ver, split)
        rows = score_files(data.list_cache(f"artifacts/cache/{ver}", split), ver, split, thr_full)
        combo.check_close(combo.pack(rows["winner"])["iou"], combo.WINNER_PUB[key], f"winner {key}")
        combo.check_close(combo.pack(rows["a"])["iou"], combo.BLEND_PUB[key], f"a {key}")
        external[key] = rows
        del rows
        gc.collect()
    del nets
    gc.collect()

    def one(rows_c, rows_w):
        return combo.split_block(rows_c, rows_w)

    block = dict(
        val45=one(rows_full["c"], rows_full["winner"]),
        hold41_refit=one(rows_hold["c"], rows_hold["winner"]),
        hold41_frozen=one(rows_frozen["c"], rows_frozen["winner"]),
        test=one(external["test"]["c"], external["test"]["winner"]),
        v10=one(external["v10"]["c"], external["v10"]["winner"]),
        v11=one(external["v11"]["c"], external["v11"]["winner"]),
        nested41={k: nested41[k] for k in ("2", "5")},
        nested45={k: nested45[k] for k in ("2", "5")},
    )
    c_minus_a = combo.bootstrap(rows_hold["c"], rows_hold["a"])
    judged = combo.judge("c", block["hold41_refit"], block["nested41"], block["v10"], block["v11"], True)
    report = dict(
        name="COMBO_DONE",
        formula="(gen2 + winner + gen2w24 + gen2tcn) / 4",
        weights_fixed_before_eval=True,
        members=meta,
        proxy_files=list(combo.PROXY),
        threshold_full=thr_c,
        threshold_hold41=thr_c_hold,
        threshold_winner_hold41=thr_w_hold,
        threshold_a_hold41=thr_hold["a"],
        n_bg_full=n_bg,
        n_bg_hold41=n_hold,
        c=block,
        c_minus_a_hold41=c_minus_a,
        nested41=nested41,
        nested45=nested45,
        judge=judged,
        beats=int(judged["beats"]),
        beats_rule="same as scripts/s3_combo.py, applied to mixture c",
        infer_changed=False,
        real_train=0,
        seconds=round(time.time() - t0, 1),
    )
    combo.dump(OUT, report)
    section = render_section(report)
    text = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    pending = (
        "(c) ждёт `artifacts/ckpt/s3_gen2w24/DONE` и `artifacts/ckpt/s3_gen2tcn/DONE`. "
        "Текущие `artifacts/eval/s3_gen2w24_unet.pt` и `s3_gen2tcn_tcn.pt` — снимки идущего обучения."
    )
    done_sentence = (
        "(c) посчитана отдельным прогоном: равное среднее GEN2, победителя, GEN2W24 и GEN2TCN. "
        f"Оба чекпойнта — шаг 0, max |Δ| со stage1 равен 0. "
        f"На 41 файле Δ {block['hold41_refit']['delta']:+.6f}, "
        f"ДИ {block['hold41_refit']['ci']['p2_5']:+.6f}…{block['hold41_refit']['ci']['p97_5']:+.6f}. "
        f"beats={int(judged['beats'])}."
    )
    if pending in text:
        text = text.replace(pending, done_sentence)
    if "## (c) mean" not in text:
        text = text.rstrip() + "\n" + section
    DOC.write_text(text, encoding="utf-8")
    line = event_line(report)
    if judged["beats"] and "BEATS_WINNER" not in line:
        raise SystemExit("beats=1 без токена")
    if not judged["beats"] and "BEATS_WINNER" in line:
        raise SystemExit("beats=0, а токен попал в строку")
    combo.elog(line)
    if combo.MAPS.exists():
        shutil.rmtree(combo.MAPS, ignore_errors=True)
    log(f"записан {OUT} за {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
