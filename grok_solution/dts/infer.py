"""Сквозной инференс (CPU): сырое поле T(t,x) -> карта вероятности аномалии -> зоны -> тип зоны."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from . import grid, features, models, events as E, postproc as PP

import os
ART = Path(os.environ.get("DTS_MODELS", Path(__file__).resolve().parents[1] / "models_release"))

_ZONES = ("cc", "pdrs_hybrid")
_STATIC = ("off", "on")
_MAE = ("off", "on")
_GEN2 = ("off", "on")
_HYBRID_FALLBACK = {
    "mode": "seed",
    "floor": 0.7,
    "r_saddle": 0.3,
    "region_median": 0.5,
    "fallback": False,
}


def hybrid_spec(path=None):
    """Структурные параметры гибрида PDRS. seed и thr в рантайме равны порогу анализатора."""
    p = Path(path) if path else Path(__file__).resolve().parents[1] / "artifacts" / "eval" / "pdrs_hybrid_best_params.json"
    spec = dict(_HYBRID_FALLBACK)
    if p.is_file():
        raw = json.loads(p.read_text())
        if isinstance(raw, dict) and isinstance(raw.get("best_params"), dict):
            raw = raw["best_params"]
        if isinstance(raw, dict):
            for key in ("mode", "floor", "r_saddle", "region_median", "fallback",
                        "seed", "thr", "min_cells", "max_gap_s", "peak_size"):
                if key in raw:
                    spec[key] = raw[key]
    if spec.get("mode") not in ("seed", "refine"):
        spec["mode"] = "seed"
    spec["fallback"] = bool(spec.get("fallback", False))
    return spec


def _pipeline_from_pdrs(zones, x, t):
    """Схема зоны победителя: id с нуля, start_m/end_m, t_start_s/t_end_s. Полярность — как у PDRS."""
    out = []
    for i, z in enumerate(zones):
        t0, t1 = int(z["t0"]), int(z["t1"])
        x0, x1 = int(z["x0"]), int(z["x1"])
        out.append(dict(
            id=i, t0=t0, t1=t1, x0=x0, x1=x1,
            start_m=float(x[x0]), end_m=float(x[x1 - 1]),
            t_start_s=float(t[t0]), t_end_s=float(t[t1 - 1]),
            n_px=int(z["n_px"]), polarity=z["polarity"], score=float(z["score"]),
        ))
    return out


def pipeline_zones(score, hot, valid, x, t, thr, zones="cc"):
    """cc — тот же zones_from_prob, что у победителя. pdrs_hybrid — extract_zones_hybrid и та же схема."""
    if zones == "cc":
        return E.zones_from_prob(score, hot, valid, x, t, thr)
    if zones != "pdrs_hybrid":
        raise ValueError("zones must be 'cc' or 'pdrs_hybrid'")
    from .pdrs import extract_zones_hybrid
    spec = hybrid_spec()
    mode = spec["mode"]
    seed = float(thr) if mode == "seed" else float(spec.get("seed", thr))
    kw = dict(
        mode=mode, thr=float(thr), seed=seed,
        floor=float(spec["floor"]), r_saddle=float(spec["r_saddle"]),
        region_median=float(spec["region_median"]), fallback=bool(spec["fallback"]),
    )
    if "min_cells" in spec:
        kw["min_cells"] = int(spec["min_cells"])
    if "max_gap_s" in spec:
        kw["max_gap_s"] = float(spec["max_gap_s"])
    if "peak_size" in spec:
        kw["peak_size"] = tuple(spec["peak_size"])
    _, raw = extract_zones_hybrid(score, valid, hot, t, x, **kw)
    return _pipeline_from_pdrs(raw, x, t)


def static_spec(path=None):
    """Выбранное слияние STATIC2. Порог и режим — из val[selected] в s3_static2.json.

    Калибровка только на synth v8 val. Реальные CSV и карточки этот словарь не меняют.
    """
    p = Path(path) if path else Path(__file__).resolve().parents[1] / "artifacts" / "static" / "s3_static2.json"
    raw = json.loads(Path(p).read_text())
    name = raw["selected"]
    block = raw["val"][name]
    return {
        "name": name,
        "mode": block["mode"],
        "scale": float(block["scale"]),
        "gate": float(block["gate"]),
        "threshold": float(block["threshold"]),
        "ckpt": raw["ckpt"],
        "step": int(raw["step"]),
    }


def static_ckpt_path(spec, root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    ck = Path(spec["ckpt"])
    return ck if ck.is_absolute() else root / ck


def load_static_net(path):
    """StaticNet из чекпойнта STATIC2. Сеть не член ансамбля победителя."""
    from .static_net import StaticNet
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = StaticNet(
        cin=int(ckpt.get("cin", 9)), width=int(ckpt.get("width", 16)), pool=int(ckpt.get("pool", 80)),
        max_rows=int(ckpt.get("max_rows", 64)), n_bins=int(ckpt.get("n_bins", 0)),
    )
    net.load_state_dict(ckpt["state"])
    net.eval()
    return net


def operating_threshold(winner_thr, spec, static):
    """off — порог выпуска. on — замороженный порог слитой карты."""
    if static == "off":
        return winner_thr
    if static != "on":
        raise ValueError("static must be 'off' or 'on'")
    return float(spec["threshold"])


def apply_static_score(score, T, valid, dx, net, spec):
    """Остаток в °C на той же сетке, что и скор, затем fuse. Скор ансамбля не занижается (max/blend).

    Запись длиннее обучающего окна режется скользящими окнами по времени
    (перекрытие 1/2, max). Тренд по x считается на всей ширине и на всех кадрах,
    прореживание до max_rows — уже внутри окна. Короткая запись — прежний один проход.
    """
    from .static_net import (
        INFER_TIME_WIN, fuse, predict_map, predict_map_windows, residual_from_temperature,
    )
    max_rows = int(getattr(net, "max_rows", 256))
    rows = int(np.asarray(T).shape[0])
    win = int(INFER_TIME_WIN)
    if rows <= win:
        resid, rv = residual_from_temperature(T, valid, dx=float(dx), max_rows=max_rows)
        column_p = predict_map(net, resid, rv)
    else:
        resid, rv = residual_from_temperature(T, valid, dx=float(dx), max_rows=rows)
        column_p = predict_map_windows(net, resid, rv, win=win)
    return fuse(score, column_p, spec["mode"], float(spec["scale"]), float(spec["gate"]))


def score_with_static(score, T, valid, dx, static, net=None, spec=None):
    """off возвращает тот же объект score. on вызывает apply_static_score."""
    if static == "off":
        return score
    if static != "on":
        raise ValueError("static must be 'off' or 'on'")
    if net is None or spec is None:
        raise ValueError("static=on needs a StaticNet and static_spec")
    return apply_static_score(score, T, valid, dx, net, spec)


def mae_release_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "models_release_mae"


def mae_spec(path=None) -> dict:
    """Порог ens_mean4 и узлы CDF. Веса реконструктора лежат рядом симлинком, не копией в git."""
    p = Path(path) if path else mae_release_dir() / "stage1_thresholds.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    thr = float(raw["thresholds"]["cnn_unet"]["far_1e-3"])
    ckpt = Path(raw["ckpt"])
    if not ckpt.is_file():
        ckpt = p.parent / raw["ckpt"]
    knots = raw.get("cdf_knots")
    if not knots:
        raise ValueError("в релизе MAE нет cdf_knots")
    return dict(
        threshold=thr,
        mae_weight=float(raw.get("mae_weight", 0.2)),
        definition=raw.get("definition", ""),
        ckpt=str(ckpt),
        width=int(raw.get("width", 16)),
        band=int(raw.get("band", 8)),
        radius=int(raw.get("radius", 1)),
        cdf_knots=knots,
        method=raw.get("method", "ens_mean4"),
    )


def load_mae_model(spec: dict):
    from .mae_anom import MaskedReconstructor
    model = MaskedReconstructor(int(spec.get("width", 16)))
    state = torch.load(spec["ckpt"], map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def blend_with_mae(score, hot, s_mae, h_mae, n_members: int):
    """Равный вес члена mae: (n * средний победитель + mae) / (n + 1). При n=4 это ens_mean4."""
    n = int(n_members)
    if n < 1:
        raise ValueError("нет членов ансамбля")
    s = (np.asarray(score, np.float32) * n + np.asarray(s_mae, np.float32)) / float(n + 1)
    h = (np.asarray(hot, np.float32) * n + np.asarray(h_mae, np.float32)) / float(n + 1)
    return s.astype(np.float32), h.astype(np.float32)


def gen2_release_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "models_release_gen2"


def gen2_spec(path=None) -> dict:
    """Порог ens_blend: 0.5 * (победитель + U-Net GEN2). Фон полного v8 val, FAR 1e-3."""
    p = Path(path) if path else gen2_release_dir() / "stage1_thresholds.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    thr = float(raw["thresholds"]["cnn_unet"]["far_1e-3"])
    ckpt = Path(raw["ckpt"])
    if not ckpt.is_file():
        ckpt = p.parent / raw["ckpt"]
    return dict(
        threshold=thr,
        definition=raw.get("definition", ""),
        ckpt=str(ckpt),
        width=int(raw.get("width", 16)),
        method=raw.get("method", "ens_blend"),
    )


def load_gen2_model(spec: dict):
    model = models.build("unet", int(spec.get("width", 16)))
    state = torch.load(spec["ckpt"], map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def blend_with_gen2(score, hot, s_gen2, h_gen2):
    """Равная смесь: 0.5 * (средний скор четырёх членов + U-Net GEN2)."""
    s = 0.5 * (np.asarray(score, np.float32) + np.asarray(s_gen2, np.float32))
    h = 0.5 * (np.asarray(hot, np.float32) + np.asarray(h_gen2, np.float32))
    return s.astype(np.float32), h.astype(np.float32)


class Analyzer:
    def __init__(self, stage1=ART / "unet.pt", stage2=ART / "eventnet.pt", lgbm=ART / "lgbm.joblib",
                 thresholds=ART / "stage1_thresholds.json", width=16, op="far_ref", zones="cc", static="off",
                 mae="off", gen2="off"):
        if zones not in _ZONES:
            raise ValueError("zones must be 'cc' or 'pdrs_hybrid'")
        if static not in _STATIC:
            raise ValueError("static must be 'off' or 'on'")
        if mae not in _MAE:
            raise ValueError("mae must be 'off' or 'on'")
        if gen2 not in _GEN2:
            raise ValueError("gen2 must be 'off' or 'on'")
        if static == "on" and mae == "on":
            raise ValueError("mae=on и static=on вместе не калибровались")
        if gen2 == "on" and (static == "on" or mae == "on"):
            raise ValueError("gen2=on вместе с mae или static не калибровался")
        self.zones = zones
        self.static = static
        self.static_spec = None
        self.static_net = None
        self.mae = mae
        self.mae_spec = None
        self.mae_model = None
        self.mae_knots = None
        self.gen2 = gen2
        self.gen2_spec = None
        self.gen2_net = None
        th = json.load(open(thresholds)); self.arch = th.get("arch", "unet")
        # детектор: одна сеть или ансамбль (members: [{"kind": "unet"|"tcn"|...|"pixel_lgbm", "file":..., "width":...}])
        self.members = []
        for m in th.get("members") or [{"kind": self.arch, "file": Path(stage1).name, "width": th.get("width")}]:
            f = Path(thresholds).parent / m["file"]
            if m["kind"] == "pixel_lgbm":
                import joblib; self.members.append(("pixel_lgbm", joblib.load(f)))
            else:
                net = models.build(m["kind"], m.get("width")); net.load_state_dict(torch.load(f, map_location="cpu")); net.eval()
                self.members.append(("net", net))
        self.unet = self.members[0][1]
        ck = torch.load(stage2, map_location="cpu", weights_only=False)
        self.mu = np.asarray(ck["mu"], np.float32); self.sd = np.asarray(ck["sd"], np.float32)
        self.temperature = float(ck.get("temperature", 1.0))
        self.lgbm_temperature = float(ck.get("lgbm_temperature", 1.0))
        self.cnn_weight = float(ck.get("cnn_weight", 0.5))
        self.scalar_names = list(ck["scalar_names"]) if ck.get("scalar_names") else None
        self.two_scale = bool(ck.get("two_scale", False))
        self.stack_coef = None if ck.get("stack_coef") is None else np.asarray(ck["stack_coef"], np.float32)
        self.stack_intercept = None if ck.get("stack_intercept") is None else np.asarray(ck["stack_intercept"], np.float32)
        self.enet = models.EventNet(len(E.CLASSES), len(self.mu), cin=int(ck.get("cin", 3)), late_fusion=bool(ck.get("late_fusion", False)))
        self.enet.load_state_dict(ck["state"]); self.enet.eval()
        self.lgbm = None; self.lgbm_kind = "legacy"
        try:
            import joblib
            obj = joblib.load(lgbm)
            if isinstance(obj, dict) and "model" in obj:
                self.lgbm = obj["seeds"] if obj.get("seeds") else obj["model"]
                self.lgbm_kind = obj.get("kind", "full")
            else:
                self.lgbm = obj
        except Exception:
            pass
        self.winner_thr = float(th["thresholds"]["cnn_unet"][op])
        self.thr = self.winner_thr
        self.op = op
        if static == "on":
            self._arm_static()
        if mae == "on":
            self._arm_mae()
        if gen2 == "on":
            self._arm_gen2()

    def _arm_static(self):
        if self.static_net is not None and self.static_spec is not None:
            return
        spec = static_spec()
        self.static_spec = spec
        self.static_net = load_static_net(static_ckpt_path(spec))
        if self.static == "on":
            self.thr = float(spec["threshold"])

    def _arm_mae(self):
        if self.mae_model is not None and self.mae_spec is not None:
            return
        spec = mae_spec()
        self.mae_spec = spec
        self.mae_model = load_mae_model(spec)
        self.mae_knots = np.asarray(spec["cdf_knots"], np.float64)
        if self.mae == "on":
            self.thr = float(spec["threshold"])

    def _arm_gen2(self):
        if self.gen2_net is not None and self.gen2_spec is not None:
            return
        spec = gen2_spec()
        self.gen2_spec = spec
        self.gen2_net = load_gen2_model(spec)
        if self.gen2 == "on":
            self.thr = float(spec["threshold"])

    def _mae_member(self, X):
        from .mae_anom import TILE, member_maps
        spec = self.mae_spec
        return member_maps(
            self.mae_model, X[:2], X[2], self.mae_knots,
            tile=TILE, overlap=32, band=int(spec["band"]), radius=int(spec["radius"]), tile_batch=2,
        )

    def classify(self, X, sigma, x, t, zones):
        return predict_zone_probs(
            self.enet, self.mu, self.sd, X, sigma, x, t, zones, lgbm=self.lgbm, lgbm_kind=self.lgbm_kind,
            temperature=self.temperature, lgbm_temperature=self.lgbm_temperature, cnn_weight=self.cnn_weight,
            scalar_names=self.scalar_names, two_scale=self.two_scale,
            stack_coef=self.stack_coef, stack_intercept=self.stack_intercept)

    def run(self, T, valid, x, t, postproc=None, zones=None, static=None, mae=None, gen2=None):
        """postproc: None — dts.postproc.ENABLED только при op == far_1e-3; False/True — явно.

        Включённый вариант сдвигается так, что сравнение карты с self.thr остаётся
        калибровкой FAR 1e-3 этого варианта. Выключенный флаг не меняет score.

        zones: None — self.zones. По умолчанию 'cc' вызывает events.zones_from_prob
        с теми же аргументами. 'pdrs_hybrid' берёт замороженные floor/r_saddle/region_median,
        а seed и порог равны self.thr. Набор ключей результата тот же.

        static: None — self.static. 'off' не загружает StaticNet и не меняет score:
        порог зон — self.thr выпуска. 'on' сливает выбранную ветвь STATIC2
        (score не ниже ансамбля) и режет зоны замороженным порогом слитой карты.

        mae: None — self.mae. 'off' не загружает реконструктор и оставляет
        средний скор четырёх членов победителя тем же массивом. 'on' добавляет
        член веса 1/5 и режет зоны порогом ens_mean4.

        gen2: None — self.gen2. 'off' не загружает U-Net GEN2 и оставляет тот же
        средний скор и тот же порог. 'on' ставит 0.5 * (победитель + GEN2) и порог
        ens_blend из models_release_gen2. Вместе с mae или static не калибровался.
        """
        mode = self.zones if zones is None else zones
        if mode not in _ZONES:
            raise ValueError("zones must be 'cc' or 'pdrs_hybrid'")
        use = self.static if static is None else static
        if use not in _STATIC:
            raise ValueError("static must be 'off' or 'on'")
        use_mae = self.mae if mae is None else mae
        if use_mae not in _MAE:
            raise ValueError("mae must be 'off' or 'on'")
        if use == "on" and use_mae == "on":
            raise ValueError("mae=on и static=on вместе не калибровались")
        use_gen2 = self.gen2 if gen2 is None else gen2
        if use_gen2 not in _GEN2:
            raise ValueError("gen2 must be 'off' or 'on'")
        if use_gen2 == "on" and (use == "on" or use_mae == "on"):
            raise ValueError("gen2=on вместе с mae или static не калибровался")
        p = grid.plan(x, t)
        Tc, v = grid.resample(T, valid, p, "mean")
        X, aux = features.make_channels(Tc, v)
        ss, hh = [], []
        for kind, mdl in self.members:
            if kind == "net":
                prob = models.predict_full(mdl, X)
            else:
                from . import classic
                prob = np.moveaxis(mdl.predict_proba(classic.pixel_features(X)).reshape(X.shape[1], X.shape[2], 3), -1, 0)
            ss.append(1 - prob[0]); hh.append(prob[1] - prob[2])
        score = np.mean(ss, 0); hot = np.mean(hh, 0)
        mae_thr = None
        if use_mae == "on":
            if self.mae_model is None:
                self._arm_mae()
            if len(self.members) != 4:
                raise RuntimeError("ens_mean4 ждёт ровно 4 члена победителя")
            s_mae, h_mae = self._mae_member(X)
            score, hot = blend_with_mae(score, hot, s_mae, h_mae, len(self.members))
            mae_thr = float(self.mae_spec["threshold"])
        gen2_thr = None
        if use_gen2 == "on":
            if self.gen2_net is None:
                self._arm_gen2()
            if len(self.members) != 4:
                raise RuntimeError("ens_blend GEN2 ждёт ровно 4 члена победителя")
            prob = models.predict_full(self.gen2_net, X)
            s_g = 1.0 - prob[0]
            h_g = prob[1] - prob[2]
            score, hot = blend_with_gen2(score, hot, s_g, h_g)
            gen2_thr = float(self.gen2_spec["threshold"])
        extra_thr = mae_thr if mae_thr is not None else gen2_thr
        if PP.resolve_enabled(postproc, self.op):
            pp_thr = self.winner_thr if use == "on" else (extra_thr if extra_thr is not None else self.thr)
            score = PP.score_for_analyzer(score, v, pp_thr, True)
        zone_thr = extra_thr if extra_thr is not None else self.thr
        if use == "on":
            if self.static_net is None:
                self._arm_static()
            score = score_with_static(score, Tc, v, float(p["dx"]), "on", self.static_net, self.static_spec)
            zone_thr = operating_threshold(self.winner_thr, self.static_spec, "on")
        found = pipeline_zones(score, hot, v, p["x"], p["t"], zone_thr, mode)
        P = self.classify(X, aux["sigma_x"], p["x"], p["t"], found)
        for z, pr in zip(found, P):
            z["class"] = E.CLASSES[int(pr.argmax())]; z["class_prob"] = {c: round(float(q), 3) for c, q in zip(E.CLASSES, pr)}
            z["alarm"] = z["class"] not in ("HVAC_Diurnal", "Sensor_Artifact")
            sl = (slice(z["t0"], z["t1"]), slice(z["x0"], z["x1"]))
            z["peak_dT_C"] = float(np.max(np.abs(aux["dT_time"][sl])))
        return dict(grid=p, X=X, valid=v, score=score, hotness=hot, zones=found, sigma=aux["sigma_x"], dT=aux["dT_time"])


def rezone(analyzer, out, thr, zones="pdrs_hybrid"):
    """Зоны при другом пороге тем же экстрактором, что Analyzer.run.

    domain.rethreshold всегда зовёт zones_from_prob. Для pdrs_hybrid его не вызывать:
    иначе флаг зон откатывается к cc. Класс и тревога — domain.annotate_zones.
    """
    if zones not in _ZONES:
        raise ValueError("zones must be 'cc' or 'pdrs_hybrid'")
    from . import domain
    p = out["grid"]
    found = pipeline_zones(out["score"], out["hotness"], out["valid"], p["x"], p["t"], thr, zones)
    domain.annotate_zones(analyzer, out["X"], out["sigma"], p["x"], p["t"], found, out["dT"])
    new = dict(out)
    new["zones"] = found
    new["threshold"] = float(thr)
    return new


def _softmax_np(dec):
    z = dec - np.max(dec, axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def predict_zone_probs(enet, mu, sd, X, sigma, x, t, zones, lgbm=None, lgbm_kind="legacy", temperature=1.0,
                       lgbm_temperature=1.0, cnn_weight=0.5, scalar_names=None, two_scale=False,
                       stack_coef=None, stack_intercept=None):
    """Вероятности классов по зонам. Старый чекпоинт (T=1, вес 0.5, legacy LGBM) совпадает с прежним classify."""
    if not zones:
        return []
    imgs, sc = [], []
    for z in zones:
        im, f = E.crop_and_features(X, sigma, x, t, z["start_m"], z["end_m"], z["t_start_s"], z["t_end_s"])
        imgs.append(im); sc.append(f)
    imgs = np.stack(imgs); base = np.stack(sc).astype(np.float32)
    names = list(scalar_names) if scalar_names else list(E.SCALAR_NAMES)
    scal = base if names == list(E.SCALAR_NAMES) else E.named_scalars(base, imgs, zones, names)
    mu = np.asarray(mu, np.float32); sd = np.asarray(sd, np.float32)
    S = torch.from_numpy(((scal - mu) / sd).astype(np.float32))
    xt = torch.from_numpy(imgs)

    def prep(im):
        return models.two_scale(im) if two_scale else im

    with torch.no_grad():
        lo = enet(prep(xt), S)
        lo = 0.5 * (lo + enet(prep(torch.flip(xt, [3])), S))
        lo = lo / max(float(temperature), 1e-2)
        P = torch.softmax(lo, 1).numpy()
    gbms = [] if lgbm is None else (list(lgbm) if isinstance(lgbm, (list, tuple)) else [lgbm])
    if gbms:
        ctx = E.context_from_zones(zones)
        feat = E.lgbm_matrix(base, imgs, ctx, kind=lgbm_kind)
        n_in = getattr(gbms[0], "n_features_in_", None)
        if n_in is not None and feat.shape[1] != n_in:
            for kind in ("legacy", "full", "shape", "context"):
                try:
                    alt = E.lgbm_matrix(base, imgs, ctx, kind=kind)
                except ValueError:
                    continue
                if alt.shape[1] == n_in:
                    feat = alt
                    break
            else:
                return P
        acc = None
        for m in gbms:
            pr = np.asarray(m.predict_proba(feat), np.float64)
            acc = pr if acc is None else acc + pr
        Pl = E.temper_proba(acc / len(gbms), lgbm_temperature)
        if stack_coef is not None:
            dec = np.concatenate([P, Pl], 1) @ np.asarray(stack_coef, np.float64).T
            if stack_intercept is not None:
                dec = dec + np.asarray(stack_intercept, np.float64)
            P = _softmax_np(dec)
        else:
            w = float(np.clip(cnn_weight, 0.0, 1.0))
            P = w * P + (1.0 - w) * Pl
    return P


# Текущий выпуск (docs/RELEASE_DECISION.md): ens_gen2 = 0.5*(ens_u2ft_tcn_lgbm + GEN2 U-Net).
# Конструктор Analyzer() по умолчанию оставлен gen2="off" (= прежний выпуск, откат и совместимость
# с флагами mae/static); выпуск собирайте через release_analyzer().
RELEASE = "ens_gen2"
RELEASE_FLAGS = dict(op="far_1e-3", zones="cc", static="off", mae="off", gen2="on")


def release_analyzer(**overrides) -> "Analyzer":
    """Analyzer текущего выпуска ens_gen2. Откат: release_analyzer(gen2="off")."""
    kw = dict(RELEASE_FLAGS); kw.update(overrides)
    return Analyzer(**kw)
