"""Extract architecture specs of every NN in the repo -> docs/ARCH_SPECS.{md,json}.
CPU only, read-only on checkpoints. Run: nice -19 python scripts/arch_specs.py"""
from __future__ import annotations
import json, sys, traceback, importlib
from pathlib import Path
import torch, torch.nn as nn
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]
from dts import models as M

H, W = 128, 256  # training tile (T rows x X cols) on canonical grid 0.25 m x 120 s
CH3 = "ch0 dT_time (T - common-mode(t) - static profile(x)) / sigma(x); ch1 dT_space (T - rolling median over x 40 m - common-mode) / sigma(x); ch2 valid mask"

def x3(c=3, h=H, w=W): return torch.randn(1, c, h, w)

def load(net, path, key=None, prefix=None):
    if not path: return "no checkpoint"
    p = ROOT / path
    if not p.exists(): return f"missing {path}"
    sd = torch.load(p, map_location="cpu", weights_only=False)
    if key: sd = sd[key]
    if prefix: sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if isinstance(sd, dict) and "model" in sd and "optim" in sd: sd = sd["model"]  # resume blob
    net.load_state_dict(sd, strict=True)
    return f"loaded {path} (strict)"

def attrs(m):
    d = {}
    for a in ("in_channels", "out_channels", "kernel_size", "stride", "dilation", "padding", "in_features", "out_features",
              "num_features", "output_size", "scale_factor", "mode", "p", "num_heads", "embed_dim", "hidden_size", "groups"):
        if hasattr(m, a):
            v = getattr(m, a)
            if isinstance(v, (int, float, str, tuple, list)) or v is None: d[a] = v
    return d

BLOCKS = ("TimeGate", "AxisAttention", "ConvGRUCell", "Integrate", "SmoothReference", "DilatedReference", "TransformerEncoderLayer")

def trace(net, inputs, kwargs=None):
    rows, seen, hooks = [], set(), []
    names = {m: n for n, m in net.named_modules()}
    def hook(m, i, o):
        if id(m) in seen: return
        seen.add(id(m))
        o0 = o[0] if isinstance(o, (tuple, list)) else (next(iter(o.values())) if isinstance(o, dict) else o)
        rows.append(dict(name=names[m], type=type(m).__name__, **attrs(m),
                         in_shape=list(i[0].shape) if i and torch.is_tensor(i[0]) else None,
                         out_shape=list(o0.shape) if torch.is_tensor(o0) else None,
                         params=sum(p.numel() for p in m.parameters(recurse=False))))
    for n, m in net.named_modules():
        leaf = len(list(m.children())) == 0
        if (leaf or type(m).__name__ in BLOCKS) and n:
            hooks.append(m.register_forward_hook(hook))
    import torch.nn.functional as _F
    _mp = _F.max_pool2d; _cnt = [0]
    def mp(x, kernel_size, *a, **k):
        y = _mp(x, kernel_size, *a, **k); _cnt[0] += 1
        st = k.get("stride", a[0] if a else None) or kernel_size
        rows.append(dict(name=f"F.max_pool2d#{_cnt[0]}", type="MaxPool2d (functional)", kernel_size=kernel_size, stride=st,
                         in_shape=list(x.shape), out_shape=list(y.shape), params=0))
        return y
    net.eval()
    _F.max_pool2d = mp
    try:
        with torch.no_grad(): out = net(*inputs, **(kwargs or {}))
    finally:
        _F.max_pool2d = _mp
    for h in hooks: h.remove()
    for r in rows:
        if r["type"] == "TimeGate":
            r["desc"] = ("TimeGate(c): s = mean_x(x) -> (B,C,T); g = sigmoid(Conv1d(C,C,k=5,pad=2)(s)) -> (B,C,T,1); "
                         "out = x * g (squeeze-excitation along time; params in child .fc)")
    covered = set()
    for m in net.modules():
        if id(m) in seen: covered |= {id(p) for p in m.parameters(recurse=False)}
    rest = [(n, p) for n, p in net.named_parameters() if id(p) not in covered]
    if rest:
        rows.append(dict(name="(direct nn.Parameter of container modules)", type="learned tensors: queries / pos-emb / MHA in_proj / loss weights",
                         desc="; ".join(f"{n} {list(p.shape)}" for n, p in rest[:20]) + (" ..." if len(rest) > 20 else ""),
                         in_shape=None, out_shape=None, params=sum(p.numel() for _, p in rest)))
    def shp(o):
        if torch.is_tensor(o): return list(o.shape)
        if isinstance(o, dict): return {k: shp(v) for k, v in o.items()}
        if isinstance(o, (tuple, list)): return [shp(v) for v in o]
        return str(type(o))
    return rows, shp(out)

UNET_SKIPS = ["e1 -> cat -> d1", "e2 -> cat -> d2", "e3 -> cat -> d3", "e4 -> TimeGate -> u3"]
UNET_FLOW = "e1(k7x3) -mp2-> e2(k7x3) -mp2-> e3 -mp2-> e4 -> TimeGate -> u3(convT2) ++e3 -> d3 -> u2 ++e2 -> d2 -> u1 ++e1 -> d1 -> head 1x1 (3 cls: bg/heat/cool), p_anom = 1 - softmax_bg"

def S(name, status, build, inputs, ckpt=None, key=None, prefix=None, skips=None, flow="", input_desc=CH3, head="", kwargs=None, note=""):
    return dict(name=name, status=status, build=build, inputs=inputs, ckpt=ckpt, key=key, prefix=prefix, skips=skips or [],
                flow=flow, input_desc=input_desc, head=head, kwargs=kwargs, note=note)

def specs():
    from dts import axial, convgru, mae_anom, flowprior, centernet, contour, membank, static_net
    L = []
    U = lambda w: (lambda: M.DTSUNet(w=w))
    L.append(S("unet_w24 (release det0)", "RELEASE member of ens_u2ft_tcn_lgbm (ens val IoU 0.387 / test 0.405 @FAR1e-3)", U(24), lambda: [x3()],
               "models_release/det0_unet.pt", skips=UNET_SKIPS, flow=UNET_FLOW, head="Conv1x1 -> 3 logits"))
    L.append(S("unet_v810ft (release det1, width 16)", "RELEASE member; also frozen encoder for CenterNet/contour/membank/realnet", U(16), lambda: [x3()],
               "models_release/det1_unet.pt", skips=UNET_SKIPS, flow=UNET_FLOW, head="Conv1x1 -> 3 logits"))
    L.append(S("tcn (release det2)", "RELEASE member", lambda: M.TCN(), lambda: [x3()], "models_release/det2_tcn.pt",
               flow="6x [Conv(5x3, dil_t=1,2,4,8,16,32)+BN+ReLU], no pooling, receptive field along t = 1+4*(63)=253 rows (~8.4 h)", head="Conv1x1 -> 3 logits"))
    L.append(S("unet_cnsym (ens_cnsym member)", "REJECTED (unet_cnsym val 0.358; ens_cnsym val 0.384 < 0.387)", U(16), lambda: [x3()],
               "artifacts/stage1/unet_cnsym.pt", skips=UNET_SKIPS, flow=UNET_FLOW + "; trained on heat+cool symmetric synthetic"))
    L.append(S("unet_dom (ens_dom)", "REJECTED (val 0.390/test 0.406, nested check failed; noise from real PSD)", U(16), lambda: [x3()],
               "models_release_dom/unet_dom.pt", skips=UNET_SKIPS, flow=UNET_FLOW))
    L.append(S("distill student", "see docs/S3_DISTILL.md (student DTSUNet w16 from ensemble teacher)", U(16), lambda: [x3()],
               "artifacts/ckpt/s3_distill/best.pt", skips=UNET_SKIPS, flow=UNET_FLOW + "; KL+BCE to ens_u2ft_tcn_lgbm soft map"))
    L.append(S("MAE masked reconstructor (ens_mean4 member)", "CANDIDATE (flag mae=on, not default): ens_mean4 val 0.3995/test 0.4336/v10 0.4299 vs winner 0.3867/0.4053/0.4039; same-fold nested +0.0068 (k5) +0.0100 (k2). NOTE: checkpoint = INIT weights: best_epoch=-1 means no epoch beat the epoch-0 (pre-training) proxy, so the saved state is the initialisation; head is zero-initialised (hat=0), so the MAE score reduces exactly to CDF(|dT_space|) and ens_mean4 = (4*winner + CDF|dT_space|)/5, a fixed post-filter (artifacts/eval/s3_mae_why.json, docs/S3_MAE.md); trained seeds unstable (-0.031..+0.014); stays flag mae=on, not default",
               lambda: mae_anom.MaskedReconstructor(16),
               lambda: [torch.randn(1, 2, H, W), torch.ones(1, 1, H, W), torch.zeros(1, 1, H, W)], "artifacts/eval/s3_mae_recon.pt",
               skips=["e1 + up1(d2) -> d1 (additive)", "e2 + up2(e3) -> d2 (additive)"],
               flow="in = cat[ch*vis, vis, mask] (4ch) -> e1(5x3,3x3) -> e2(s2) -> e3(s2) -> up2 +e2 -> d2 -> up1 +e1 -> d1 -> ctx(dil2,dil4) -> head 1x1 -> 2ch recon * valid",
               input_desc="ch (2: dT_time,dT_space) masked by spatial bands; valid; mask (1=hidden). Anomaly score = |recon residual| on hidden cells, 4 mask phases",
               head="Conv1x1 -> 2 (reconstructed dT_time, dT_space)",
               note="ens_mean4 = (4 x ens_u2ft_tcn_lgbm score + MAE residual score)/5, fixed a priori; threshold FAR 1e-3 on v8 val background -> see S3_MAE.md"))
    L.append(S("GEN2 U-Net (physics generator finetune)", "RELEASE member of ens_gen2 (=mean(gen2, ens_u2ft_tcn_lgbm); val 0.4106 / test 0.4579 / v10 0.4576 vs 0.3867/0.4053/0.4039; held-out41 +0.018 (2 seeds); nested k5 +0.031 on all 45 v8 val seq (incl. 4 checkpoint-selection proxy files) vs +0.051 on the 41 held-out seq (clean, used for the decision); thr 0.8580; see RELEASE_DECISION.md)", U(16), lambda: [x3()],
               "models_release_gen2/s3_gen2_unet.pt", skips=UNET_SKIPS, flow=UNET_FLOW + "; finetuned from unet_v810ft on synth v8+v10 train with on-the-fly dts/gen2.py: narrow static hot spots 2-8 m +3..+24C, broad slow profile shifts, HVAC cycles 20-90 min, instrument -20C dips as negative class; BN frozen",
               head="Conv1x1 -> 3 logits"))
    L.append(S("GEN2 U-Net seed 2 (replication)", "REPLICATION, not in release (mean(seed2,winner) held-out41 +0.018, test/v10/v11 +0.049 CI excl 0)", U(16), lambda: [x3()],
               "artifacts/eval/s3_gen2_unet_s2.pt", skips=UNET_SKIPS, flow=UNET_FLOW + "; identical GEN2 recipe, seed 20261018, proxy on v10 val",
               head="Conv1x1 -> 3 logits"))
    L.append(S("GEN2-W24 U-Net (gen2 finetune of unet_w24)", "REJECTED (v10 proxy: init 0.366, finetuned 0.289-0.292; best=step0)", U(24), lambda: [x3()],
               "artifacts/eval/s3_gen2w24_unet.pt", skips=UNET_SKIPS, flow=UNET_FLOW + "; same dts/gen2.py physics-generator finetune as GEN2, from unet_w24, BN frozen, proxy on v10 val",
               head="Conv1x1 -> 3 logits"))
    L.append(S("GEN2-TCN (gen2 finetune of release tcn)", "REJECTED (v10 proxy: init 0.368, finetuned 0.295; best_step=0)", lambda: M.TCN(), lambda: [x3()],
               "artifacts/eval/s3_gen2tcn_tcn.pt", flow="release TCN architecture; finetuned with dts/gen2.py on synth v8+v10 train, proxy on v10 val",
               head="as release tcn"))
    L.append(S("GEN3 U-Net (gen3 physics generator finetune)", "REJECTED (hold41 mean_w +0.0002 CI covers0; nested_k5 -0.0018; gen3 alone val41 0.142; see S3_GEN3.md)", U(16), lambda: [x3()],
               "artifacts/eval/s3_gen3_unet.pt", skips=UNET_SKIPS, flow=UNET_FLOW + "; finetuned from s3_gen2_unet.pt on synth v8+v10 train with dts/gen3.py (GEN2 phenomena + cold_static, weak<=2C hot/cold, splice_step artifact); BN frozen; seed 20261019",
               head="Conv1x1 -> 3 logits"))
    def fuse():
        mod = importlib.import_module("scripts.s3_fusenet_net"); return mod.FuseNet()
    L.append(S("FuseNet (learned score fusion)", "REJECTED (val 0.351 vs 0.387; nested k5 -0.052)", fuse, lambda: [torch.randn(1, 4, H, W)],
               "artifacts/eval/s3_fusenet.pt", flow="in = cat[s_winner, CDF|dT_space|, |dT_time|/12, valid] -> Conv3x3 16 ReLU -> Conv3x3 16 ReLU -> Conv3x3 8 ReLU -> Conv3x3 1 (logit)",
               input_desc="4ch per-pixel fusion inputs", head="1 logit"))
    L.append(S("axial-attention U-Net", "REJECTED (docs/S3_AXIAL.md)", lambda: axial.AxialUNet(w=16), lambda: [x3()],
               "artifacts/eval/s3_axial_model.pt", skips=UNET_SKIPS, flow="DTSUNet w16 with AxialBottleneck (time attention + x attention, 4 heads, MLP x2) at e4"))
    L.append(S("ConvGRU U-Net", "REJECTED (docs/S3_CONVGRU.md)", lambda: convgru.ConvGRUNet(w=16), lambda: [x3()],
               "artifacts/eval/s3_convgru_unet.pt", skips=UNET_SKIPS, flow="DTSUNet w16 + ConvGRU over time (kernel 1x3 along x) after TimeGate in bottleneck"))
    def mtl():
        mod = importlib.import_module("scripts.s3_mtl_train"); return mod.MTLUNet(M.DTSUNet(w=16))
    L.append(S("multi-task U-Net (seg + amplitude mu/logvar)", "REJECTED (docs/S3_MTL.md)", mtl, lambda: [x3()], "artifacts/eval/s3_mtl_unet.pt",
               skips=UNET_SKIPS + ["d1 -> amp head 1x1 (2: mu, log sigma^2)"], flow=UNET_FLOW, kwargs={"aux": True}))
    def bnd():
        mod = importlib.import_module("scripts.s3_boundary"); return mod.BoundaryUNet(16)
    L.append(S("boundary U-Net (deep supervision)", "REJECTED (docs/S3_BOUNDARY.md)", bnd, lambda: [x3()], "artifacts/eval/s3_boundary_unet.pt",
               skips=UNET_SKIPS + ["d2 -> side2 1x1 (3)", "d3 -> side3 1x1 (3)"], flow=UNET_FLOW, kwargs={"deep": True}))
    L.append(S("CenterNet head (on frozen unet_v810ft e2+e3)", "REJECTED (ens val 0.349 vs 0.387)",
               lambda: centernet.CenterNetHead(cin=96, mid=32), lambda: [torch.randn(1, 96, H // 2, W // 2)], "artifacts/ckpt/centernet2/last.pt", key="head",
               flow="feat = cat[e2 (2w=32ch, /2), up_nearest(e3) (4w=64ch)] = 96ch at stride 2 -> 1x1 -> dilated 3x3 stack -> heads hm(2 cls hot/cold), wh(2), off(2)",
               input_desc="96-ch feature map of frozen unet_v810ft encoder at stride 2 (T/2 x X/2)", head="hm sigmoid 2ch, wh 2ch (dt,dx), off 2ch"))
    L.append(S("contour model (frozen encoder + query head)", "REJECTED (ens val 0.3855 vs 0.3867)", lambda: contour.ContourNet(), lambda: [x3()],
               "artifacts/ckpt/s3_contour/last.pt", key="model",
               flow="FrozenEncoder(e1,e2 of unet_v810ft, 32ch) -> ContourHead: proj 1x1 64 -> 16 learned queries cross-attn (q,k,v Linear 64) -> FF -> box(4), polygon offsets (8 verts x2), objectness, hot/cold",
               head="per query: box 4, offsets 16, obj 1, hot 1; rasterized to map"))
    L.append(S("membank (PatchCore-style memory bank)", "REJECTED (val 0.053, blend 0.381)", lambda: membank.MemBankNet(), lambda: [x3()],
               "artifacts/ckpt/s3_membank/last.pt", key="model", flow="FrozenEncoder(e1..e2 w16) -> adapter 1x1 32->16 -> agg conv 9x3 -> kNN (k=3) vs coreset 1024 -> ScoreHead MLP 1x1"))
    L.append(S("flow prior (rectified flow velocity U-Net)", "REJECTED (ens val 0.382 vs 0.387)", lambda: flowprior.VelocityNet(w=16),
               lambda: [torch.randn(1, 1, H, W), torch.ones(1, 1, H, W), torch.full((1,), 0.5)], "artifacts/eval/s3_flow_prior.pt",
               skips=UNET_SKIPS, flow="in = cat[x_t (1ch), valid, t-map] (3ch) -> DTSUNet-shaped body w16 -> head 1x1 -> velocity 1ch * valid",
               input_desc="x_t = t*x1 + (1-t)*z on dT (1ch); valid; scalar t broadcast", head="Conv1x1 -> 1 velocity"))
    sn = static_net
    L.append(S("STATIC branch StaticNet6 (1D over x)", "OFF by default flag static=on (synthetic 62/87 hits, real 0/16)",
               lambda: sn.StaticNet6(cin=5, width=16, max_rows=256), lambda: [torch.randn(1, 5, 1024)], "artifacts/static/static_net6.pt", key="state",
               flow="per-column temperature features (5) over long window -> DilatedReference (dil 1..73, smooth background) -> cat[x, x-ref] -> in_proj k5 -> ResBlock1d dil 1,2,4,8,16,32,64,73 -> head 1x1 -> logit per x",
               input_desc="(5, X) per-column stats of raw T over up to 256 rows (30-min / 2-h windows), feat_scale 10 C", head="Conv1d 1x1 -> 1 logit per column"))
    L.append(S("STATIC gate", "OFF (part of static=on)", lambda: sn.StaticGate(cin=sn.GATE_CIN, width=16), lambda: [torch.randn(1, sn.GATE_CIN, 1024)],
               "artifacts/static/static_gate.pt", key="state", input_desc=f"({sn.GATE_CIN}, X) gate features", head="sigmoid gate g; fused = max(s, g*p)"))
    L.append(S("dilated FCN", "REJECTED (experiments/dilated)", lambda: M.DilatedFCN(), lambda: [x3()], "artifacts/stage1/dilated.pt",
               flow="6x [Conv3x3 dil 1,2,4,8,16,1 + BN + ReLU] w32 -> head 1x1 3"))
    L.append(S("multiscale U-Net (unet_ms)", "REJECTED (val 0.275 alone; experiments/unet_ms)", lambda: M.DTSUNetMS(w=16), lambda: [x3()],
               "artifacts/stage1/unet_ms.pt", skips=UNET_SKIPS, flow="Integrate (fixed avg windows 7x15, 21x61 -> +4 ch, /sqrt(N)) -> DTSUNet cin=7 w16"))
    def realnet():
        mod = importlib.import_module("realnet.realnet"); net = mod.RealNet(M.DTSUNet(w=16))
        ck = torch.load(ROOT / "artifacts/realnet/model_bc.pt", map_location="cpu", weights_only=False)
        net.unet.load_state_dict(ck["unet"]); net.mask_head.load_state_dict(ck["mask"])
        class Wrap(nn.Module):
            def __init__(s): super().__init__(); s.m = net
            def forward(s, x): return s.m.unet(x), s.m.mask_pred(x)
        return Wrap()
    L.append(S("RealNet (U-Net + MaskHead)", "REJECTED (experiments/realnet)", realnet, lambda: [x3()], None,
               skips=UNET_SKIPS + ["d1 -> MaskHead (ring conv 3x3 dil (5,9))"], flow=UNET_FLOW + " + MaskHead on d1"))
    def unsup():
        mod = importlib.import_module("unsup.unsup")
        ck = torch.load(ROOT / "artifacts/unsup/model_synth_v8.pt", map_location="cpu", weights_only=False); cfg = ck["config"]
        return mod.BlindSpotForecaster(cin=cfg.get("cin", 3), width=cfg.get("width", 8), bottleneck=cfg.get("bottleneck", 4),
                                       blind_t=cfg.get("blind_t", 15), blind_x=cfg.get("blind_x", 41))
    L.append(S("unsup blind-spot forecaster", "REJECTED (experiments/unsup)", unsup, lambda: [x3()], "artifacts/unsup/model_synth_v8.pt", key="state_dict",
               flow="predicts each cell from context outside blind box 15x41; residual = anomaly"))
    def ssl():
        mod = importlib.import_module("ssl.ssl") if False else None
        spec = importlib.util.spec_from_file_location("exp_ssl", ROOT / "experiments/ssl/ssl.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        ck = torch.load(ROOT / "artifacts/stage1/ssl_ae.pt", map_location="cpu", weights_only=False); cfg = ck.get("config", {})
        kw = {k: cfg[k] for k in ("width", "k", "dilations", "side_k", "cin") if k in cfg}
        return m.BlindSpotAE(**kw)
    L.append(S("SSL blind-spot autoencoder", "REJECTED (experiments/ssl, TZ_AE)", ssl, lambda: [x3()], "artifacts/stage1/ssl_ae.pt", key="state_dict",
               flow="directional causal branches (DirConv, 4 directions) excluding center cell -> fuse -> reconstruct"))
    L.append(S("EventNet (stage-2 classifier)", "RELEASE stage 2 (CNN+MLP, blended 0.5 with LightGBM)",
               lambda: M.EventNet(7, _nscal()), lambda: [torch.randn(1, 3, 32, 64), torch.randn(1, _nscal())], "models_release/eventnet.pt", key="state",
               flow="crop 32x64x3 -> Conv3x3 24 BN ReLU MP2 -> Conv 48 MP2 -> Conv 96 -> AdaptiveAvg 2x2 -> 384 ; scalars -> Linear 32 ; cat -> Dropout .2 -> Linear 64 -> Linear 7",
               input_desc="event crop resized to 32(t) x 64(x), 3 channels; + scalar features", head="7 classes: " + "Local_Heating, Extended_Heating, Contact_Trace, Local_Cooling, Extended_Cooling, HVAC_Diurnal, Sensor_Artifact"))
    def emb():
        mod = importlib.import_module("scripts.s3_st2_embed"); return mod.EmbedFusion(_nscal(), 0)
    L.append(S("EmbedFusion (stage-2 variant)", "REJECTED (docs/S3_ST2.md)", emb, lambda: [torch.randn(1, 3, 32, 64), torch.randn(1, _nscal())]))
    def catcc():
        from dts import semi_contrastive as sc
        net = sc.CATCC(width=64)
        class Wrap(nn.Module):
            def __init__(s): super().__init__(); s.m = net
            def forward(s, t, x): return s.m.embed(t, x)
        return Wrap()
    L.append(S("CA-TCC encoder (semi-supervised, TZ E)", "REJECTED/disabled (5% labels: 0.67 vs LGBM 0.84)", catcc,
               lambda: [torch.randn(4, 3, 64), torch.randn(4, 3, 64)],
               flow="temporal tower Conv1d 32-48-64 (MP2 x3) & spatial tower; cross-attention (q from time, k/v from x); transformer over time tokens; embed = cat[time ctx, x mean] (128)",
               input_desc="event window: temporal series (3, 64) and spatial profile (3, 64)", head="embedding 128 -> Head MLP 64 -> 7 classes"))
    return L

def _nscal():
    ck = torch.load(ROOT / "models_release/eventnet.pt", map_location="cpu", weights_only=False)
    return int(len(ck["mu"]))

def main():
    out = []
    for s in specs():
        rec = {k: v for k, v in s.items() if k not in ("build", "inputs", "key", "prefix", "kwargs")}
        try:
            net = s["build"]()
            rec["load"] = load(net, s["ckpt"], s["key"], s["prefix"]) if s["ckpt"] else ("loaded artifacts/realnet/model_bc.pt (unet+mask, strict)" if "RealNet" in s["name"] else "no checkpoint (built from code defaults)")
            rec["params_total"] = sum(p.numel() for p in net.parameters())
            rec["params_trainable"] = sum(p.numel() for p in net.parameters() if p.requires_grad)
            inp = s["inputs"]()
            rec["input_shapes"] = [list(t.shape) for t in inp]
            rec["layers"], rec["output_shape"] = trace(net, inp, s["kwargs"])
            rec["module_tree"] = str(net)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"; traceback.print_exc()
        print(rec["name"], rec.get("params_total"), rec.get("output_shape"), rec.get("error", ""), rec.get("load"), flush=True)
        out.append(rec)
    (ROOT / "docs/ARCH_SPECS_raw.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))

if __name__ == "__main__":
    main()
