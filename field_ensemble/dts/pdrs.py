"""PDRS-style zone extraction on neural probability maps (TZ task B, neural variant).

Instead of |Z| of a classical detector, the field is the ensemble probability s(t,x) in [0,1]
(U-Net+TCN+pixel-LGBM). Seeds = local maxima with s >= seed; all fronts grow simultaneously
(marker watershed on -s, 4-connectivity, restricted to s >= floor and valid cells, time gaps
never bridged); neighbouring clusters merge if their saddle s > r_saddle * min(peaks);
zones with median s < region_median are dropped. Polarity from sign of h (hot>0).
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

FOUR = ndi.generate_binary_structure(2, 1)


def extract_zones_pdrs(s, valid, h=None, t=None, x=None, *, seed=0.86, floor=0.4,
                       r_saddle=0.6, region_median=0.6, min_cells=12, max_gap_s=300.0,
                       peak_size=(5, 9)):
    s = np.where(valid, np.asarray(s, np.float32), 0.0)
    mask = valid & (s >= floor)
    if t is not None and len(t) > 1:  # never bridge time gaps: cut mask rows after a gap
        gap = np.where(np.diff(t) > max_gap_s)[0]
        cut = np.zeros(len(t), bool); cut[gap] = True
        mask = mask & ~cut[:, None]
    mx = ndi.maximum_filter(s, size=peak_size)
    peaks = mask & (s >= seed) & (s == mx)
    markers, n = ndi.label(peaks, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros(s.shape, np.int32), []
    lab = watershed(-s, markers, mask=mask, connectivity=1)
    peak_val = ndi.maximum(s, lab, index=np.arange(n + 1)); peak_val[0] = 0
    # saddle merging: for each pair of 4-adjacent labels take max of min(s_a, s_b) across boundary
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    sad = {}
    for ax in (0, 1):
        a = lab.take(range(lab.shape[ax] - 1), axis=ax); b = lab.take(range(1, lab.shape[ax]), axis=ax)
        sa = s.take(range(s.shape[ax] - 1), axis=ax); sb = s.take(range(1, s.shape[ax]), axis=ax)
        m = (a > 0) & (b > 0) & (a != b)
        if not m.any():
            continue
        aa, bb, v = a[m], b[m], np.minimum(sa[m], sb[m])
        lo, hi = np.minimum(aa, bb), np.maximum(aa, bb)
        key = lo.astype(np.int64) * (n + 1) + hi
        order = np.argsort(key); key, v = key[order], v[order]
        uk, idx = np.unique(key, return_index=True)
        vm = np.maximum.reduceat(v, idx)
        for k, val in zip(uk, vm):
            sad[(int(k // (n + 1)), int(k % (n + 1)))] = float(val)
    for (a, b), val in sorted(sad.items(), key=lambda kv: -kv[1]):
        if val > r_saddle * min(peak_val[a], peak_val[b]):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n + 1)])
    lab = roots[lab]
    out = np.zeros_like(lab, dtype=np.int32); zones = []; k = 0
    objs = ndi.find_objects(lab)
    for r, sl in enumerate(objs, 1):
        if sl is None:
            continue
        m = lab[sl] == r
        vals = s[sl][m]
        if m.sum() < min_cells or np.median(vals) < region_median:
            continue
        k += 1; out[sl][m] = k
        pol = "hot"
        if h is not None:
            pol = "hot" if float(np.mean(np.asarray(h, np.float32)[sl][m])) >= 0 else "cold"
        z = dict(zone=k, polarity=pol, n_px=int(m.sum()), score=float(vals.max()),
                 t0=sl[0].start, t1=sl[0].stop, x0=sl[1].start, x1=sl[1].stop)
        if t is not None: z.update(t_start_s=float(t[sl[0].start]), t_end_s=float(t[sl[0].stop - 1]))
        if x is not None: z.update(x_start_m=float(x[sl[1].start]), x_end_m=float(x[sl[1].stop - 1]))
        zones.append(z)
    return out, zones


def extract_zones_cc(s, valid, h=None, *, thr=0.86, min_cells=12):
    """Reference: threshold + connected components (current pipeline)."""
    lab, n = ndi.label(valid & (s >= thr), structure=FOUR)
    out = np.zeros_like(lab); zones = []; k = 0
    for r, sl in enumerate(ndi.find_objects(lab), 1):
        m = lab[sl] == r
        if m.sum() < min_cells: continue
        k += 1; out[sl][m] = k
        pol = "hot" if h is None or float(np.mean(np.asarray(h, np.float32)[sl][m])) >= 0 else "cold"
        zones.append(dict(zone=k, polarity=pol, n_px=int(m.sum()), t0=sl[0].start, t1=sl[0].stop,
                          x0=sl[1].start, x1=sl[1].stop))
    return out, zones


def _paste_zone(out, zones, mask, template, t, x):
    """Append one zone with a fresh id. mask is a full-frame boolean."""
    if not np.any(mask):
        return
    k = len(zones) + 1
    out[mask] = k
    ts = np.flatnonzero(mask.any(1))
    xs = np.flatnonzero(mask.any(0))
    z = dict(template)
    z["zone"] = k
    z["n_px"] = int(mask.sum())
    z["t0"], z["t1"] = int(ts[0]), int(ts[-1]) + 1
    z["x0"], z["x1"] = int(xs[0]), int(xs[-1]) + 1
    if t is not None:
        z["t_start_s"] = float(t[z["t0"]])
        z["t_end_s"] = float(t[z["t1"] - 1])
    if x is not None:
        z["x_start_m"] = float(x[z["x0"]])
        z["x_end_m"] = float(x[z["x1"] - 1])
    zones.append(z)


def extract_zones_hybrid(s, valid, h=None, t=None, x=None, *, mode="seed",
                         thr=0.8608087301264272, seed=None, floor=0.7,
                         r_saddle=0.3, region_median=0.5, min_cells=12,
                         max_gap_s=300.0, peak_size=(5, 9), fallback=True):
    """Hybrid of threshold-CC and PDRS. Does not change extract_zones_pdrs.

    mode="seed": PDRS on the full valid mask with the given seed (default seed is `thr`,
    the far 1e-3 operating point). Growth still stops at `floor`.

    mode="refine": CC at `thr` is the support. PDRS (seed/floor = `seed` or `floor`)
    resegments inside that support. A CC component that receives no surviving PDRS
    subzone is copied back when fallback=True, so a weak blob above `thr` is not dropped.
    """
    if mode not in ("seed", "refine"):
        raise ValueError("mode must be 'seed' or 'refine'")
    sd = float(thr if seed is None else seed)
    kw = dict(seed=sd, floor=float(floor), r_saddle=float(r_saddle),
              region_median=float(region_median), min_cells=int(min_cells),
              max_gap_s=float(max_gap_s), peak_size=tuple(peak_size))
    if mode == "seed":
        return extract_zones_pdrs(s, valid, h, t, x, **kw)
    cc_lab, cc_zones = extract_zones_cc(s, valid, h, thr=float(thr), min_cells=int(min_cells))
    support = np.asarray(valid, bool) & (cc_lab > 0)
    p_lab, p_zones = extract_zones_pdrs(s, support, h, t, x, **kw)
    if not fallback or not cc_zones:
        return p_lab, p_zones
    out = np.array(p_lab, dtype=np.int32, copy=True)
    zones = [dict(z) for z in p_zones]
    for z in cc_zones:
        comp = cc_lab == z["zone"]
        if (p_lab[comp] > 0).any():
            continue
        _paste_zone(out, zones, comp, z, t, x)
    return out, zones
