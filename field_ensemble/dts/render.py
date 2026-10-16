import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

COL = {"Local_Heating": "red", "Extended_Heating": "orange", "Contact_Trace": "magenta", "Local_Cooling": "blue",
       "Extended_Cooling": "cyan", "HVAC_Diurnal": "green", "Sensor_Artifact": "gray"}


def render(o, path, title=""):
    p = o["grid"]; dT = np.where(o["valid"], o["dT"], np.nan)
    v = np.nanpercentile(np.abs(dT), 99.5) if np.isfinite(dT).any() else 1
    ext = [p["x"][0], p["x"][-1], (p["t"][-1] - p["t"][0]) / 3600, 0]
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    ax[0].imshow(dT, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, extent=ext, interpolation="nearest")
    ax[1].imshow(o["score"], aspect="auto", cmap="magma", vmin=0, vmax=1, extent=ext, interpolation="nearest")
    ax[0].set_title(f"{title}: ΔT, °C + зоны"); ax[1].set_title("P(аномалия), U-Net")
    t0 = p["t"][0]
    for z in o["zones"]:
        ax[0].add_patch(Rectangle((z["start_m"], (z["t_start_s"] - t0) / 3600), max(z["end_m"] - z["start_m"], .5),
                                  (z["t_end_s"] - z["t_start_s"]) / 3600 + .03, fill=False, ec=COL[z["class"]], lw=1.2 if z["alarm"] else .6,
                                  ls="-" if z["alarm"] else ":"))
    for k, c in COL.items(): ax[0].plot([], [], color=c, label=k)
    ax[0].legend(fontsize=7, loc="lower right"); ax[0].set_ylabel("ч от начала"); ax[0].set_xlabel("x, м"); ax[1].set_xlabel("x, м")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()
