import pandas as pd, numpy as np
d = pd.read_csv("artifacts/eval/cards2_review.csv"); d["shape"] = d["shape"].fillna(""); d["notes"] = d.notes.fillna("")
ms = ["winner", "ens_dom", "ens_cnsym", "pdrs_hybrid", "baseline"]
def verdict(r):
    hi, lo, s, ext, cr, nz = r.dT_hi, r.dT_lo, r.snr, r.extent_m, r.com_range, r.noise
    if lo < -12 and abs(lo) > 2 * hi:
        return "unclear", "низк.", f"провал {lo:.0f} °C — для помещения нефизично как воздух; вероятнее артефакт прибора (сварка/коннектор/изгиб, ступенька затухания) или участок вне помещения"
    if cr > 3 * max(nz, .3) * 2 and r.x1 - r.x0 < 25:
        pol = "нагрев" if hi >= -lo else "охлаждение"
        return "yes", "выс.", f"узкая зона {r.x1-r.x0:.0f} м (≈{(r.x1-r.x0):.0f} разрешающих длин), средняя T внутри меняется на {cr:.1f} °C при шуме {nz:.2f} °C — локальный источник ({pol}: оборудование/нагреватель/солнце)"
    if s >= 6 and ext >= 2:
        pol = "нагрев" if hi >= -lo else "охлаждение"
        big = "сильный (>10 °C) — оборудование/нагреватель" if max(hi, -lo) > 10 else "умеренный, типично для оборудования, солнечного пятна или струи HVAC"
        return "yes", "выс." if s >= 9 else "средн.", f"{pol} {max(hi,-lo):.1f} °C, SNR {s:.0f}, протяжённость {ext:.0f} м ≫ 1 м разрешения — {big}"
    if s >= 4.5 and ext >= 1:
        return "unclear", "средн.", f"отклонение {max(hi,-lo):.1f} °C (SNR {s:.1f}) на {ext:.0f} м — слабое событие на границе обнаружимости (типичный сквозняк/HVAC-цикл или остаток суточного хода)"
    return "no", "средн." if s < 3.6 else "низк.", f"пик |ΔT| {max(hi,-lo):.1f} °C ≈ {s:.1f}σ при шуме {nz:.2f} °C — для окна такого размера это ожидаемый максимум чистого шума; размах средней T {cr:.1f} °C"
V = d.apply(verdict, axis=1, result_type="expand"); d[["mine", "conf", "why"]] = V
L = ["# Разбор карточек оператора, партия 2", "",
     "Только оценка. Карточки ни для обучения, ни для калибровки, ни для подбора порогов не использовались. Ответы оператора — быстрое мнение по графику, не эталон.", "",
     f"Новых карточек: {len(d)} (записи 1–6); дублей со старыми 20 карточками (68e4…, 185b…) по (record, meters, time) и по id: {int(d.dup.sum())}. "
     f"Ответы: yes {sum(d.user=='yes')}, no {sum(d.user=='no')}, unclear {sum(d.user=='unclear')}.", "",
     "Модели: зоны-тревоги готовых прогонов по полным файлам (`real_zones.csv` — лидер ens_u2ft_tcn_lgbm, `real_dom.csv` — ens_dom, `release_tmp_ens_cnsym/zones.csv` — ens_cnsym, `real_pdrs_hybrid.csv` — PDRS-гибрид на лидере), классический baseline (median_x3+spatial_z) пересчитан. «Нашла» = есть хотя бы одна тревожная зона, пересекающая рамку карточки. Порог прежний (FAR 1e-3 на синтетической валидации).", "",
     "Физические признаки (скрипт `scripts/cards_review2.py`): ΔT = T минус статический профиль минус синфазная составляющая; шум σ — по разностям соседних кадров; SNR = 99.5-перцентиль |ΔT| / σ (для чистого шума в окне ожидается ≈3); протяжённость — длина участка, где |ΔT| > max(4σ, 0.5 °C); размах — изменение средней T внутри рамки во времени (ловит статичные горячие точки, которые вычитание профиля съедает).", "",
     "## Согласие с оператором", "", "| модель | yes найдено | no ложных | unclear отмечено | совпадение с оператором на yes/no |", "|---|---|---|---|---|"]
yn = d[d.user != "unclear"]
for m in ms + ["mine"]:
    f = (d[m] == "yes") if m == "mine" else (d[m] == 1)
    fy = f[yn.index]; acc = ((fy & (yn.user == "yes")) | (~fy & (yn.user == "no"))).mean()
    L.append(f"| {m if m!='mine' else 'мой разбор (yes)'} | {int(f[d.user=='yes'].sum())}/{sum(d.user=='yes')} | {int(f[d.user=='no'].sum())}/{sum(d.user=='no')} | {int(f[d.user=='unclear'].sum())}/{sum(d.user=='unclear')} | {acc:.2f} |")
L += ["", "### По типам (среди yes, тип может быть составным)", "", "| тип | n | " + " | ".join(ms) + " | мой yes |", "|---|---|" + "---|" * (len(ms) + 1)]
y = d[d.user == "yes"]
for s in ["Local_Heating", "Local_Cooling", "Periodic_Thermal", "Broad_Shift"]:
    k = y[y["shape"].str.contains(s)]
    L.append(f"| {s} | {len(k)} | " + " | ".join(str(int(k[m].sum())) for m in ms) + f" | {int((k.mine=='yes').sum())} |")
L += ["", "### Мой разбор против оператора", ""]
ct = pd.crosstab(d.user, d.mine); L += ["| оператор \\ я | " + " | ".join(ct.columns) + " |", "|---|" + "---|" * len(ct.columns)]
for i, r in ct.iterrows(): L.append(f"| {i} | " + " | ".join(str(v) for v in r) + " |")
L += ["", "## Выводы", "",
"- Нейросетевые модели консервативны: при FAR 1e-3 они почти не отмечают ответы «no» (3–4 из 19), но находят лишь 14–19 из 91 «yes». Baseline отмечает 60/91, но и почти все unclear-окна; на шумной записи 5 (σ≈0.9 °C, вдвое выше остальных) он срабатывает почти везде.",
"- Local_Cooling (17 карточек): нейросети не нашли ни одной. Большая часть таких карточек — это −1…−2.5 °C при σ 0.3–0.9 °C, то есть SNR около 3, на уровне шума. Это та же физическая граница слабых событий, что и в синтетике.",
"- Пропуски на ярких событиях. Узкие (3–5 м) долгие окна записи 1 (x≈417, 425, 491, 803, 812, 880–894) и записи 5 (x≈387, 410, 793–999, 1146–1381; +6…+13 °C): статичная или медленная горячая точка уходит в «статический профиль», а детектор настроен на появление и исчезновение. Это главный реальный изъян, который показали карточки.",
"- Провалы −20…−25 °C в записи 1 (x≈75–135, 375–460) нефизичны для воздуха в помещении. Я считаю их артефактом прибора или участком трассы вне помещения, а не Periodic_Thermal.",
"- Оператор и модели по-разному понимают рамку: оператор отвечает про рамку целиком («слева Periodic_Thermal»), а модели — про пиксели внутри. Поэтому часть «unclear» на деле означает «yes по краю».", "",
"## Все карточки", "", "Рисунки: `docs/figures/cards2/<id>.png` (пунктир — рамка карточки; лайм — лидер, оранжевый — ens_dom, пурпурный — ens_cnsym, голубой — PDRS, серый пунктир — baseline).", "",
"| id | оператор | лидер/dom/cnsym/pdrs/base | σ, °C | ΔT+ / ΔT− | SNR | моё мнение | увер. | почему |", "|---|---|---|---|---|---|---|---|---|"]
for _, r in d.iterrows():
    u = r.user + (f" ({r['shape']})" if r["shape"] else "")
    dis = " **≠**" if (r.user in ("yes", "no") and r.mine in ("yes", "no") and r.user != r.mine) else ""
    L.append(f"| [{r.id}](figures/cards2/{r.id}.png) | {u} | {r.winner}/{r.ens_dom}/{r.ens_cnsym}/{r.pdrs_hybrid}/{r.baseline} | {r.noise} | {r.dT_hi}/{r.dT_lo} | {r.snr} | {r.mine}{dis} | {r.conf} | {r.why} |")
open("docs/CARDS_REVIEW_2.md", "w").write("\n".join(L) + "\n")
d.to_csv("artifacts/eval/cards2_review.csv", index=False)
dd = d[(d.user.isin(["yes","no"])) & (d.mine.isin(["yes","no"])) & (d.user != d.mine)]
print("\n".join(L[:40])); print(len(dd)); print(dd[["id","user","shape","mine","why"]].to_csv(sep="|",index=False))
