"""Descriptive changes within a saved profile window, never quality scores."""

import html


def profile_changes(profile):
    processed = profile.get("processed_temperature_c")
    if processed is None:
        return None
    raw, axes = profile["temperature_c"], profile["x_m"]
    if len(processed) != len(raw) or len(axes) != len(raw):
        raise ValueError("Profile shape differs")
    pairs = []
    missing_raw = missing_processed = 0
    for index, (x, before, after) in enumerate(zip(axes, raw, processed)):
        if before is None:
            missing_raw += 1
        elif after is None:
            missing_processed += 1
        else:
            pairs.append(
                {
                    "index": index,
                    "x_m": x,
                    "original_c": before,
                    "processed_c": after,
                    "delta_c": after - before,
                }
            )
    absolute = [abs(point["delta_c"]) for point in pairs]
    return {
        "schema_version": "dts-profile-changes-v1",
        "threshold_c": 0.01,
        "total_points": len(raw),
        "paired_points": len(pairs),
        "missing_original": missing_raw,
        "missing_processed": missing_processed,
        "changed_points": sum(value > 0.01 for value in absolute),
        "max_absolute_change_c": max(absolute) if absolute else None,
        "mean_absolute_change_c": sum(absolute) / len(absolute) if absolute else None,
        "top_points": sorted(
            pairs, key=lambda point: (-abs(point["delta_c"]), point["index"])
        )[:5],
    }


def changes_html(changes):
    if changes is None:
        return ""
    notice = "Порог 0,01 °C — правило отображения, не точность датчика. Изменения сигнала не подтверждают качество очистки."
    if not changes["paired_points"]:
        content = "<p>Сопоставимых точек нет.</p>"
    else:
        content = (
            f"<p>Сопоставимых точек: {changes['paired_points']} из {changes['total_points']}. "
            f"Изменились больше 0,01 °C: {changes['changed_points']}. "
            f"Максимальное абсолютное изменение: {changes['max_absolute_change_c']} °C; "
            f"среднее абсолютное изменение: {changes['mean_absolute_change_c']} °C.</p>"
            '<div style="overflow:auto"><table><thead><tr><th>Расстояние, м</th><th>Исходник, °C</th><th>Обработка, °C</th><th>Разность, °C</th></tr></thead><tbody>'
            + "".join(
                "<tr>"
                + "".join(
                    f"<td>{html.escape(str(point[key]))}</td>"
                    for key in ("x_m", "original_c", "processed_c", "delta_c")
                )
                + "</tr>"
                for point in changes["top_points"]
            )
            + "</tbody></table></div>"
        )
    return (
        "<section><h2>Что изменила обработка</h2>"
        + content
        + f"<p>Пропуски исходника: {changes['missing_original']}; пропуски обработки среди валидных исходных точек: {changes['missing_processed']}.</p><p>{notice}</p></section>"
    )
