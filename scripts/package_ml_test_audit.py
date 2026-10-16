"""Bundle the completed audit, small retrained weights and their provenance."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
summary = json.loads((ROOT / "docs/evidence/ml_test_reproduction.json").read_text(encoding="utf-8"))
assert summary["all_six_complete"], "finish all six native runs before handoff"
assert all(row["patch_median_ablation"] is not None for row in summary["records"])
assert all(row["saved_outputs_verified"] for row in summary["records"])
assert all(row["weights_verification"] and row["weights_verification"]["full_saved_cnn_score_reproduced_exactly"]
           for row in summary["records"])
files = [(ROOT / "docs/ML_TEST_REVIEW.md", "docs/ML_TEST_REVIEW.md")]
files += [(path, path.relative_to(ROOT).as_posix()) for path in sorted((ROOT / "docs/evidence").glob("ml_test_*.json")) if path.name != "ml_test_archive.json"]
files += [(ROOT / "docs" / name, "docs/" + name) for name in ("REAL_DATA_WORK.md", "REAL_DATA_GOAL.md")]
files += [(path, path.relative_to(ROOT).as_posix()) for path in sorted((ROOT / "docs/images/real-data").glob("*.png"))]
files += [(path, path.relative_to(ROOT).as_posix()) for path in sorted((ROOT / "research/ml-test-review").rglob("*")) if path.is_file()]
files += [(ROOT / "research/real-data-v2/real_masked_ridge.json", "research/real-data-v2/real_masked_ridge.json")]
files += [(path, path.relative_to(ROOT).as_posix()) for path in sorted((ROOT / "docs/images/ml-test-review").glob("*.png"))]
files += [(path, path.relative_to(ROOT).as_posix()) for path in sorted((ROOT / "scripts").glob("*ml_test*.py"))]
for name in summary["completed_records"]:
    directory = ROOT / "artifacts/ml-test-audit/reproduced" / name
    files += [(directory / filename, f"reproduced/{name}/{filename}")
              for filename in ("receipt.json", "cnn_weights.pt", "cnn_ablation.json", "weights_verification.json")]
files.append((ROOT / "artifacts/ml-test-audit/external-tests.xml", "external-tests.xml"))
manifest = {name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
            for path, name in files}
readme = """# Захару: независимый аудит ml-test

Начните с docs/ML_TEST_REVIEW.md. Там вывод, сравнение подходов, графики,
разбор REPORT.md и конкретные ограничения. Числа и проверки сохранены
в docs/evidence/ml_test_*.json. MANIFEST.json проверяет файлы архива.

В reproduced/1..6/cnn_weights.pt лежат маленькие state_dict CNN нашего
повторного CPU-обучения на соответствующем реальном CSV, без синтетического
обучения. Это НЕ исходные веса Захара/Grok и НЕ модель, которую нужно сразу
подключать к production. Веса Isolation Forest здесь не сохранялись.
Для каждого файла receipt.json фиксирует source CSV SHA, commit, версии,
результаты и SHA весов. В общем evidence есть cnn_inference_contract с
центром/масштабом residual, окнами и порогом этого ретроспективного прогона.
Загружать с torch.load(..., map_location='cpu', weights_only=True), затем
load_state_dict в исходную ConvAutoencoder замороженного commit.

Исходный репозиторий: https://github.com/PiorlZ/ml-test
Commit: 33e12ac19b7b6318dc48b5f3e8c1239aaaf9244e
Архив не дублирует исходные CSV, большой REPORT.md, runtime и большие карты.
Они нужны отдельно для полного повторения. Скрипты предназначены для
рабочей копии haklolarp с native artifacts, а не для запуска прямо из архива;
расположение входов описано в обзоре. Разбор не изменял чужие исходники.

Шесть real-прогонов и штатные unit tests не являются независимой оценкой
precision/recall. Экспертной истины здесь нет; partial cores — development.
"""
destination = ROOT / "artifacts/zakhar-ml-test-audit.zip"
with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path, name in files:
        archive.write(path, name)
    archive.writestr("README.md", readme)
    archive.writestr("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
with zipfile.ZipFile(destination) as archive:
    assert archive.testzip() is None
    for name, value in manifest.items():
        assert hashlib.sha256(archive.read(name)).hexdigest() == value["sha256"]
receipt = {"archive": destination.name, "bytes": destination.stat().st_size,
           "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "verified_files": len(files),
           "native_records": 6, "original_author_weights": False,
           "retrained_cnn_weights_verified": 6, "expert_truth_available": False}
(ROOT / "docs/evidence/ml_test_archive.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8", newline="\n")
print(json.dumps(receipt))
