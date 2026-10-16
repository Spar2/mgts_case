"""PatchCore / DFM tiles through Anomalib.

The memory is tiles whose spatial-Z tail stays under the gate. A hotspot that
fills less than half a tile leaves the median near the background, so the
median is not the split. Suspect tiles are scored only against that memory.
ImportError is raised when anomalib is absent. An empty memory raises
RuntimeError instead of scoring a tile against itself.
"""

from __future__ import annotations

import numpy as np

from fiber.anomaly.detect import spatial_z
from fiber.models.ssl_common import live_mask, torch_device


def _tiles(field: np.ndarray, mask: np.ndarray, tile: int) -> list[tuple[int, int, np.ndarray]]:
    found = []
    time_count, distance = field.shape
    if time_count < tile or distance < tile:
        return found
    normal = np.nan_to_num(field, nan=0.0).astype(np.float32)
    for t0 in range(0, time_count - tile + 1, tile):
        for x0 in range(0, distance - tile + 1, tile):
            patch = normal[t0:t0 + tile, x0:x0 + tile]
            if float(mask[t0:t0 + tile, x0:x0 + tile].mean()) < 0.8:
                continue
            found.append((t0, x0, patch))
    return found


# Share of cells with |Z| >= normal_z that still counts as background.
# local_scene puts all 147 hot cells in two tiles whose median |Z| is
# 0.36-0.52, and a 12-cell cold piece is 4.7% of its tile. Both are
# above this cap. Quiet tiles on that scene stay under it.
_MAX_QUIET_FRACTION = 0.02


def _quiet_tile(window: np.ndarray, normal_z: float) -> bool:
    finite = np.abs(np.asarray(window, dtype=np.float64))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return False
    tail = float(np.quantile(finite, 0.95))
    fraction = float(np.mean(finite >= normal_z))
    return tail <= normal_z and fraction <= _MAX_QUIET_FRACTION


def _rgb(patch: np.ndarray) -> np.ndarray:
    centered = patch - float(np.median(patch))
    scale = float(np.std(centered))
    scale = scale if scale > 1e-3 else 1.0
    gray = np.clip(centered / (3.0 * scale), -1.0, 1.0)
    gray = ((gray + 1.0) * 127.5).astype(np.float32)
    return np.stack([gray, gray, gray], axis=0)


def image_anomaly_score(
    field: np.ndarray,
    valid: np.ndarray,
    x_m: np.ndarray,
    *,
    tile: int = 16,
    device: str | None = None,
    normal_z: float = 2.0,
) -> np.ndarray:
    try:
        import torch
        from anomalib.models import Patchcore
    except ImportError as exc:
        raise ImportError("anomalib is not installed") from exc

    values = np.asarray(field, dtype=np.float32)
    mask = live_mask(field, valid)
    score = np.zeros(values.shape, dtype=np.float32)
    chosen = torch_device(device)
    z = spatial_z(np.where(mask, values, 0.0), mask, x_m)[0]
    patches = _tiles(values, mask, tile)
    if len(patches) < 2:
        return score
    normal = []
    rest = []
    for t0, x0, patch in patches:
        window = z[t0:t0 + tile, x0:x0 + tile]
        item = (t0, x0, _rgb(patch))
        if _quiet_tile(window, normal_z):
            normal.append(item)
        else:
            rest.append(item)
    if not rest:
        return score
    if not normal:
        raise RuntimeError("PatchCore memory is empty: no tile has a low spatial-Z tail")
    try:
        model = Patchcore()
        model.to(chosen)
        model.eval()
    except Exception as exc:
        raise RuntimeError(f"Anomalib Patchcore could not be built: {exc}") from exc
    # Patchcore's public fit path needs a datamodule. Use the backbone embedding
    # when the model exposes one, and otherwise refuse to invent a score.
    embedder = getattr(model, "model", None)
    if embedder is None or not hasattr(embedder, "feature_extractor"):
        raise RuntimeError("Anomalib Patchcore has no feature extractor in this version")
    extractor = embedder.feature_extractor
    extractor.to(chosen)
    extractor.eval()

    def _embed(items: list[tuple[int, int, np.ndarray]]) -> torch.Tensor:
        batch = torch.from_numpy(np.stack([item[2] for item in items])).to(chosen)
        with torch.no_grad():
            features = extractor(batch)
        if isinstance(features, dict):
            layers = [value.flatten(1) for value in features.values()]
            return torch.cat(layers, dim=1)
        return features.flatten(1)

    memory = _embed(normal)
    query_items = rest
    query = _embed(query_items)
    distance = torch.cdist(query, memory).min(dim=1).values.detach().cpu().numpy()
    for (t0, x0, _patch), value in zip(query_items, distance, strict=True):
        residual = values[t0:t0 + tile, x0:x0 + tile] - float(np.median(values[t0:t0 + tile, x0:x0 + tile]))
        signed = np.sign(residual).astype(np.float32)
        signed[signed == 0] = 1.0
        score[t0:t0 + tile, x0:x0 + tile] = signed * np.float32(value)
    score[~mask] = 0.0
    finite = np.abs(score[mask])
    if finite.size:
        mad = 1.4826 * float(np.median(np.abs(finite - np.median(finite))))
        mad = mad if mad > 1e-6 else 1.0
        score /= np.float32(mad)
        score[~mask] = 0.0
    return score
