"""CPU reference pipeline with explicit coverage and shape hypotheses."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from fiber.anomaly import extract_zones, spatial_z
from fiber.anomaly.detect import detection_status, spatial_availability
from fiber.classify.rules import classify_zones
from fiber.denoise.dsp import median_x
from fiber.grid import validate_axis
from fiber.quality import quality_mask


@dataclass(frozen=True)
class PipelineConfig:
    threshold_z: float = 3.5
    inner_m: float = 8.0
    outer_m: float = 40.0
    bin_m: float = 2.0
    floor_c: float = .30
    max_gap_s: float = 300.0
    min_width_m: float = 3.0
    min_profiles: int = 6
    min_cells: int = 24

    def __post_init__(self):
        values = asdict(self)
        if any(not np.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError('pipeline parameters must be positive and finite')
        if self.inner_m >= self.outer_m:
            raise ValueError('inner_m must be smaller than outer_m')
        if not isinstance(self.min_profiles, int) or not isinstance(self.min_cells, int):
            raise ValueError('minimum counts must be integers')

    def manifest(self):
        body = {'schema': 'dts_reference_v2', 'clean': 'median_x3', 'detector': 'spatial_z_halo',
                'analysis_mode': 'offline_batch',
                'config': asdict(self), 'class_source': 'shape_rules_hypothesis',
                'expert_confirmed': False, 'trained_classifier_operational': False}
        body['config_sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        return body


def analyze_field(raw, x_m, time_s, *, valid=None, time_iso=None, config=None):
    config = config or PipelineConfig()
    x = validate_axis(x_m, 'x_m')
    times = validate_axis(time_s, 'time_s')
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape != (len(times),len(x)) or len(x) < 2:
        raise ValueError('raw must have shape [len(time_s), len(x_m)], at least two coordinates')
    physical_valid = quality_mask(raw)['valid']
    if valid is not None:
        if np.shape(valid) != raw.shape:
            raise ValueError('valid shape must agree with raw')
        physical_valid &= np.asarray(valid, dtype=bool)
    iso = [str(float(t)) for t in times] if time_iso is None else list(time_iso)
    if len(iso) != len(times):
        raise ValueError('time_iso must agree with time_s')
    clean = median_x(raw, physical_valid)
    z, background, scale = spatial_z(clean, physical_valid, x, inner_m=config.inner_m,
        outer_m=config.outer_m, bin_m=config.bin_m, floor_c=config.floor_c)
    available = spatial_availability(clean, physical_valid, background)
    zones = extract_zones(z, available, x, iso, times, thresh=config.threshold_z,
        min_width_m=config.min_width_m, min_profiles=config.min_profiles,
        min_cells=config.min_cells, contrast_c=clean-background, max_gap_s=config.max_gap_s)
    measured = x[physical_valid.any(axis=0)]
    if len(measured)>1:
        labels = classify_zones(zones, float(measured[-1]-measured[0]))
    else:
        labels = ['Unknown']*len(zones)
    for zone, label in zip(zones, labels, strict=True):
        zone['shape_hypothesis'] = label
        zone['label_source'] = 'shape_rules'
        zone['expert_confirmed'] = False
    return {
        **detection_status(len(zones), physical_valid, available), 'zones': zones,
        'manifest': config.manifest(), 'clean': clean, 'z': z,
        'background': background, 'scale': scale, 'score_available': available,
    }
