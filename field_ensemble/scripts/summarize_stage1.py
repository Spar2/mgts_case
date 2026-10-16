import json, sys
r = json.load(open(sys.argv[1])); print('far_ref', round(r['far_ref'], 4), 'base_test_far', round(r.get('base_test_far', 0), 4))
print('threshold-free:', {k: {kk: round(vv, 3) for kk, vv in v.items() if vv is not None} for k, v in r.get('threshold_free', {}).items()})
rows = list(r['test'].items()) + [('MATCHED ' + k, v) for k, v in r.get('test_far_matched', {}).items()]
for k, v in rows:
    if 'far_1e-4' in k: continue
    w = v.get('weak_events') or {}
    print(f"{k:42s}", {kk: round(float(v[kk]), 3) for kk in ['far', 'precision', 'recall', 'f1', 'iou']}, 'hot', round(v.get('hot', {}).get('iou', 0), 3),
          'cold', round(v.get('cold', {}).get('iou', 0), 3), 'weakR', round(w.get('recall', 0), 3), 'weakPx', round(w.get('pixel_recall', 0), 3),
          'fz/kmh', round(v.get('zones', {}).get('false_zones_per_km_h', -1), 2))
