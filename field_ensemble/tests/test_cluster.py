"""Имена кластеров, эмбеддинг EventNet и выбор k. Реальные CSV не читаются."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import cluster_real as C
from dts import events


def test_names_follow_duration_width_and_sign():
    hot = dict(dur_h=0.4, width_m=2.0, sign=1.0, acf_peak=0.05, rise_frac=0.2, fall_frac=0.2, plateau_frac=0.3)
    cold = dict(dur_h=12.0, width_m=40.0, sign=-1.0, acf_peak=0.05, rise_frac=0.2, fall_frac=0.2, plateau_frac=0.5)
    periodic = dict(dur_h=2.0, width_m=10.0, sign=1.0, acf_peak=0.55, rise_frac=0.5, fall_frac=0.2, plateau_frac=0.2)
    assert C.name_cluster(hot) == "кратковременный узкий нагрев"
    assert C.name_cluster(cold) == "длительное обширное охлаждение"
    assert C.name_cluster(periodic) == "многочасовой умеренный нагрев, с периодичностью, медленный подъём"
    filled = dict(dur_h=0.4, width_m=2.0, sign=1.0, acf_peak=0.1, rise_frac=0.0, fall_frac=0.0, plateau_frac=1.0)
    assert C.name_cluster(filled) == "кратковременный узкий нагрев, без спада внутри бокса"


def test_embedding_is_input_of_last_head_layer():
    torch = pytest.importorskip("torch")
    from dts.models import EventNet
    torch.manual_seed(0)
    torch.set_num_threads(2)
    net = EventNet(7, len(events.SCALAR_NAMES))
    net.eval()
    imgs = np.zeros((4, 3, 32, 64), np.float32)
    imgs[:, 0, 8:20, 10:30] = 1
    scal = np.zeros((4, len(events.SCALAR_NAMES)), np.float32)
    scal[:, events.SCALAR_NAMES.index("sign")] = 1
    emb = C.eventnet_embedding(net, imgs, scal, np.zeros(len(events.SCALAR_NAMES)), np.ones(len(events.SCALAR_NAMES)))
    assert emb.shape == (4, net.head[-1].in_features)
    assert emb.shape[1] == 64
    assert np.isfinite(emb).all()


def test_kmeans_separates_two_blobs_and_ari_is_one_for_a_copy():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.15, (30, 6))
    b = rng.normal(4, 0.15, (30, 6))
    X = np.vstack([a, b])
    k, labels, rows = C.select_kmeans(X, ks=range(2, 6), seed=0)
    assert k == 2
    assert max(r["silhouette"] for r in rows) == max(r["silhouette"] for r in rows if r["k"] == 2)
    # два облака не перемешаны
    assert len(set(labels[:30])) == 1 and len(set(labels[30:])) == 1
    assert C.adjusted_rand(labels, labels) == pytest.approx(1.0)
    agg, sil = C.agglomerative_labels(X, 2)
    assert sil > 0.5
    assert C.adjusted_rand(labels, agg) == pytest.approx(1.0)


def test_cluster_summary_uses_physical_medians_not_class_names():
    scal = np.zeros((4, len(events.SCALAR_NAMES)), np.float32)
    # две короткие узкие горячие и две длинные широкие холодные
    scal[:2, events.SCALAR_NAMES.index("log_dur_h")] = np.log1p(0.3)
    scal[:2, events.SCALAR_NAMES.index("log_width_m")] = np.log1p(1.5)
    scal[:2, events.SCALAR_NAMES.index("sign")] = 1
    scal[:2, events.SCALAR_NAMES.index("amp_med")] = 2
    scal[:2, events.SCALAR_NAMES.index("rise_frac")] = 0.2
    scal[:2, events.SCALAR_NAMES.index("fall_frac")] = 0.2
    scal[2:, events.SCALAR_NAMES.index("log_dur_h")] = np.log1p(10)
    scal[2:, events.SCALAR_NAMES.index("log_width_m")] = np.log1p(40)
    scal[2:, events.SCALAR_NAMES.index("sign")] = -1
    scal[2:, events.SCALAR_NAMES.index("amp_med")] = -3
    scal[2:, events.SCALAR_NAMES.index("rise_frac")] = 0.2
    scal[2:, events.SCALAR_NAMES.index("fall_frac")] = 0.2
    phys = [C.physical_row(s) for s in scal]
    metas = [dict(zone_class="Sensor_Artifact"), dict(zone_class="Sensor_Artifact"),
             dict(zone_class="Local_Heating"), dict(zone_class="Local_Heating")]
    rows = C.cluster_summaries(metas, phys, np.array([0, 0, 1, 1]))
    by_id = {r["id"]: r for r in rows}
    assert by_id[0]["name"] == "кратковременный узкий нагрев"
    assert by_id[1]["name"] == "длительное обширное охлаждение"
    # имя не взято из класса классификатора (он здесь противоположный)
    assert by_id[0]["dominant_class"] == "Sensor_Artifact"
    assert "Sensor" not in by_id[0]["name"]
    order, table = C.contingency(metas, np.array([0, 0, 1, 1]))
    assert table.shape == (2, len(events.CLASSES))
    assert table.sum() == 4
    assert order == [0, 1]
