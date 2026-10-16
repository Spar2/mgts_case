"""The review page shows one window and does not label the shortlist."""
import json

import numpy as np

from fiber.shortlist import BUCKETS
from fiber.shortlist_review import QUESTIONS, preview_window, render_page


def test_preview_puts_the_hot_core_above_the_flanks():
    clean = np.full((20, 40), 20.0, dtype=np.float64)
    clean[:, 15:20] = 30.0
    valid = np.ones(clean.shape, dtype=bool)
    x_m = np.arange(40, dtype=np.float64)
    stamps = [f"2026-01-01T00:{index:02d}:00+00:00" for index in range(20)]
    view = preview_window(
        clean, valid, x_m, stamps, [15.0, 19.0], [stamps[2], stamps[10]], pad_m=5.0,
    )
    assert view["core_median_c"] == 30.0
    assert view["side_median_c"] == 20.0
    assert view["core_cols"][0] < view["core_cols"][1]
    assert view["heat"]
    assert all(len(row) == len(view["heat"][0]) for row in view["heat"])
    assert len(view["profile_x"]) == len(view["profile_c"])


def test_review_page_lists_every_bucket_question_and_leaves_answers_blank():
    assert set(QUESTIONS) == set(BUCKETS)
    cards = []
    for index, bucket in enumerate(BUCKETS):
        cards.append({
            "id": f"ask-{index:04d}",
            "record": str(index + 1),
            "source_sha256": "abc",
            "question": QUESTIONS[bucket],
            "reason": f"reason for {bucket}",
            "meters": [1.0, 2.0],
            "time": ["2026-01-01T00:00:00+00:00", "2026-01-01T01:00:00+00:00"],
            "tz_missing": False,
            "heat": [[20.0, 21.0], [22.0, None]],
            "core_rows": [0, 2],
            "core_cols": [0, 1],
            "core_x": [1.0, 2.0],
            "profile_x": [1.0, 2.0],
            "profile_c": [20.0, 21.0],
            "core_median_c": 21.0,
            "side_median_c": 20.0,
        })
    page = render_page(cards)
    assert "shortlist.json" in page
    assert "Скачать ответы" in page
    assert "Сохранить и дальше" in page
    assert 'data-action="save-and-next"' in page
    assert 'data-presence="yes"' in page
    assert 'data-presence="no"' in page
    assert 'data-presence="unclear"' in page
    assert "let index = 0;" in page
    assert "index += 1;" in page
    assert "card.reason" in page
    assert "expert_confirmed: true" not in page
    assert '"expert_confirmed": true' not in page
    for card in cards:
        assert card["id"] in page
        assert card["question"] in page
        assert card["reason"] in page
    payload = page.split("const CARDS = ", 1)[1].split(";\nconst KEY", 1)[0]
    loaded = json.loads(payload)
    assert len(loaded) == len(BUCKETS)
    assert "expert_confirmed" not in loaded[0]
    assert loaded[0]["reason"] == cards[0]["reason"]
