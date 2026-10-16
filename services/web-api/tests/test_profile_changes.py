import copy
import json
from pathlib import Path

import pytest
from workbench.profile_changes import changes_html, profile_changes

FIXTURES = json.loads(
    (
        Path(__file__).resolve().parents[3]
        / "apps/web/src/profile-changes.fixtures.json"
    ).read_text()
)


@pytest.mark.parametrize("fixture", FIXTURES, ids=[f["name"] for f in FIXTURES])
def test_shared_change_contract(fixture):
    profile = copy.deepcopy(fixture["profile"])
    result = profile_changes(profile)
    assert result == fixture["expected"]
    assert profile == fixture["profile"]
    rendered = changes_html(result)
    if result is None:
        assert rendered == ""
    else:
        assert "Что изменила обработка" in rendered
        assert "не подтверждают качество очистки" in rendered
        if not result["paired_points"]:
            assert "Сопоставимых точек нет" in rendered


def test_changes_reject_truncation():
    with pytest.raises(ValueError, match="shape"):
        profile_changes(
            {"x_m": [0], "temperature_c": [1], "processed_temperature_c": []}
        )
