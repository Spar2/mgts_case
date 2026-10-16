import argparse
import json
from pathlib import Path

from workbench.api import app

p = argparse.ArgumentParser()
p.add_argument("--check", action="store_true")
args = p.parse_args()
target = Path(__file__).resolve().parents[1] / "services/web-api/openapi.json"
text = json.dumps(app.openapi(), ensure_ascii=False, indent=2)
if args.check:
    assert target.read_text(encoding="utf-8") == text, (
        "OpenAPI drift: regenerate contract and client types"
    )
else:
    target.write_text(text, encoding="utf-8")
