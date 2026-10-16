"""One-time streaming CSV preparation; no interpolation or label generation."""

import argparse
import json
from pathlib import Path

from .queue_builder import build_queue
from .vendor.data import ingest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv-root", type=Path, required=True)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--queue", type=Path)
    a = p.parse_args()
    for i in range(1, 7):
        r = ingest(a.csv_root / f"{i}.csv", a.records)
        print(
            json.dumps(
                {
                    "source": r.manifest["source_name"],
                    "shape": list(r.raw_c.shape),
                    "path": str(r.path),
                }
            ),
            flush=True,
        )
    if a.queue:
        print(build_queue(a.records, a.queue), flush=True)


if __name__ == "__main__":
    main()
