"""Read-only localhost timings. First request may already be cached: report honestly."""

import argparse
import json
import statistics
import struct
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx


def request(client, path, **kwargs):
    start = time.perf_counter()
    response = client.get(path, **kwargs)
    response.raise_for_status()
    return response, (time.perf_counter() - start) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    parser.add_argument(
        "--output", type=Path, default=Path(".runtime/api-benchmark.json")
    )
    args = parser.parse_args()
    if urlsplit(args.url).hostname not in ("127.0.0.1", "localhost"):
        parser.error("Use the local pilot for this benchmark")
    result = {
        "at": datetime.now(timezone.utc).isoformat(),
        "url": args.url,
        "conditions": "loopback, sequential GET, gzip accepted; first request may be cached; concurrent replay/browser QA possible",
        "overview_shape": [127, 190],
        "records": [],
    }
    with httpx.Client(base_url=args.url, timeout=120, trust_env=False) as client:
        records = client.get("/api/v1/records").json()
        for record in records:
            path = f"/api/v1/records/{record['id']}/tiles"
            params = {
                "x0": record["x_bounds"][0],
                "x1": record["x_bounds"][1],
                "t0": record["time_bounds"][0],
                "t1": record["time_bounds"][1],
                "width": 190,
                "height": 127,
            }
            tile, first = request(client, path, params=params)
            head_length = struct.unpack_from("<I", tile.content)[0]
            metadata = json.loads(tile.content[4 : 4 + head_length])
            warm = [request(client, path, params=params)[1] for _ in range(10)]
            row, column = [n // 2 for n in record["shape"]]
            slices, slice_ms = request(
                client,
                f"/api/v1/records/{record['id']}/slices",
                params={"row": row, "column": column},
            )
            result["records"].append(
                {
                    "source": record["source"],
                    "source_sha256": record["source_sha256"],
                    "native_shape": record["shape"],
                    "aggregation": metadata["aggregation"],
                    "first_overview_ms": round(first, 2),
                    "warm_median_ms": round(statistics.median(warm), 2),
                    "warm_max_ms": round(max(warm), 2),
                    "overview_decoded_bytes": len(tile.content),
                    "overview_content_length": tile.headers.get("content-length"),
                    "slice_ms": round(slice_ms, 2),
                    "slice_decoded_bytes": len(slices.content),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
