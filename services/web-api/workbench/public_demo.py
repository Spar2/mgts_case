"""Bounded anonymous surface. Reports never create journal entries."""

import hashlib
import html
import json
import logging
import math
import os
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

PUBLIC_PATHS = {
    "/api/public/v1/package",
    "/api/public/v1/report",
    "/api/public/v1/access-requests",
}


class DemoReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: Literal["present", "absent", "ambiguous", "unknown"] = "unknown"
    notes: str = Field(default="", max_length=4000)


class AccessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=100)
    contact: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=2000)


class CloseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["approved", "declined"]


class PublicProvenance(BaseModel):
    method: str
    method_version: str
    training_origin: str
    scope: str
    calculated_at: str
    limitations: str


class PublicPackage(BaseModel):
    schema_version: Literal["dts-public-demo-v1"]
    candidate: Literal["review-00018"]
    source: Literal["5.csv"]
    source_sha256: str
    queue_sha256: str
    package_sha256: str
    row_bounds: tuple[int, int]
    column_bounds: tuple[int, int]
    x_m: list[float]
    time_s: list[float]
    timestamps: list[str]
    timezone: None
    segment_id: list[int]
    raw_c: list[list[float | None]]
    processed_c: list[list[float | None]]
    invalid_reason: list[list[int]]
    provenance: PublicProvenance


def load_package():
    root = os.environ.get("DTS_PUBLIC_PACKAGE")
    if not root:
        logger.warning("key=DTS_PUBLIC_PACKAGE configured=false code=package_unavailable")
        return None
    try:
        root = Path(root)
        encoded = (root / "package.json").read_bytes()
        manifest = json.loads((root / "manifest.json").read_bytes())
        if (
            hashlib.sha256(encoded).hexdigest() != manifest["sha256"]
            or len(encoded) != manifest["bytes"]
        ):
            logger.error("key=DTS_PUBLIC_PACKAGE configured=true code=package_unavailable reason=integrity")
            raise ValueError("Public package integrity mismatch")
        package = json.loads(encoded)
        if (
            package["schema_version"] != "dts-public-demo-v1"
            or package["row_bounds"] != [1419, 1470]
            or package["column_bounds"] != [798, 884]
        ):
            logger.error("key=DTS_PUBLIC_PACKAGE configured=true code=package_unavailable reason=window")
            raise ValueError("Unapproved public window")
        expected = {
            "schema_version",
            "candidate",
            "source",
            "source_sha256",
            "queue_sha256",
            "row_bounds",
            "column_bounds",
            "x_m",
            "time_s",
            "timestamps",
            "timezone",
            "segment_id",
            "raw_c",
            "invalid_reason",
            "processed_c",
            "provenance",
        }
        if (
            set(package) != expected
            or package["source_sha256"]
            != "98c1d629ed59478c81fa58b60e891f5587ef9d64ad435655e717a27f64767e46"
        ):
            logger.error("key=DTS_PUBLIC_PACKAGE configured=true code=package_unavailable reason=fields")
            raise ValueError("Unexpected public package fields or source")
        if (
            package["x_m"] != list(range(798, 884))
            or len(package["time_s"]) != 51
            or len(package["timestamps"]) != 51
            or len(package["segment_id"]) != 51
            or package["timezone"] is not None
        ):
            logger.error("key=DTS_PUBLIC_PACKAGE configured=true code=package_unavailable reason=coordinates")
            raise ValueError("Invalid public coordinates")
        if any(not math.isfinite(t) for t in package["time_s"]) or any(
            b <= a for a, b in zip(package["time_s"], package["time_s"][1:])
        ):
            logger.error("key=DTS_PUBLIC_PACKAGE configured=true code=package_unavailable reason=timeline")
            raise ValueError("Invalid public timeline")
        for key in ("raw_c", "invalid_reason", "processed_c"):
            if len(package[key]) != 51 or any(len(row) != 86 for row in package[key]):
                logger.error("key=DTS_PUBLIC_PACKAGE configured=true code=package_unavailable reason=shape")
                raise ValueError("Invalid public matrix shape")
        for raw, mask, processed in zip(
            package["raw_c"], package["invalid_reason"], package["processed_c"]
        ):
            for value, reason, cleaned in zip(raw, mask, processed):
                if (
                    reason not in (0, 1, 2, 4)
                    or (reason and (value is not None or cleaned is not None))
                    or (
                        not reason
                        and (
                            value is None
                            or cleaned is None
                            or not math.isfinite(value)
                            or not math.isfinite(cleaned)
                            or value == -300
                        )
                    )
                ):
                    logger.error("key=DTS_PUBLIC_PACKAGE configured=true code=package_unavailable reason=mask")
                    raise ValueError("Invalid public quality mask")
        return {**package, "package_sha256": manifest["sha256"]}
    except UnicodeDecodeError as exc:
        logger.error("key=DTS_PUBLIC_PACKAGE configured=true code=package_unavailable reason=unreadable")
        raise ValueError(f"Public package integrity mismatch: {exc}")
    except ValueError:
        logger.error("key=DTS_PUBLIC_PACKAGE configured=true code=package_unavailable reason=integrity")
        raise
    except (OSError, KeyError) as exc:
        logger.error("key=DTS_PUBLIC_PACKAGE configured=true code=package_unavailable reason=unreadable")
        raise ValueError(f"Public package integrity mismatch: {exc}")


def install_public(app):
    from .account_lifecycle import admin

    def package():
        value = app.state.public_package
        if value is None:
            raise HTTPException(503, "Published demo package is not configured")
        return value

    @app.get("/api/public/v1/package", response_model=PublicPackage)
    def read_package():
        return package()

    @app.post("/api/public/v1/report", response_class=HTMLResponse)
    def report(body: DemoReport, request: Request):
        app.state.limiter.check("public-report", request.client.host, 20)
        data = package()
        rows = "".join(
            "<tr><td>"
            + html.escape(timestamp)
            + "</td>"
            + "".join(
                "<td>" + (str(v) if v is not None else "Пропуск") + "</td>" for v in row
            )
            + "</tr>"
            for timestamp, row in zip(data["timestamps"], data["raw_c"])
        )
        return HTMLResponse(
            "<!doctype html><html lang='ru'><meta charset='utf-8'><title>DTS — демо</title><h1>Демонстрационный ответ — не экспертная метка</h1><p>5.csv / review-00018 · только опубликованное окно</p>"
            f"<p>SHA источника: {data['source_sha256']}</p><p>SHA пакета: {data['package_sha256']}</p><p>Ответ: {html.escape(body.answer)}</p><p>{html.escape(body.notes)}</p>"
            "<p>Часовой пояс неизвестен. median_x3 не классифицирует событие.</p><table><caption>Исходные значения, °C</caption><thead><tr><th>Время</th>"
            + "".join(f"<th>{x} м</th>" for x in data["x_m"])
            + "</tr></thead><tbody>"
            + rows
            + "</tbody></table></html>"
        )

    @app.post("/api/public/v1/access-requests", status_code=202)
    def access(body: AccessRequest, request: Request):
        app.state.limiter.check("access-request", request.client.host, 3)
        if not app.state.platform:
            raise HTTPException(503, "Access requests temporarily unavailable")
        if not all(v.strip() for v in (body.name, body.contact, body.reason)):
            raise HTTPException(422, "Заполните все поля")
        key, now = uuid.uuid4().hex, time.time()
        with app.state.platform.connect() as db:
            db.execute(
                "DELETE FROM access_requests WHERE closed<%s", (now - 30 * 86400,)
            )
            db.execute(
                "INSERT INTO access_requests(id,name,contact,reason,created) VALUES(%s,%s,%s,%s,%s)",
                (
                    key,
                    body.name.strip(),
                    body.contact.strip(),
                    body.reason.strip(),
                    now,
                ),
            )
        return {"received": True, "id": key, "access_granted": False}

    @app.get("/api/v2/access-requests")
    def requests():
        if admin()["project_id"] != "local":
            raise HTTPException(403, "Local project administrator required")
        with app.state.platform.connect() as db:
            db.execute(
                "DELETE FROM access_requests WHERE closed<%s",
                (time.time() - 30 * 86400,),
            )
            return [
                dict(zip(("id", "name", "contact", "reason", "created", "status"), row))
                for row in db.execute(
                    "SELECT id,name,contact,reason,created,status FROM access_requests ORDER BY created DESC LIMIT 200"
                )
            ]

    @app.post("/api/v2/access-requests/{key}/close")
    def close(key: str, body: CloseRequest):
        if admin()["project_id"] != "local":
            raise HTTPException(403, "Local project administrator required")
        with app.state.platform.connect() as db:
            if not db.execute(
                "UPDATE access_requests SET status=%s,closed=%s WHERE id=%s AND closed IS NULL RETURNING id",
                (body.status, time.time(), key),
            ).fetchone():
                raise HTTPException(404, "Pending request not found")
        return {"closed": key, "access_granted": False}
