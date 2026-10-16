"""Immutable, project-scoped evidence snapshots and portable offline bundles."""

import hashlib
import html
import io
import json
import time
import uuid
import zipfile
from typing import Annotated

from fastapi import Header, HTTPException, Query
from fastapi.responses import Response
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field

from .platform import principal
from .profile_changes import profile_changes
from .workspace import report_html
from .workspace_routes import Region, checked_profile_run


def encoded(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def export_run(run):
    from .workspace_routes import RunDetail

    allowed = set(RunDetail.model_fields) | {
        "parameters",
        "calculated_at",
        "updated_at",
    }
    return {key: value for key, value in run.items() if key in allowed}


class ReviewSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    package: str = Field(max_length=100)
    review_id: str = Field(min_length=1, max_length=100)
    queue_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    expected_revision: int = Field(ge=0)


class EvidenceCaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    title: str = Field(min_length=1, max_length=150)
    notes: str = Field(default="", max_length=8000)
    region: Region
    profile: int = Field(ge=0)
    source_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    run_id: str | None = Field(default=None, min_length=1, max_length=128)
    analysis_column: int | None = Field(default=None, ge=0)
    analysis_zone_id: str | None = Field(default=None, max_length=100)
    analysis_layer: str | None = Field(default=None, max_length=100)
    review: ReviewSelection | None = None


class EvidenceCaseSummary(BaseModel):
    id: str
    title: str
    notes: str
    project_id: str
    account_id: str
    created_at: float
    source_sha256: str
    region: Region
    profile: int
    run_id: str | None
    method: str | None
    training_origin: str | None
    review: dict | None
    snapshot_sha256: str


class EvidenceCaseDetail(EvidenceCaseSummary):
    result: dict
    source_available: bool
    research_available: bool


def make_bundle(case):
    snapshot = case["result"]
    if digest(encoded(snapshot)) != case["snapshot_sha256"]:
        raise ValueError("Case snapshot integrity differs")
    evidence = snapshot["evidence"]
    report_bytes = encoded(evidence)
    html_page = report_html(evidence).replace(
        "<title>",
        f'<meta name="dts-evidence-sha256" content="{digest(report_bytes)}"><title>',
        1,
    )
    context = evidence.get("case", {})
    html_page = html_page.replace(
        "</h1>",
        "</h1><h2>"
        + html.escape(context.get("title", case["title"]))
        + "</h2><p>"
        + html.escape(context.get("notes", case["notes"]))
        + "</p>",
        1,
    )
    files = {
        "evidence.json": report_bytes,
        "evidence.html": html_page.encode("utf-8"),
        "README.txt": (
            "DTS Workbench — сохранённый кейс\n"
            "Откройте evidence.html без сервера. Точные данные: evidence.json.\n"
            "Проверка: python scripts/verify-evidence.py --record /path/to/native-record --bundle case.zip\n"
            "Нужны зависимости web-api и подготовленная исходная запись. ZIP не содержит весь CSV.\n"
            "SHA обеспечивают сверку с manifest, не цифровую подпись или доказательство авторства.\n"
            "Качество модели и физическая причина не установлены; метки не создавались.\n"
        ).encode(),
    }
    if snapshot["run"] is not None:
        files["run.json"] = encoded(snapshot["run"])
    manifest = {
        "schema_version": "dts-case-bundle-v1",
        "case_id": case["id"],
        "source_sha256": case["source_sha256"],
        "snapshot_sha256": case["snapshot_sha256"],
        "run_id": case["run_id"],
        "review": case["review"],
        "files": {
            name: {"sha256": digest(data), "bytes": len(data)}
            for name, data in files.items()
        },
    }
    files["manifest.json"] = encoded(manifest)
    if sum(map(len, files.values())) > 16 * 1024 * 1024:
        raise ValueError("Bundle exceeds 16 MiB")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return output.getvalue()


def install_cases(app, describe):
    from .control_routes import CaseUpdate

    def access():
        user = principal.get()
        if not user:
            raise HTTPException(401, "Войдите для сохранения кейса")
        if not app.state.platform:
            raise HTTPException(503, "Для кейсов требуется PostgreSQL")
        return user, app.state.platform

    def bind_analysis_view(body, evidence, run):
        if body.analysis_layer is None:
            return
        if not run:
            raise HTTPException(409, "A saved run is required for analysis view")
        layers = set(evidence.get("analysis", {}).get("layers", {}))
        if evidence["profile"].get("processed_temperature_c") is not None:
            layers.update({"processed_temperature_c", "difference"})
        layers.add("raw")
        if body.analysis_layer not in layers:
            raise HTTPException(409, "Selected layer is unavailable")
        column = (
            body.analysis_column
            if body.analysis_column is not None
            else body.region.column_start
        )
        if not body.region.column_start <= column < body.region.column_end:
            raise HTTPException(409, "Selected point is outside snapshot")
        if body.analysis_zone_id:
            zone = next(
                (z for z in run.get("zones", []) if z["id"] == body.analysis_zone_id),
                None,
            )
            if not zone or any(
                zone[k] != body.region.model_dump()[k]
                for k in ("row_start", "row_end", "column_start", "column_end")
            ):
                raise HTTPException(409, "Selected zone differs from snapshot window")
        evidence["analysis_view"] = {
            "schema_version": "dts-analysis-view-v1",
            "run_id": body.run_id,
            "source_sha256": body.source_sha256,
            "region": body.region.model_dump(),
            "row": body.profile,
            "column": column,
            "layer": body.analysis_layer,
            "zone_id": body.analysis_zone_id,
        }

    def prepare(body, user):
        if user["project_id"] != "local":
            raise HTTPException(403, "Native archive belongs to the local project")
        evidence = describe(body.region, body.profile)
        if evidence["source_sha256"] != body.source_sha256:
            raise HTTPException(
                409, "Версия источника изменилась. Обновите исследование."
            )
        evidence.update(runs=[], profile_runs=[])
        run = None
        if body.run_id:
            if len(body.run_id) == 64:
                from .analysis_results import snapshot_analysis

                run, saved = snapshot_analysis(
                    app,
                    body.run_id,
                    body.region,
                    body.profile,
                    body.analysis_column,
                    body.analysis_zone_id,
                )
                if run["source_sha256"] != body.source_sha256:
                    raise HTTPException(409, "Analysis source SHA differs")
                evidence["analysis"] = saved
                metadata = {key: value for key, value in run.items() if key != "result"}
                evidence["runs"] = evidence["profile_runs"] = [metadata]
                processed = saved["layers"].get("processed_temperature_c")
                if processed:
                    evidence["profile"].update(
                        processing_run_id=body.run_id,
                        processed_temperature_c=processed["profile"],
                    )
                    evidence["processing_changes"] = profile_changes(
                        evidence["profile"]
                    )
                bind_analysis_view(body, evidence, run)
                return evidence, run
            run, values = checked_profile_run(
                app.state.workspace,
                body.run_id,
                body.region,
                body.profile,
                body.source_sha256,
                app.state.store.get(body.region.record_id).raw_c.shape[1],
            )
            # Export the existing public run contract, never arbitrary workspace internals.
            run = export_run(run)
            metadata = {key: value for key, value in run.items() if key != "result"}
            evidence["runs"] = evidence["profile_runs"] = [metadata]
            evidence["profile"].update(
                processing_run_id=body.run_id,
                processed_temperature_c=[
                    value if raw is not None else None
                    for value, raw in zip(
                        values[body.region.column_start : body.region.column_end],
                        evidence["profile"]["temperature_c"],
                    )
                ],
            )
            evidence["processing_changes"] = profile_changes(evidence["profile"])
        bind_analysis_view(body, evidence, run)
        return evidence, run

    def journal(body, user):
        if body.review is None:
            return None, None
        if user["role"] not in ("expert", "admin"):
            raise HTTPException(
                403, "Для включения своего экспертного ответа нужна роль эксперта"
            )
        from .review_packages import resolve_queue
        q = resolve_queue(app,body.review.package)
        card = q.cards.get(body.review.review_id)
        if not q or q.sha != body.review.queue_sha256 or not card:
            raise HTTPException(409, "Версия экспертной очереди изменилась")
        if not (
            card["source"].removesuffix(".csv") == body.region.record_id
            and card["evidence"]["source_sha256"] == body.source_sha256
            and card["row_bounds"] == [body.region.row_start, body.region.row_end]
            and card["column_bounds"]
            == [body.region.column_start, body.region.column_end]
        ):
            raise HTTPException(409, "Ответ относится к другому окну")
        return q, f"{user['project_id']}:{q.sha}:{user['id']}"

    def answer_in(db, body, namespace):
        if namespace is None:
            return None
        db.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (namespace,))
        row = db.execute(
            "SELECT payload FROM reviews WHERE namespace=%s AND review_id=%s ORDER BY revision DESC LIMIT 1",
            (namespace, body.review.review_id),
        ).fetchone()
        answer = row[0] if row else None
        if (answer["revision"] if answer else 0) != body.review.expected_revision:
            raise HTTPException(
                409,
                "Ответ изменился. Обновите предпросмотр и ознакомьтесь с новой ревизией.",
            )
        return answer

    @app.post("/api/v2/cases/preview")
    def preview(body: EvidenceCaseRequest):
        user, platform = access()
        evidence, run = prepare(body, user)
        _, namespace = journal(body, user)
        with platform.connect() as db:
            answer = answer_in(db, body, namespace)
        return {
            "source_sha256": evidence["source_sha256"],
            "region": body.region,
            "profile": body.profile,
            "run_id": body.run_id,
            "method": run.get("method") if run else None,
            "expert_answer": answer,
            "automatic_training_labels_created": 0,
        }

    @app.post(
        "/api/v2/cases", response_model=EvidenceCaseDetail | dict, status_code=201
    )
    def create(
        body: EvidenceCaseRequest | CaseUpdate,
        response: Response,
        idempotency_key: Annotated[
            str | None, Header(min_length=1, max_length=128, pattern=r"^[!-~]+$")
        ] = None,
    ):
        if isinstance(body, CaseUpdate):
            response.status_code = 200
            return app.state.case_assignment_update(body)
        user, platform = access()
        if not body.title.strip():
            raise HTTPException(422, "Название кейса не должно быть пустым")
        if user['role'] not in ('expert','analyst','admin'):
            raise HTTPException(403,'Viewer can read evidence; saving requires expert or analyst role')
        request_id = (
            digest(encoded([user["project_id"], user["id"], idempotency_key]))
            if idempotency_key
            else None
        )
        request_body = body.model_dump()
        for name in ("analysis_column", "analysis_zone_id", "analysis_layer"):
            if request_body[name] is None:
                request_body.pop(name)
        body_digest = digest(encoded(request_body))
        with platform.connect() as db:
            if request_id:
                db.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                    ("evidence-case-request:" + request_id,),
                )
                previous = db.execute(
                    "SELECT payload FROM documents WHERE kind='evidence_case_request' AND id=%s",
                    (request_id,),
                ).fetchone()
                if previous:
                    if previous[0]["body_sha256"] != body_digest:
                        raise HTTPException(
                            409, "Ключ сохранения уже использован с другим содержимым"
                        )
                    return detail(previous[0]["case_id"])
            evidence, run = prepare(body, user)
            _, namespace = journal(body, user)
            answer = answer_in(db, body, namespace)
            evidence["expert_answer"] = answer
            if body.review:
                evidence["queue_sha256"] = body.review.queue_sha256
            case_id, created_at = uuid.uuid4().hex, time.time()
            evidence["case"] = {
                "id": case_id,
                "title": body.title.strip(),
                "notes": body.notes,
                "created_at": created_at,
            }
            snapshot = {"evidence": evidence, "run": run}
            if len(encoded(snapshot)) > 4 * 1024 * 1024:
                raise HTTPException(413, "Снимок превышает 4 MiB")
            review = (
                {
                    **body.review.model_dump(),
                    "account_id": user["id"],
                    "revision": answer["revision"],
                }
                if answer
                else None
            )
            case = {
                "id": case_id,
                "title": body.title.strip(),
                "notes": body.notes,
                "project_id": user["project_id"],
                "account_id": user["id"],
                "created_at": created_at,
                "source_sha256": body.source_sha256,
                "region": body.region.model_dump(),
                "profile": body.profile,
                "run_id": body.run_id,
                "method": run.get("method") if run else None,
                "training_origin": run.get("training_origin") if run else None,
                "review": review,
                "snapshot_sha256": digest(encoded(snapshot)),
            }
            stored = {
                **case,
                "updated_at": case["created_at"],
                "result_ref": platform.blobs.put(snapshot),
            }
            # One insertion, in the same transaction and journal lock as revision validation.
            db.execute(
                "INSERT INTO documents VALUES('evidence_case',%s,%s,%s)",
                (case["id"], Jsonb(stored), stored["updated_at"]),
            )
            if request_id:
                db.execute(
                    "INSERT INTO documents VALUES('evidence_case_request',%s,%s,%s)",
                    (
                        request_id,
                        Jsonb(
                            {
                                "project_id": user["project_id"],
                                "account_id": user["id"],
                                "body_sha256": body_digest,
                                "case_id": case_id,
                            }
                        ),
                        created_at,
                    ),
                )
            db.execute(
                "INSERT INTO audit(project_id,actor,action,resource,at) VALUES(%s,%s,'case.created',%s,%s)",
                (user["project_id"], user["id"], case["id"], case["created_at"]),
            )
        return {
            **case,
            "result": snapshot,
            "source_available": True,
            "research_available": True,
        }

    def can_read(item, user, platform):
        if item["project_id"] != user["project_id"]:
            return False
        review = item.get("review")
        if not review or item["account_id"] == user["id"] or user["role"] in ("admin","analyst"):
            return True
        # Preserve independence: another expert sees the answer only after their own response.
        if user["role"] not in ("expert", "admin"):
            return False
        namespace = f"{user['project_id']}:{review['queue_sha256']}:{user['id']}"
        with platform.connect() as db:
            return bool(
                db.execute(
                    "SELECT 1 FROM reviews WHERE namespace=%s AND review_id=%s LIMIT 1",
                    (namespace, review["review_id"]),
                ).fetchone()
            )

    def get_case(key):
        user, platform = access()
        item = platform.get("evidence_case", key)
        if not can_read(item, user, platform):
            raise HTTPException(
                403, "Самостоятельный ответ ещё не сохранён; чужой ответ закрыт"
            )
        if digest(encoded(item["result"])) != item["snapshot_sha256"]:
            raise HTTPException(409, "Нарушена целостность снимка кейса")
        return item

    @app.get("/api/v2/cases", response_model=list[EvidenceCaseSummary] | dict)
    def listing(
        limit: int = Query(50, ge=1, le=100),
        offset: int = Query(0, ge=0),
        package: str | None = None,
    ):
        if package is not None:
            return app.state.case_assignment_list(package, limit, offset)
        user, platform = access()
        return [
            item
            for item in platform.list("evidence_case", limit, offset)
            if can_read(item, user, platform)
        ]

    @app.get("/api/v2/cases/{key}", response_model=EvidenceCaseDetail)
    def detail(key: str):
        item = get_case(key)
        native = app.state.store.records.get(item["region"]["record_id"])
        available = bool(
            native and native.manifest["source_sha256"] == item["source_sha256"]
        )
        research_available = available
        if available and item["run_id"]:
            try:
                if len(item["run_id"]) == 64:
                    from .analysis_results import snapshot_analysis

                    current_run, _ = snapshot_analysis(
                        app,
                        item["run_id"],
                        Region.model_validate(item["region"]),
                        item["profile"],
                        item["result"]["evidence"]["analysis"].get("column"),
                        item["result"]["evidence"]["analysis"].get("zone_id"),
                    )
                    research_available = current_run == item["result"]["run"]
                    return {
                        **item,
                        "source_available": available,
                        "research_available": research_available,
                    }
                current_run, _ = checked_profile_run(
                    app.state.workspace,
                    item["run_id"],
                    Region.model_validate(item["region"]),
                    item["profile"],
                    item["source_sha256"],
                    native.raw_c.shape[1],
                )
                research_available = export_run(current_run) == item["result"]["run"]
            except (HTTPException, KeyError, OSError, ValueError):
                research_available = False
        return {
            **item,
            "source_available": available,
            "research_available": research_available,
        }

    @app.get("/api/v2/cases/{key}/bundle")
    def bundle(key: str):
        try:
            payload = make_bundle(get_case(key))
        except ValueError:
            raise HTTPException(409, "Пакет кейса недоступен или повреждён") from None
        return Response(
            payload,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="dts-case-{key}.zip"',
                "Cache-Control": "private, no-store",
            },
        )
