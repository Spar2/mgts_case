"""Reject untrusted request contexts before touching sessions or data storage."""

import os
from pathlib import PurePosixPath
from urllib.parse import urlsplit

from fastapi.responses import JSONResponse

AUTH_BODY_PATHS = {
    "/api/v2/auth/login",
    "/api/v2/auth/activate",
    "/api/v2/auth/recover",
}
ASSET_SUFFIXES = {
    ".js",
    ".css",
    ".woff",
    ".woff2",
    ".ttf",
    ".svg",
    ".png",
    ".ico",
    ".jpg",
    ".jpeg",
    ".webp",
    ".avif",
    ".wasm",
}
LANDING_ASSETS = frozenset({
    "/landing/mgts-original.jpg",
    "/landing/larp-corp-original.jpg",
    "/landing/visual-data.json",
})


def public_asset(path):
    return path in LANDING_ASSETS or (
        path.startswith("/assets/")
        and PurePosixPath(path).suffix.lower() in ASSET_SUFFIXES
        and not any(p.startswith(".") for p in path.split("/") if p)
        and "\\" not in path
    )


def protect_response(response, static_asset=False):
    response.headers["Cache-Control"] = (
        "public, max-age=31536000, immutable" if static_asset else "no-store"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = (
        "frame-ancestors 'none'; base-uri 'self'; object-src 'none'"
    )
    return response


def reject_untrusted(request):
    path = request.scope.get("path", "")
    if path.startswith("/assets/") and not public_asset(path):
        return protect_response(
            JSONResponse({"detail": "Asset not found"}, status_code=404)
        )
    raw_host = request.headers.get("host", "")
    try:
        parsed = urlsplit("//" + raw_host)
        # Accessing port also validates malformed/non-numeric port syntax.
        _ = parsed.port
        host = parsed.hostname
    except ValueError:
        host = None
    allowed = {
        v.strip().lower()
        for v in os.environ.get(
            "DTS_ALLOWED_HOSTS", "localhost,127.0.0.1,::1,testserver"
        ).split(",")
    }
    if (
        not raw_host
        or any(ord(c) <= 32 or c in "/\\?#@" for c in raw_host)
        or host not in allowed
    ):
        return protect_response(
            JSONResponse({"detail": "Untrusted host"}, status_code=403)
        )
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        origins = os.environ.get(
            "DTS_ALLOWED_ORIGINS",
            "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:8081,http://localhost:8081,http://127.0.0.1:8091,http://localhost:8091",
        ).split(",")
        if request.headers.get("sec-fetch-site") == "cross-site" or (
            origin and origin not in origins
        ):
            return protect_response(
                JSONResponse({"detail": "Untrusted origin"}, status_code=403)
            )
        if (
            request.url.path in AUTH_BODY_PATHS
            and request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            != "application/json"
        ):
            return protect_response(
                JSONResponse({"detail": "JSON request required"}, status_code=415)
            )
    return None
