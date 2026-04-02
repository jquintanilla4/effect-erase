from __future__ import annotations

from fastapi import Request


def _first_forwarded_value(value: str | None) -> str | None:
    if not value:
        return None
    return value.split(",", 1)[0].strip() or None


def _forwarded_pairs(value: str | None) -> dict[str, str]:
    if not value:
        return {}

    pairs: dict[str, str] = {}
    for part in value.split(",", 1)[0].split(";"):
        if "=" not in part:
            continue
        key, raw_value = part.split("=", 1)
        pairs[key.strip().lower()] = raw_value.strip().strip('"')
    return pairs


def public_base_url(request: Request) -> str:
    forwarded = _forwarded_pairs(request.headers.get("forwarded"))
    # Reverse proxies often terminate TLS before forwarding plain HTTP to the
    # worker, so prefer forwarded scheme/host when they are present.
    scheme = (
        _first_forwarded_value(request.headers.get("x-forwarded-proto"))
        or forwarded.get("proto")
        or request.url.scheme
    )
    host = (
        _first_forwarded_value(request.headers.get("x-forwarded-host"))
        or forwarded.get("host")
        or request.headers.get("host")
        or request.url.netloc
    )
    return f"{scheme}://{host}"
