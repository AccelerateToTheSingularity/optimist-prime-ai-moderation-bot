"""Request-origin checks for local-only HTTP operator interfaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def local_http_authority(server_address: tuple[Any, ...]) -> str:
    """Build the exact Host authority used to reach a local HTTP server."""
    host = str(server_address[0]).strip()
    port = int(server_address[1])
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"{host}:{port}"


def is_local_http_request(
    headers: Mapping[str, str],
    server_address: tuple[Any, ...],
    *,
    require_origin: bool = False,
) -> bool:
    """Reject DNS-rebinding Hosts and browser requests from other origins."""
    authority = local_http_authority(server_address)
    host = str(headers.get("Host", "")).strip()
    if host.casefold() != authority.casefold():
        return False

    origin = headers.get("Origin")
    if origin is None:
        return not require_origin
    expected_origin = f"http://{authority}"
    return str(origin).strip().casefold() == expected_origin.casefold()
