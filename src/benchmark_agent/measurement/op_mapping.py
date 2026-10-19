from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit

Operation = Literal["write", "read", "delete"]

ENDPOINT_TO_OPERATION: dict[str, Operation] = {
    "/api/v2/write": "write",
    "/api/v2/query": "read",
    "/api/v2/delete": "delete",
    "/write": "write",
    "/query": "read",
    "/tessera/v1/ingest": "write",
    "/tessera/v1/fetch": "read",
    "/tessera/v1/purge": "delete",
}


def map_path_to_operation(path: str) -> Operation | None:
    clean_path = urlsplit(path).path.rstrip("/") or "/"
    for endpoint, operation in ENDPOINT_TO_OPERATION.items():
        if clean_path == endpoint or clean_path.startswith(f"{endpoint}/"):
            return operation
    return None
