from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

_ALLOWED_METHODS = {"GET", "HEAD", "POST"}
_POST_ALLOWED_PREFIXES = ("/api/v2/query", "/api/v2/signin")
_FLUX_WRITE_RE = re.compile(r"\bto\s*\(", re.IGNORECASE)
_HEALTH_PATHS = {"tessera": "/tessera/v1/health", "kestrel": "/v1/status"}
_POST_PREFIXES: dict[str, tuple[str, ...]] = {"tessera": ("/tessera/v1/fetch",), "kestrel": ()}
_AUTH_HEADERS = {"tessera": "X-Tessera-Key", "kestrel": "X-Api-Token"}


class SutProbeToolkit:
    def __init__(
        self,
        *,
        base_url: str,
        max_calls: int = 8,
        timeout_seconds: float = 5.0,
        max_response_chars: int = 4000,
        auth_token: str | None = None,
        api: str = "influx",
    ) -> None:
        self._api = api
        self._health_path = _HEALTH_PATHS.get(api, "/health")
        self._post_allowed_prefixes = _POST_PREFIXES.get(api, _POST_ALLOWED_PREFIXES)
        self._auth_header = _AUTH_HEADERS.get(api, "Authorization")
        self._base_url = base_url.rstrip("/") + "/"
        self._max_calls = max_calls
        self._timeout_seconds = timeout_seconds
        self._max_response_chars = max_response_chars
        self._auth_token = auth_token
        self._calls = 0
        self._base_parts = urlsplit(self._base_url)

    @property
    def remaining_calls(self) -> int:
        return max(0, self._max_calls - self._calls)

    @property
    def max_calls(self) -> int:
        return self._max_calls

    def call(self, tool_name: str, args: dict[str, Any] | None = None) -> str:
        arguments = args or {}
        try:
            self._consume()
        except RuntimeError as error:
            return str(error)

        if tool_name == "sut_health":
            return self._request("GET", self._health_path)
        if tool_name == "http_request":
            method = str(arguments.get("method", "GET"))
            path = str(arguments.get("path", "/"))
            body = arguments.get("body")
            headers = arguments.get("headers")
            extra_headers = headers if isinstance(headers, dict) else None
            body_text = None if body is None else str(body)
            return self._request(method, path, body=body_text, headers=extra_headers)
        return f"Unknown probe tool: {tool_name}"

    def _consume(self) -> None:
        if self._calls >= self._max_calls:
            raise RuntimeError(f"L1 probing budget exhausted ({self._max_calls} calls).")
        self._calls += 1

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> str:
        normalized_method = method.upper()
        if normalized_method not in _ALLOWED_METHODS:
            return f"Rejected method {method!r}. Allowed: {sorted(_ALLOWED_METHODS)}"
        try:
            url = self._safe_url(path)
        except ValueError as error:
            return str(error)
        if normalized_method == "POST" and not urlsplit(url).path.startswith(
            self._post_allowed_prefixes
        ):
            if not self._post_allowed_prefixes:
                return "Rejected: POST is not needed for probing this API (reads are GET requests)."
            return (
                "Rejected: POST is only allowed for "
                + " and ".join(self._post_allowed_prefixes)
                + " during probing."
            )
        if normalized_method == "POST" and body and _FLUX_WRITE_RE.search(body):
            return "Rejected: Flux `to()` writes data; probing must stay read-only."

        request_headers: dict[str, str] = {}
        if self._auth_token:
            request_headers[self._auth_header] = (
                f"Token {self._auth_token}" if self._api == "influx" else self._auth_token
            )
        request_headers.update({str(key): str(value) for key, value in (headers or {}).items()})
        try:
            response = httpx.request(
                normalized_method,
                url,
                content=body.encode("utf-8") if body else None,
                headers=request_headers or None,
                timeout=self._timeout_seconds,
            )
        except httpx.HTTPError as error:
            return f"{normalized_method} {path} failed: {error}"

        text = response.text
        if len(text) > self._max_response_chars:
            text = text[: self._max_response_chars] + "\n...[truncated]"
        return (
            f"{normalized_method} {path} -> HTTP {response.status_code}\n{text or '(empty body)'}"
        )

    def _safe_url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            raise ValueError("Absolute URLs are not allowed. Use a path on the configured SUT.")
        relative = path if path.startswith("/") else f"/{path}"
        url = urljoin(self._base_url, relative.lstrip("/"))
        parts = urlsplit(url)
        if parts.scheme != self._base_parts.scheme or parts.netloc != self._base_parts.netloc:
            raise ValueError("Probe target escaped the configured SUT host.")
        return url
