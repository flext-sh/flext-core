"""Git remote credential redaction."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_SSH_SCHEMES = frozenset({"ssh", "git+ssh"})
_SENSITIVE_QUERY_KEYS = frozenset({
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "bearer",
    "client_secret",
    "id_token",
    "jwt",
    "key",
    "oauth_token",
    "password",
    "passwd",
    "private_key",
    "private_token",
    "refresh_token",
    "secret",
    "token",
})


def _redact_component(component: str) -> str:
    if not component or "=" not in component:
        return component
    pairs = parse_qsl(component, keep_blank_values=True)
    redacted = [
        (key, "REDACTED" if key.casefold() in _SENSITIVE_QUERY_KEYS else value)
        for key, value in pairs
    ]
    return urlencode(redacted) if redacted != pairs else component


def redact_origin_remote(url: str) -> str:
    """Remove credentials and sensitive query values from a remote URL."""
    value = url.strip()
    parsed = urlsplit(value)
    userinfo, marker, host = parsed.netloc.rpartition("@")
    netloc = (
        f"{userinfo}@{host}"
        if marker and parsed.scheme in _SSH_SCHEMES and ":" not in userinfo
        else host
        if marker
        else parsed.netloc
    )
    query = _redact_component(parsed.query)
    fragment = _redact_component(parsed.fragment)
    return (
        value
        if (netloc, query, fragment) == (parsed.netloc, parsed.query, parsed.fragment)
        else urlunsplit((parsed.scheme, netloc, parsed.path, query, fragment))
    )


def canonical_origin_remote(url: str) -> str:
    """Return one transport-stable repository identity for a remote URL."""
    value = redact_origin_remote(url)
    parsed = urlsplit(value)
    if parsed.scheme in {"git", "git+ssh", "http", "https", "ssh"} and parsed.netloc:
        path = parsed.path.rstrip("/").removesuffix(".git")
        return urlunsplit((
            parsed.scheme,
            parsed.netloc,
            path,
            parsed.query,
            parsed.fragment,
        ))
    if "@" in value.partition(":")[0]:
        return value.rstrip("/").removesuffix(".git")
    return value


__all__: list[str] = ["canonical_origin_remote", "redact_origin_remote"]
