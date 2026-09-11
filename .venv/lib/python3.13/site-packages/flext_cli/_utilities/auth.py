"""CLI auth helpers shared through ``u.Cli``."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from flext_cli import c, m, p, r, t


class FlextCliUtilitiesAuth:
    """Authentication utility helpers for services and commands."""

    @staticmethod
    def auth_token_file_path(token_file: str | None) -> Path:
        """Resolve configured token file path with canonical default."""
        if isinstance(token_file, str) and token_file.strip():
            return Path(token_file)
        return Path.home() / ".flext" / "auth_token.json"

    @staticmethod
    def auth_validate_credentials(username: str, password: str) -> p.Result[bool]:
        """Validate direct username/password credentials."""
        if not username.strip():
            return r[bool].fail("Username cannot be empty")
        if not password.strip():
            return r[bool].fail("Password cannot be empty")
        return r[bool].ok(True)

    @staticmethod
    def auth_extract_token(payload: t.JsonValue) -> p.Result[str]:
        """Extract auth token from JSON payload mapping."""
        if not isinstance(payload, Mapping):
            return r[str].fail("Token file must contain a mapping")
        try:
            token_payload = m.Cli.AuthCredentialsPayload.model_validate(payload)
        except c.ValidationError:
            return r[str].fail("Token file does not contain a valid token")
        if not token_payload.token:
            return r[str].fail("Token file does not contain a valid token")
        return r[str].ok(token_payload.token)


__all__: t.MutableSequenceOf[str] = ["FlextCliUtilitiesAuth"]
