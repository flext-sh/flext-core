"""FLEXT CLI - Auth Abstraction Layer.

This is the ONLY file in the entire FLEXT ecosystem allowed to import Typer/Click.
All CLI framework functionality is exposed through this unified interface.

Implementation: Uses Typer as the backend framework. Since Typer is built on Click,
it generates Click-compatible commands internally.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import secrets

from flext_cli import c, m, p, r, s, settings, t, u
from flext_cli.services.file_tools import FlextCliFileTools


class FlextCliAuth(s):
    """Unified Typer/Click abstraction marker for the FLEXT CLI ecosystem.

    Container and logger are provided by x via MRO.
    """

    def validate_credentials(self, username: str, password: str) -> p.Result[bool]:
        """Validate direct username/password credentials."""
        return u.Cli.auth_validate_credentials(username, password)

    def save_auth_token(self, token: str) -> p.Result[bool]:
        """Persist an authentication token using the public file facade."""
        if not token.strip():
            return r[bool].fail(
                c.Cli.VALIDATION_MSG_FIELD_CANNOT_BE_EMPTY.format(field_name="token")
            )
        token_file_path = u.Cli.auth_token_file_path(settings.cli_token_file)
        return FlextCliFileTools.write_json_file(
            token_file_path, {c.Cli.DICT_KEY_AUTH_TOKEN: token}
        )

    def fetch_auth_token(self) -> p.Result[str]:
        """Load the persisted authentication token from the configured token file."""
        token_file_path = u.Cli.auth_token_file_path(settings.cli_token_file)
        return FlextCliFileTools.read_json_file(token_file_path).flat_map(
            u.Cli.auth_extract_token
        )

    def authenticate(self, credentials: t.StrMapping) -> p.Result[str]:
        """Authenticate with a token or username/password and persist the token."""
        try:
            payload = m.Cli.AuthCredentialsPayload.model_validate(credentials)
        except c.ValidationError:
            return r[str].fail(c.Cli.ERR_INVALID_CREDENTIALS)
        return self._resolve_token(payload).flat_map(self._persist_token)

    def _resolve_token(self, payload: m.Cli.AuthCredentialsPayload) -> p.Result[str]:
        if payload.token:
            return r[str].ok(payload.token)
        return (
            self
            .validate_credentials(payload.username, payload.password)
            .map_error(lambda err: err or c.Cli.ERR_INVALID_CREDENTIALS)
            .map(lambda _ok: secrets.token_urlsafe(32))
        )

    def _persist_token(self, token: str) -> p.Result[str]:
        return (
            self
            .save_auth_token(token)
            .map_error(
                lambda err: (
                    err
                    or c.Cli.ERR_AUTH_SAVE_FAILED.format(error=c.Cli.ERR_UNKNOWN_ERROR)
                )
            )
            .map(lambda _ok: token)
        )

    def clear_auth_tokens(self) -> p.Result[bool]:
        """Delete the configured authentication token file if present."""
        token_file = u.Cli.auth_token_file_path(settings.cli_token_file)
        if not token_file.exists():
            return r[bool].ok(True)
        return u.Cli.files_delete(token_file)


__all__: t.MutableSequenceOf[str] = ["FlextCliAuth"]
