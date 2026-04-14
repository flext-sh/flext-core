"""Structlog proxy context variable implementation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping

import structlog.contextvars

from flext_core import FlextModelsContextTokens, t


class FlextModelsContextProxyVar:
    """Namespace for structlog proxy context variable."""

    class StructlogProxyContextVar[T: t.ValueOrModel | t.ConfigMap]:
        """ContextVar-like proxy using structlog as backend (single source of truth).

        Delegates ALL operations to structlog's contextvar storage ensuring
        FlextContext.Variables and FlextLogger use THE SAME underlying storage.

        """

        def __init__(self, key: str, default: T | None = None) -> None:
            super().__init__()
            self._key = key
            self._default: T | None = default

        @staticmethod
        def reset(token: FlextModelsContextTokens.StructlogProxyToken) -> None:
            """Reset to previous value using token."""
            if token.previous_value is None:
                structlog.contextvars.unbind_contextvars(token.key)
            else:
                _ = structlog.contextvars.bind_contextvars(**{
                    token.key: token.previous_value,
                })

        def get(self) -> t.ValueOrModel | t.ConfigMap | None:
            """Get current value from structlog context."""
            contextvars_data = structlog.contextvars.get_contextvars()
            structlog_context: Mapping[str, t.ValueOrModel | t.ConfigMap] = (
                contextvars_data
            )
            if self._key not in structlog_context:
                return self._default
            value = structlog_context[self._key]
            if value is None:
                return self._default
            return value

        def set(self, value: T | None) -> FlextModelsContextTokens.StructlogProxyToken:
            """Set value in structlog context."""
            current_value = self.get()
            if value is not None:
                _ = structlog.contextvars.bind_contextvars(**{self._key: value})
            else:
                structlog.contextvars.unbind_contextvars(self._key)
            prev_value: t.ValueOrModel | t.ConfigMap | None = current_value
            return FlextModelsContextTokens.StructlogProxyToken(
                key=self._key,
                previous_value=prev_value,
            )


__all__: list[str] = ["FlextModelsContextProxyVar"]
