"""Structured logging factory helpers."""

from __future__ import annotations

from flext_core import FlextProtocols as p, FlextTypes as t

from .flextlogger_part_04 import (
    FlextUtilitiesLogging as FlextUtilitiesLoggingPart04,
)


class FlextUtilitiesLogging(FlextUtilitiesLoggingPart04):
    @classmethod
    def fetch_logger(cls, name: str) -> p.Logger:
        """Fetch the canonical public logger wrapper."""
        return cls.create_module_logger(name)

    @classmethod
    def create_module_logger(
        cls,
        name: str,
        *,
        context: t.MappingKV[str, t.JsonPayload | None] | None = None,
    ) -> p.Logger:
        """Create a logger instance for a module."""
        cls.ensure_structlog_configured()
        merged_context: t.MutableJsonMapping = {}
        if context is not None:
            merged_context.update(
                cls.to_container_context({
                    key: value for key, value in context.items() if value is not None
                }),
            )
        logger: p.Logger = cls(name, context=merged_context)
        return logger


__all__: list[str] = ["FlextUtilitiesLogging"]
