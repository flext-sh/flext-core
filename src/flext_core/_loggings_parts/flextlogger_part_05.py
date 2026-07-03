"""Structured logging factory helpers."""

from __future__ import annotations

import inspect

from flext_core import FlextConstants as c, FlextProtocols as p, FlextTypes as t

from .flextlogger_part_04 import (
    FlextLogger as FlextLoggerPart04,
)


class FlextLogger(FlextLoggerPart04):
    @classmethod
    def fetch_logger(cls, name: str | None = None) -> p.Logger:
        """Fetch the canonical public logger wrapper."""
        resolved_name = name
        if resolved_name is None:
            frame = inspect.currentframe()
            if frame is not None and frame.f_back is not None:
                resolved_name = frame.f_back.f_globals.get("__name__", __name__)
            else:
                resolved_name = __name__
        return cls.create_module_logger(resolved_name)

    @classmethod
    def create_module_logger(
        cls,
        name: str = "flext",
        *,
        context: t.MappingKV[str, t.JsonPayload | None] | None = None,
        **legacy_context: t.JsonPayload,
    ) -> p.Logger:
        """Create a logger instance for a module."""
        cls.ensure_structlog_configured()
        merged_context: t.MutableJsonMapping = dict(
            cls.to_container_context({
                key: value for key, value in legacy_context.items() if value is not None
            }),
        )
        if context is not None:
            merged_context.update(
                cls.to_container_context({
                    key: value for key, value in context.items() if value is not None
                }),
            )
        logger: p.Logger = cls(name, context=merged_context)
        return logger

    @classmethod
    def for_container(
        cls,
        container: p.Container,
        level: str | None = None,
        **context: t.JsonPayload,
    ) -> p.Logger:
        """Create logger configured for a specific container."""
        if level is None:
            settings: p.Settings | None
            try:
                settings = container.settings
            except c.EXC_ATTR_RUNTIME_TYPE:
                settings = None
            level = getattr(settings, "log_level", "INFO")
        logger: p.Logger = cls(f"container_{id(container)}")
        if context:
            return logger.bind(**context)
        return logger


__all__: list[str] = ["FlextLogger"]
