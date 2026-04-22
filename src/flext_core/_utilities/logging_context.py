"""Logging context binding and value normalization.

Extracted from FlextLogger as an MRO mixin to keep the facade under
the 200-line cap (AGENTS.md §3.1).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import logging
import types
from collections.abc import (
    Mapping,
)
from contextlib import suppress
from pathlib import Path
from typing import ClassVar

from flext_core import (
    FlextModelsPydantic,
    FlextRuntime,
    FlextUtilitiesCollection,
    FlextUtilitiesGuardsTypeCore,
    FlextUtilitiesLoggingConfig,
    c,
    e,
    p,
    r,
    t,
)


class FlextUtilitiesLoggingContext(FlextUtilitiesLoggingConfig):
    """Context binding, value normalization, and source path helpers."""

    _scoped_contexts: ClassVar[t.ScopedContainerRegistry]
    _level_contexts: ClassVar[t.ScopedContainerRegistry]

    @classmethod
    def bind_context(cls, scope: str, **context: t.RuntimeData) -> p.Result[bool]:
        """Bind context variables to a specific scope."""
        try:
            cls._scoped_contexts.setdefault(scope, {})
            current_context: t.FlatContainerMapping = {
                key: cls._to_container_value(value)
                for key, value in cls._scoped_contexts[scope].items()
            }
            incoming_context: t.FlatContainerMapping = {
                key: cls._to_container_value(value) for key, value in context.items()
            }
            current_context_obj: Mapping[str, t.MetadataValue] = {
                str(key): FlextRuntime.normalize_to_metadata(value)
                for key, value in current_context.items()
            }
            incoming_context_obj: Mapping[str, t.MetadataValue] = {
                str(key): FlextRuntime.normalize_to_metadata(value)
                for key, value in incoming_context.items()
            }
            merge_result = FlextUtilitiesCollection.merge_mappings(
                incoming_context_obj,
                current_context_obj,
                strategy="deep",
            )
            merged_value = merge_result.unwrap_or(current_context_obj)
            merged_context: t.MutableFlatContainerMapping = {}
            for key, value in merged_value.items():
                merged_context[str(key)] = cls._to_container_value(value)
            cls._scoped_contexts[scope] = merged_context
            cls.structlog().contextvars.bind_contextvars(**context)
            return r[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            return cls._fail_logging_operation(
                f"bind context for scope '{scope}'",
                exc,
            )

    @classmethod
    def bind_global_context(cls, **context: t.RuntimeData) -> p.Result[bool]:
        """Bind context globally using structlog contextvars."""
        try:
            normalized_context = cls._to_container_context(context)
            cls.structlog().contextvars.bind_contextvars(**normalized_context)
            return r[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            return cls._fail_logging_operation("bind global context", exc)

    @classmethod
    def clear_global_context(cls) -> p.Result[bool]:
        """Clear global logging context and cached scoped bindings."""
        try:
            cls.structlog().contextvars.clear_contextvars()
            cls._scoped_contexts.clear()
            cls._level_contexts.clear()
            return r[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            return cls._fail_logging_operation("clear global context", exc)

    @classmethod
    def clear_scope(cls, scope: str) -> p.Result[bool]:
        """Clear all context variables for a specific scope."""
        try:
            if scope in cls._scoped_contexts:
                keys = list(cls._scoped_contexts[scope].keys())
                if keys:
                    cls.structlog().contextvars.unbind_contextvars(*keys)
                cls._scoped_contexts[scope].clear()
            return r[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            return cls._fail_logging_operation(f"clear scope '{scope}'", exc)

    @classmethod
    def unbind_global_context(cls, *keys: str) -> p.Result[bool]:
        """Unbind specific keys from global context."""
        try:
            unbind_keys: t.StrSequence = [str(key) for key in keys]
            cls.structlog().contextvars.unbind_contextvars(*unbind_keys)
            return r[bool].ok(True)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as exc:
            return cls._fail_logging_operation("unbind global context", exc)

    @staticmethod
    def _to_container_value(
        value: t.LogValue | t.Container | t.RuntimeData | t.MetadataValue | None,
    ) -> t.Container:
        """Normalize value to Container (internal helper)."""
        if isinstance(value, Exception):
            return str(value)
        if value is None:
            return ""
        if isinstance(value, FlextModelsPydantic.BaseModel):
            return str(value.model_dump())
        if isinstance(value, p.Model):
            return str(value)
        model_dump_attr = getattr(value, "model_dump", None)
        if callable(model_dump_attr):
            return str(model_dump_attr())
        if FlextUtilitiesGuardsTypeCore.scalar(value) or isinstance(value, Path):
            return value
        flattened = FlextRuntime.normalize_to_metadata(value)
        return (
            flattened
            if FlextUtilitiesGuardsTypeCore.scalar(flattened)
            else str(flattened)
        )

    @staticmethod
    def _to_scalar_value(
        value: (
            t.LogValue
            | t.Container
            | t.RuntimeData
            | t.ContainerCarrier
            | t.MetadataValue
            | None
        ),
    ) -> t.Scalar:
        if value is None:
            return ""
        if isinstance(value, Exception):
            return str(value)
        if isinstance(value, FlextModelsPydantic.BaseModel):
            return str(value.model_dump())
        model_dump_attr = getattr(value, "model_dump", None)
        if callable(model_dump_attr):
            return str(model_dump_attr())
        if isinstance(value, (list, tuple, dict, Mapping)):
            return str(value)
        if FlextUtilitiesGuardsTypeCore.scalar(value):
            return value
        return str(value)

    @staticmethod
    def _to_container_context(
        context: Mapping[str, t.LogValue | t.Container | t.RuntimeData],
    ) -> t.FlatContainerMapping:
        """Convert mapping to container context using normalization."""
        return {
            key: FlextUtilitiesLoggingContext._to_container_value(value)
            for key, value in context.items()
        }

    @classmethod
    def _to_scalar_context(
        cls,
        context: Mapping[str, t.LogValue | t.Container | t.RuntimeData | None],
    ) -> t.ScalarMapping:
        return {key: cls._to_scalar_value(value) for key, value in context.items()}

    @staticmethod
    def _extract_class_name(frame: types.FrameType) -> str | None:
        """Extract class name from frame locals or qualname."""
        if "self" in frame.f_locals:
            self_obj = frame.f_locals["self"]
            if hasattr(self_obj, "__class__"):
                class_name: str = self_obj.__class__.__name__
                return class_name
        if hasattr(frame.f_code, "co_qualname"):
            qualname = frame.f_code.co_qualname
            if "." in qualname:
                parts = qualname.rsplit(".", 1)
                if len(parts) == c.LEVEL_PREFIX_PARTS_COUNT:
                    potential_class = parts[0]
                    if potential_class and potential_class[0].isupper():
                        return potential_class
        return None

    @staticmethod
    def _find_workspace_root(abs_path: Path) -> Path | None:
        """Find workspace root by looking for common markers."""
        current = abs_path.parent
        markers = ["pyproject.toml", ".git", "poetry.lock"]
        for _ in range(10):
            if any((current / marker).exists() for marker in markers):
                return current
            if current == current.parent:
                break
            current = current.parent
        return None

    @staticmethod
    def _caller_source_path() -> str | None:
        """Get source file path with line, class and method context."""
        try:
            caller_frame = FlextUtilitiesLoggingContext._calling_frame()
            if caller_frame is None:
                return None
            filename = caller_frame.f_code.co_filename
            abs_path = Path(filename).resolve()
            workspace_root = FlextUtilitiesLoggingContext._find_workspace_root(abs_path)
            if workspace_root is None:
                return None
            relative_path = abs_path.relative_to(workspace_root)
            if relative_path.parts and relative_path.parts[0] == ".venv":
                return None
            file_path = str(relative_path)
            line_number = caller_frame.f_lineno + 1
            method_name = caller_frame.f_code.co_name
            class_name = FlextUtilitiesLoggingContext._extract_class_name(caller_frame)
            source_parts = [f"{file_path}:{line_number}"]
            if class_name and method_name:
                source_parts.append(f"{class_name}.{method_name}")
            elif method_name and method_name != "<module>":
                source_parts.append(method_name)
            return " ".join(source_parts) if len(source_parts) > 1 else source_parts[0]
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            FlextUtilitiesLoggingContext._report_internal_logging_failure(
                "get_caller_source_path",
                exc,
            )
            return None

    @staticmethod
    def _calling_frame() -> types.FrameType | None:
        """Get the calling frame 4 levels up the stack."""
        frame = inspect.currentframe()
        if not frame:
            return None
        for _ in range(4):
            frame = frame.f_back
            if not frame:
                return None
        return frame

    @staticmethod
    def _report_internal_logging_failure(operation: str, exc: Exception) -> None:
        with suppress(AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            FlextUtilitiesLoggingContext.structlog().get_logger("flext_core").warning(
                "Internal logger operation failed",
                operation=operation,
                error=exc,
                exception_type=exc.__class__.__name__,
                exception_message=str(exc),
            )

    @staticmethod
    def _fail_logging_operation(operation: str, exc: Exception) -> p.Result[bool]:
        """Return the canonical failed result for a logger operation."""
        return e.fail_operation(operation, exc)

    @staticmethod
    def _should_include_stack_trace() -> bool:
        try:
            return logging.getLogger().getEffectiveLevel() <= logging.DEBUG
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            FlextUtilitiesLoggingContext._report_internal_logging_failure(
                "should_include_stack_trace",
                exc,
            )
            return True


__all__: list[str] = ["FlextUtilitiesLoggingContext"]
