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
from contextlib import suppress
from pathlib import Path
from typing import ClassVar

from flext_core import (
    FlextConstants as c,
    FlextExceptions as e,
    FlextProtocols as p,
    FlextResult as r,
    FlextRuntime,
    FlextTypes as t,
    FlextUtilitiesCollection,
    FlextUtilitiesLoggingConfig,
)


class FlextUtilitiesLoggingContext(FlextUtilitiesLoggingConfig):
    """Context binding, value normalization, and source path helpers."""

    _scoped_contexts: ClassVar[t.ScopedContainerRegistry]
    _level_contexts: ClassVar[t.ScopedContainerRegistry]

    @classmethod
    def bind_context(cls, scope: str, **context: t.JsonPayload) -> p.Result[bool]:
        """Bind context variables to a specific scope."""
        try:
            cls._scoped_contexts.setdefault(scope, {})
            current_context = {
                key: cls._to_container_value(value)
                for key, value in cls._scoped_contexts[scope].items()
            }
            incoming_context = {
                key: cls._to_container_value(value) for key, value in context.items()
            }
            current_context_obj: t.MappingKV[str, t.JsonValue] = {
                key: FlextRuntime.normalize_to_metadata(value)
                for key, value in current_context.items()
            }
            incoming_context_obj: t.MappingKV[str, t.JsonValue] = {
                key: FlextRuntime.normalize_to_metadata(value)
                for key, value in incoming_context.items()
            }
            merge_result = FlextUtilitiesCollection.merge_mappings(
                incoming_context_obj,
                current_context_obj,
                strategy=c.MergeStrategy.DEEP,
            )
            merged_value = merge_result.unwrap_or(current_context_obj)
            merged_context: t.MutableJsonMapping = {}
            for key, value in merged_value.items():
                merged_context[key] = cls._to_container_value(value)
            cls._scoped_contexts[scope] = merged_context
            cls.structlog().contextvars.bind_contextvars(**context)
            return r[bool].ok(True)
        except c.CONTEXT_EXCEPTIONS as exc:
            return e.fail_operation(
                f"{c.LoggingOperation.BIND_CONTEXT} '{scope}'",
                exc,
            )

    @classmethod
    def bind_global_context(cls, **context: t.JsonPayload) -> p.Result[bool]:
        """Bind context globally using structlog contextvars."""
        try:
            normalized_context = cls.to_container_context(context)
            cls.structlog().contextvars.bind_contextvars(**normalized_context)
            return r[bool].ok(True)
        except c.CONTEXT_EXCEPTIONS as exc:
            return e.fail_operation(c.LoggingOperation.BIND_GLOBAL, exc)

    @classmethod
    def clear_global_context(cls) -> p.Result[bool]:
        """Clear global logging context and cached scoped bindings."""
        try:
            cls.structlog().contextvars.clear_contextvars()
            cls._scoped_contexts.clear()
            cls._level_contexts.clear()
            return r[bool].ok(True)
        except c.CONTEXT_EXCEPTIONS as exc:
            return e.fail_operation(c.LoggingOperation.CLEAR_GLOBAL, exc)

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
        except c.CONTEXT_EXCEPTIONS as exc:
            return e.fail_operation(f"{c.LoggingOperation.CLEAR_SCOPE} '{scope}'", exc)

    @classmethod
    def unbind_global_context(cls, *keys: str) -> p.Result[bool]:
        """Unbind specific keys from global context."""
        try:
            unbind_keys: t.StrSequence = list(keys)
            cls.structlog().contextvars.unbind_contextvars(*unbind_keys)
            return r[bool].ok(True)
        except c.CONTEXT_EXCEPTIONS as exc:
            return e.fail_operation(c.LoggingOperation.UNBIND_GLOBAL, exc)

    @staticmethod
    def _to_container_value(
        value: t.LogValue | t.JsonValue | t.JsonPayload | None,
    ) -> t.JsonValue:
        """Normalize value to Container (internal helper)."""
        if isinstance(value, Exception):
            return str(value)
        return FlextRuntime.normalize_to_json_value(value)

    @staticmethod
    def to_container_context(
        context: t.MappingKV[str, t.LogValue | t.JsonValue | t.JsonPayload],
    ) -> t.JsonMapping:
        """Convert mapping to container context using normalization."""
        return {
            key: FlextUtilitiesLoggingContext._to_container_value(value)
            for key, value in context.items()
        }

    @classmethod
    def _to_scalar_context(
        cls,
        context: t.MappingKV[str, t.LogValue | t.JsonValue | t.JsonPayload | None],
    ) -> t.JsonMapping:
        validated: t.JsonMapping = t.json_mapping_adapter().validate_python(
            {key: cls._to_container_value(value) for key, value in context.items()},
        )
        return validated

    @staticmethod
    def _extract_class_name(frame: types.FrameType) -> str | None:
        """Extract class name from frame locals or qualname."""
        if c.FRAME_SELF_KEY in frame.f_locals:
            self_obj = frame.f_locals[c.FRAME_SELF_KEY]
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
        for _ in range(10):
            if any((current / marker).exists() for marker in c.WORKSPACE_ROOT_MARKERS):
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
            if relative_path.parts and relative_path.parts[0] == c.VENV_DIR_NAME:
                return None
            file_path = str(relative_path)
            line_number = caller_frame.f_lineno
            method_name = caller_frame.f_code.co_name
            class_name = FlextUtilitiesLoggingContext._extract_class_name(caller_frame)
            source_parts = [f"{file_path}:{line_number}"]
            if class_name and method_name:
                source_parts.append(f"{class_name}.{method_name}")
            elif method_name and method_name != c.MODULE_FRAME_NAME:
                source_parts.append(method_name)
            return " ".join(source_parts) if len(source_parts) > 1 else source_parts[0]
        except c.EXC_ATTR_RUNTIME_TYPE as exc:
            FlextUtilitiesLoggingContext._report_internal_logging_failure(
                c.LoggingOperation.GET_CALLER_SOURCE,
                exc,
            )
            return None

    @staticmethod
    def _calling_frame() -> types.FrameType | None:
        """Walk the stack backward and return the first frame outside the logging machinery.

        Generic: skips any frame whose source file path matches one of
        ``c.LOGGING_INTERNAL_PATH_FRAGMENTS``. The first frame outside is the
        true caller regardless of how many internal wrappers are involved.
        """
        frame = inspect.currentframe()
        if frame is None:
            return None
        skip = c.LOGGING_INTERNAL_PATH_FRAGMENTS
        while frame is not None:
            filename = frame.f_code.co_filename
            if not any(fragment in filename for fragment in skip):
                return frame
            frame = frame.f_back
        return None

    @staticmethod
    def _report_internal_logging_failure(operation: str, exc: Exception) -> None:
        with suppress(*c.CONTEXT_EXCEPTIONS):
            FlextUtilitiesLoggingContext.structlog().fetch_logger(
                c.LOGGER_NAME_FLEXT_CORE
            ).warning(
                c.LOG_INTERNAL_OPERATION_FAILED,
                operation=operation,
                error=exc,
                exception_type=exc.__class__.__name__,
                exception_message=str(exc),
            )

    @staticmethod
    def _should_include_stack_trace() -> bool:
        try:
            return logging.getLogger().getEffectiveLevel() <= logging.DEBUG
        except c.EXC_ATTR_RUNTIME_TYPE as exc:
            FlextUtilitiesLoggingContext._report_internal_logging_failure(
                c.LoggingOperation.SHOULD_INCLUDE_STACK,
                exc,
            )
            return True


__all__: t.MutableSequenceOf[str] = ["FlextUtilitiesLoggingContext"]
