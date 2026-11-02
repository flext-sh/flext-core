"""FlextProtocolsConfig - declarative config loader protocol (ADR-005).

Behavioral contract for a config loader. flext-core ships a minimal stdlib
implementation (``u.config_load`` / ``u.config_merge`` / ``u.config_env_override``);
``flext-cli`` provides the advanced multi-format loader that also satisfies this
protocol. Callers depend on the protocol, not the concrete class (DIP).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from flext_core._protocols import FlextProtocolsLogging
    from flext_core._typings import FlextTypesServices, FlextTypingBase


from typing import Protocol, runtime_checkable

from flext_core._protocols import FlextProtocolsBase

if TYPE_CHECKING:
    from pathlib import Path

    from flext_core import t
    from flext_core._protocols import FlextProtocolsResult


class FlextProtocolsConfig:
    """Protocols for declarative config loading and env override."""

    @runtime_checkable
    class ConfigLoader(Protocol):
        """Structural contract for loading a config source into a mapping."""

        def config_load(self, path: Path) -> FlextProtocolsResult.Result[t.JsonMapping]:
            """Load and parse a config source into a validated mapping."""
            ...

        def config_merge(
            self,
            base: t.JsonMapping,
            override: t.JsonMapping,
        ) -> t.JsonMapping:
            """Deep-merge ``override`` onto ``base``, returning a new mapping."""
            ...

    @runtime_checkable
    class AsyncLogJob(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def text(self) -> str | None:
            """Exact queued text, or None for a native flush barrier."""
            ...

        @property
        def completion(self) -> FlextTypesServices.LoggingWriteCompletion:
            """Native completion receiving this job's result or first failure."""
            ...

    @runtime_checkable
    class StructlogOptions(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def log_level(self) -> int | None:
            """Effective structlog level number."""
            ...

        @property
        def console_renderer(self) -> bool | None:
            """Use the console renderer."""
            ...

        @property
        def processing_stages(self) -> Sequence[FlextProtocolsLogging.LoggingStage]:
            """Ordered typed event stages executed before rendering."""
            ...

        @property
        def observers(self) -> Sequence[FlextProtocolsLogging.LoggingObserver]:
            """Typed observers notified with each sanitized event."""
            ...

        @property
        def handlers(self) -> Sequence[FlextProtocolsLogging.LoggingHandler]:
            """Native logging handlers attached to the runtime."""
            ...

        @property
        def redacted_keys(self) -> Sequence[str]:
            """Context keys whose values never reach the rendered event."""
            ...

        @property
        def redacted_values(self) -> Sequence[str]:
            """Exact values scrubbed from every rendered event."""
            ...

        @property
        def wrapper_class_factory(
            self,
        ) -> FlextProtocolsLogging.LoggerWrapperFactory | None:
            """Factory building the bound-logger wrapper class."""
            ...

        @property
        def logger_factory(self) -> FlextProtocolsLogging.OutputLoggerFactory | None:
            """Factory building the logger."""
            ...

        @property
        def cache_logger_on_first_use(self) -> bool:
            """Cache the logger on first use."""
            ...

        @property
        def async_logging(self) -> bool | None:
            """Write rendered log events through the asynchronous writer."""
            ...

    @runtime_checkable
    class ConfigDocument(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def data(self) -> FlextTypingBase.JsonMapping:
            """Parsed config mapping (execution parametrization)."""
            ...

        @property
        def source_path(self) -> str | None:
            """Absolute path of the config source."""
            ...

        @property
        def schema_ref(self) -> str | None:
            """Path of the JSON Schema validating this document."""
            ...

    @runtime_checkable
    class ModelDumpOptions(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def by_alias(self) -> bool | None:
            """Serialize using field aliases."""
            ...

        @property
        def exclude_none(self) -> bool | None:
            """Exclude None-valued fields."""
            ...

        @property
        def exclude_unset(self) -> bool | None:
            """Exclude fields not explicitly set."""
            ...

        @property
        def exclude_defaults(self) -> bool | None:
            """Exclude fields matching defaults."""
            ...

        @property
        def include(self) -> set[str] | None:
            """Whitelist of field names to include."""
            ...

        @property
        def exclude(self) -> set[str] | None:
            """Blacklist of field names to exclude."""
            ...

    @runtime_checkable
    class ParseOptions[T](FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def strict(self) -> bool | None:
            """Reject coercions; fail on type mismatch."""
            ...

        @property
        def case_insensitive(self) -> bool | None:
            """Normalize case before parsing."""
            ...

        @property
        def default(self) -> T | None:
            """Fallback value when parsing fails."""
            ...

        @property
        def default_factory(self) -> Callable[[], T] | None:
            """Factory producing fallback value."""
            ...

        @property
        def field_name(self) -> str | None:
            """Source field name for error context."""
            ...

    @runtime_checkable
    class RetryOptions(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def max_attempts(self) -> int | None:
            """Maximum number of retry attempts."""
            ...

        @property
        def delay_seconds(self) -> float | None:
            """Initial delay between retries in seconds."""
            ...

    @runtime_checkable
    class LoggingRecord(FlextProtocolsBase.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def message(self) -> str:
            """Rendered event message text."""
            ...

        @property
        def level(self) -> str:
            """Event level name."""
            ...

        @property
        def context(self) -> FlextTypingBase.JsonMapping:
            """Sanitized structured event context."""
            ...

        @property
        def exception(self) -> BaseException | None:
            """Original local-only cause; never serialized into logging context."""
            ...


__all__: list[str] = ["FlextProtocolsConfig"]
