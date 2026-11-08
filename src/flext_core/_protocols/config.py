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


from typing import Protocol, runtime_checkable

from flext_core import t
from flext_core._protocols import FlextProtocolsBase

if TYPE_CHECKING:
    from pathlib import Path

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
    class AsyncLogJob(FlextProtocolsBase.ContractModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def text(self) -> str | None:
            """Exact queued text, or None for a native flush barrier."""
            ...

        @property
        def completion(self) -> t.LoggingWriteCompletion:
            """Native completion receiving this job's result or first failure."""
            ...

    @runtime_checkable
    class StructlogOptions(FlextProtocolsBase.ArbitraryTypesModel, Protocol):
        """Structural data contract projected from the owning model."""

        log_level: int | None
        console_renderer: bool | None
        processing_stages: Sequence[FlextProtocolsLogging.LoggingStage]
        observers: Sequence[FlextProtocolsLogging.LoggingObserver]
        handlers: Sequence[FlextProtocolsLogging.LoggingHandler]
        redacted_keys: Sequence[str]
        redacted_values: Sequence[str]
        wrapper_class_factory: FlextProtocolsLogging.LoggerWrapperFactory | None
        logger_factory: FlextProtocolsLogging.OutputLoggerFactory | None
        cache_logger_on_first_use: bool
        async_logging: bool | None

    @runtime_checkable
    class ConfigDocument(FlextProtocolsBase.FrozenModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def data(self) -> t.JsonMapping:
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
    class ModelDumpOptions(FlextProtocolsBase.FlexibleInternalModel, Protocol):
        """Structural data contract projected from the owning model."""

        by_alias: bool | None
        exclude_none: bool | None
        exclude_unset: bool | None
        exclude_defaults: bool | None
        include: set[str] | None
        exclude: set[str] | None
        exclude_computed_fields: bool | None
        round_trip: bool | None

    @runtime_checkable
    class ParseOptions[T](FlextProtocolsBase.FlexibleInternalModel, Protocol):
        """Structural data contract projected from the owning model."""

        strict: bool | None
        case_insensitive: bool | None
        default: T | None
        default_factory: Callable[[], T] | None
        field_name: str | None

    @runtime_checkable
    class RetryOptions(FlextProtocolsBase.FlexibleInternalModel, Protocol):
        """Structural data contract projected from the owning model."""

        max_attempts: int | None
        delay_seconds: float | None

    @runtime_checkable
    class LoggingRecord(FlextProtocolsBase.ContractModel, Protocol):
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
        def context(self) -> t.JsonMapping:
            """Sanitized structured event context."""
            ...

        @property
        def exception(self) -> BaseException | None:
            """Original local-only cause; never serialized into logging context."""
            ...


__all__: list[str] = ["FlextProtocolsConfig"]
