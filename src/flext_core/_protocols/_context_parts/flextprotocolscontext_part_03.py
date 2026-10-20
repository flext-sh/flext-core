"""FlextProtocolsContext - context and bootstrap protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from flext_core import t
from flext_core._protocols import FlextProtocolsHandler, FlextProtocolsSettings
from flext_core._protocols._context_parts.flextprotocolscontext_part_02 import (
    FlextProtocolsContextPart02,
)

if TYPE_CHECKING:
    from flext_core._protocols import FlextProtocolsContainer


class FlextProtocolsContextPart03(FlextProtocolsContextPart02):
    @runtime_checkable
    class ContextProxyToken(Protocol):
        """Stored structlog value needed to restore a context variable."""

        @property
        def key(self) -> str: ...

        @property
        def previous_value(self) -> t.JsonPayload | datetime | None: ...

    @runtime_checkable
    class ContextProxy[ValueT: t.JsonPayload | datetime](Protocol):
        """Context-variable operations sharing structlog's backing storage."""

        def get(self) -> t.JsonPayload | datetime | None: ...

        def set(
            self, value: ValueT | None
        ) -> FlextProtocolsContextPart03.ContextProxyToken: ...

        @staticmethod
        def reset(token: FlextProtocolsContextPart03.ContextProxyToken) -> None: ...

    @runtime_checkable
    class ContextMetadata(Protocol):
        """Metadata snapshot copied when creating a context runtime."""

        @property
        def attributes(self) -> t.JsonMapping: ...

        def model_copy(self, *, deep: bool = False) -> Self: ...

    @runtime_checkable
    class ContextStatistics(Protocol):
        """Validated counters and operation metrics."""

        @property
        def sets(self) -> int: ...

        @property
        def gets(self) -> int: ...

        @property
        def removes(self) -> int: ...

        @property
        def clears(self) -> int: ...

        @property
        def operations(self) -> t.JsonMapping: ...

        def model_copy(
            self,
            *,
            update: t.MappingKV[str, t.JsonPayload] | None = None,
            deep: bool = False,
        ) -> Self: ...

    @runtime_checkable
    class ContextRuntimeState(Protocol):
        """Data capabilities consumed by context runtime helpers."""

        @property
        def metadata(self) -> FlextProtocolsContextPart03.ContextMetadata: ...

        @property
        def hooks(self) -> t.ContextHookMap: ...

        @property
        def statistics(self) -> FlextProtocolsContextPart03.ContextStatistics: ...

        @property
        def active(self) -> bool: ...

        @property
        def suspended(self) -> bool: ...

        @property
        def scope_vars(
            self,
        ) -> t.MappingKV[
            str,
            ContextVar[FlextProtocolsContainer.MutableRootDict[t.JsonPayload] | None],
        ]: ...

        def model_copy(
            self,
            *,
            update: t.MappingKV[
                str,
                FlextProtocolsContextPart03.ContextStatistics
                | t.MappingKV[
                    str,
                    ContextVar[
                        FlextProtocolsContainer.MutableRootDict[t.JsonPayload] | None
                    ],
                ],
            ]
            | None = None,
            deep: bool = False,
        ) -> Self: ...

    @runtime_checkable
    class ServiceRuntime(Protocol):
        """Actual scoped runtime selected by a component's composition root."""

        @property
        def settings(self) -> FlextProtocolsSettings.Settings: ...

        @property
        def context(self) -> FlextProtocolsContextPart03.Context: ...

        @property
        def container(self) -> FlextProtocolsContainer.Container: ...

        @property
        def dispatcher(self) -> FlextProtocolsHandler.Dispatcher: ...

    @runtime_checkable
    class RuntimeBootstrapOptions(Protocol):
        """Runtime bootstrap options a service base declares for its runtime."""

        @property
        def settings(self) -> FlextProtocolsSettings.Settings | None: ...

        @property
        def settings_type(self) -> FlextProtocolsSettings.SettingsType | None: ...

        @property
        def settings_overrides(self) -> t.MappingKV[str, t.JsonValue] | None: ...

        @property
        def context(self) -> FlextProtocolsContextPart03.Context | None: ...

        @property
        def container(self) -> FlextProtocolsContainer.Container | None: ...

        @property
        def dispatcher(self) -> FlextProtocolsHandler.Dispatcher | None: ...


__all__: list[str] = ["FlextProtocolsContextPart03"]
