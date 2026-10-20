"""FlextProtocolsContext - context and bootstrap protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime
from typing import Protocol, Self, runtime_checkable

from flext_core import t
from flext_core._protocols._context_parts import flextprotocolscontext_part_02


class FlextProtocolsContext(flextprotocolscontext_part_02.FlextProtocolsContext):
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
        ) -> FlextProtocolsContext.ContextProxyToken: ...

        @staticmethod
        def reset(token: FlextProtocolsContext.ContextProxyToken) -> None: ...

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
        def metadata(self) -> FlextProtocolsContext.ContextMetadata: ...

        @property
        def hooks(self) -> t.ContextHookMap: ...

        @property
        def statistics(self) -> FlextProtocolsContext.ContextStatistics: ...

        @property
        def active(self) -> bool: ...

        @property
        def suspended(self) -> bool: ...

        @property
        def scope_vars(
            self,
        ) -> t.MappingKV[
            str, ContextVar[FlextProtocolsContext.MutableRootDict[t.JsonPayload] | None]
        ]: ...

        def model_copy(
            self,
            *,
            update: t.MappingKV[
                str,
                FlextProtocolsContext.ContextStatistics
                | t.MappingKV[
                    str,
                    ContextVar[
                        FlextProtocolsContext.MutableRootDict[t.JsonPayload] | None
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
        def settings(self) -> FlextProtocolsContext.Settings: ...

        @property
        def context(self) -> FlextProtocolsContext.Context: ...

        @property
        def container(self) -> FlextProtocolsContext.Container: ...

        @property
        def dispatcher(self) -> FlextProtocolsContext.Dispatcher: ...

    @runtime_checkable
    class RuntimeBootstrapOptions(Protocol):
        """Runtime bootstrap options a service base declares for its runtime."""

        @property
        def settings(self) -> FlextProtocolsContext.Settings | None: ...

        @property
        def settings_type(self) -> FlextProtocolsContext.SettingsType | None: ...

        @property
        def settings_overrides(self) -> t.MappingKV[str, t.JsonValue] | None: ...

        @property
        def context(self) -> FlextProtocolsContext.Context | None: ...

        @property
        def container(self) -> FlextProtocolsContext.Container | None: ...

        @property
        def dispatcher(self) -> FlextProtocolsContext.Dispatcher | None: ...


__all__: list[str] = ["FlextProtocolsContext"]
