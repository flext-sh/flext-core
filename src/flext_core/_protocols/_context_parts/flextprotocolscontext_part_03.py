"""FlextProtocolsContext - context and bootstrap protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core._protocols._context_parts import flextprotocolscontext_part_02

if TYPE_CHECKING:
    from flext_core import p, t


class FlextProtocolsContext(flextprotocolscontext_part_02.FlextProtocolsContext):
    @runtime_checkable
    class ServiceRuntime(Protocol):
        """Actual scoped runtime selected by a component's composition root."""

        @property
        def settings(self) -> p.Settings: ...

        @property
        def context(self) -> p.Context: ...

        @property
        def container(self) -> p.Container: ...

        @property
        def dispatcher(self) -> p.Dispatcher: ...

    @runtime_checkable
    class RuntimeBootstrapOptions(Protocol):
        """Runtime bootstrap options a service base declares for its runtime."""

        @property
        def settings(self) -> p.Settings | None: ...

        @property
        def settings_type(self) -> t.SettingsClass | None: ...

        @property
        def settings_overrides(self) -> t.ScalarMapping | None: ...

        @property
        def context(self) -> p.Context | None: ...

        @property
        def container(self) -> p.Container | None: ...

        @property
        def dispatcher(self) -> p.Dispatcher | None: ...


__all__: list[str] = ["FlextProtocolsContext"]
