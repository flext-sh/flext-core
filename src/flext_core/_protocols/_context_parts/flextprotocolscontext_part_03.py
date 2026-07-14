"""FlextProtocolsContext - context and bootstrap protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from types import ModuleType

if TYPE_CHECKING:
    # mro-wkii.17.26 (codex): the context protocol is part of the root p
    # composition, so its postponed self-facade annotations cannot load p.
    from flext_core import p, t

from .flextprotocolscontext_part_02 import (
    FlextProtocolsContext as FlextProtocolsContextPart02,
)


class FlextProtocolsContext(FlextProtocolsContextPart02):
    class RuntimeBootstrapOptions(Protocol):
        """Runtime bootstrap options for service initialization."""

        settings: p.Settings | None
        settings_type: type | None
        settings_overrides: t.ScalarMapping | None
        context: p.Context | None
        dispatcher: p.Dispatcher | None
        registry: p.Registry | None
        subproject: str | None
        services: t.MappingKV[str, t.RegisterableService] | None
        factories: t.MappingKV[str, t.FactoryCallable] | None
        resources: t.MappingKV[str, t.ResourceCallable] | None
        container_overrides: t.ScalarMapping | None
        wire_modules: t.SequenceOf[ModuleType | str] | None
        wire_packages: t.StrSequence | None
        wire_classes: t.SequenceOf[type] | None


__all__: list[str] = ["FlextProtocolsContext"]
