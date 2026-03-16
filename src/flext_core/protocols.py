"""Runtime-checkable structural typing protocols - Thin MRO facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._protocols import (
    FlextProtocolsBase,
    FlextProtocolsConfig,
    FlextProtocolsContext,
    FlextProtocolsDI,
    FlextProtocolsHandler,
    FlextProtocolsLogging,
    FlextProtocolsMetrics,
    FlextProtocolsResult,
    FlextProtocolsService,
)
from flext_core._protocols.introspection import _ProtocolIntrospection
from flext_core._protocols.metaclass import (
    FlextProtocolsMetaclassUtilities,
    ProtocolModel,
    ProtocolModelMeta,
    ProtocolSettings,
    _CombinedModelMeta,
)


class FlextProtocols(
    FlextProtocolsBase,
    FlextProtocolsContext,
    FlextProtocolsResult,
    FlextProtocolsConfig,
    FlextProtocolsDI,
    FlextProtocolsService,
    FlextProtocolsHandler,
    FlextProtocolsMetrics,
    FlextProtocolsLogging,
    FlextProtocolsMetaclassUtilities,
):
    """Runtime-checkable structural typing protocols for FLEXT framework."""

    ProtocolModelMeta = ProtocolModelMeta
    ProtocolModel = ProtocolModel
    ProtocolSettings = ProtocolSettings
    _CombinedModelMeta = _CombinedModelMeta
    _ProtocolIntrospection = _ProtocolIntrospection


p = FlextProtocols
__all__ = ["FlextProtocols", "p"]
