"""Runtime-checkable structural typing protocols - Thin MRO facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._models.namespace import FlextModelsNamespace
from flext_core._protocols.base import FlextProtocolsBase
from flext_core._protocols.container import FlextProtocolsContainer
from flext_core._protocols.context import FlextProtocolsContext
from flext_core._protocols.handler import FlextProtocolsHandler
from flext_core._protocols.logging import FlextProtocolsLogging
from flext_core._protocols.pydantic import FlextProtocolsPydantic
from flext_core._protocols.registry import FlextProtocolsRegistry
from flext_core._protocols.result import FlextProtocolsResult
from flext_core._protocols.service import FlextProtocolsService
from flext_core._protocols.settings import FlextProtocolsSettings


class FlextProtocols(
    FlextProtocolsBase,
    FlextProtocolsContext,
    FlextProtocolsResult,
    FlextProtocolsSettings,
    FlextProtocolsContainer,
    FlextProtocolsService,
    FlextProtocolsHandler,
    FlextProtocolsLogging,
    FlextProtocolsPydantic,
    FlextProtocolsRegistry,
    FlextModelsNamespace,
):
    """Runtime-checkable structural typing protocols for FLEXT framework."""


__all__: list[str] = ["FlextProtocols", "p"]

p = FlextProtocols
