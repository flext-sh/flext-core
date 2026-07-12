"""Runtime-checkable structural typing protocols - Thin MRO facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from ._protocols.base import FlextProtocolsBase
from ._protocols.config import FlextProtocolsConfig
from ._protocols.container import FlextProtocolsContainer
from ._protocols.context import FlextProtocolsContext
from ._protocols.handler import FlextProtocolsHandler
from ._protocols.logging import FlextProtocolsLogging
from ._protocols.project_metadata import FlextProtocolsProjectMetadata
from ._protocols.pydantic import FlextProtocolsPydantic
from ._protocols.registry import FlextProtocolsRegistry
from ._protocols.result import FlextProtocolsResult
from ._protocols.service import FlextProtocolsService
from ._protocols.settings import FlextProtocolsSettings


class FlextProtocols(
    FlextProtocolsBase,
    FlextProtocolsConfig,
    FlextProtocolsContext,
    FlextProtocolsResult,
    FlextProtocolsSettings,
    FlextProtocolsContainer,
    FlextProtocolsService,
    FlextProtocolsHandler,
    FlextProtocolsLogging,
    FlextProtocolsProjectMetadata,
    FlextProtocolsPydantic,
    FlextProtocolsRegistry,
):
    """Runtime-checkable structural typing protocols for FLEXT framework."""


__all__: list[str] = ["FlextProtocols", "p"]

p = FlextProtocols
