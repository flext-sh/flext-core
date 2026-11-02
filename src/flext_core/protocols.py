# AUTO-GENERATED FILE — Regenerate with: make gen
"""Runtime-checkable structural typing protocols - Thin MRO facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._protocols import (
    FlextProtocolsBase,
    FlextProtocolsCollectionModels,
    FlextProtocolsConfig,
    FlextProtocolsContainer,
    FlextProtocolsContainers,
    FlextProtocolsContext,
    FlextProtocolsCqrs,
    FlextProtocolsDomainEvent,
    FlextProtocolsEntity,
    FlextProtocolsErrors,
    FlextProtocolsFlextContext,
    FlextProtocolsFlextMixins,
    FlextProtocolsHandler,
    FlextProtocolsLogging,
    FlextProtocolsOptions,
    FlextProtocolsProjectMetadata,
    FlextProtocolsPydantic,
    FlextProtocolsRegistry,
    FlextProtocolsResult,
    FlextProtocolsService,
    FlextProtocolsSettings,
)
from flext_core._protocols.exception_params import FlextProtocolsExceptionParams


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
    FlextProtocolsCollectionModels,
    FlextProtocolsContainers,
    FlextProtocolsCqrs,
    FlextProtocolsDomainEvent,
    FlextProtocolsEntity,
    FlextProtocolsErrors,
    FlextProtocolsFlextContext,
    FlextProtocolsFlextMixins,
    FlextProtocolsOptions,
    FlextProtocolsExceptionParams,
):
    """Runtime-checkable structural typing protocols for FLEXT framework.

    The ``p`` letter alias binds to this class; protocols compose via MRO so
    consumers can reference any contract as ``p.<Protocol>`` while
    implementations are checked structurally rather than by inheritance.
    """


p = FlextProtocols
__all__: list[str] = ["FlextProtocols", "p"]
