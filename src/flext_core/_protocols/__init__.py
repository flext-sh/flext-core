# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Protocols package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._protocols.base import FlextProtocolsBase
    from flext_core._protocols.collection_models import FlextProtocolsCollectionModels
    from flext_core._protocols.config import FlextProtocolsConfig
    from flext_core._protocols.container import FlextProtocolsContainer
    from flext_core._protocols.containers import FlextProtocolsContainers
    from flext_core._protocols.context import FlextProtocolsContext
    from flext_core._protocols.cqrs import FlextProtocolsCqrs
    from flext_core._protocols.domain_event import FlextProtocolsDomainEvent
    from flext_core._protocols.entity import FlextProtocolsEntity
    from flext_core._protocols.errors import FlextProtocolsErrors
    from flext_core._protocols.exception_params import FlextProtocolsExceptionParams
    from flext_core._protocols.flext_context import FlextProtocolsFlextContext
    from flext_core._protocols.flext_mixins import FlextProtocolsFlextMixins
    from flext_core._protocols.handler import FlextProtocolsHandler
    from flext_core._protocols.loggings import FlextProtocolsLogging
    from flext_core._protocols.options import FlextProtocolsOptions
    from flext_core._protocols.project_metadata import FlextProtocolsProjectMetadata
    from flext_core._protocols.pydantic import FlextProtocolsPydantic
    from flext_core._protocols.registry import FlextProtocolsRegistry
    from flext_core._protocols.result import FlextProtocolsResult
    from flext_core._protocols.service import FlextProtocolsService
    from flext_core._protocols.settings import FlextProtocolsSettings


__all__: tuple[str, ...] = (
    "FlextProtocolsBase",
    "FlextProtocolsCollectionModels",
    "FlextProtocolsConfig",
    "FlextProtocolsContainer",
    "FlextProtocolsContainers",
    "FlextProtocolsContext",
    "FlextProtocolsCqrs",
    "FlextProtocolsDomainEvent",
    "FlextProtocolsEntity",
    "FlextProtocolsErrors",
    "FlextProtocolsExceptionParams",
    "FlextProtocolsFlextContext",
    "FlextProtocolsFlextMixins",
    "FlextProtocolsHandler",
    "FlextProtocolsLogging",
    "FlextProtocolsOptions",
    "FlextProtocolsProjectMetadata",
    "FlextProtocolsPydantic",
    "FlextProtocolsRegistry",
    "FlextProtocolsResult",
    "FlextProtocolsService",
    "FlextProtocolsSettings",
)

install_lazy_exports(
    __name__,
    globals(),
    {
        "FlextProtocolsBase": ".base",
        "FlextProtocolsCollectionModels": ".collection_models",
        "FlextProtocolsConfig": ".config",
        "FlextProtocolsContainer": ".container",
        "FlextProtocolsContainers": ".containers",
        "FlextProtocolsContext": ".context",
        "FlextProtocolsCqrs": ".cqrs",
        "FlextProtocolsDomainEvent": ".domain_event",
        "FlextProtocolsEntity": ".entity",
        "FlextProtocolsErrors": ".errors",
        "FlextProtocolsExceptionParams": ".exception_params",
        "FlextProtocolsFlextContext": ".flext_context",
        "FlextProtocolsFlextMixins": ".flext_mixins",
        "FlextProtocolsHandler": ".handler",
        "FlextProtocolsLogging": ".loggings",
        "FlextProtocolsOptions": ".options",
        "FlextProtocolsProjectMetadata": ".project_metadata",
        "FlextProtocolsPydantic": ".pydantic",
        "FlextProtocolsRegistry": ".registry",
        "FlextProtocolsResult": ".result",
        "FlextProtocolsService": ".service",
        "FlextProtocolsSettings": ".settings",
    },
    public_exports=__all__,
)
