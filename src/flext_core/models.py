"""DDD base models with Pydantic v2 validation and dispatcher-first CQRS.

Expose ``FlextModels`` as the façade for entities, value objects, aggregates,
commands, queries, and domain events that integrate directly with the
dispatcher-driven CQRS layer. Concrete implementations live in the
``models`` subpackage and are organized for clear validation, serialization,
and event collection.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from ._models.base import FlextModelsBase
from ._models.builder import FlextModelsBuilder
from ._models.collections import FlextModelsCollections
from ._models.config import FlextModelsConfig
from ._models.container import FlextModelsContainer
from ._models.containers import FlextModelsContainers
from ._models.context import FlextModelsContext
from ._models.cqrs import FlextModelsCqrs
from ._models.dispatcher import FlextModelsDispatcher
from ._models.domain_event import FlextModelsDomainEvent
from ._models.enforcement import FlextModelsEnforcement
from ._models.entity import FlextModelsEntity
from ._models.errors import FlextModelsErrors
from ._models.exception_params import FlextModelsExceptionParams
from ._models.handler import FlextModelsHandler
from ._models.namespace import FlextModelsNamespace
from ._models.project_metadata import FlextModelsProjectMetadata
from ._models.pydantic import FlextModelsPydantic
from ._models.registry import FlextModelsRegistry
from ._models.service import FlextModelsService
from ._models.settings import FlextModelsSettings


class FlextModels(
    FlextModelsBase,
    FlextModelsBuilder,
    FlextModelsCollections,
    FlextModelsContainers,
    FlextModelsConfig,
    FlextModelsContainer,
    FlextModelsContext,
    FlextModelsCqrs,
    FlextModelsDispatcher,
    FlextModelsDomainEvent,
    FlextModelsEnforcement,
    FlextModelsEntity,
    FlextModelsErrors,
    FlextModelsHandler,
    FlextModelsProjectMetadata,
    FlextModelsRegistry,
    FlextModelsService,
    FlextModelsSettings,
    FlextModelsExceptionParams,
    FlextModelsNamespace,
    FlextModelsPydantic,
):
    """Facade that groups DDD building blocks for CQRS-ready domains.

    Architecture: Domain layer helper
    Provides strongly typed base classes for entities, aggregates, commands,
    queries, and domain events so dispatcher handlers can enforce invariants,
    collect domain events, and validate inputs through Pydantic v2.

    Core concepts
    - Entity: Domain value object with identity and lifecycle controls.
    - Value: Immutable value objects for pure operations.
    - AggregateRoot: Consistency boundary that aggregates events.
    - Command/Query: Message shapes consumed by dispatcher handlers.
    - DomainEvent: Stored and published through dispatcher pipelines.

    Pydantic v2 integration supplies BaseModel validation, computed fields,
    and JSON-ready serialization for all exported types.
    """


m = FlextModels


__all__: list[str] = ["FlextModels", "m"]
