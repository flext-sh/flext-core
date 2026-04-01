"""DDD base models with Pydantic v2 validation and dispatcher-first CQRS.

Expose ``FlextModels`` as the façade for entities, value objects, aggregates,
commands, queries, and domain events that integrate directly with the
dispatcher-driven CQRS layer. Concrete implementations live in the
``_models`` subpackage and are organized for clear validation, serialization,
and event collection.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._models.base import FlextModelFoundation
from flext_core._models.collections import FlextModelsCollections
from flext_core._models.container import FlextModelsContainer
from flext_core._models.containers import FlextModelsContainers
from flext_core._models.context import FlextModelsContext
from flext_core._models.cqrs import FlextModelsCqrs
from flext_core._models.decorators import FlextModelsDecorators
from flext_core._models.dispatcher import FlextModelsDispatcher
from flext_core._models.domain_event import FlextModelsDomainEvent
from flext_core._models.entity import FlextModelsEntity
from flext_core._models.errors import FlextModelsErrors
from flext_core._models.exception_params import FlextModelsExceptionParams
from flext_core._models.generic import FlextGenericModels
from flext_core._models.handler import FlextModelsHandler
from flext_core._models.service import FlextModelsService
from flext_core._models.settings import FlextModelsConfig


class FlextModels(
    FlextModelFoundation,
    FlextModelsCollections,
    FlextModelsContainer,
    FlextModelsContainers,
    FlextModelsContext,
    FlextModelsCqrs,
    FlextModelsDecorators,
    FlextModelsDispatcher,
    FlextModelsDomainEvent,
    FlextModelsEntity,
    FlextModelsErrors,
    FlextGenericModels,
    FlextModelsHandler,
    FlextModelsService,
    FlextModelsConfig,
    FlextModelsExceptionParams,
):
    """Facade that groups DDD building blocks for CQRS-ready domains.

    Architecture: Domain layer helper
    Provides strongly typed base classes for entities, aggregates, commands,
    queries, and domain events so dispatcher handlers can enforce invariants,
    collect domain events, and validate inputs through Pydantic v2.

    Core concepts
    - Entity: Domain t.NormalizedValue with identity and lifecycle controls.
    - Value: Immutable value objects for pure operations.
    - AggregateRoot: Consistency boundary that aggregates events.
    - Command/Query: Message shapes consumed by dispatcher handlers.
    - DomainEvent: Stored and published through dispatcher pipelines.

    Pydantic v2 integration supplies BaseModel validation, computed fields,
    and JSON-ready serialization for all exported types.
    """


m = FlextModels

__all__ = ["FlextModels", "m"]
