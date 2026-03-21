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

from flext_core import (
    FlextGenericModels,
    FlextModelFoundation,
    FlextModelsCollections,
    FlextModelsConfig,
    FlextModelsContainer,
    FlextModelsContainers,
    FlextModelsContext,
    FlextModelsCqrs,
    FlextModelsDecorators,
    FlextModelsDispatcher,
    FlextModelsEntity,
    FlextModelsHandler,
    FlextModelsResult,
    FlextModelsService,
)


class FlextModels(
    FlextModelFoundation,
    FlextModelsCollections,
    FlextModelsContainer,
    FlextModelsContainers,
    FlextModelsContext,
    FlextModelsCqrs,
    FlextModelsDecorators,
    FlextModelsDispatcher,
    FlextModelsEntity,
    FlextGenericModels,
    FlextModelsHandler,
    FlextModelsResult,
    FlextModelsService,
    FlextModelsConfig,
):
    """Facade that groups DDD building blocks for CQRS-ready domains.

    Architecture: Domain layer helper
    Provides strongly typed base classes for entities, aggregates, commands,
    queries, and domain events so dispatcher handlers can enforce invariants,
    collect domain events, and validate inputs through Pydantic v2.

    Core concepts
    - Entity: Domain object with identity and lifecycle controls.
    - Value: Immutable value objects for pure operations.
    - AggregateRoot: Consistency boundary that aggregates events.
    - Command/Query: Message shapes consumed by dispatcher handlers.
    - DomainEvent: Stored and published through dispatcher pipelines.

    Pydantic v2 integration supplies BaseModel validation, computed fields,
    and JSON-ready serialization for all exported types.
    """

    pass


m = FlextModels

__all__ = ["FlextModels", "m"]
