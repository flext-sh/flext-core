"""Contracts module for FLEXT Core."""

from flext_core.contracts.lifecycle_protocols import (
    AsyncContextManagerProtocol,
    HealthCheckProtocol,
    LifecycleProtocol,
    ResultProtocol,
    ServiceLifecycle,
    is_initializable,
    is_shutdownable,
)
from flext_core.contracts.repository_contracts import (
    EntityId,
    EntityInterface,
    RepositoryInterface,
    UnitOfWorkInterface,
)

__all__ = [
    "AsyncContextManagerProtocol",
    "EntityId",
    "EntityInterface",
    "HealthCheckProtocol",
    "LifecycleProtocol",
    "RepositoryInterface",
    "ResultProtocol",
    "ServiceLifecycle",
    "UnitOfWorkInterface",
    "is_initializable",
    "is_shutdownable",
]
