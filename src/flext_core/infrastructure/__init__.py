"""Infrastructure layer - DIP-compliant concrete implementations.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Dependency Inversion Principle compliant infrastructure layer that provides
concrete implementations of domain abstractions. Infrastructure depends on
domain abstractions, never the reverse.
"""

from __future__ import annotations

from flext_core.infrastructure.adapters import StandardLibraryLoggerAdapter
from flext_core.infrastructure.adapters import create_logger_adapter

# gRPC base classes moved to flext-grpc module for better separation
from flext_core.infrastructure.memory import InMemoryRepository
from flext_core.infrastructure.protocols import BaseAsyncContextManager
from flext_core.infrastructure.protocols import BaseInfrastructureService
from flext_core.infrastructure.protocols import CacheProtocol
from flext_core.infrastructure.protocols import ConfigurationProviderProtocol
from flext_core.infrastructure.protocols import ConnectionProtocol
from flext_core.infrastructure.protocols import EventPublishingProtocol
from flext_core.infrastructure.protocols import LoggingProtocol
from flext_core.infrastructure.protocols import PersistenceProtocol
from flext_core.infrastructure.protocols import SerializationProtocol

__all__ = [
    "BaseAsyncContextManager",
    # gRPC service patterns moved to flext-grpc
    # Abstract base classes
    "BaseInfrastructureService",
    "CacheProtocol",
    "ConfigurationProviderProtocol",
    "ConnectionProtocol",
    "EventPublishingProtocol",
    # Concrete implementations
    "InMemoryRepository",
    "LoggingProtocol",
    # DIP-compliant protocols (abstractions)
    "PersistenceProtocol",
    "SerializationProtocol",
    # Adapters
    "StandardLibraryLoggerAdapter",
    "create_logger_adapter",
]
