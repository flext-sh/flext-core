"""Core domain module for FLEXT framework.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module contains the core domain logic and exceptions that are technology-agnostic
and serve as the foundation for all FLEXT projects.

ONLY ABSTRACT/GENERIC DOMAIN LOGIC - No technology-specific implementations.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import TypeVar

# Type variables for generic repository interface
T = TypeVar("T")
ID = TypeVar("ID")

# ==============================================================================
# DOMAIN EXCEPTIONS - ABSTRACT AND GENERIC
# ==============================================================================


class DomainError(Exception):
    """Base domain exception for all FLEXT domain errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(DomainError):
    """Domain validation error for input validation failures."""


class RepositoryError(DomainError):
    """Repository operation error for data access failures."""


class NotFoundError(DomainError):
    """Entity not found error for missing resources."""


class ServiceError(DomainError):
    """Service layer error for business logic failures."""

    def __init__(
        self,
        error_code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.error_code = error_code


class ServiceConnectionError(DomainError):
    """Service connection error for external service communication failures."""


class DatabaseError(DomainError):
    """Database operation error for database-related failures."""


class AuthenticationError(DomainError):
    """Authentication error for invalid credentials."""


class AuthorizationError(DomainError):
    """Authorization error for insufficient permissions."""


class DataError(DomainError):
    """Data processing error for data transformation/validation failures."""


class TransformationError(DomainError):
    """Data transformation error for transformation pipeline failures."""


class SchemaError(DomainError):
    """Schema validation or compatibility error."""


class ConfigurationError(DomainError):
    """Configuration error for invalid or missing configuration."""


class APIError(DomainError):
    """API error for external API communication failures."""


class ExternalServiceError(DomainError):
    """External service error for third-party service failures."""


# ==============================================================================
# ABSTRACT DOMAIN SERVICES
# ==============================================================================


class DomainService(ABC):
    """Abstract base class for domain services."""

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the domain service logic."""


class EventPublisher(ABC):
    """Abstract event publisher for domain events."""

    @abstractmethod
    def publish(self, event: Any) -> None:
        """Publish a domain event."""


class Repository[T, ID](ABC):
    """Abstract repository interface for data access."""

    @abstractmethod
    async def save(self, entity: T) -> T:
        """Save an entity."""

    @abstractmethod
    async def find_by_id(self, entity_id: ID) -> T | None:
        """Find an entity by ID."""

    @abstractmethod
    async def delete(self, entity_id: ID) -> bool:
        """Delete an entity by ID."""

    @abstractmethod
    async def find_all(self) -> list[T]:
        """Find all entities."""

    @abstractmethod
    async def count(self) -> int:
        """Count total entities."""


# ==============================================================================
# EXPORTS - ONLY ABSTRACT/GENERIC DOMAIN ELEMENTS
# ==============================================================================

__all__ = [
    # Type variables
    "ID",
    "APIError",
    "AuthenticationError",
    "AuthorizationError",
    "ConfigurationError",
    "DataError",
    "DatabaseError",
    # Domain exceptions
    "DomainError",
    # Abstract services
    "DomainService",
    "EventPublisher",
    "ExternalServiceError",
    "NotFoundError",
    "Repository",
    "RepositoryError",
    "SchemaError",
    "ServiceConnectionError",
    "ServiceError",
    "T",
    "TransformationError",
    "ValidationError",
]
