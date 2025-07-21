"""Core domain abstractions.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import TypeVar

# Core domain abstractions - no direct imports from pydantic_base to avoid circular
# dependencies
# These are imported by __init__.py to provide the public API


# Domain Exceptions
class DomainError(Exception):
    """Base domain exception."""


class ValidationError(DomainError):
    """Domain validation error."""


class RepositoryError(DomainError):
    """Repository operation error."""


class NotFoundError(DomainError):
    """Entity not found error."""


class ServiceError(DomainError):
    """Service layer error."""

    def __init__(self, error_code: str, message: str) -> None:
        """Initialize service error with code and message."""
        super().__init__(message)
        self.error_code = error_code
        self.message = message


# Configuration and Connection Exceptions
class ConfigurationError(DomainError):
    """Configuration error - invalid or missing configuration."""


class ConnectionError(DomainError):  # noqa: A001
    """Connection error - failed to connect to external service."""


class DatabaseError(DomainError):
    """Database operation error."""


class AuthenticationError(DomainError):
    """Authentication error - invalid credentials."""


class AuthorizationError(DomainError):
    """Authorization error - insufficient permissions."""


# Data Processing Exceptions
class DataError(DomainError):
    """Data processing error."""


class SchemaError(DomainError):
    """Schema validation or compatibility error."""


class TransformationError(DomainError):
    """Data transformation error."""


# Integration Exceptions
class APIError(DomainError):
    """API operation error."""

    def __init__(
        self,
        status_code: int,
        message: str,
        response_data: dict[str, Any] | None = None,
    ) -> None:
        """Initialize API error with status code and message."""
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}


class ExternalServiceError(DomainError):
    """External service integration error."""


class TimeoutError(DomainError):  # noqa: A001
    """Operation timeout error."""


# LDAP Specific Exceptions
class LDAPError(DomainError):
    """LDAP operation error."""


class LDIFError(DomainError):
    """LDIF processing error."""


# Oracle Specific Exceptions
class OracleError(DomainError):
    """Oracle database error."""


class OICError(DomainError):
    """Oracle Integration Cloud error."""


class WMSError(DomainError):
    """WMS (Warehouse Management System) error."""


# Singer Protocol Exceptions
class SingerError(DomainError):
    """Singer protocol error."""


class TapError(DomainError):
    """Tap (data extraction) error."""


class TargetError(DomainError):
    """Target (data loading) error."""


# Meltano Exceptions
class MeltanoError(DomainError):
    """Meltano integration error."""


class PluginError(DomainError):
    """Plugin system error."""


# Modern type variables
T = TypeVar("T")
ID = TypeVar("ID")


class Repository[T, ID](ABC):
    """Repository interface."""

    @abstractmethod
    async def save(self, entity: T) -> T:
        """Save entity."""
        ...

    @abstractmethod
    async def get_by_id(self, entity_id: ID) -> T | None:
        """Get entity by ID.

        Args:
            entity_id: ID of entity to retrieve

        Returns:
            Entity if found, None otherwise

        Raises:
            RepositoryError: If get operation fails

        """
        ...

    @abstractmethod
    async def delete(self, entity_id: ID) -> bool:
        """Delete entity by ID."""
        ...

    @abstractmethod
    async def find_all(self) -> list[T]:
        """Find all entities.

        Returns:
            List of all entities

        Raises:
            RepositoryError: If find operation fails

        """
        ...

    @abstractmethod
    async def count(self) -> int:
        """Count total entities.

        Returns:
            Number of entities

        Raises:
            RepositoryError: If count operation fails

        """
        ...


# ==============================================================================
# DOMAIN SERVICE ABSTRACTIONS - DIP COMPLIANCE
# ==============================================================================


class DomainService[T](ABC):
    """Base domain service interface."""

    @abstractmethod
    async def validate(self, entity: T) -> bool:
        """Validate domain entity.

        Args:
            entity: Entity to validate

        Returns:
            True if valid, False otherwise

        Raises:
            ValidationError: If validation logic fails

        """
        ...


class EventPublisher(ABC):
    """Domain event publisher interface."""

    @abstractmethod
    async def publish(self, event: Any) -> None:
        """Publish domain event.

        Args:
            event: Domain event to publish

        Raises:
            ServiceError: If publish operation fails

        """
        ...


__all__ = [
    "APIError",
    "AuthenticationError",
    "AuthorizationError",
    "ConfigurationError",
    "ConnectionError",
    "DataError",
    "DatabaseError",
    # Domain exceptions
    "DomainError",
    "DomainService",
    "EventPublisher",
    "ExternalServiceError",
    "LDAPError",
    "LDIFError",
    "MeltanoError",
    "NotFoundError",
    "OICError",
    "OracleError",
    "PluginError",
    # DIP-compliant abstractions
    "Repository",
    "RepositoryError",
    "SchemaError",
    "ServiceError",
    "SingerError",
    "TapError",
    "TargetError",
    "TimeoutError",
    "TransformationError",
    "ValidationError",
    "WMSError",
]
