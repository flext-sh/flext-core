"""Core domain abstractions.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
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


# Modern type variables
T = TypeVar("T")
ID = TypeVar("ID")


class Repository(ABC):
    """Repository interface."""

    @abstractmethod
    async def save(self, entity: T) -> T:
        """Save entity."""
        ...

    @abstractmethod
    async def get(self, entity_id: ID) -> T | None:
        """Get entity by ID."""
        ...

    @abstractmethod
    async def delete(self, entity_id: ID) -> bool:
        """Delete entity by ID."""
        ...

    @abstractmethod
    async def find_all(self) -> list[T]:
        """Find all entities."""
        ...
