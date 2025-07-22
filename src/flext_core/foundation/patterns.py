"""Architectural Patterns - Fundamental Design Patterns.

Provides the core architectural patterns used throughout FLEXT.
These are abstract pattern implementations that ensure consistency.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING
from typing import Any
from typing import Self

if TYPE_CHECKING:
    import types
    from collections.abc import Callable


# Type variables for pattern generics
class ResultPattern[T, TError: Exception](abc.ABC):
    """Result Pattern for Type-Safe Error Handling.

    Provides a way to handle operations that can fail without
    using exceptions for control flow.

    🎯 PRINCIPLES:
    - Type-safe error handling
    - Explicit failure cases
    - Composable operations
    - No hidden exceptions

    Example:
        result = some_operation()
        if result.success:
            value = result.value
        else:
            error = result.error

    """

    @property
    @abc.abstractmethod
    def is_success(self) -> bool:
        """True if the operation succeeded."""

    @property
    @abc.abstractmethod
    def is_failure(self) -> bool:
        """True if the operation failed."""

    @property
    @abc.abstractmethod
    def value(self) -> T:
        """The success value. Only access if is_success is True."""

    @property
    @abc.abstractmethod
    def error(self) -> TError:
        """The error value. Only access if is_failure is True."""

    @abc.abstractmethod
    def map(self, func: Callable[[T], Any]) -> ResultPattern[Any, TError]:
        """Transform the value if successful, leave error unchanged."""

    @abc.abstractmethod
    def flat_map(
        self,
        func: Callable[[T], ResultPattern[Any, TError]],
    ) -> ResultPattern[Any, TError]:
        """Chain operations that can fail."""


class SpecificationPattern[T](abc.ABC):
    """Specification Pattern for Business Rules.

    Encapsulates business rules in a composable and testable way.
    Useful for validation, filtering, and selection.

    🎯 PRINCIPLES:
    - Encapsulates business rules
    - Composable with AND, OR, NOT
    - Testable in isolation
    - Reusable across contexts

    Example:
        spec = IsActiveUser() & HasPermission("read")
        if spec.is_satisfied_by(user):
            # Allow access

    """

    @abc.abstractmethod
    def is_satisfied_by(self, candidate: T) -> bool:
        """Check if the candidate satisfies this specification."""

    def and_(self, other: SpecificationPattern[T]) -> SpecificationPattern[T]:
        """Combine specifications with logical AND."""
        return AndSpecification(self, other)

    def or_(self, other: SpecificationPattern[T]) -> SpecificationPattern[T]:
        """Combine specifications with logical OR."""
        return OrSpecification(self, other)

    def not_(self) -> SpecificationPattern[T]:
        """Negate this specification with logical NOT."""
        return NotSpecification(self)

    def __and__(self, other: SpecificationPattern[T]) -> SpecificationPattern[T]:
        """Allow using & operator for AND operations."""
        return self.and_(other)

    def __or__(self, other: SpecificationPattern[T]) -> SpecificationPattern[T]:
        """Allow using | operator for OR operations."""
        return self.or_(other)

    def __invert__(self) -> SpecificationPattern[T]:
        """Allow using ~ operator for NOT operations."""
        return self.not_()


class AndSpecification[T](SpecificationPattern[T]):
    """Combines two specifications with logical AND."""

    def __init__(
        self,
        left: SpecificationPattern[T],
        right: SpecificationPattern[T],
    ) -> None:
        self._left = left
        self._right = right

    def is_satisfied_by(self, candidate: T) -> bool:
        """Both specifications must be satisfied."""
        return self._left.is_satisfied_by(candidate) and self._right.is_satisfied_by(
            candidate,
        )


class OrSpecification[T](SpecificationPattern[T]):
    """Combines two specifications with logical OR."""

    def __init__(
        self,
        left: SpecificationPattern[T],
        right: SpecificationPattern[T],
    ) -> None:
        self._left = left
        self._right = right

    def is_satisfied_by(self, candidate: T) -> bool:
        """Either specification must be satisfied."""
        return self._left.is_satisfied_by(candidate) or self._right.is_satisfied_by(
            candidate,
        )


class NotSpecification[T](SpecificationPattern[T]):
    """Negates a specification with logical NOT."""

    def __init__(self, specification: SpecificationPattern[T]) -> None:
        self._specification = specification

    def is_satisfied_by(self, candidate: T) -> bool:
        """Check that the specification is NOT satisfied."""
        return not self._specification.is_satisfied_by(candidate)


class UnitOfWorkPattern(abc.ABC):
    """Unit of Work Pattern for Transaction Management.

    Maintains a list of objects affected by a business transaction
    and coordinates writing out changes and resolving concurrency issues.

    🎯 PRINCIPLES:
    - Transaction boundary
    - Tracks changes
    - Coordinates persistence
    - Ensures consistency
    """

    @abc.abstractmethod
    async def commit(self) -> None:
        """Commit all changes made during this unit of work."""

    @abc.abstractmethod
    async def rollback(self) -> None:
        """Rollback all changes made during this unit of work."""

    @abc.abstractmethod
    def register_new(self, entity: Any) -> None:
        """Register a new entity to be inserted."""

    @abc.abstractmethod
    def register_dirty(self, entity: Any) -> None:
        """Register an entity that has been modified."""

    @abc.abstractmethod
    def register_removed(self, entity: Any) -> None:
        """Register an entity to be deleted."""

    async def __aenter__(self) -> Self:
        """Support async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Auto-commit on success, rollback on exception."""
        if exc_type is None:
            await self.commit()
        else:
            await self.rollback()


class ObserverPattern(abc.ABC):
    """Observer Pattern for Event Handling.

    Defines a one-to-many dependency between objects so that when one
    object changes state, all dependents are notified automatically.

    🎯 PRINCIPLES:
    - Loose coupling
    - Event-driven architecture
    - Multiple observers
    - Automatic notification
    """

    @abc.abstractmethod
    def notify(self, event: Any) -> None:
        """Handle notification of an event."""
