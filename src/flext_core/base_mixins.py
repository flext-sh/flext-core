"""Base mixin abstractions following SOLID principles.

This module provides abstract base classes for mixin patterns used across
the FLEXT ecosystem. Concrete mixins implementations are in mixins.py.

Classes:
    FlextAbstractMixin: Base class for all mixins.
    FlextAbstractTimestampMixin: Abstract timestamp tracking.
    FlextAbstractLoggableMixin: Abstract logging functionality.
    FlextAbstractValidatableMixin: Abstract validation patterns.
    FlextAbstractSerializableMixin: Abstract serialization patterns.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flext_core.protocols import FlextLoggerProtocol
    from flext_core.result import FlextResult
    from flext_core.typings import TEntityId

# =============================================================================
# ABSTRACT MIXIN BASE
# =============================================================================


class FlextAbstractMixin(ABC):
    """Abstract base class for all FLEXT mixins following SOLID principles.

    Provides foundation for implementing mixins with proper separation
    of concerns and dependency inversion.
    """

    @abstractmethod
    def mixin_setup(self) -> None:
        """Set up mixin - must be implemented by concrete mixins."""
        ...

    def __init__(self) -> None:
        """Initialize abstract mixin."""
        self._mixin_initialized = True


# =============================================================================
# TIMESTAMP ABSTRACTIONS
# =============================================================================


class FlextAbstractTimestampMixin(FlextAbstractMixin):
    """Abstract timestamp mixin for entity time tracking."""

    @abstractmethod
    def update_timestamp(self) -> None:
        """Update timestamp - must be implemented by subclasses."""
        ...

    @abstractmethod
    def get_timestamp(self) -> float:
        """Get timestamp - must be implemented by subclasses."""
        ...

    def mixin_setup(self) -> None:
        """Set up timestamp mixin."""
        self.update_timestamp()


# =============================================================================
# IDENTIFICATION ABSTRACTIONS
# =============================================================================


class FlextAbstractIdentifiableMixin(FlextAbstractMixin):
    """Abstract identifiable mixin for entity identification."""

    @abstractmethod
    def get_id(self) -> TEntityId:
        """Get entity ID - must be implemented by subclasses."""
        ...

    @abstractmethod
    def set_id(self, entity_id: TEntityId) -> None:
        """Set entity ID - must be implemented by subclasses."""
        ...

    def mixin_setup(self) -> None:
        """Set up identifiable mixin."""


# =============================================================================
# LOGGING ABSTRACTIONS
# =============================================================================


class FlextAbstractLoggableMixin(FlextAbstractMixin):
    """Abstract loggable mixin for entity logging."""

    @property
    @abstractmethod
    def logger(self) -> FlextLoggerProtocol:
        """Get logger instance - must be implemented by subclasses."""
        ...

    @abstractmethod
    def log_operation(self, operation: str, **kwargs: object) -> None:
        """Log operation - must be implemented by subclasses."""
        ...

    def mixin_setup(self) -> None:
        """Set up loggable mixin."""


# =============================================================================
# VALIDATION ABSTRACTIONS
# =============================================================================


class FlextAbstractValidatableMixin(FlextAbstractMixin):
    """Abstract validatable mixin for entity validation."""

    @abstractmethod
    def validate(self) -> FlextResult[None]:
        """Validate entity - must be implemented by subclasses."""
        ...

    @property
    @abstractmethod
    def is_valid(self) -> bool:  # pragma: no cover - abstract property declaration
        """Check if entity is valid - must be implemented by subclasses."""
        raise NotImplementedError

    def mixin_setup(self) -> None:
        """Set up validatable mixin."""


# =============================================================================
# SERIALIZATION ABSTRACTIONS
# =============================================================================


class FlextAbstractSerializableMixin(FlextAbstractMixin):
    """Abstract serializable mixin for entity serialization."""

    @abstractmethod
    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary - must be implemented by subclasses."""
        ...

    @abstractmethod
    def load_from_dict(self, data: dict[str, object]) -> None:
        """Load from dictionary - must be implemented by subclasses."""
        ...

    def mixin_setup(self) -> None:
        """Set up serializable mixin."""


# =============================================================================
# ENTITY ABSTRACTIONS
# =============================================================================


class FlextAbstractEntityMixin(FlextAbstractMixin):
    """Abstract entity mixin for domain entities."""

    @abstractmethod
    def get_domain_events(self) -> list[object]:
        """Get domain events - must be implemented by subclasses."""
        ...

    @abstractmethod
    def clear_domain_events(self) -> None:
        """Clear domain events - must be implemented by subclasses."""
        ...

    def mixin_setup(self) -> None:
        """Set up entity mixin."""


# =============================================================================
# SERVICE ABSTRACTIONS
# =============================================================================


class FlextAbstractServiceMixin(FlextAbstractMixin):
    """Abstract service mixin for service classes."""

    @abstractmethod
    def get_service_name(self) -> str:
        """Get service name - must be implemented by subclasses."""
        ...

    @abstractmethod
    def initialize_service(self) -> FlextResult[None]:
        """Initialize service - must be implemented by subclasses."""
        ...

    def mixin_setup(self) -> None:
        """Set up service mixin."""


# =============================================================================
# TIMING ABSTRACTIONS
# =============================================================================


class FlextAbstractTimingMixin(FlextAbstractMixin):
    """Abstract timing mixin for performance tracking."""

    @abstractmethod
    def start_timing(self) -> None:
        """Start timing - must be implemented by subclasses."""
        ...

    @abstractmethod
    def stop_timing(self) -> float:
        """Stop timing and return duration - must be implemented by subclasses."""
        ...

    def mixin_setup(self) -> None:
        """Set up timing mixin."""


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__: list[str] = [
    "FlextAbstractEntityMixin",
    "FlextAbstractIdentifiableMixin",
    "FlextAbstractLoggableMixin",
    "FlextAbstractMixin",
    "FlextAbstractSerializableMixin",
    "FlextAbstractServiceMixin",
    "FlextAbstractTimestampMixin",
    "FlextAbstractTimingMixin",
    "FlextAbstractValidatableMixin",
]
