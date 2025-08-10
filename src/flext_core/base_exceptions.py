"""Base exception abstractions following SOLID principles.

This module provides abstract base classes for exception patterns used across
the FLEXT ecosystem. Concrete implementations are in exceptions.py.

Classes:
    FlextAbstractError: Base class for all exceptions.
    FlextAbstractValidationError: Abstract validation exception.
    FlextAbstractBusinessError: Abstract business rule exception.
    FlextAbstractInfrastructureError: Abstract infrastructure exception.
    FlextAbstractConfigurationError: Abstract configuration exception.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flext_core.typings import TAnyDict

# =============================================================================
# ABSTRACT ERROR BASE
# =============================================================================


class FlextAbstractError(ABC, Exception):
    """Abstract base class for all FLEXT exceptions following SOLID principles.

    Provides foundation for implementing exceptions with proper separation
    of concerns and dependency inversion.
    """

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        """Initialize abstract error."""
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self._get_default_error_code()
        self.context = context or {}

    def __str__(self) -> str:  # pragma: no cover - trivial
        """Return human-readable exception message."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message

    @property
    @abstractmethod
    def error_category(self) -> str:
        """Get error category - must be implemented by subclasses."""
        ...

    @abstractmethod
    def _get_default_error_code(self) -> str:
        """Get default error code - must be implemented by subclasses."""
        ...

    @abstractmethod
    def to_dict(self) -> TAnyDict:
        """Convert exception to dictionary - must be implemented by subclasses."""
        ...


class FlextAbstractValidationError(FlextAbstractError):
    """Abstract validation error for input validation."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        validation_details: dict[str, object] | None = None,
        error_code: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        """Initialize abstract validation error."""
        super().__init__(message, error_code, context)
        self.field = field
        self.validation_details = validation_details or {}

    @property
    def error_category(self) -> str:
        """Get error category."""
        return "VALIDATION"

    def _get_default_error_code(self) -> str:
        """Get default error code."""
        return "VALIDATION_ERROR"


class FlextAbstractBusinessError(FlextAbstractError):
    """Abstract business error for business rule violations."""

    def __init__(
        self,
        message: str,
        business_rule: str | None = None,
        error_code: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize abstract business error."""
        super().__init__(message, error_code, kwargs)
        self.business_rule = business_rule

    @property
    def error_category(self) -> str:
        """Get error category."""
        return "BUSINESS"

    def _get_default_error_code(self) -> str:
        """Get default error code."""
        return "BUSINESS_ERROR"


class FlextAbstractInfrastructureError(FlextAbstractError):
    """Abstract infrastructure error for infrastructure issues."""

    def __init__(
        self,
        message: str,
        service: str | None = None,
        error_code: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize abstract infrastructure error."""
        super().__init__(message, error_code, kwargs)
        self.service = service

    @property
    def error_category(self) -> str:
        """Get error category."""
        return "INFRASTRUCTURE"

    def _get_default_error_code(self) -> str:
        """Get default error code."""
        return "INFRASTRUCTURE_ERROR"


class FlextAbstractConfigurationError(FlextAbstractError):
    """Abstract configuration error for configuration issues."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        error_code: str | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize abstract configuration error."""
        super().__init__(message, error_code, kwargs)
        self.config_key = config_key

    @property
    def error_category(self) -> str:
        """Get error category."""
        return "CONFIGURATION"

    def _get_default_error_code(self) -> str:
        """Get default error code."""
        return "CONFIG_ERROR"


class FlextAbstractErrorFactory(ABC):
    """Abstract factory for creating exceptions."""

    @abstractmethod
    def create_validation_error(
        self,
        message: str,
        **kwargs: object,
    ) -> FlextAbstractValidationError:
        """Create validation error - must be implemented by subclasses."""
        ...

    @abstractmethod
    def create_business_error(
        self,
        message: str,
        **kwargs: object,
    ) -> FlextAbstractBusinessError:
        """Create business error - must be implemented by subclasses."""
        ...

    @abstractmethod
    def create_infrastructure_error(
        self,
        message: str,
        **kwargs: object,
    ) -> FlextAbstractInfrastructureError:
        """Create infrastructure error - must be implemented by subclasses."""
        ...

    @abstractmethod
    def create_configuration_error(
        self,
        message: str,
        **kwargs: object,
    ) -> FlextAbstractConfigurationError:
        """Create configuration error - must be implemented by subclasses."""
        ...


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__: list[str] = [
    "FlextAbstractBusinessError",
    "FlextAbstractConfigurationError",
    "FlextAbstractError",
    "FlextAbstractErrorFactory",
    "FlextAbstractInfrastructureError",
    "FlextAbstractValidationError",
]
