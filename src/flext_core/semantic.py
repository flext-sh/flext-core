"""FLEXT Harmonized Semantic Architecture - Single Source of Truth.

Provides unified semantic patterns for constants, models, observability,
and error handling across the FLEXT ecosystem.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .constants import FlextConstants
from .result import FlextResult

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

# =============================================================================
# LAYER 1: MODEL SEMANTICS - Domain Foundation
# =============================================================================


class FlextSemanticModel:
    """Unified model patterns using constants from single source."""

    # FOUNDATION CLASSES - 4 core types using semantic constants
    class Foundation:
        """Core domain modeling patterns."""

        @staticmethod
        def get_base_config() -> dict[str, object]:
            """Get base model configuration using semantic constants."""
            return {
                "extra": FlextConstants.Models.EXTRA_FORBID,
                "validate_assignment": FlextConstants.Models.VALIDATE_ASSIGNMENT,
                "use_enum_values": FlextConstants.Models.USE_ENUM_VALUES,
                "str_strip_whitespace": FlextConstants.Models.STR_STRIP_WHITESPACE,
                "str_max_length": FlextConstants.Limits.MAX_STRING_LENGTH,
                "arbitrary_types_allowed": (
                    FlextConstants.Models.ARBITRARY_TYPES_ALLOWED
                ),
                "validate_default": FlextConstants.Models.VALIDATE_DEFAULT,
            }

    # NAMESPACE ORGANIZATION - Semantic grouping
    class Namespace:
        """Domain organization namespaces."""

        class FlextData:
            """Data-related model namespace."""

        class FlextAuth:
            """Authentication model namespace."""

        class FlextService:
            """Service model namespace."""

        class FlextInfrastructure:
            """Infrastructure model namespace."""

    # FACTORY PATTERNS - Semantic creation
    class Factory:
        """Model creation with semantic defaults."""

        @staticmethod
        def create_model_with_defaults(**kwargs: object) -> dict[str, object]:
            """Create model data with semantic defaults."""
            return {
                "timeout": FlextConstants.Defaults.TIMEOUT,
                "status": FlextConstants.Status.ACTIVE,
                **kwargs,
            }

        @staticmethod
        def validate_with_business_rules(instance: object) -> FlextResult[None]:
            """Validate instance using business rules."""
            if hasattr(instance, "validate_business_rules"):
                result = instance.validate_business_rules()
                if isinstance(result, FlextResult):
                    return result
            return FlextResult.ok(None)


# =============================================================================
# LAYER 2: OBSERVABILITY SEMANTICS - Interface Foundation
# =============================================================================


class FlextSemanticObservability:
    """Unified observability patterns using semantic constants."""

    # PROTOCOL INTERFACES - Foundation contracts
    class Protocol:
        """Observability interface contracts."""

        @runtime_checkable
        class Logger(Protocol):
            """Logging interface using semantic constants."""

            def trace(self, message: str, **context: object) -> None:
                """Log trace using semantic log levels."""

            def debug(self, message: str, **context: object) -> None:
                """Log debug using semantic log levels."""

            def info(self, message: str, **context: object) -> None:
                """Log info using semantic log levels."""

            def warn(self, message: str, **context: object) -> None:
                """Log warn using semantic log levels."""

            def error(
                self,
                message: str,
                *,
                error_code: str | None = None,
                exception: Exception | None = None,
                **context: object,
            ) -> None:
                """Log error using semantic error codes."""

            def fatal(self, message: str, **context: object) -> None:
                """Log fatal using semantic log levels."""

            def audit(
                self,
                message: str,
                *,
                user_id: str | None = None,
                action: str | None = None,
                resource: str | None = None,
                outcome: str | None = None,
                **context: object,
            ) -> None:
                """Audit logging with semantic structure."""

        @runtime_checkable
        class SpanProtocol(Protocol):
            """Span interface for tracing."""

            def add_context(self, key: str, value: object) -> None:
                """Add context to span."""

            def add_error(self, error: Exception) -> None:
                """Add error to span."""

        @runtime_checkable
        class Tracer(Protocol):
            """Tracing interface using semantic constants."""

            def business_span(
                self,
                operation_name: str,
                **context: object,
            ) -> AbstractContextManager[object]:
                """Create business operation span using semantic span types."""

            def technical_span(
                self,
                operation_name: str,
                *,
                component: str | None = None,
                resource: str | None = None,
                **context: object,
            ) -> AbstractContextManager[object]:
                """Technical operation span using semantic span types."""

        @runtime_checkable
        class Metrics(Protocol):
            """Metrics interface using semantic constants."""

            def increment(
                self,
                metric_name: str,
                value: int = 1,
                tags: dict[str, str] | None = None,
            ) -> None:
                """Increment counter using semantic metric types."""

            def histogram(
                self,
                metric_name: str,
                value: float,
                tags: dict[str, str] | None = None,
            ) -> None:
                """Record histogram using semantic metric types."""

            def gauge(
                self,
                metric_name: str,
                value: float,
                tags: dict[str, str] | None = None,
            ) -> None:
                """Set gauge using semantic metric types."""

        @runtime_checkable
        class Observability(Protocol):
            """Complete observability interface."""

            @property
            def log(self) -> FlextSemanticObservability.Protocol.Logger:
                """Access logging component."""

            @property
            def trace(self) -> FlextSemanticObservability.Protocol.Tracer:
                """Access tracing component."""

            @property
            def metrics(self) -> FlextSemanticObservability.Protocol.Metrics:
                """Access metrics component."""

    # FACTORY FUNCTIONS - Instance creation
    class Factory:
        """Observability factory using semantic constants."""

        @staticmethod
        def get_minimal_observability() -> object:
            """Get minimal observability implementation."""
            # Dynamic import to avoid circular dependencies
            observability_module = importlib.import_module(
                ".observability",
                package=__package__,
            )
            return observability_module.get_observability()

        @staticmethod
        def configure_observability(
            service_name: str,
            *,
            log_level: str = FlextConstants.Observability.DEFAULT_LOG_LEVEL,
        ) -> object:
            """Configure observability using semantic constants."""
            # Dynamic import to avoid circular dependencies
            observability_module = importlib.import_module(
                ".observability",
                package=__package__,
            )
            return observability_module.configure_minimal_observability(
                service_name,
                log_level=log_level,
            )


# =============================================================================
# LAYER 3: ERROR SEMANTICS - Exception Foundation
# =============================================================================


class FlextSemanticError:
    """Unified error patterns using semantic constants."""

    # ERROR HIERARCHY - Semantic classification
    class Hierarchy:
        """Error hierarchy using semantic error codes."""

        class FlextError(Exception):
            """Base error for entire FLEXT ecosystem."""

            def __init__(
                self,
                message: str,
                *,
                error_code: str = FlextConstants.Errors.GENERIC_ERROR,
                context: dict[str, object] | None = None,
                cause: Exception | None = None,
            ) -> None:
                super().__init__(message)
                self.message = message
                self.error_code = error_code
                self.context = context or {}
                self.cause = cause

        class FlextBusinessError(FlextError):
            """Business logic errors using semantic constants."""

            def __init__(self, message: str, **kwargs: object) -> None:
                context_val = kwargs.pop("context", None)
                cause_val = kwargs.pop("cause", None)
                # Type-safe context and cause handling
                context_dict = context_val if isinstance(context_val, dict) else None
                cause_exception = (
                    cause_val if isinstance(cause_val, Exception) else None
                )
                super().__init__(
                    message,
                    error_code=FlextConstants.Errors.BUSINESS_RULE_ERROR,
                    context=context_dict,
                    cause=cause_exception,
                )

        class FlextTechnicalError(FlextError):
            """Technical/infrastructure errors using semantic constants."""

            def __init__(self, message: str, **kwargs: object) -> None:
                context_val = kwargs.pop("context", None)
                cause_val = kwargs.pop("cause", None)
                context_dict = context_val if isinstance(context_val, dict) else None
                cause_exception = (
                    cause_val if isinstance(cause_val, Exception) else None
                )
                super().__init__(
                    message,
                    error_code=FlextConstants.Errors.CONNECTION_ERROR,
                    context=context_dict,
                    cause=cause_exception,
                )

        class FlextValidationError(FlextError):
            """Data validation errors using semantic constants."""

            def __init__(self, message: str, **kwargs: object) -> None:
                context_val = kwargs.pop("context", None)
                cause_val = kwargs.pop("cause", None)
                context_dict = context_val if isinstance(context_val, dict) else None
                cause_exception = (
                    cause_val if isinstance(cause_val, Exception) else None
                )
                super().__init__(
                    message,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    context=context_dict,
                    cause=cause_exception,
                )

        class FlextSecurityError(FlextError):
            """Security-related errors using semantic constants."""

            def __init__(self, message: str, **kwargs: object) -> None:
                context_val = kwargs.pop("context", None)
                cause_val = kwargs.pop("cause", None)
                context_dict = context_val if isinstance(context_val, dict) else None
                cause_exception = (
                    cause_val if isinstance(cause_val, Exception) else None
                )
                super().__init__(
                    message,
                    error_code=FlextConstants.Errors.AUTHENTICATION_ERROR,
                    context=context_dict,
                    cause=cause_exception,
                )

    # NAMESPACE ERRORS - Domain-specific (defined after hierarchy)
    class Namespace:
        """Domain-specific errors using semantic hierarchy."""

        # These will be defined after the FlextSemanticError class is complete

    # ERROR FACTORY - Semantic error creation
    class Factory:
        """Error factory using semantic constants."""

        @staticmethod
        def create_business_error(
            message: str,
            *,
            context: dict[str, object] | None = None,
            cause: Exception | None = None,
        ) -> FlextSemanticError.Hierarchy.FlextBusinessError:
            """Create business error using semantic constants."""
            return FlextSemanticError.Hierarchy.FlextBusinessError(
                message,
                context=context,
                cause=cause,
            )

        @staticmethod
        def create_validation_error(
            message: str,
            *,
            field_name: str | None = None,
            field_value: object = None,
            context: dict[str, object] | None = None,
        ) -> FlextSemanticError.Hierarchy.FlextValidationError:
            """Create validation error using semantic constants."""
            validation_context = context or {}
            if field_name:
                validation_context["field_name"] = field_name
            if field_value is not None:
                validation_context["field_value"] = field_value

            return FlextSemanticError.Hierarchy.FlextValidationError(
                message,
                context=validation_context,
            )

        @staticmethod
        def from_exception(
            exception: Exception,
            *,
            message: str | None = None,
            context: dict[str, object] | None = None,
        ) -> FlextSemanticError.Hierarchy.FlextError:
            """Convert standard exception to FLEXT error."""
            error_message = message or str(exception)

            # Classify exception type using semantic constants
            if isinstance(exception, ValueError):
                return FlextSemanticError.Hierarchy.FlextValidationError(
                    error_message,
                    context=context,
                    cause=exception,
                )
            if isinstance(exception, ConnectionError):
                return FlextSemanticError.Hierarchy.FlextTechnicalError(
                    error_message,
                    context=context,
                    cause=exception,
                )
            return FlextSemanticError.Hierarchy.FlextError(
                error_message,
                context=context,
                cause=exception,
            )


# =============================================================================
# UNIFIED SEMANTIC API - Single source of truth
# =============================================================================


class FlextSemantic:
    """Unified semantic API for entire FLEXT ecosystem.

    Single entry point providing access to all semantic patterns:
    constants, domain models, observability interfaces, and hierarchical
    error handling with semantic consistency across the ecosystem.
    """

    # Layer 0: Constants (Single source of truth)
    Constants = FlextConstants

    # Layer 1: Models (Domain foundation)
    Models = FlextSemanticModel

    # Layer 2: Observability (Interface foundation)
    Observability = FlextSemanticObservability

    # Layer 3: Errors (Exception foundation)
    Errors = FlextSemanticError


# =============================================================================
# CONVENIENCE EXPORTS - Direct access to most common types
# =============================================================================

# Most commonly used error types
FlextError = FlextSemanticError.Hierarchy.FlextError
FlextBusinessError = FlextSemanticError.Hierarchy.FlextBusinessError
FlextValidationError = FlextSemanticError.Hierarchy.FlextValidationError

# Most commonly used functions
get_observability = FlextSemanticObservability.Factory.get_minimal_observability
configure_observability = FlextSemanticObservability.Factory.configure_observability
create_business_error = FlextSemanticError.Factory.create_business_error
create_validation_error = FlextSemanticError.Factory.create_validation_error

# =============================================================================
# EXPORTS - Minimal unified API
# =============================================================================

__all__ = [
    "FlextBusinessError",
    "FlextError",
    "FlextSemantic",
    "FlextValidationError",
    "configure_observability",
    "create_business_error",
    "create_validation_error",
    "get_observability",
]

# Total exports: 8 (vs. 93+ in previous fragmented approach)
