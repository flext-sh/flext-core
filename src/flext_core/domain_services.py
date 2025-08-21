"""Domain-Driven Design domain services implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from pydantic import ConfigDict

from flext_core.mixins import FlextSerializableMixin
from flext_core.models import FlextModel
from flext_core.result import FlextResult

# Type alias for flexible operation callables
OperationType = (
    Callable[[], object]
    | Callable[[object], object]
    | Callable[[object, object], object]
    | Callable[[object, object, object], object]
)

# =============================================================================
# FLEXT DOMAIN SERVICE - Public DDD Domain Service implementation
# =============================================================================


class FlextDomainService[TDomainResult](
    FlextModel,
    FlextSerializableMixin,
    ABC,
):
    """Abstract domain service for stateless cross-entity operations.

    Provides foundation for complex business operations that span multiple
    entities or aggregates with validation and serialization support.
    """

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,  # Allow non-Pydantic types like FlextDbOracleApi
    )

    # Mixin functionality is now inherited properly:
    # - Serialization methods from FlextSerializableMixin

    def is_valid(self) -> bool:
        """Check if domain service is valid (compatibility with FlextValidatableMixin)."""
        validation_result = self.validate_business_rules()
        return validation_result.is_success

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate domain service business rules (override in subclasses)."""
        return FlextResult[None].ok(None)

    @abstractmethod
    def execute(self) -> FlextResult[TDomainResult]:
        """Execute the domain service operation.

        Must be implemented by concrete services.
        """
        raise NotImplementedError

    def validate_config(self) -> FlextResult[None]:
        """Validate service configuration - override in subclasses.

        Default implementation returns success. Override to add custom validation.
        """
        return FlextResult[None].ok(None)

    def execute_operation(
        self,
        operation_name: str,
        operation: object,
        *args: object,
        **kwargs: object,
    ) -> FlextResult[object]:
        """Execute operation with standard error handling and logging.

        Args:
            operation_name: Name of the operation for logging
            operation: Operation to execute
            *args: Arguments to pass to the operation
            **kwargs: Keyword arguments to pass to the operation

        Returns:
            Result of the operation

        """
        try:
            # Validate configuration first
            config_result = self.validate_config()
            if config_result.is_failure:
                error_message = config_result.error or "Configuration validation failed"
                return FlextResult[object].fail(error_message)

            # Execute operation
            if not callable(operation):
                return FlextResult[object].fail(
                    f"Operation {operation_name} is not callable"
                )
            result = operation(*args, **kwargs)
            return FlextResult[object].ok(result)
        except (RuntimeError, ValueError, TypeError) as e:
            return FlextResult[object].fail(f"Operation {operation_name} failed: {e}")

    def get_service_info(self) -> dict[str, object]:
        """Get service information for monitoring."""
        return {
            "service_type": self.__class__.__name__,
            "config_valid": self.validate_config().is_success,
        }


# Export API
__all__: list[str] = ["FlextDomainService"]
