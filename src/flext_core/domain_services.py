"""Domain-Driven Design domain services implementation.

Provides enterprise-grade domain service patterns following DDD principles
with stateless cross-entity operations, business logic orchestration, and
type-safe error handling using FLEXT foundation patterns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import override

from pydantic import ConfigDict

from flext_core.constants import FlextConstants
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.utilities import FlextUtilities

# =============================================================================
# FLEXT DOMAIN SERVICE - Public DDD Domain Service implementation
# =============================================================================


class FlextDomainService[TDomainResult](
    FlextModels.BaseConfig,
    FlextMixins.Serializable,
    FlextMixins.Loggable,
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

    # Mixin functionality is now inherited via FlextMixins.Serializable

    def is_valid(self) -> bool:
        """Check if domain service is valid using foundation patterns."""
        try:
            validation_result = self.validate_business_rules()
            return validation_result.is_success
        except Exception:
            # Use FlextUtilities for error logging if needed
            return False

    @override
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
        """Execute operation with standard error handling using foundation patterns.

        Args:
            operation_name: Name of the operation for logging
            operation: Operation to execute
            *args: Arguments to pass to the operation
            **kwargs: Keyword arguments to pass to the operation

        Returns:
            FlextResult containing the operation result or error

        """
        try:
            # Validate configuration first
            config_result = self.validate_config()
            if config_result.is_failure:
                error_message = (
                    config_result.error
                    or f"{FlextConstants.Messages.VALIDATION_FAILED}: Configuration validation failed"
                )
                return FlextResult[object].fail(
                    error_message, error_code=FlextConstants.Errors.VALIDATION_ERROR
                )

            # Validate operation is callable and execute
            if not callable(operation):
                return FlextResult[object].fail(
                    f"{FlextConstants.Messages.OPERATION_FAILED}: Operation {operation_name} is not callable",
                    error_code=FlextConstants.Errors.OPERATION_ERROR,
                )

            # Execute the callable operation - MyPy should understand this is reachable
            result = operation(*args, **kwargs)
            return FlextResult[object].ok(result)

        except (RuntimeError, ValueError, TypeError) as e:
            return FlextResult[object].fail(
                f"{FlextConstants.Messages.OPERATION_FAILED}: Operation {operation_name} failed: {e}",
                error_code=FlextConstants.Errors.EXCEPTION_ERROR,
            )
        except Exception as e:
            # Catch any other exceptions using FlextConstants
            return FlextResult[object].fail(
                f"{FlextConstants.Messages.UNKNOWN_ERROR}: Unexpected error in {operation_name}: {e}",
                error_code=FlextConstants.Errors.UNKNOWN_ERROR,
            )

    def get_service_info(self) -> dict[str, object]:
        """Get service information for monitoring using foundation patterns."""
        return {
            "service_type": self.__class__.__name__,
            "service_id": f"service_{self.__class__.__name__.lower()}_{FlextUtilities.Generators.generate_id()}",
            "config_valid": self.validate_config().is_success,
            "business_rules_valid": self.validate_business_rules().is_success,
            "timestamp": FlextUtilities.Generators.generate_iso_timestamp(),
        }


# Export API
__all__: list[str] = [
    "FlextDomainService",  # Main domain service base class
]
