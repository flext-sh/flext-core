"""Domain service base class for FLEXT applications.

FlextService[T] supplies validation, dependency injection, and railway-style
result handling for domain services that participate in CQRS flows. It relies
on structural typing to satisfy ``FlextProtocols.Service`` and aligns with the
dispatcher-centric architecture.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from pydantic import computed_field

from flext_core.exceptions import FlextExceptions
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextService[TDomainResult](
    FlextModels.ArbitraryTypesModel,
    FlextMixins,
    ABC,
):
    """Base class for domain services used in CQRS flows.

    Subclasses implement ``execute`` to run business logic and return
    ``FlextResult`` values. The base provides validation hooks, dependency
    injection, and context-aware logging while remaining protocol compliant via
    structural typing.
    """

    def __init__(
        self,
        **data: FlextTypes.ScalarValue
        | Sequence[FlextTypes.ScalarValue]
        | Mapping[str, FlextTypes.ScalarValue],
    ) -> None:
        """Initialize service with configuration data.

        Sets up the service instance with optional configuration parameters
        passed through **data. Delegates to parent classes for proper
        initialization of mixins, models, and infrastructure components.

        Args:
            **data: Configuration parameters for service initialization

        """
        super().__init__(**data)

    @abstractmethod
    def execute(self) -> FlextResult[TDomainResult]:
        """Execute domain service logic.

        This is the core business logic method that must be implemented by all
        concrete service subclasses. It contains the actual domain operations,
        business rules, and result generation logic specific to each service.

        The method should follow railway-oriented programming principles,
        returning ``FlextResult[T]`` for consistent error handling and success
        indication.

        Returns:
            FlextResult[TDomainResult]: Success with domain result or failure
                with error details

        Note:
            Implementations should focus on business logic only. Infrastructure
            concerns like validation and error handling are handled by the base class.

        """
        ...

    @computed_field
    def result(self) -> TDomainResult:
        """Get execution result with lazy evaluation.

        Computed property that executes the service and returns the result value.
        Uses Pydantic's computed_field for caching and lazy evaluation. Raises
        exception on failure to maintain synchronous error handling.

        Returns:
            TDomainResult: The successful execution result

        Raises:
            FlextExceptions.BaseError: When execution fails

        Example:
            >>> service = MyService()
            >>> result_value = service.result  # Executes and returns value

        """
        result = self.execute()
        if result.is_success:
            return result.value
        raise FlextExceptions.BaseError(result.error or "Service execution failed")

    def validate_business_rules(self) -> FlextResult[bool]:
        """Validate business rules with extensible validation pipeline.

        Base method for business rule validation that can be overridden by subclasses
        to implement custom validation logic. By default, returns success. Subclasses
        should extend this method to add domain-specific business rule validation.

        The validation follows railway-oriented programming principles, allowing
        for complex validation pipelines that can fail early or accumulate errors.

        Returns:
            FlextResult[bool]: Success (True) if all business rules pass, failure with error details

        Example:
            >>> class ValidatedService(FlextService[Data]):
            ...     def validate_business_rules(self) -> FlextResult[bool]:
            ...         if not self.has_required_data():
            ...             return FlextResult[bool].fail("Missing required data")
            ...         return FlextResult[bool].ok(True)

        """
        # Base implementation - accept all (no validation)
        # Subclasses should override for specific business rules
        return FlextResult[bool].ok(True)

    def is_valid(self) -> bool:
        """Check if service is in valid state for execution.

        Performs business rule validation and returns boolean result.
        Catches exceptions during validation to ensure safe state checking.
        Used by infrastructure components to determine if service can execute.

        Returns:
            bool: True if service is valid and ready for execution

        Example:
            >>> service = MyService()
            >>> if service.is_valid():
            ...     result = service.execute()

        """
        try:
            return self.validate_business_rules().is_success
        except Exception:
            # Validation failed due to exception - consider invalid
            return False

    def get_service_info(self) -> Mapping[str, FlextTypes.FlexibleValue]:
        """Get service metadata and configuration information.

        Returns comprehensive metadata about the service instance including
        type information and execution parameters.
        Used by monitoring, logging, and debugging infrastructure.

        Returns:
            Mapping[str, FlextTypes.FlexibleValue]: Service metadata dictionary containing:
                - service_type: Class name of the service
                - Additional metadata can be added by subclasses

        Example:
            >>> service = MyService()
            >>> info = service.get_service_info()
            >>> print(f"Service: {info['service_type']}")

        """
        return {
            "service_type": self.__class__.__name__,
        }


__all__ = ["FlextService"]
