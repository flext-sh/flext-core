"""FlextService - Domain Service Base Class Module.

This module provides FlextService[T], a base class for implementing domain
services with infrastructure support including dependency injection, validation,
type-safe result handling, and auto-execution patterns. Implements structural
typing via FlextProtocols.Service through duck typing, providing a foundation
for CQRS command and query services throughout the FLEXT ecosystem.

Scope: Domain service base class, auto-execution pattern, business rule validation,
service metadata, type-safe execution infrastructure, railway-oriented programming
with FlextResult, and dependency injection support.

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
    """Domain Service Base Class for FLEXT ecosystem.

    Provides comprehensive infrastructure support for implementing domain services
    with type-safe execution, dependency injection, business rule validation,
    and auto-execution patterns. Implements structural typing via FlextProtocols.Service
    through duck typing (no inheritance required), serving as the foundation for
    CQRS command and query services throughout the FLEXT ecosystem.

    Core Features:
    - Abstract base class with generic type parameters for type-safe results
    - Railway-oriented programming with FlextResult for error handling
    - Auto-execution pattern for immediate service execution on instantiation
    - Business rule validation with extensible validation pipeline
    - Dependency injection support through FlextMixins
    - Pydantic integration for configuration and validation
    - Service metadata and introspection capabilities

    Architecture:
    - Single class with nested service execution logic
    - DRY principle applied through centralized result handling
    - SOLID principles: Single Responsibility for domain service execution
    - Railway pattern for consistent error handling without exceptions
    - Structural typing for protocol compliance without inheritance

    Type Parameters:
    - TDomainResult: The type of result returned by service execution

    Usage Examples:
        >>> # Standard service usage
        >>> class UserService(FlextService[User]):
        ...     def execute(self) -> FlextResult[User]:
        ...         # Domain logic here
        ...         user = User(id=1, name="John")
        ...         return self.ok(user)
        >>>
        >>> service = UserService()
        >>> result = service.execute()
        >>> if result.is_success:
        ...     user = result.value
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
        """Execute domain service logic - abstract method to be implemented by subclasses.

        This is the core business logic method that must be implemented by all
        concrete service subclasses. It contains the actual domain operations,
        business rules, and result generation logic specific to each service.

        The method should follow railway-oriented programming principles,
        returning FlextResult[T] for consistent error handling and success indication.

        Returns:
            FlextResult[TDomainResult]: Success with domain result or failure with error details

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
