# ruff: disable=E402
"""Domain service abstractions supporting the 1.0.0 alignment pillar.

These bases codify the service ergonomics described in ``README.md`` and
``docs/architecture.md``: immutable models, context-aware logging, and
``FlextResult`` contracts that remain stable throughout the 1.x lifecycle.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    override,
)

from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextService[TDomainResult](
    FlextModels.ArbitraryTypesModel,
    FlextMixins,
    ABC,
):
    """Domain service base using railway patterns with Pydantic models.

    **Function**: Domain service base class for business logic with full infrastructure
        - Abstract execute() method for domain operations (Domain.Service protocol)
        - Business rule validation with FlextResult (Domain.Service protocol)
        - Configuration validation and management (Domain.Service protocol)
        - Dependency injection via FlextMixins
        - Context propagation via FlextMixins
        - Structured logging via FlextMixins
        - Performance tracking via FlextMixins
        - Configuration access via FlextMixins.Configurable
        - Operation execution with timeout support (Domain.Service protocol)
        - Batch processing for multiple operations
        - Performance metrics collection and tracking
        - Service information and metadata access (Domain.Service protocol)
        - Context-aware logging integration
        - Serialization support via FlextMixins
        - Type-safe generic result handling

    **Uses**: Core FLEXT infrastructure for services
        - FlextResult[T] for railway pattern error handling
        - FlextModels.ArbitraryTypesModel for Pydantic base
        - FlextMixins for serialization and logging
        - FlextConstants for defaults and error codes
        - FlextProtocols.Domain.Service for protocol compliance
        - Pydantic Generic[T] for type-safe operations
        - abc.ABC for abstract base class pattern
        - signal module for timeout enforcement
        - datetime for timestamp operations
        - contextmanager for context scopes
        - Protocol for callable interfaces

    **How to use**: Domain service implementation patterns
        ```python
        from flext_core import FlextService, FlextResult


        # Example 1: Implement domain service (Domain.Service protocol)
        class UserService(FlextService[User]):
            name: str = "UserService"
            version: str = "1.0.0"

            def execute(self) -> FlextResult[User]:
                # Validate business rules first (Domain.Service protocol)
                validation = self.validate_business_rules()
                if validation.is_failure:
                    return FlextResult[User].fail(validation.error)

                # Execute domain logic
                user = User(id="123", name="John")
                return FlextResult[User].ok(user)

            def validate_business_rules(self) -> FlextResult[None]:
                # Business rule validation (Domain.Service protocol)
                if not self.name:
                    return FlextResult[None].fail("Name required")
                return FlextResult[None].ok(None)


        # Example 2: Protocol compliance check
        from flext_core.protocols import FlextProtocols

        service = UserService()
        # Verify protocol implementation at runtime
        assert isinstance(service, FlextProtocols.Domain.Service)

        # Use protocol-defined methods
        if service.is_valid():
            result = service.execute()


        # Example 3: Execute service operation
        service = UserService()
        result = service.execute()
        if result.is_success:
            user = result.unwrap()

        # Example 4: Execute with timeout
        operation_request = OperationExecutionRequest(
            operation_callable=lambda: service.execute(),
            timeout_seconds=FlextConstants.Defaults.OPERATION_TIMEOUT_SECONDS,
        )
        result = service.execute_operation(operation_request)

        # Example 5: Validate configuration (Domain.Service protocol)
        config_result = service.validate_config()
        if config_result.is_failure:
            print(f"Config error: {config_result.error}")

        # Example 6: Get service information (Domain.Service protocol)
        info = service.get_service_info()
        print(f"Service: {info['name']} v{info['version']}")

        # Example 7: Batch operation execution
        operations = [op1, op2, op3]
        results = [service.execute() for _ in operations]

        # Example 8: Check if service is valid (Domain.Service protocol)
        if service.is_valid():
            result = service.execute()
        ```

    Attributes:
        model_config: Pydantic configuration dict.

    Note:
        All services must implement execute() method (Domain.Service protocol).
        Generic type TDomainResult provides type safety.
        Services inherit serialization from FlextMixins.
        Business rule validation returns FlextResult (Domain.Service protocol).
        Configuration validation is separate from rules (Domain.Service protocol).
        Protocol compliance ensures ecosystem-wide consistency.

    Warning:
        Execute method must be implemented by subclasses.
        Timeout operations require signal support (Unix-like).
        Batch operations do not provide transaction semantics.
        Service validation does not guarantee execution success.

    Example:
        Complete domain service implementation with protocol compliance:

        >>> class OrderService(FlextService[Order]):
        ...     def execute(self) -> FlextResult[Order]:
        ...         return FlextResult[Order].ok(Order())
        >>> service = OrderService()
        >>> # Verify protocol implementation
        >>> from flext_core.protocols import FlextProtocols
        >>> assert isinstance(service, FlextProtocols.Domain.Service)
        >>> result = service.execute()
        >>> print(result.is_success)
        True

    See Also:
        FlextProtocols.Domain.Service: Protocol definition for domain services.
        FlextResult: For railway pattern error handling.
        FlextModels: For domain model definitions.
        FlextMixins: For serialization and logging.
        FlextHandlers: For handler implementation patterns.

    **IMPLEMENTATION NOTES**:
    - Implements FlextProtocols.Domain.Service for ecosystem consistency
    - Abstract domain service base class with railway patterns
    - Comprehensive validation and execution patterns
    - Timeout and retry mechanisms with signal handling
    - Batch operation support with error accumulation
    - Metrics collection integration
    - Resource management patterns with automatic cleanup
    - Protocol compliance verified at runtime

    """

    # Dependency injection attributes provided by FlextMixins
    # - container: FlextContainer (via FlextMixins)
    # - context: object (via FlextMixins)
    # - logger: FlextLogger (via FlextMixins)
    # - config: object (via FlextMixins.Configurable)
    # - _track_operation: context manager (via FlextMixins)

    _bus: object | None = None  # FlextBus type to avoid circular import

    @override
    def __init__(self, **data: object) -> None:
        """Initialize domain service with Pydantic validation and infrastructure."""
        super().__init__(**data)
        # Initialize service infrastructure if needed
        self._init_service(service_name=self.__class__.__name__)

        # Enrich context with service metadata (Phase 1 enhancement)
        # This automatically adds service information to all logs
        self._enrich_context(
            service_type=self.__class__.__name__,
            service_module=self.__class__.__module__,
        )

    # =============================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses (Domain.Service protocol)
    # =============================================================================

    @abstractmethod
    def execute(self) -> FlextResult[TDomainResult]:
        """Execute the main domain operation (Domain.Service protocol).

        Returns:
            FlextResult[TDomainResult]: Success with domain result or failure with error

        """

    # =============================================================================
    # VALIDATION METHODS (Domain.Service protocol)
    # =============================================================================

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules for the domain service (Domain.Service protocol).

        Returns:
            FlextResult[None]: Success if valid, failure with error details

        """
        return FlextResult[None].ok(None)


__all__: FlextTypes.StringList = [
    "FlextService",
]
