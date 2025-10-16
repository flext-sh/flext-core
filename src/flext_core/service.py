# ruff: disable=E402
"""Domain service base class with dependency injection and validation.

This module provides FlextService[T], a base class for implementing domain
services with comprehensive infrastructure support including dependency
injection, context management, logging, and validation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import concurrent.futures
import signal
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import cast, override

from pydantic import computed_field

from flext_core.config import FlextConfig
from flext_core.container import FlextContainer
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextService[TDomainResult](
    FlextModels.ArbitraryTypesModel,
    FlextMixins,
    ABC,
):
    """Base class for domain services with dependency injection and validation.

    Implements FlextProtocols.Service through structural typing. All subclasses
    automatically satisfy the Service protocol by implementing the required methods:
    - execute() - main domain operation (abstract)
    - validate_business_rules() - business rule validation
    - validate_config() - configuration validation
    - is_valid() - validity check
    - get_service_info() - metadata retrieval

    Provides abstract base class for implementing domain services with
    comprehensive infrastructure support including dependency injection,
    context management, logging, and validation.

    Features:
    - Abstract execute() method for domain operations
    - Business rule validation with FlextResult
    - Configuration validation and management
    - Dependency injection via FlextMixins
    - Context propagation and correlation
    - Structured logging integration
    - Performance tracking and metrics
    - Operation execution with timeout support

    Usage:
        >>> from flext_core.service import FlextService
        >>> from flext_core.result import FlextResult
        >>> from flext_core.protocols import FlextProtocols
        >>>
        >>> class UserService(FlextService[User]):
        ...     def execute(self) -> FlextResult[User]:
        ...         return FlextResult[User].ok(User(name="John"))
        >>> service = UserService()
        >>> # FlextService instances satisfy FlextProtocols.Service
        >>> assert isinstance(service, FlextProtocols.Service)
    """

    # Dependency injection attributes provided by FlextMixins
    # - container: FlextContainer (via FlextMixins)
    # - context: object (via FlextMixins)
    # - logger: FlextLogger (via FlextMixins)
    # - config: object (via FlextMixins)
    # - track: context manager (via FlextMixins)

    _bus: object | None = None  # FlextBus type to avoid circular import

    @computed_field  # Pydantic v2 computed_field already provides property behavior
    def service_config(self) -> FlextConfig:
        """Automatic config binding via Pydantic v2 computed_field.

        Pure Pydantic v2 pattern - no wrappers, no descriptors, no boilerplate.
        Config is transparently available via computed property.

        Example:
            >>> class OrderService(FlextService[Order]):
            ...     def execute(self) -> FlextResult[Order]:
            ...         # Config automatically available
            ...         timeout = self.service_config.timeout
            ...         return FlextResult[Order].ok(Order())

        Returns:
            FlextConfig: Global configuration instance

        """
        return FlextConfig.get_global_instance()

    def __init_subclass__(cls) -> None:
        """Automatic service registration on subclass definition.

        Pure Python 3.13+ pattern - no wrappers, no helpers, no boilerplate.
        Services are transparently registered in global container when class is defined.

        Example:
            >>> class UserService(FlextService[User]):  # ← Auto-registered
            ...     def execute(self) -> FlextResult[User]:
            ...         return FlextResult[User].ok(User(name="John"))

            >>> # Service already registered - no manual calls needed
            >>> container = FlextContainer.get_global()
            >>> service_result = container.get("UserService")
            >>> assert service_result.is_success

        """
        super().__init_subclass__()

        service_name = cls.__name__
        container = FlextContainer.get_global()
        container.register_factory(service_name, cls)

    @override
    def __init__(self, **data: object) -> None:
        """Initialize domain service with Pydantic validation and infrastructure.

        Automatic infrastructure provided transparently:
        - Service registration: via __init_subclass__ (class definition time)
        - Container access: via FlextMixins.container property
        - Logger access: via FlextMixins.logger property
        - Context access: via FlextMixins.context property
        - Config access: via FlextMixins.config property

        No manual setup needed - pure Python 3.13+ patterns.
        """
        super().__init__(**data)
        # AUTOMATIC: All infrastructure via properties (zero boilerplate)

    # =============================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses (Domain.Service protocol)
    # =============================================================================

    @abstractmethod
    def execute(self) -> FlextResult[TDomainResult]:
        """Execute the main domain operation (Domain.Service protocol).

        Returns:
            FlextResult[TDomainResult]: Success with domain result or failure with error

        """

    def execute_with_context_cleanup(self) -> FlextResult[TDomainResult]:
        """Execute operation with automatic scoped context cleanup.

        This method wraps execute() with automatic cleanup of operation-scoped
        logging context, preventing context accumulation while preserving
        request and application-level context.

        Returns:
            FlextResult[TDomainResult]: Result from execute() with guaranteed context cleanup

        Usage:
            >>> service = MyService()
            >>> result = service.execute_with_context_cleanup()
            >>> # Operation context cleared, request context (correlation_id) preserved

        Note:
            - Recommended for calling services from CLI/API boundaries
            - Clears operation scope only (preserves request and application scopes)
            - Request-level context (correlation_id) persists across service calls
            - Application-level context (app name, version) persists for lifetime

        """
        try:
            # Execute the service operation
            return self.execute()
        finally:
            # CRITICAL: Clean up operation-scoped context to prevent accumulation
            # Preserves request context (correlation_id) and application context
            self._clear_operation_context()

    # =============================================================================
    # VALIDATION METHODS (Domain.Service protocol)
    # =============================================================================

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules for the domain service (Domain.Service protocol).

        Returns:
            FlextResult[None]: Success if valid, failure with error details

        """
        return FlextResult[None].ok(None)

    def validate_config(self) -> FlextResult[None]:
        """Validate service configuration (Domain.Service protocol).

        Returns:
            FlextResult[None]: Success if configuration is valid, failure with error details

        """
        return FlextResult[None].ok(None)

    def is_valid(self) -> bool:
        """Check if the domain service is in a valid state (Domain.Service protocol).

        Returns:
            bool: True if the service is valid and ready for operations, False otherwise

        """
        # Check business rules and configuration
        try:
            business_rules = self.validate_business_rules()
            config = self.validate_config()
            return business_rules.is_success and config.is_success
        except Exception:
            # If validation raises an exception, the service is not valid
            return False

    def get_service_info(self) -> FlextTypes.Dict:
        """Get service information and metadata (Domain.Service protocol).

        Returns:
            FlextTypes.Dict: Service information dictionary with basic service type info.

        """
        return {
            "service_type": self.__class__.__name__,
        }

    # =============================================================================
    # OPERATION EXECUTION METHODS (Domain.Service protocol)
    # =============================================================================

    def execute_operation(
        self,
        request: FlextModels.OperationExecutionRequest,
    ) -> FlextResult[TDomainResult]:
        """Execute operation with validation, timeout, retry, and monitoring (Domain.Service protocol).

        Validates business rules and configuration before executing the operation.

        Args:
            request: Operation execution request with callable, arguments, and configuration

        Returns:
            FlextResult[TDomainResult]: Success with operation result or failure with validation/execution error

        """
        operation_name = request.operation_name or "unnamed_operation"
        with self.track(operation_name):
            self._propagate_context(operation_name)

            self.logger.info(
                f"Executing operation: {request.operation_name}",
                extra={
                    "timeout_seconds": request.timeout_seconds,
                    "has_retry_config": bool(request.retry_config),
                    "correlation_id": self._get_correlation_id(),
                },
            )

            # Validate business rules before execution (Domain.Service protocol)
            business_rules_result = self.validate_business_rules()
            if business_rules_result.is_failure:
                self.logger.error(
                    f"Business rules validation failed for operation: {request.operation_name}",
                    extra={"error": business_rules_result.error},
                )
                return FlextResult[TDomainResult].fail(
                    f"Business rules validation failed: {business_rules_result.error}"
                )

            # Validate configuration (Domain.Service protocol)
            config_result = self.validate_config()
            if config_result.is_failure:
                self.logger.error(
                    f"Configuration validation failed for operation: {request.operation_name}",
                    extra={"error": config_result.error},
                )
                return FlextResult[TDomainResult].fail(
                    f"Configuration validation failed: {config_result.error}"
                )

            # Validate keyword_arguments is a dict
            if not isinstance(request.keyword_arguments, dict):
                return FlextResult[TDomainResult].fail(
                    f"Invalid keyword arguments: expected dict, got {type(request.keyword_arguments).__name__}"
                )

            # Execute with retry logic if configured
            retry_config = request.retry_config or {}

            # Validate retry config types
            max_attempts_raw = retry_config.get("max_attempts", 1) or 1
            if not isinstance(max_attempts_raw, int):
                return FlextResult[TDomainResult].fail(
                    f"Invalid retry configuration: max_attempts must be an integer, got {type(max_attempts_raw).__name__}"
                )

            initial_delay_raw = retry_config.get("initial_delay_seconds", 0.1) or 0.1
            if not isinstance(initial_delay_raw, (int, float)):
                return FlextResult[TDomainResult].fail(
                    f"Invalid retry configuration: initial_delay_seconds must be numeric, got {type(initial_delay_raw).__name__}"
                )

            max_delay_raw = retry_config.get("max_delay_seconds", 60.0) or 60.0
            if not isinstance(max_delay_raw, (int, float)):
                return FlextResult[TDomainResult].fail(
                    f"Invalid retry configuration: max_delay_seconds must be numeric, got {type(max_delay_raw).__name__}"
                )

            # Validate backoff_multiplier if present
            backoff_multiplier_raw = retry_config.get("backoff_multiplier")
            if backoff_multiplier_raw is not None:
                if not isinstance(backoff_multiplier_raw, (int, float)):
                    return FlextResult[TDomainResult].fail(
                        f"Invalid retry configuration: backoff_multiplier must be numeric, got {type(backoff_multiplier_raw).__name__}"
                    )
                if backoff_multiplier_raw < 1.0:
                    return FlextResult[TDomainResult].fail(
                        "Invalid retry configuration: backoff_multiplier must be >= 1.0"
                    )

            max_attempts: int = max_attempts_raw
            initial_delay: float = cast("float", initial_delay_raw)
            max_delay: float = cast("float", max_delay_raw)
            exponential_backoff: bool = cast(
                "bool", retry_config.get("exponential_backoff", False)
            )
            retry_on_exceptions_raw = retry_config.get(
                "retry_on_exceptions", [Exception]
            )
            retry_on_exceptions: list[type[Exception]] = cast(
                "list[type[Exception]]", retry_on_exceptions_raw or [Exception]
            )

            last_exception: Exception | None = None

            for attempt in range(max_attempts):
                try:
                    # Filter out None values from arguments
                    filtered_args = [
                        v for v in request.arguments.values() if v is not None
                    ]

                    # Apply timeout if specified
                    if request.timeout_seconds and request.timeout_seconds > 0:
                        if not callable(request.operation_callable):
                            return FlextResult[TDomainResult].fail(
                                f"operation_callable must be callable, got {type(request.operation_callable)}"
                            )
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=1
                        ) as executor:
                            future = executor.submit(
                                request.operation_callable,
                                *filtered_args,
                                **request.keyword_arguments,
                            )
                            try:
                                result = future.result(timeout=request.timeout_seconds)
                            except concurrent.futures.TimeoutError:
                                return FlextResult[TDomainResult].fail(
                                    f"Operation {request.operation_name} timed out after {request.timeout_seconds} seconds"
                                )
                    else:
                        # Execute the operation without timeout
                        if not callable(request.operation_callable):
                            return FlextResult[TDomainResult].fail(
                                f"operation_callable must be callable, got {type(request.operation_callable)}"
                            )
                        result = request.operation_callable(
                            *filtered_args, **request.keyword_arguments
                        )

                    self.logger.info(
                        f"Operation completed successfully: {request.operation_name}"
                    )

                    # If result is already a FlextResult, return it directly
                    if isinstance(result, FlextResult):
                        return cast("FlextResult[TDomainResult]", result)

                    result_value: TDomainResult = cast("TDomainResult", result)
                    return FlextResult[TDomainResult].ok(result_value)

                except Exception as e:
                    last_exception = e

                    # Check if we should retry this exception
                    should_retry = any(
                        isinstance(e, exc_type) for exc_type in retry_on_exceptions
                    )

                    if not should_retry or attempt >= max_attempts - 1:
                        self.logger.exception(
                            f"Operation execution failed: {request.operation_name}",
                            extra={"error": str(e), "error_type": type(e).__name__},
                        )
                        return FlextResult[TDomainResult].fail(
                            f"Operation {request.operation_name} failed: {e}"
                        )

                    # Calculate delay for retry
                    if exponential_backoff:
                        delay = min(initial_delay * (2**attempt), max_delay)
                    else:
                        delay = min(initial_delay, max_delay)

                    self.logger.warning(
                        f"Operation {request.operation_name} failed (attempt {attempt + 1}/{max_attempts}), retrying in {delay}s",
                        extra={"error": str(e), "error_type": type(e).__name__},
                    )

                    time.sleep(delay)

            # Should not reach here, but handle it
            if last_exception:
                return FlextResult[TDomainResult].fail(
                    f"Operation {request.operation_name} failed: {last_exception}"
                )
            return FlextResult[TDomainResult].fail(
                f"Operation {request.operation_name} failed after {max_attempts} attempts"
            )

    def execute_with_full_validation(
        self, _request: FlextModels.DomainServiceExecutionRequest
    ) -> FlextResult[TDomainResult]:
        """Execute operation with full validation including business rules and config.

        Args:
            request: Domain service execution request

        Returns:
            FlextResult[TDomainResult]: Success with operation result or failure with validation/execution error

        """
        # Full validation: business rules + config + execution
        business_rules_result = self.validate_business_rules()
        if business_rules_result.is_failure:
            return FlextResult[TDomainResult].fail(
                f"Business rules validation failed: {business_rules_result.error}"
            )

        config_result = self.validate_config()
        if config_result.is_failure:
            return FlextResult[TDomainResult].fail(
                f"Configuration validation failed: {config_result.error}"
            )

        # Execute the operation and cast to object result type for API compatibility
        return self.execute()

    def execute_conditionally(
        self, request: FlextModels.ConditionalExecutionRequest
    ) -> FlextResult[TDomainResult]:
        """Execute operation conditionally based on the provided condition.

        Args:
            request: Conditional execution request

        Returns:
            FlextResult[TDomainResult]: Success with domain result or failure

        """
        # Evaluate condition
        try:
            condition_met = bool(request.condition(self))
        except Exception as e:
            return FlextResult[TDomainResult].fail(f"Condition evaluation failed: {e}")

        if not condition_met:
            # Condition not met, check if there's a false action
            if hasattr(request, "false_action") and request.false_action is not None:
                try:
                    false_result: object = None
                    if callable(request.false_action):
                        false_result = request.false_action(self)
                        # If the action returns a FlextResult, return it directly
                        if isinstance(false_result, FlextResult):
                            false_flext_result: FlextResult[TDomainResult] = (
                                false_result
                            )
                            return false_flext_result
                        false_result_value: TDomainResult = cast(
                            "TDomainResult", false_result
                        )
                        return FlextResult[TDomainResult].ok(false_result_value)
                    false_action_value: TDomainResult = cast(
                        "TDomainResult", request.false_action
                    )
                    return FlextResult[TDomainResult].ok(false_action_value)
                except Exception as e:
                    return FlextResult[TDomainResult].fail(
                        f"False action execution failed: {e}"
                    )
            else:
                return FlextResult[TDomainResult].fail("Condition not met")

        # Condition met, check if there's a true action
        if hasattr(request, "true_action") and request.true_action is not None:
            try:
                if callable(request.true_action):
                    true_result = request.true_action(self)
                    # If the action returns a FlextResult, return it directly
                    if isinstance(true_result, FlextResult):
                        true_flext_result: FlextResult[TDomainResult] = true_result
                        return true_flext_result
                    true_result_value: TDomainResult = cast(
                        "TDomainResult", true_result
                    )
                    return FlextResult[TDomainResult].ok(true_result_value)
                true_action_value: TDomainResult = cast(
                    "TDomainResult", request.true_action
                )
                return FlextResult[TDomainResult].ok(true_action_value)
            except Exception as e:
                return FlextResult[TDomainResult].fail(
                    f"True action execution failed: {e}"
                )

        # No specific action, execute the default operation
        return self.execute()

    def execute_with_timeout(
        self, timeout_seconds: float
    ) -> FlextResult[TDomainResult]:
        """Execute operation with timeout handling.

        Args:
            timeout_seconds: Maximum execution time in seconds

        Returns:
            FlextResult[TDomainResult]: Success with result or failure with timeout error

        """

        def timeout_handler(_signum: object, _frame: object) -> None:
            msg = f"Operation timed out after {timeout_seconds} seconds"
            raise TimeoutError(msg)

        # Set up the timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))

        try:
            return self.execute()
        except TimeoutError as e:
            return FlextResult[TDomainResult].fail(str(e))
        finally:
            # Restore the old handler and cancel the alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    # Helper classes for advanced service operations
    class _ExecutionHelper:
        """Helper class for execution-related operations utilities."""

        @staticmethod
        def prepare_execution_context(
            service: FlextService[
                object
            ],  # Generic service - works with any TDomainResult
        ) -> FlextTypes.Dict:
            """Prepare execution context for a service."""
            context: FlextTypes.Dict = {
                "service_type": service.__class__.__name__,
                "service_name": getattr(service, "_service_name", None),
                "timestamp": datetime.now(UTC),
            }
            return context

        @staticmethod
        def cleanup_execution_context(
            service: FlextService[object],
            context: FlextTypes.Dict,  # Generic service
        ) -> None:
            """Clean up execution context after operation."""
            # Basic cleanup - could be extended for more complex operations

    class _MetadataHelper:
        """Helper class for metadata extraction and formatting utilities."""

        @staticmethod
        def extract_service_metadata(
            service: FlextService[object],
            *,
            include_timestamps: bool = True,  # Generic service
        ) -> FlextTypes.Dict:
            """Extract metadata from a service instance."""
            metadata: FlextTypes.Dict = {
                "service_class": service.__class__.__name__,
                "service_name": getattr(service, "_service_name", None),
                "service_module": service.__class__.__module__,
            }

            if include_timestamps:
                now = datetime.now(UTC)
                metadata["created_at"] = now
                metadata["extracted_at"] = now

            return metadata

        @staticmethod
        def format_service_info(
            _service: FlextService[object],
            metadata: FlextTypes.Dict,  # Generic service
        ) -> str:
            """Format service information for display."""
            return f"Service: {metadata.get('service_type', 'Unknown')} ({metadata.get('service_name', 'unnamed')})"


__all__: FlextTypes.StringList = [
    "FlextService",
]
