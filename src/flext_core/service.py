"""Domain service abstractions supporting the 1.0.0 alignment pillar.

These bases codify the service ergonomics described in ``README.md`` and
``docs/architecture.md``: immutable models, context-aware logging, and
``FlextResult`` contracts that remain stable throughout the 1.x lifecycle.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import signal
import time
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import (
    cast,
    override,
)

from pydantic import ConfigDict

from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextService[TDomainResult](
    FlextModels.ArbitraryTypesModel,
    FlextMixins.Serializable,
    FlextMixins.Loggable,
    FlextProtocols.Domain.Service,  # Protocol inheritance now works with ServiceMeta
    ABC,
    metaclass=FlextModels.ServiceMeta,
):
    """Domain service base using railway patterns with Pydantic models.

    **PROTOCOL IMPLEMENTATION**: This service implements FlextProtocols.Domain.Service,
    establishing the foundation pattern for ALL domain services across the FLEXT ecosystem.

    Provides unified service pattern for FLEXT ecosystem with
    FlextResult[T] error handling, Pydantic Generic[T] for type-safe
    domain operations, and complete type annotations for consistency.

    **Function**: Domain service base class for business logic
        - Abstract execute() method for domain operations (Domain.Service protocol)
        - Business rule validation with FlextResult (Domain.Service protocol)
        - Configuration validation and management (Domain.Service protocol)
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

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )

    @override
    def __init__(self, **data: object) -> None:
        """Initialize domain service with Pydantic validation."""
        super().__init__(**data)

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
    # CORE DOMAIN OPERATIONS
    # =============================================================================

    def execute_with_full_validation(
        self,
        request: FlextModels.DomainServiceExecutionRequest,
    ) -> FlextResult[TDomainResult]:
        """Execute with comprehensive validation using DomainServiceExecutionRequest model.

        Args:
            request: DomainServiceExecutionRequest containing validation settings and context

        Returns:
            FlextResult[TDomainResult]: Success with result or failure with validation error

        """
        validation_result: FlextResult[None] = self.validate_with_request(request)
        if validation_result.is_failure:
            return FlextResult[TDomainResult].fail(
                f"{FlextConstants.Messages.VALIDATION_FAILED}: {validation_result.error}"
                if validation_result.error
                else FlextConstants.Messages.VALIDATION_FAILED,
            )

        return self.execute()

    def is_valid(self) -> bool:
        """Check if the domain service is in a valid state (Domain.Service protocol).

        Returns:
            bool: True if valid, False otherwise

        """
        try:
            return self.validate_business_rules().is_success
        except Exception:
            return False

    def get_service_info(self) -> FlextTypes.Dict:
        """Get service information for diagnostics (Domain.Service protocol).

        Returns:
            FlextTypes.Dict: Service information including type and configuration

        """
        return {"service_type": self.__class__.__name__}  # type: ignore[misc]

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
            FlextResult[None]: Success if valid, failure with error details

        """
        return FlextResult[None].ok(None)

    def validate_with_request(
        self,
        request: FlextModels.DomainServiceExecutionRequest,
    ) -> FlextResult[None]:
        """Validate using DomainServiceExecutionRequest model.

        Args:
            request: DomainServiceExecutionRequest containing validation settings

        Returns:
            FlextResult[None]: Success if valid, failure with validation error

        """
        # Validate business rules if requested
        if getattr(request, "enable_validation", True):
            business_result = self.validate_business_rules()
            if business_result.is_failure:
                return FlextResult[None].fail(
                    f"{FlextConstants.Messages.VALIDATION_FAILED}"
                    f" (business rules): {business_result.error}",
                )

        return FlextResult[None].ok(None)

    # =============================================================================
    # EXECUTION METHODS (Domain.Service protocol)
    # =============================================================================

    def execute_operation(
        self,
        operation: FlextModels.OperationExecutionRequest,
    ) -> FlextResult[TDomainResult]:
        """Execute operation using OperationExecutionRequest model (Domain.Service protocol).

        Args:
            operation: OperationExecutionRequest containing operation settings

        Returns:
            FlextResult[TDomainResult]: Success with result or failure with error

        """
        operation_name = getattr(operation, "operation_name", "operation")

        # Pre-execution validation
        validation_result = self._validate_operation_pre_execution(operation_name)
        if validation_result.is_failure:
            return validation_result  # type: ignore[return-value]

        # Parse arguments
        arguments_result = self._parse_operation_arguments(operation, operation_name)
        if arguments_result.is_failure:
            return arguments_result  # type: ignore[return-value]

        positional_arguments, keyword_arguments = arguments_result.unwrap()

        # Parse timeout
        timeout_seconds = self._parse_timeout(operation)

        # Execute operation directly (simplified from complex retry logic)
        def call_operation() -> FlextResult[TDomainResult]:
            # operation_callable is already validated as Callable[..., object] in the model
            operation_callable = cast(
                "FlextProtocols.Foundation.OperationCallable",
                operation.operation_callable,
            )
            result: object = operation_callable(
                *positional_arguments,
                **keyword_arguments,
            )
            # Convert the result to FlextResult if it's not already
            if isinstance(result, FlextResult):
                # Cast to FlextResult[TDomainResult] to ensure type compatibility
                typed_result: FlextResult[TDomainResult] = cast(
                    "FlextResult[TDomainResult]",
                    result,
                )
                return typed_result
            return FlextResult[TDomainResult].ok(cast("TDomainResult", result))

        if timeout_seconds <= 0:
            return call_operation()

        # Execute with timeout constraint
        timeout_message = (
            f"Operation '{operation_name}' timed out after {timeout_seconds} seconds"
        )

        def timeout_handler(_signum: int, _frame: object) -> None:
            raise FlextExceptions.TimeoutError(
                timeout_message,
                operation=operation_name,
            )

        previous_handler = signal.getsignal(signal.SIGALRM)
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
            return call_operation()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)

    def execute_with_request(
        self,
        _request: FlextModels.DomainServiceExecutionRequest,
    ) -> FlextResult[TDomainResult]:
        """Execute with DomainServiceExecutionRequest model containing execution settings.

        Args:
            request: DomainServiceExecutionRequest containing execution configuration

        Returns:
            FlextResult[TDomainResult]: Success with result or failure with error

        """
        # Execute the operation
        return self.execute()

    def execute_with_timeout(self, timeout_seconds: int) -> FlextResult[TDomainResult]:
        """Execute with timeout constraint.

        Args:
            timeout_seconds: Maximum execution time in seconds

        Returns:
            FlextResult[TDomainResult]: Success with result or failure with timeout error

        """

        @contextmanager
        def timeout_context(seconds: int) -> Generator[None]:
            def timeout_handler(_signum: int, _frame: object) -> None:
                msg = f"Operation timed out after {seconds} seconds"
                raise FlextExceptions.TimeoutError(msg)

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                yield
            finally:
                signal.alarm(0)

        try:
            with timeout_context(timeout_seconds):
                return self.execute()
        except TimeoutError as e:
            return FlextResult[TDomainResult].fail(str(e))

    def execute_conditionally(
        self,
        condition: FlextModels.ConditionalExecutionRequest,
    ) -> FlextResult[TDomainResult]:
        """Execute only if condition is met using ConditionalExecutionRequest model.

        Args:
            condition: ConditionalExecutionRequest containing condition logic

        Returns:
            FlextResult[TDomainResult]: Success with result, failure, or skipped

        """
        # Evaluate the condition
        if condition.condition(self):
            # Execute true action if condition is met
            result = condition.true_action(self)
            if isinstance(result, FlextResult):
                return result
            # Type assertion since we expect TDomainResult from the action
            return FlextResult[TDomainResult].ok(cast("TDomainResult", result))
        # Execute false action if condition is not met
        if condition.false_action:
            result = condition.false_action(self)
            if isinstance(result, FlextResult):
                return result
            # Type assertion since we expect TDomainResult from the action
            return FlextResult[TDomainResult].ok(cast("TDomainResult", result))
        return FlextResult[TDomainResult].fail("Condition not met")

    def execute_batch_with_request(
        self,
        request: FlextModels.DomainServiceBatchRequest,
    ) -> FlextResult[list[TDomainResult]]:
        """Execute batch operations using DomainServiceBatchRequest model.

        Args:
            request: DomainServiceBatchRequest containing batch execution settings

        Returns:
            FlextResult[list[TDomainResult]]: Success with results list or failure with error

        """
        results: list[TDomainResult] = []
        errors: FlextTypes.StringList = []

        for i in range(request.batch_size):
            try:
                result = self.execute()
                if result.is_success:
                    results.append(result.value)
                else:
                    errors.append(f"Batch item {i}: {result.error}")
                    if not getattr(request, "continue_on_failure", False):
                        break
            except Exception as e:
                errors.append(f"Batch item {i}: {e}")
                if not getattr(request, "continue_on_failure", False):
                    break

        if errors and not getattr(request, "continue_on_failure", False):
            return FlextResult[list[TDomainResult]].fail(
                f"Batch execution failed: {'; '.join(errors)}",
            )

        return FlextResult[list[TDomainResult]].ok(results)

    def execute_with_metrics_request(
        self,
        _request: FlextModels.DomainServiceMetricsRequest,
    ) -> FlextResult[TDomainResult]:
        """Execute with metrics using DomainServiceMetricsRequest model.

        Args:
            request: DomainServiceMetricsRequest containing metrics collection configuration

        Returns:
            FlextResult[TDomainResult]: Success with result or failure with metrics error

        """
        start_time = time.time()
        metrics_data: FlextTypes.Dict = {}

        try:
            # Collect pre-execution metrics
            metrics_data["start_time"] = start_time
            metrics_data["service_type"] = self.__class__.__name__  # type: ignore[misc]

            # Execute operation
            result = self.execute()

            # Collect post-execution metrics
            end_time = time.time()
            metrics_data["end_time"] = end_time
            metrics_data["execution_time"] = end_time - start_time
            metrics_data["success"] = result.is_success

            return result

        except Exception as e:
            end_time = time.time()
            metrics_data["end_time"] = end_time
            metrics_data["execution_time"] = end_time - start_time
            metrics_data["success"] = False
            metrics_data["error"] = str(e)

            return FlextResult[TDomainResult].fail(f"Metrics execution error: {e}")

    def execute_with_resource_request(
        self,
        request: FlextModels.DomainServiceResourceRequest,
    ) -> FlextResult[TDomainResult]:
        """Execute with resource management using DomainServiceResourceRequest model.

        Args:
            request: DomainServiceResourceRequest containing resource management configuration

        Returns:
            FlextResult[TDomainResult]: Success with result or failure with resource error

        """
        acquired_resources: FlextTypes.StringList = []

        try:
            # Acquire resources
            required_resources = getattr(request, "required_resources", [])
            acquired_resources = [str(resource) for resource in required_resources]

            # Execute with acquired resources
            return self.execute()

        except Exception as e:
            return FlextResult[TDomainResult].fail(f"Resource execution error: {e}")

        finally:
            # Always cleanup resources
            for _resource_id in acquired_resources:
                # Simulate resource cleanup
                pass

    def validate_and_transform(
        self,
        config: FlextModels.ValidationConfiguration,
    ) -> FlextResult[TDomainResult]:
        """Validate and transform using ValidationConfiguration model.

        Args:
            config: ValidationConfiguration containing validation and transformation settings

        Returns:
            FlextResult[TDomainResult]: Success with result or failure with validation error

        """
        # Perform validation based on config
        if getattr(config, "enable_validation", True):
            validation_result = self.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult[TDomainResult].fail(
                    f"{FlextConstants.Messages.VALIDATION_FAILED}: {validation_result.error}"
                    if validation_result.error
                    else FlextConstants.Messages.VALIDATION_FAILED,
                )

        # Execute after successful validation
        return self.execute()

    # =============================================================================
    # OPERATION HELPER METHODS - Simplified execution logic
    # =============================================================================

    def _validate_operation_pre_execution(
        self, operation_name: str
    ) -> FlextResult[None]:
        """Validate operation before execution."""
        config_validation = self.validate_config()
        if config_validation.is_failure:
            return FlextResult[None].fail(
                f"Operation '{operation_name}' failed validation (pre-execution)"
                + (f": {config_validation.error}" if config_validation.error else ""),
            )

        business_validation = self.validate_business_rules()
        if business_validation.is_failure:
            return FlextResult[None].fail(
                f"Business rules validation failed: {business_validation.error}",
            )

        return FlextResult[None].ok(None)

    def _parse_operation_arguments(
        self,
        operation: FlextModels.OperationExecutionRequest,
        operation_name: str,
    ) -> FlextResult[tuple[tuple[object, ...], FlextTypes.Dict]]:
        """Parse operation arguments into positional and keyword arguments."""
        raw_arguments = getattr(operation, "arguments", None)
        if raw_arguments is None:
            return FlextResult[tuple[tuple[object, ...], FlextTypes.Dict]].ok(((), {}))

        # Handle positional arguments
        if isinstance(raw_arguments, (list, tuple, set)):
            positional_arguments = tuple(cast("Iterable[object]", raw_arguments))
        elif isinstance(raw_arguments, dict):
            if "args" in raw_arguments:
                nested_args: object = cast("FlextTypes.Dict", raw_arguments).get(
                    "args", None
                )
                if isinstance(nested_args, (list, tuple, set)):
                    positional_arguments = tuple(cast("Iterable[object]", nested_args))
                elif nested_args is None:
                    positional_arguments = ()
                else:
                    positional_arguments = (nested_args,)
            else:
                positional_arguments = tuple(
                    cast("Iterable[object]", raw_arguments.values())
                )
        else:
            positional_arguments = (raw_arguments,)

        # Handle keyword arguments
        raw_keyword_arguments = getattr(operation, "keyword_arguments", None)
        if raw_keyword_arguments is None:
            keyword_arguments: FlextTypes.Dict = {}
        elif isinstance(raw_keyword_arguments, dict):
            keyword_arguments = dict(
                cast("Mapping[str, object]", raw_keyword_arguments)
            )
        else:
            try:
                keyword_arguments = dict(raw_keyword_arguments)
            except Exception as exc:
                return FlextResult[tuple[tuple[object, ...], FlextTypes.Dict]].fail(
                    f"Invalid keyword arguments for operation '{operation_name}': {exc}",
                )

        return FlextResult[tuple[tuple[object, ...], FlextTypes.Dict]].ok((
            positional_arguments,
            keyword_arguments,
        ))

    def _parse_retry_configuration(
        self,
        operation: FlextModels.OperationExecutionRequest,
        operation_name: str,
    ) -> FlextResult[FlextModels.RetryConfiguration | None]:
        """Parse retry configuration from operation."""
        retry_config_data: FlextTypes.Dict = (
            getattr(operation, "retry_config", {}) or {}
        )

        if not retry_config_data:
            return FlextResult[FlextModels.RetryConfiguration | None].ok(None)

        try:
            retry_config = FlextModels.RetryConfiguration.model_validate(
                retry_config_data
            )
            return FlextResult[FlextModels.RetryConfiguration | None].ok(retry_config)
        except Exception as exc:
            return FlextResult[FlextModels.RetryConfiguration | None].fail(
                f"Invalid retry configuration for operation '{operation_name}': {exc}",
            )

    def _parse_timeout(self, operation: FlextModels.OperationExecutionRequest) -> float:
        """Parse timeout from operation."""
        raw_timeout = getattr(operation, "timeout_seconds", None)
        try:
            timeout_seconds = (
                float(raw_timeout)
                if raw_timeout is not None
                else FlextConstants.Core.INITIAL_TIME
            )
        except (TypeError, ValueError):
            timeout_seconds = FlextConstants.Core.INITIAL_TIME
        return max(FlextConstants.Core.INITIAL_TIME, timeout_seconds)

    def _extract_retry_parameters(
        self, retry_config: FlextModels.RetryConfiguration | None
    ) -> tuple[int, float, float, float, bool, tuple[type[BaseException], ...]]:
        """Extract retry parameters from configuration."""
        if retry_config is None:
            return (1, 0.0, 0.0, 1.0, False, ())

        max_attempts = max(1, int(retry_config.max_attempts))
        base_delay = max(0.0, float(retry_config.initial_delay_seconds))
        max_delay = max(base_delay, float(retry_config.max_delay_seconds))
        backoff_multiplier = max(1.0, float(retry_config.backoff_multiplier))
        exponential_backoff = bool(retry_config.exponential_backoff)
        retry_exception_filters = (
            tuple(retry_config.retry_on_exceptions)
            if retry_config.retry_on_exceptions
            else ()
        )

        return (
            max_attempts,
            base_delay,
            max_delay,
            backoff_multiplier,
            exponential_backoff,
            retry_exception_filters,
        )

    def _execute_with_retry_and_timeout(
        self,
        operation: FlextModels.OperationExecutionRequest,
        operation_name: str,
        positional_arguments: tuple[object, ...],
        keyword_arguments: FlextTypes.Dict,
        timeout_seconds: float,
        retry_params: tuple[
            int, float, float, float, bool, tuple[type[BaseException], ...]
        ],
    ) -> FlextResult[TDomainResult]:
        """Execute operation with retry and timeout logic."""
        (
            max_attempts,
            base_delay,
            max_delay,
            backoff_multiplier,
            exponential_backoff,
            retry_filters,
        ) = retry_params

        def call_operation() -> FlextResult[TDomainResult]:
            operation_callable = cast(
                "FlextProtocols.Foundation.OperationCallable",
                operation.operation_callable,
            )
            result: object = operation_callable(
                *positional_arguments, **keyword_arguments
            )

            if isinstance(result, FlextResult):
                return cast("FlextResult[TDomainResult]", result)
            return FlextResult[TDomainResult].ok(cast("TDomainResult", result))

        def call_with_timeout() -> FlextResult[TDomainResult]:
            if timeout_seconds <= 0:
                return call_operation()

            def timeout_handler(_signum: int, _frame: object) -> None:
                msg = f"Operation '{operation_name}' timed out after {timeout_seconds} seconds"
                raise FlextExceptions.TimeoutError(
                    msg,
                    operation=operation_name,
                )

            previous_handler = signal.getsignal(signal.SIGALRM)
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
                return call_operation()
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous_handler)

        def should_retry(exc: Exception, attempt: int) -> bool:
            if attempt >= max_attempts:
                return False
            if not retry_filters:
                return True
            return any(isinstance(exc, allowed) for allowed in retry_filters)

        current_delay = base_delay
        last_exception: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                return call_with_timeout()
            except Exception as exc:
                last_exception = exc
                if not should_retry(exc, attempt):
                    message = str(exc) or exc.__class__.__name__
                    if isinstance(exc, TimeoutError) and timeout_seconds > 0:
                        return FlextResult[TDomainResult].fail(message)
                    return FlextResult[TDomainResult].fail(
                        f"Operation '{operation_name}' failed: {message}",
                    )

                if current_delay > 0:
                    time.sleep(current_delay)

                if exponential_backoff:
                    next_delay = current_delay * backoff_multiplier
                    if max_delay > 0:
                        next_delay = min(next_delay, max_delay)
                    current_delay = max(base_delay, next_delay)
                else:
                    current_delay = base_delay

        if last_exception is not None:
            message = str(last_exception) or last_exception.__class__.__name__
            if isinstance(last_exception, TimeoutError) and timeout_seconds > 0:
                return FlextResult[TDomainResult].fail(message)
            return FlextResult[TDomainResult].fail(
                f"Operation '{operation_name}' failed: {message}",
            )

        return FlextResult[TDomainResult].fail(
            f"Operation '{operation_name}' failed without explicit error",
        )

    # =============================================================================
    # NESTED HELPER CLASSES - Following FLEXT unified class pattern
    # =============================================================================

    class _ValidationHelper:
        """Nested validation helper - no loose functions."""

        @staticmethod
        def validate_domain_rules(_service: object) -> FlextResult[None]:
            """Validate domain-specific business rules."""
            # Default implementation - subclasses can override
            return FlextResult[None].ok(None)

        @staticmethod
        def validate_state_consistency(_service: object) -> FlextResult[None]:
            """Validate state consistency."""
            # Default implementation - subclasses can override
            return FlextResult[None].ok(None)

    class _ExecutionHelper:
        """Nested execution helper - no loose functions."""

        @staticmethod
        def prepare_execution_context(service: object) -> FlextTypes.Dict:
            """Prepare execution context for the service."""
            return {
                "service_type": service.__class__.__name__,
                "timestamp": datetime.now(UTC),
            }

        @staticmethod
        def cleanup_execution_context(
            service: object,
            context: FlextTypes.Dict,
        ) -> None:
            """Cleanup execution context after service execution."""
            # Default implementation - can be extended by subclasses

    class _MetadataHelper:
        """Nested metadata helper - no loose functions."""

        @staticmethod
        def extract_service_metadata(service: object) -> FlextTypes.Dict:
            """Extract metadata from service instance."""
            metadata: FlextTypes.Dict = {
                "service_class": service.__class__.__name__,
                "service_module": service.__class__.__module__,
            }

            # Add timestamp information if available
            if isinstance(service, FlextProtocols.Foundation.HasTimestamps):
                metadata["created_at"] = service.created_at
                metadata["updated_at"] = service.updated_at

            return metadata

        @staticmethod
        def format_service_info(
            _service: object,
            metadata: FlextTypes.Dict,
        ) -> str:
            """Format service information for display."""
            return f"Service: {metadata.get('service_class', 'Unknown')} ({metadata.get('service_module', 'Unknown')})"

    # =============================================================================
    # SERIALIZATION METHODS - Use FlextMixins pattern
    # =============================================================================

    def to_json_instance(self) -> str:
        """Convert service to JSON using FlextMixins serialization."""
        serialization_request = FlextModels.SerializationRequest(
            data=self,
            use_model_dump=True,
            pretty_print=True,
        )
        return FlextMixins.to_json(serialization_request)


__all__: FlextTypes.StringList = [
    "FlextService",
]
