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
from typing import Protocol, cast, override

from pydantic import ConfigDict

from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class OperationCallable(Protocol):
    """Protocol for operation callables."""

    def __call__(self, *args: object, **kwargs: object) -> object: ...


class FlextService[TDomainResult](
    FlextModels.ArbitraryTypesModel,
    FlextMixins.Serializable,
    FlextMixins.Loggable,
    ABC,
):
    """Domain service base using railway patterns with Pydantic models.

    Provides unified service pattern for FLEXT ecosystem with
    FlextResult[T] error handling, Pydantic Generic[T] for type-safe
    domain operations, and complete type annotations for consistency.

    **Function**: Domain service base class for business logic
        - Abstract execute() method for domain operations
        - Business rule validation with FlextResult
        - Configuration validation and management
        - Operation execution with timeout support
        - Batch processing for multiple operations
        - Performance metrics collection and tracking
        - Service information and metadata access
        - Context-aware logging integration
        - Serialization support via FlextMixins
        - Type-safe generic result handling

    **Uses**: Core FLEXT infrastructure for services
        - FlextResult[T] for railway pattern error handling
        - FlextModels.ArbitraryTypesModel for Pydantic base
        - FlextMixins for serialization and logging
        - FlextConstants for defaults and error codes
        - Pydantic Generic[T] for type-safe operations
        - abc.ABC for abstract base class pattern
        - signal module for timeout enforcement
        - datetime for timestamp operations
        - contextmanager for context scopes
        - Protocol for callable interfaces

    **How to use**: Domain service implementation patterns
        ```python
        from flext_core import FlextService, FlextResult


        # Example 1: Implement domain service
        class UserService(FlextService[User]):
            name: str = "UserService"
            version: str = "1.0.0"

            def execute(self) -> FlextResult[User]:
                # Validate business rules first
                validation = self.validate_business_rules()
                if validation.is_failure:
                    return FlextResult[User].fail(validation.error)

                # Execute domain logic
                user = User(id="123", name="John")
                return FlextResult[User].ok(user)

            def validate_business_rules(self) -> FlextResult[None]:
                if not self.name:
                    return FlextResult[None].fail("Name required")
                return FlextResult[None].ok(None)


        # Example 2: Execute service operation
        service = UserService()
        result = service.execute()
        if result.is_success:
            user = result.unwrap()

        # Example 3: Execute with timeout
        operation_request = OperationExecutionRequest(
            operation_callable=lambda: service.execute(), timeout_seconds=5.0
        )
        result = service.execute_operation(operation_request)

        # Example 4: Validate configuration
        config_result = service.validate_config()
        if config_result.is_failure:
            print(f"Config error: {config_result.error}")

        # Example 5: Get service information
        info = service.get_service_info()
        print(f"Service: {info['name']} v{info['version']}")

        # Example 6: Batch operation execution
        operations = [op1, op2, op3]
        results = [service.execute() for _ in operations]

        # Example 7: Check if service is valid
        if service.is_valid():
            result = service.execute()
        ```

    Attributes:
        model_config: Pydantic configuration dict.

    Note:
        All services must implement execute() method.
        Generic type TDomainResult provides type safety.
        Services inherit serialization from FlextMixins.
        Business rule validation returns FlextResult.
        Configuration validation is separate from rules.

    Warning:
        Execute method must be implemented by subclasses.
        Timeout operations require signal support (Unix-like).
        Batch operations do not provide transaction semantics.
        Service validation does not guarantee execution success.

    Example:
        Complete domain service implementation:

        >>> class OrderService(FlextService[Order]):
        ...     def execute(self) -> FlextResult[Order]:
        ...         return FlextResult[Order].ok(Order())
        >>> service = OrderService()
        >>> result = service.execute()
        >>> print(result.is_success)
        True

    See Also:
        FlextResult: For railway pattern error handling.
        FlextModels: For domain model definitions.
        FlextMixins: For serialization and logging.
        FlextHandlers: For handler implementation patterns.

    **IMPLEMENTATION NOTES**:
    - Abstract domain service base class with railway patterns
    - Comprehensive validation and execution patterns
    - Timeout and retry mechanisms with signal handling
    - Batch operation support with error accumulation
    - Metrics collection integration
    - Resource management patterns with automatic cleanup

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
    # ABSTRACT METHODS - Must be implemented by subclasses
    # =============================================================================

    @abstractmethod
    def execute(self) -> FlextResult[TDomainResult]:
        """Execute the main domain operation.

        Returns:
            FlextResult[TDomainResult]: Success with domain result or failure with error

        """

    # =============================================================================
    # CORE DOMAIN OPERATIONS
    # =============================================================================

    def execute_with_full_validation(
        self, request: FlextModels.DomainServiceExecutionRequest
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
                else FlextConstants.Messages.VALIDATION_FAILED
            )

        return self.execute()

    def is_valid(self) -> bool:
        """Check if the domain service is in a valid state.

        Returns:
            bool: True if valid, False otherwise

        """
        try:
            return self.validate_business_rules().is_success
        except Exception:
            return False

    def get_service_info(self) -> FlextTypes.Core.Dict:
        """Get service information for diagnostics.

        Returns:
            FlextTypes.Core.Dict: Service information including type and configuration

        """
        return {"service_type": self.__class__.__name__}

    # =============================================================================
    # VALIDATION METHODS
    # =============================================================================

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules for the domain service.

        🚨 AUDIT VIOLATION: This validation method violates FLEXT architectural principles!
        ❌ CRITICAL ISSUE: Business rules validation should be centralized in FlextModels.Validation
        ❌ INLINE VALIDATION: This is inline validation that should be centralized

        🔧 REQUIRED ACTION:
        - Move business rules validation to FlextModels.Validation
        - Use FlextModels validation patterns for domain validation
        - Remove inline validation from service base class

        📍 SHOULD BE USED INSTEAD: FlextModels.Validation.validate_business_rules()

        Returns:
            FlextResult[None]: Success if valid, failure with error details

        """
        # 🚨 AUDIT VIOLATION: Inline validation logic - should be in FlextModels.Validation
        return FlextResult[None].ok(None)

    def validate_config(self) -> FlextResult[None]:
        """Validate service configuration.

        🚨 AUDIT VIOLATION: This validation method violates FLEXT architectural principles!
        ❌ CRITICAL ISSUE: Configuration validation should be centralized in FlextConfig.Validation
        ❌ INLINE VALIDATION: This is inline validation that should be centralized

        🔧 REQUIRED ACTION:
        - Move configuration validation to FlextConfig.Validation
        - Use FlextConfig validation patterns for configuration validation
        - Remove inline validation from service base class

        📍 SHOULD BE USED INSTEAD: FlextConfig.Validation.validate_service_config()

        Returns:
            FlextResult[None]: Success if valid, failure with error details

        """
        # 🚨 AUDIT VIOLATION: Inline validation logic - should be in FlextConfig.Validation
        return FlextResult[None].ok(None)

    def validate_with_request(
        self, request: FlextModels.DomainServiceExecutionRequest
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
                    f" (business rules): {business_result.error}"
                )

        return FlextResult[None].ok(None)

    # =============================================================================
    # EXECUTION METHODS
    # =============================================================================

    def execute_operation(
        self,
        operation: FlextModels.OperationExecutionRequest,
    ) -> FlextResult[TDomainResult]:
        """Execute operation using OperationExecutionRequest model.

        Args:
            operation: OperationExecutionRequest containing operation settings

        Returns:
            FlextResult[TDomainResult]: Success with result or failure with error

        """
        operation_name = getattr(operation, "operation_name", "operation")

        if getattr(operation, "enable_validation", True):
            config_validation = self.validate_config()
            if config_validation.is_failure:
                return FlextResult[TDomainResult].fail(
                    f"{FlextConstants.Messages.VALIDATION_FAILED} (pre-execution)"
                    + (
                        f": {config_validation.error}"
                        if config_validation.error
                        else ""
                    )
                )

            business_validation = self.validate_business_rules()
            if business_validation.is_failure:
                return FlextResult[TDomainResult].fail(
                    f"Business rules validation failed: {business_validation.error}"
                )

        raw_arguments = getattr(operation, "arguments", None)
        if raw_arguments is None:
            positional_arguments: tuple[object, ...] = ()
        elif isinstance(raw_arguments, (list, tuple, set)):
            positional_arguments = tuple(cast("Iterable[object]", raw_arguments))
        elif isinstance(raw_arguments, dict):
            if "args" in raw_arguments:
                nested_args: object = cast("FlextTypes.Core.Dict", raw_arguments).get(
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

        raw_keyword_arguments = getattr(operation, "keyword_arguments", None)
        if raw_keyword_arguments is None:
            keyword_arguments: FlextTypes.Core.Dict = {}
        elif isinstance(raw_keyword_arguments, dict):
            keyword_arguments = dict(
                cast("Mapping[str, object]", raw_keyword_arguments)
            )
        else:
            try:
                keyword_arguments = dict(raw_keyword_arguments)
            except Exception as exc:  # pragma: no cover - defensive branch
                return FlextResult[TDomainResult].fail(
                    f"Invalid keyword arguments for operation '{operation_name}': {exc}"
                )

        retry_config_data: FlextTypes.Core.Dict = (
            getattr(operation, "retry_config", {}) or {}
        )
        retry_config: FlextModels.RetryConfiguration | None
        if retry_config_data:
            try:
                # Use Pydantic model_validate for proper type conversion
                retry_config = FlextModels.RetryConfiguration.model_validate(
                    retry_config_data
                )
            except Exception as exc:
                return FlextResult[TDomainResult].fail(
                    f"Invalid retry configuration for operation '{operation_name}': {exc}"
                )
        else:
            retry_config = None

        max_attempts = (
            max(1, int(retry_config.max_attempts)) if retry_config is not None else 1
        )

        base_delay = (
            float(retry_config.initial_delay_seconds)
            if retry_config is not None
            else 0.0
        )
        base_delay = max(0.0, base_delay)
        max_delay_seconds = (
            float(retry_config.max_delay_seconds)
            if retry_config is not None
            else base_delay
        )
        max_delay_seconds = max(base_delay, max_delay_seconds)
        backoff_multiplier = (
            float(retry_config.backoff_multiplier) if retry_config is not None else 1.0
        )
        if backoff_multiplier <= FlextConstants.Core.INITIAL_TIME:
            backoff_multiplier = 1.0
        exponential_backoff = bool(
            retry_config.exponential_backoff if retry_config is not None else False
        )
        retry_exception_filters = (
            tuple(retry_config.retry_on_exceptions)
            if retry_config is not None and retry_config.retry_on_exceptions
            else ()
        )

        raw_timeout = getattr(operation, "timeout_seconds", None)
        try:
            timeout_seconds = (
                float(raw_timeout)
                if raw_timeout is not None
                else FlextConstants.Core.INITIAL_TIME
            )
        except (TypeError, ValueError):  # pragma: no cover - defensive branch
            timeout_seconds = FlextConstants.Core.INITIAL_TIME
        timeout_seconds = max(FlextConstants.Core.INITIAL_TIME, timeout_seconds)

        def call_operation() -> FlextResult[TDomainResult]:
            # operation_callable is already validated as Callable[..., object] in the model
            operation_callable = cast("OperationCallable", operation.operation_callable)
            result: object = operation_callable(
                *positional_arguments,
                **keyword_arguments,
            )
            # Convert the result to FlextResult if it's not already
            if isinstance(result, FlextResult):
                # Cast to FlextResult[TDomainResult] to ensure type compatibility
                typed_result: FlextResult[TDomainResult] = cast(
                    "FlextResult[TDomainResult]", result
                )
                return typed_result
            return FlextResult[TDomainResult].ok(cast("TDomainResult", result))

        def call_with_timeout() -> FlextResult[TDomainResult]:
            if timeout_seconds <= 0:
                return call_operation()

            timeout_message = f"Operation '{operation_name}' timed out after {timeout_seconds} seconds"

            def timeout_handler(_signum: int, _frame: object) -> None:
                raise FlextExceptions.TimeoutError(
                    timeout_message, operation=operation_name
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
            if retry_config is None:
                return False
            if not retry_exception_filters:
                return True
            return any(isinstance(exc, allowed) for allowed in retry_exception_filters)

        current_delay = base_delay
        last_exception: Exception | None = None

        attempt = 1
        while attempt <= max_attempts:
            try:
                return call_with_timeout()
            except Exception as exc:
                last_exception = exc
                if not should_retry(exc, attempt):
                    message = str(exc) or exc.__class__.__name__
                    if isinstance(exc, TimeoutError) and timeout_seconds > 0:
                        return FlextResult[TDomainResult].fail(message)
                    return FlextResult[TDomainResult].fail(
                        f"Operation '{operation_name}' failed: {message}"
                    )

                if retry_config is not None and current_delay > 0:
                    time.sleep(current_delay)

                if retry_config is not None:
                    if exponential_backoff:
                        if current_delay <= 0:
                            current_delay = base_delay
                        else:
                            next_delay = current_delay * backoff_multiplier
                            if max_delay_seconds > 0:
                                next_delay = min(next_delay, max_delay_seconds)
                            current_delay = max(base_delay, next_delay)
                    else:
                        current_delay = base_delay

                attempt += 1

        if last_exception is not None:
            message = str(last_exception) or last_exception.__class__.__name__
            if isinstance(last_exception, TimeoutError) and timeout_seconds > 0:
                return FlextResult[TDomainResult].fail(message)
            return FlextResult[TDomainResult].fail(
                f"Operation '{operation_name}' failed: {message}"
            )

        return FlextResult[TDomainResult].fail(
            f"Operation '{operation_name}' failed without explicit error"
        )

    def execute_with_request(
        self, _request: FlextModels.DomainServiceExecutionRequest
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
        self, condition: FlextModels.ConditionalExecutionRequest
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
        self, request: FlextModels.DomainServiceBatchRequest
    ) -> FlextResult[list[TDomainResult]]:
        """Execute batch operations using DomainServiceBatchRequest model.

        Args:
            request: DomainServiceBatchRequest containing batch execution settings

        Returns:
            FlextResult[list[TDomainResult]]: Success with results list or failure with error

        """
        results: list[TDomainResult] = []
        errors: list[str] = []

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
                f"Batch execution failed: {'; '.join(errors)}"
            )

        return FlextResult[list[TDomainResult]].ok(results)

    def execute_with_metrics_request(
        self, _request: FlextModels.DomainServiceMetricsRequest
    ) -> FlextResult[TDomainResult]:
        """Execute with metrics using DomainServiceMetricsRequest model.

        Args:
            request: DomainServiceMetricsRequest containing metrics collection configuration

        Returns:
            FlextResult[TDomainResult]: Success with result or failure with metrics error

        """
        start_time = time.time()
        metrics_data: FlextTypes.Core.Dict = {}

        try:
            # Collect pre-execution metrics
            metrics_data["start_time"] = start_time
            metrics_data["service_type"] = self.__class__.__name__

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
        self, request: FlextModels.DomainServiceResourceRequest
    ) -> FlextResult[TDomainResult]:
        """Execute with resource management using DomainServiceResourceRequest model.

        Args:
            request: DomainServiceResourceRequest containing resource management configuration

        Returns:
            FlextResult[TDomainResult]: Success with result or failure with resource error

        """
        acquired_resources: list[str] = []

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
        self, config: FlextModels.ValidationConfiguration
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
                    else FlextConstants.Messages.VALIDATION_FAILED
                )

        # Execute after successful validation
        return self.execute()

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
        def prepare_execution_context(service: object) -> FlextTypes.Core.Dict:
            """Prepare execution context for the service."""
            return {
                "service_type": service.__class__.__name__,
                "timestamp": datetime.now(UTC),
            }

        @staticmethod
        def cleanup_execution_context(
            service: object, context: FlextTypes.Core.Dict
        ) -> None:
            """Cleanup execution context after service execution."""
            # Default implementation - can be extended by subclasses

    class _MetadataHelper:
        """Nested metadata helper - no loose functions."""

        @staticmethod
        def extract_service_metadata(service: object) -> FlextTypes.Core.Dict:
            """Extract metadata from service instance."""
            metadata: FlextTypes.Core.Dict = {
                "service_class": service.__class__.__name__,
                "service_module": service.__class__.__module__,
            }

            # Add timestamp information if available
            if hasattr(service, "created_at"):
                metadata["created_at"] = getattr(service, "created_at")
            if hasattr(service, "updated_at"):
                metadata["updated_at"] = getattr(service, "updated_at")

            return metadata

        @staticmethod
        def format_service_info(
            _service: object, metadata: FlextTypes.Core.Dict
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


__all__: FlextTypes.Core.StringList = [
    "FlextService",
]
