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
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime

from pydantic import ConfigDict

from flext_core.constants import FlextConstants
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextDomainService[TDomainResult](
    FlextModels.ArbitraryTypesModel,
    FlextMixins.Serializable,
    FlextMixins.Loggable,
    ABC,
):
    """Domain service base using railway patterns with Pydantic models.

    Provides unified service pattern for FLEXT ecosystem:
    - FlextResult[T] error handling with railway patterns
    - Pydantic Generic[T] for type-safe domain operations
    - Complete type annotations for ecosystem consistency
    - Nested helper classes instead of loose functions
    - Direct implementation without delegation layers

    Generic[TDomainResult] provides type safety for execute() return types.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )

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
        validation_result = self.validate_with_request(request)
        if validation_result.is_failure:
            return FlextResult[TDomainResult].fail(
                f"{FlextConstants.Messages.VALIDATION_FAILED}: {validation_result.error}"
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

    def get_service_info(self) -> dict[str, object]:
        """Get service information for diagnostics.

        Returns:
            dict[str, object]: Service information including type and configuration

        """
        return {"service_type": self.__class__.__name__}

    # =============================================================================
    # VALIDATION METHODS
    # =============================================================================

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules for the domain service.

        Returns:
            FlextResult[None]: Success if valid, failure with error details

        """
        return FlextResult[None].ok(None)

    def validate_config(self) -> FlextResult[None]:
        """Validate service configuration.

        Returns:
            FlextResult[None]: Success if valid, failure with error details

        """
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
                    (
                        f"{FlextConstants.Messages.VALIDATION_FAILED}"
                        f" (business rules): {business_result.error}"
                    )
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
        # Pre-execution validation
        if getattr(operation, "enable_validation", True):
            validation_result = self.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult[TDomainResult].fail(
                    (
                        f"{FlextConstants.Messages.VALIDATION_FAILED}"
                        f" (pre-execution): {validation_result.error}"
                    )
                )

        # Execute the main operation
        return self.execute()

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
                raise TimeoutError(msg)

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
        # Check condition by looking at enable flags
        if not getattr(condition, "enable_execution", True):
            return FlextResult[TDomainResult].fail("Execution condition not met")

        # Execute if condition is met
        return self.execute()

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
        metrics_data: dict[str, object] = {}

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
            acquired_resources = list(required_resources)

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
        def prepare_execution_context(service: object) -> dict[str, object]:
            """Prepare execution context for the service."""
            return {
                "service_type": service.__class__.__name__,
                "timestamp": datetime.now(UTC),
            }

        @staticmethod
        def cleanup_execution_context(
            service: object, context: dict[str, object]
        ) -> None:
            """Cleanup execution context after service execution."""
            # Default implementation - can be extended by subclasses

    class _MetadataHelper:
        """Nested metadata helper - no loose functions."""

        @staticmethod
        def extract_service_metadata(service: object) -> dict[str, object]:
            """Extract metadata from service instance."""
            metadata: dict[str, object] = {
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
        def format_service_info(_service: object, metadata: dict[str, object]) -> str:
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
    "FlextDomainService",
]
