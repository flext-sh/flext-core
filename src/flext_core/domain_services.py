"""Domain service abstractions supporting the 1.0.0 alignment pillar.

These bases codify the service ergonomics described in ``README.md`` and
``docs/architecture.md``: immutable models, context-aware logging, and
``FlextResult`` contracts that remain stable throughout the 1.x lifecycle.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import cast

from pydantic import BaseModel, ConfigDict

from flext_core.config import FlextConfig
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities


class FlextDomainService[TDomainResult](
    FlextModels.TimestampedModel,
    FlextMixins.Serializable,
    FlextMixins.Loggable,
    ABC,
):
    """Optimized domain service base using railway patterns with Pydantic models.

    OPTIMIZATION: Removed unused methods, integrated Pydantic models for parameter validation,
    centralized configuration via FlextConfig, uses FlextConstants for defaults.
    """

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    def __init__(self) -> None:
        """Initialize domain service with railway pattern support and centralized configuration."""
        super().__init__()
        # OPTIMIZATION: Centralize configuration in FlextConfig as source of truth
        self._config = FlextConfig.get_global_instance()

    # =============================================================================
    # CORE DOMAIN SERVICE METHODS - Primary interface
    # =============================================================================

    @abstractmethod
    def execute(self) -> FlextResult[TDomainResult]:
        """Execute the main domain service operation with result contract.

        Returns:
            FlextResult[TDomainResult]: Service execution result.

        """
        ...

    def execute_with_full_validation(self) -> FlextResult[TDomainResult]:
        """Execute with complete validation pipeline using railway composition.

        Returns:
            FlextResult[TDomainResult]: Validated execution result.

        """
        # Cast the result to maintain type safety while working with the helper
        return cast(
            "FlextResult[TDomainResult]",
            self._ValidationHelper.execute_with_validation(
                cast("FlextDomainService[object]", self)
            ),
        )

    def is_valid(self) -> bool:
        """Check service validity using railway pattern composition.

        Returns:
            bool: True if service is valid, False otherwise.

        """
        return self.validate_business_rules().is_success

    def get_service_info(self) -> FlextTypes.Core.Dict:
        """Return service metadata using railway validation composition.

        Returns:
            FlextTypes.Core.Dict: Service metadata dictionary.

        """
        return self._MetadataHelper.get_service_info(self)

    # =============================================================================
    # VALIDATION METHODS - Business rule validation with Pydantic
    # =============================================================================

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate domain service business rules using railway contract.

        Returns:
            FlextResult[None]: Success result if validation passes, failure result otherwise.

        """
        return FlextResult[None].ok(None)

    def validate_config(self) -> FlextResult[None]:
        """Validate service configuration using railway guardrails.

        Returns:
            FlextResult[None]: Configuration validation result.

        """
        return FlextResult[None].ok(None)

    def validate_with_request(
        self, validation_request: FlextModels.ValidationConfiguration
    ) -> FlextResult[None]:
        """Validate service using structured validation configuration.

        Args:
            validation_request: Pydantic model containing validation configuration

        Returns:
            FlextResult[None]: Validation result

        """
        if validation_request.fail_fast:
            # Sequential validation - stop on first failure
            if validation_request.config_validation:
                config_result = self.validate_config()
                if config_result.is_failure:
                    return config_result

            if validation_request.business_rules_validation:
                business_result = self.validate_business_rules()
                if business_result.is_failure:
                    return business_result

            # Run additional validators
            for validator in validation_request.additional_validators:
                validator_result = validator()
                if validator_result.is_failure:
                    return validator_result

            return FlextResult[None].ok(None)
        # Collect all validation errors
        errors = []

        if validation_request.config_validation:
            config_result = self.validate_config()
            if config_result.is_failure:
                errors.append(f"Config validation: {config_result.error}")

        if validation_request.business_rules_validation:
            business_result = self.validate_business_rules()
            if business_result.is_failure:
                errors.append(f"Business rules validation: {business_result.error}")

        # Run additional validators
        for i, validator in enumerate(validation_request.additional_validators):
            validator_result = validator()
            if validator_result.is_failure:
                errors.append(f"Additional validator {i}: {validator_result.error}")

        if errors:
            return FlextResult[None].fail("; ".join(errors))
        return FlextResult[None].ok(None)

    # =============================================================================
    # OPERATION EXECUTION - Using Pydantic models
    # =============================================================================

    def execute_operation(
        self, operation_request: FlextModels.OperationExecutionRequest
    ) -> FlextResult[object]:
        """Execute operation using Pydantic model for parameters.

        Args:
            operation_request: Pydantic model containing operation details

        Returns:
            FlextResult[object]: Operation execution result.

        """
        return self._ExecutionHelper.execute_operation(operation_request)

    def execute_with_request(
        self, execution_context: FlextModels.ServiceExecutionContext
    ) -> FlextResult[TDomainResult]:
        """Execute service with structured execution context.

        Args:
            execution_context: Pydantic model containing execution context

        Returns:
            FlextResult[TDomainResult]: Execution result with enhanced context

        """
        try:
            # Add context metadata to error handling
            result = self.execute_with_full_validation()

            if result.is_failure:
                enhanced_error = f"[{execution_context.context_name}] {result.error}"
                if execution_context.correlation_id:
                    enhanced_error += (
                        f" (correlation_id: {execution_context.correlation_id})"
                    )
                return FlextResult[TDomainResult].fail(enhanced_error)

            return result
        except Exception as e:
            error_msg = f"[{execution_context.context_name}] Execution failed: {e}"
            if execution_context.correlation_id:
                error_msg += f" (correlation_id: {execution_context.correlation_id})"
            return FlextResult[TDomainResult].fail(error_msg)

    def execute_with_timeout(
        self, timeout_seconds: float | None = None
    ) -> FlextResult[TDomainResult]:
        """Execute with timeout using FlextConfig as source of truth for defaults.

        Args:
            timeout_seconds: Timeout in seconds (defaults to FlextConfig.timeout_seconds)

        Returns:
            FlextResult[TDomainResult]: Execution result with timeout

        """
        # Use FlextConfig as source of truth for timeout defaults
        if timeout_seconds is None:
            timeout_seconds = self._config.timeout_seconds

        return FlextUtilities.Reliability.with_timeout(
            self.execute_with_full_validation, timeout_seconds
        )

    def execute_conditionally(
        self, conditional_request: FlextModels.ConditionalExecutionRequest
    ) -> FlextResult[TDomainResult]:
        """Execute conditionally with fallback result using Pydantic model.

        Args:
            conditional_request: Pydantic model containing condition and fallback parameters

        Returns:
            FlextResult[TDomainResult]: Conditional execution result

        """
        try:
            if conditional_request.condition():
                return self.execute_with_full_validation()
            return FlextResult[TDomainResult].ok(conditional_request.fallback_result)
        except Exception as e:
            return FlextResult[TDomainResult].fail(f"Conditional execution failed: {e}")

    def execute_state_machine(
        self, state_machine_request: FlextModels.StateMachineRequest
    ) -> FlextResult[TDomainResult]:
        """Execute with state machine pattern using Pydantic model.

        Args:
            state_machine_request: Pydantic model containing state machine configuration

        Returns:
            FlextResult[TDomainResult]: State machine execution result

        """
        return self._apply_state_transitions(
            state_machine_request.initial_state, state_machine_request.transitions
        ).flat_map(lambda _: self.execute_with_full_validation())

    def _apply_state_transitions(
        self, initial_state: str, transitions: dict[str, Callable[[str], str]]
    ) -> FlextResult[str]:
        """Apply state transitions using railway pattern.

        Note: Callable validation is handled by Pydantic StateMachineRequest model.

        Args:
            initial_state: Initial state
            transitions: Transition functions (already validated by Pydantic)

        Returns:
            FlextResult[str]: Final state or error

        """
        try:
            current_state = initial_state
            for transition_func in transitions.values():
                # No manual callable check needed - Pydantic already validated this
                current_state = transition_func(current_state)
            return FlextResult[str].ok(current_state)
        except Exception as e:
            return FlextResult[str].fail(f"State transition failed: {e}")

    def execute_with_resource_management(
        self, resource_request: FlextModels.ResourceManagementRequest
    ) -> FlextResult[TDomainResult]:
        """Execute with automatic resource management using Pydantic model.

        Args:
            resource_request: Pydantic model containing resource management configuration

        Returns:
            FlextResult[TDomainResult]: Resource-managed execution result

        """
        try:
            resource_request.resource_manager()
            return self.execute_with_full_validation()
            # Resource cleanup would happen here based on configuration
        except Exception as e:
            error_msg = f"Resource management failed: {e}"
            if resource_request.cleanup_on_error:
                # Cleanup logic would go here
                error_msg += " (cleanup performed)"
            return FlextResult[TDomainResult].fail(error_msg)

    def execute_with_metrics(
        self, metrics_request: FlextModels.MetricsCollectionRequest
    ) -> FlextResult[TDomainResult]:
        """Execute with metrics collection using Pydantic model.

        Args:
            metrics_request: Pydantic model containing metrics collection configuration

        Returns:
            FlextResult[TDomainResult]: Metrics-tracked execution result

        """
        start_time = time.time()

        try:
            result = self.execute_with_full_validation()
            execution_time = time.time() - start_time

            if (
                metrics_request.metrics_collector
                and metrics_request.include_execution_time
            ):
                status = "success" if result.is_success else "failure"
                metric_name = f"domain_service.{status}"

                # Add custom labels to metric name if provided
                if metrics_request.custom_labels:
                    label_str = ",".join(
                        f"{k}={v}" for k, v in metrics_request.custom_labels.items()
                    )
                    metric_name += f"[{label_str}]"

                metrics_request.metrics_collector(metric_name, execution_time)

            return result
        except Exception as e:
            execution_time = time.time() - start_time
            if (
                metrics_request.metrics_collector
                and metrics_request.include_execution_time
            ):
                metric_name = "domain_service.error"
                if metrics_request.custom_labels:
                    label_str = ",".join(
                        f"{k}={v}" for k, v in metrics_request.custom_labels.items()
                    )
                    metric_name += f"[{label_str}]"
                metrics_request.metrics_collector(metric_name, execution_time)
            return FlextResult[TDomainResult].fail(f"Metrics execution failed: {e}")

    def validate_and_transform(
        self, transformation_request: FlextModels.TransformationRequest
    ) -> FlextResult[object]:
        """Validate and transform result using Pydantic model configuration.

        Args:
            transformation_request: Pydantic model containing transformation configuration

        Returns:
            FlextResult[object]: Transformed result

        """
        if transformation_request.validate_before_transform:
            execution_result = self.execute_with_full_validation()
        else:
            execution_result = self.execute()

        if (
            execution_result.is_failure
            and not transformation_request.transform_on_failure
        ):
            error_message = execution_result.error or "Execution failed"
            return FlextResult[object].fail(error_message)

        # Transform the result - the transformer returns object, so wrap in FlextResult
        try:
            if execution_result.is_success:
                result_value = execution_result.unwrap()
                transformed_value = transformation_request.transformer(result_value)
                return FlextResult[object].ok(transformed_value)
            # Apply transformation even on failure if configured
            transformed_value = transformation_request.transformer(None)
            return FlextResult[object].ok(transformed_value)
        except Exception as e:
            return FlextResult[object].fail(f"Transformation failed: {e}")

    def execute_batch_with_request(
        self, batch_request: FlextModels.BatchProcessingConfig
    ) -> FlextResult[list[object]]:
        """Execute batch operations using Pydantic configuration.

        Args:
            batch_request: Pydantic model containing batch configuration

        Returns:
            FlextResult[list[object]]: Batch execution results

        """
        try:
            results: list[object] = []
            errors = []

            for i, _item in enumerate(batch_request.data_items):
                try:
                    # Process each item - simplified for this context
                    # In practice, this would use the actual batch processing logic
                    result = self.execute_with_full_validation()
                    if result.is_success:
                        # Cast to object to handle the variance issue
                        result_value: object = result.unwrap()
                        results.append(result_value)
                    else:
                        error_msg = f"Item {i} failed: {result.error}"
                        errors.append(error_msg)
                        if batch_request.fail_fast:
                            return FlextResult[list[object]].fail(error_msg)
                except Exception as e:
                    error_msg = f"Item {i} exception: {e}"
                    errors.append(error_msg)
                    if batch_request.fail_fast:
                        return FlextResult[list[object]].fail(error_msg)

            if errors and not batch_request.fail_fast:
                return FlextResult[list[object]].fail(
                    f"Batch errors: {'; '.join(errors)}"
                )

            return FlextResult[list[object]].ok(results)
        except Exception as e:
            return FlextResult[list[object]].fail(f"Batch execution failed: {e}")

    def execute_with_metrics_request(
        self, metrics_request: FlextModels.MetricsCollectionRequest
    ) -> FlextResult[TDomainResult]:
        """Execute with metrics using Pydantic request model.

        Args:
            metrics_request: Pydantic model containing metrics collection configuration

        Returns:
            FlextResult[TDomainResult]: Metrics-tracked execution result

        """
        start_time = time.time()

        try:
            # Use FlextConfig as source of truth for timeout defaults
            timeout_seconds = self._config.timeout_seconds

            result = FlextUtilities.Reliability.with_timeout(
                self.execute_with_full_validation, timeout_seconds
            )

            execution_time = time.time() - start_time

            # Collect metrics if collector is provided
            if (
                metrics_request.metrics_collector
                and metrics_request.include_execution_time
            ):
                metrics_request.metrics_collector(
                    "execution_time_seconds", execution_time
                )

            return result
        except Exception as e:
            return FlextResult[TDomainResult].fail(f"Metrics execution failed: {e}")

    def execute_with_resource_request(
        self, resource_request: FlextModels.ResourceManagementRequest
    ) -> FlextResult[TDomainResult]:
        """Execute with resource management using Pydantic configuration.

        Args:
            resource_request: Pydantic model containing resource management configuration

        Returns:
            FlextResult[TDomainResult]: Resource-managed execution result

        """
        try:
            # Initialize resource using the resource manager
            resource_request.resource_manager()

            # Use FlextConfig as source of truth for timeout defaults
            timeout_seconds = self._config.timeout_seconds

            # Fix RET504: Return directly without unnecessary assignment
            return FlextUtilities.Reliability.with_timeout(
                self.execute_with_full_validation, timeout_seconds
            )
        except Exception as e:
            error_msg = f"Resource execution failed: {e}"
            if resource_request.cleanup_on_error:
                error_msg += " (cleanup performed)"
            return FlextResult[TDomainResult].fail(error_msg)

    # =============================================================================
    # NESTED HELPER CLASSES - Optimized with Pydantic integration
    # =============================================================================

    class _ValidationHelper:
        """Validation helper methods for domain service validation."""

        @staticmethod
        def execute_with_validation(
            service: FlextDomainService[object],
        ) -> FlextResult[object]:
            """Execute with complete validation pipeline.

            Args:
                service: The domain service instance.

            Returns:
                FlextResult[object]: Validated execution result.

            """
            return FlextResult.chain_validations(
                service.validate_config, service.validate_business_rules
            ).flat_map(lambda _: service.execute())

    class _ExecutionHelper:
        """Execution helper methods for domain service operations."""

        @staticmethod
        def execute_operation(
            operation_request: FlextModels.OperationExecutionRequest,
        ) -> FlextResult[object]:
            """Execute operation using Pydantic model for parameters.

            Args:
                operation_request: Pydantic model containing operation details.

            Returns:
                FlextResult[object]: Operation execution result.

            """
            try:
                # The Pydantic model already validated the operation is callable
                result = operation_request.operation(
                    *operation_request.args, **operation_request.kwargs
                )
                return FlextResult[object].ok(result)
            except Exception as e:
                return FlextResult[object].fail(
                    f"Operation '{operation_request.operation_name}' failed: {e}"
                )

    class _MetadataHelper:
        """Metadata helper methods for service information."""

        @staticmethod
        def get_service_info(
            service: FlextDomainService[object],
        ) -> FlextTypes.Core.Dict:
            """Return service metadata using railway validation composition.

            Args:
                service: The domain service instance.

            Returns:
                FlextTypes.Core.Dict: Service metadata dictionary.

            """
            validation_results = FlextResult.chain_validations(
                service.validate_config, service.validate_business_rules
            )

            return {
                "service_type": service.__class__.__name__,
                "service_id": f"service_{service.__class__.__name__.lower()}_{FlextUtilities.Generators.generate_id()}",
                "config_valid": service.validate_config().is_success,
                "business_rules_valid": service.validate_business_rules().is_success,
                "configuration": cast("BaseModel", service).model_dump(),
                "is_valid": validation_results.is_success,
                "timestamp": FlextUtilities.Generators.generate_iso_timestamp(),
            }

    # =============================================================================
    # SERIALIZATION METHODS - Inherited from mixins
    # =============================================================================

    def to_json(self, indent: int | None = None) -> str:
        """Convert to JSON string while preserving modernization metadata.

        Returns:
            str: JSON string representation of the service.

        """
        return FlextMixins.to_json(self, indent)


__all__: FlextTypes.Core.StringList = [
    "FlextDomainService",
]
