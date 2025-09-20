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
    """Optimized domain service base using advanced railway patterns.

    Reduced complexity through monadic composition while maintaining
    identical functionality and API compatibility for FLEXT 1.0.0.
    """

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    def to_json(self, indent: int | None = None) -> str:
        """Convert to JSON string while preserving modernization metadata."""
        return FlextMixins.to_json(self, indent)

    def is_valid(self) -> bool:
        """Check service validity using railway pattern composition."""
        return self.validate_business_rules().is_success

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate domain service business rules using railway contract."""
        return FlextResult[None].ok(None)

    @abstractmethod
    def execute(self) -> FlextResult[TDomainResult]:
        """Execute the main domain service operation with result contract."""
        ...

    def validate_config(self) -> FlextResult[None]:
        """Validate service configuration using railway guardrails."""
        return FlextResult[None].ok(None)

    def execute_operation(
        self,
        operation_name: str,
        operation: object,
        *args: object,
        **kwargs: object,
    ) -> FlextResult[object]:
        """Execute operation using optimized railway composition."""
        return (
            FlextResult.ok(operation)
            >> (lambda op: self._validate_callable(op, operation_name))
            >> (lambda _: self._safe_execute_operation(operation, *args, **kwargs))
        )

    def _validate_callable(
        self, operation: object, operation_name: str
    ) -> FlextResult[object]:
        """Validate callable using filter pattern."""
        return FlextResult.ok(operation).filter(
            callable, f"Operation {operation_name} is not callable"
        )

    def _safe_execute_operation(
        self, operation: object, *args: object, **kwargs: object
    ) -> FlextResult[object]:
        """Execute operation using FlextResult.from_exception pattern."""

        def execute() -> object:
            if not callable(operation):
                msg = "Operation is not callable"
                raise TypeError(msg)
            return operation(*args, **kwargs)

        return FlextResult.from_exception(execute)

    def get_service_info(self) -> FlextTypes.Core.Dict:
        """Return service metadata using railway validation composition."""
        validation_results = FlextResult.chain_validations(
            self.validate_config, self.validate_business_rules
        )

        return {
            "service_type": self.__class__.__name__,
            "service_id": f"service_{self.__class__.__name__.lower()}_{FlextUtilities.Generators.generate_id()}",
            "config_valid": self.validate_config().is_success,
            "business_rules_valid": self.validate_business_rules().is_success,
            "configuration": cast("BaseModel", self).model_dump(),
            "is_valid": validation_results.is_success,
            "timestamp": FlextUtilities.Generators.generate_iso_timestamp(),
        }

    def execute_with_full_validation(self) -> FlextResult[TDomainResult]:
        """Execute with complete validation pipeline using railway composition."""
        return FlextResult.chain_validations(
            self.validate_config, self.validate_business_rules
        ) >> (lambda _: self.execute())

    def execute_batch_operations(
        self, operations: list[Callable[[], FlextResult[object]]]
    ) -> FlextResult[list[object]]:
        """Execute multiple operations using traverse pattern."""
        return FlextResult.traverse(operations, lambda op: op())

    def retry_execute(self, max_attempts: int = 3) -> FlextResult[TDomainResult]:
        """Retry execution with railway pattern backoff."""
        return FlextUtilities.Reliability.retry_with_backoff(
            self.execute, max_retries=max_attempts
        )

    # === NEW ADVANCED DOMAIN SERVICE PATTERNS ===

    def execute_with_context(self, context: str) -> FlextResult[TDomainResult]:
        """Execute with enhanced error context using railway patterns."""
        return self.execute_with_full_validation().with_context(
            lambda error: f"[{context}] {error}"
        )

    def execute_with_timeout(
        self, timeout_seconds: float = 30.0
    ) -> FlextResult[TDomainResult]:
        """Execute with timeout using advanced railway patterns."""
        return FlextResult.ok(None).with_timeout(
            timeout_seconds, lambda _: self.execute_with_full_validation()
        )

    def execute_with_fallback(
        self, *fallback_services: Callable[[], FlextResult[TDomainResult]]
    ) -> FlextResult[TDomainResult]:
        """Execute with fallback services using railway patterns."""
        return FlextUtilities.Reliability.with_fallback(
            self.execute_with_full_validation, *fallback_services
        )

    def execute_conditionally(
        self,
        condition: Callable[[object], bool],
        precondition_validator: Callable[[], FlextResult[object]] | None = None,
    ) -> FlextResult[TDomainResult]:
        """Execute conditionally based on validation and business rules."""
        if precondition_validator:
            return precondition_validator().when(condition) >> (
                lambda _: self.execute_with_full_validation()
            )
        return self.validate_business_rules().when(lambda _: condition(self)) >> (
            lambda _: self.execute()
        )

    def execute_state_machine(
        self,
        state_transitions: dict[
            str, Callable[[TDomainResult], FlextResult[TDomainResult]]
        ],
        initial_state: str = "start",
    ) -> FlextResult[TDomainResult]:
        """Execute as state machine using railway pattern transitions."""
        return self.execute_with_full_validation().transition(
            lambda result: self._apply_state_transitions(
                result, state_transitions, initial_state
            )
        )

    def _apply_state_transitions(
        self,
        initial_result: TDomainResult,
        transitions: dict[str, Callable[[TDomainResult], FlextResult[TDomainResult]]],
        current_state: str,
    ) -> FlextResult[TDomainResult]:
        """Apply state machine transitions using railway patterns."""
        if current_state not in transitions:
            return FlextResult[TDomainResult].fail(
                f"Invalid state transition: {current_state}"
            )

        return transitions[current_state](initial_result)

    def execute_with_resource_management[TResource](
        self,
        resource_factory: Callable[[], TResource],
        cleanup_func: Callable[[TResource], None] | None = None,
    ) -> FlextResult[TDomainResult]:
        """Execute with automatic resource management using railway patterns."""
        return FlextResult.ok(None).with_resource(
            resource_factory,
            lambda _, resource: self.execute_with_full_validation(),
            cleanup_func,
        )

    def execute_parallel_validations(
        self,
        *additional_validators: Callable[[], FlextResult[None]],
        fail_fast: bool = True,
    ) -> FlextResult[TDomainResult]:
        """Execute with parallel validation pipeline."""
        all_validators = [
            self.validate_config,
            self.validate_business_rules,
            *additional_validators,
        ]

        if fail_fast:
            return FlextResult.chain_validations(*all_validators) >> (
                lambda _: self.execute()
            )
        validation_results = [validator() for validator in all_validators]
        return FlextResult.accumulate_errors(*validation_results) >> (
            lambda _: self.execute()
        )

    def execute_with_metrics(
        self,
        metrics_collector: Callable[[str, float], None] | None = None,
    ) -> FlextResult[TDomainResult]:
        """Execute with performance metrics collection using railway patterns."""

        def timed_execution() -> TDomainResult:
            start_time = time.time()
            result = self.execute_with_full_validation()
            execution_time = time.time() - start_time

            if metrics_collector:
                metrics_collector(
                    f"{self.__class__.__name__}_execution_time", execution_time
                )

            # Extract the value from the result or raise if failed
            return result.unwrap()

        return FlextResult.from_exception(timed_execution)

    def execute_with_circuit_breaker(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> FlextResult[TDomainResult]:
        """Execute with circuit breaker pattern using railway composition."""
        return FlextUtilities.Reliability.circuit_breaker(
            self.execute_with_full_validation,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )

    def compose_with[TCompose](
        self,
        other_service: Callable[[TDomainResult], FlextResult[TCompose]],
    ) -> FlextResult[TCompose]:
        """Compose this service with another service using monadic composition."""
        return self.execute_with_full_validation() >> other_service

    def validate_and_transform[TTransform](
        self,
        validator: Callable[[TDomainResult], FlextResult[None]],
        transformer: Callable[[TDomainResult], FlextResult[TTransform]],
    ) -> FlextResult[TTransform]:
        """Validate result then transform using railway patterns."""
        return self.execute_with_full_validation().validate_and_execute(
            validator, transformer
        )


__all__: FlextTypes.Core.StringList = [
    "FlextDomainService",
]
