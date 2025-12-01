"""Validation utilities extracted from FlextModels.

This module contains the Validation nested class from FlextModels.
It should NOT be imported directly - use FlextModels.Validation instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time as time_module
from collections.abc import Callable
from datetime import datetime
from typing import cast

from pydantic import BaseModel

from flext_core.constants import FlextConstants
from flext_core.protocols import FlextProtocols
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes


class FlextModelsValidation:
    """Validation utility functions."""

    @staticmethod
    def validate_business_rules(
        model: FlextTypes.GeneralValueType,
        *rules: Callable[
            [FlextTypes.GeneralValueType],
            FlextProtocols.ResultProtocol[FlextTypes.GeneralValueType],
        ],
    ) -> FlextProtocols.ResultProtocol[FlextTypes.GeneralValueType]:
        """Validate business rules with railway patterns.

        Args:
            model: The model to validate
            *rules: Business rule validation functions

        Returns:
            ResultProtocol: Validated model or accumulated errors

        Example:
            ```python
            def check_age_business_rule(user: User) -> FlextResult[bool]:
                return (
                    FlextResult[bool].ok(True)
                    if user.age >= 18
                    else FlextResult[bool].fail("Must be 18+")
                )


            result = FlextModels.Validation.validate_business_rules(
                user_model,
                check_age_business_rule,
            )
            ```

        """
        # Validate all rules and return model if all pass
        for rule in rules:
            result = rule(model)
            if result.is_failure:
                base_msg = "Validation failed"
                error_msg = (
                    f"{base_msg}: {result.error}"
                    if result.error
                    else f"{base_msg} (validation rule failed)"
                )
                return FlextRuntime.result_fail(error_msg)

        return FlextRuntime.result_ok(model)

    @staticmethod
    def validate_cross_fields(
        model: FlextTypes.GeneralValueType,
        field_validators: FlextTypes.Types.FieldValidatorMapping,
    ) -> FlextProtocols.ResultProtocol[FlextTypes.GeneralValueType]:
        """Validate cross-field dependencies with railway patterns.

        Args:
            model: The model to validate
            field_validators: Field name to validator mapping

        Returns:
            ResultProtocol: Validated model or accumulated errors

        Example:
            ```python
            result = FlextModels.Validation.validate_cross_fields(
                order_model,
                {
                    "start_date": lambda o: validate_date_range(
                        o.start_date, o.end_date
                    ),
                    "end_date": lambda o: validate_date_range(o.start_date, o.end_date),
                    "amount": lambda o: validate_amount_range(o.amount, o.currency),
                },
            )
            ```

        """
        validation_results: list[FlextProtocols.ResultProtocol[bool]] = [
            cast("FlextProtocols.ResultProtocol[bool]", validator(model))
            for validator in field_validators.values()
        ]

        errors = [
            result.error
            for result in validation_results
            if result.is_failure and result.error
        ]

        if errors:
            # Type assertion: errors contains only non-None strings due to filter above
            error_messages = [str(err) for err in errors]
            return FlextRuntime.result_fail(
                f"Cross-field validation failed: {'; '.join(error_messages)}",
                error_code="CROSS_FIELD_VALIDATION_FAILED",
                error_data={"field_errors": error_messages},
            )

        return FlextRuntime.result_ok(model)

    @staticmethod
    def validate_performance(
        model: BaseModel,
        max_validation_time_ms: int | None = None,
    ) -> FlextProtocols.ResultProtocol[BaseModel]:
        """Validate model with performance constraints.

        Args:
            model: The model to validate
            max_validation_time_ms: Maximum validation time in milliseconds

        Returns:
            ResultProtocol: Validated model or performance error

        Example:
            ```python
            result = FlextModels.Validation.validate_performance(
                complex_model, max_validation_time_ms=50
            )
            ```

        """
        # Use config value if not provided
        if max_validation_time_ms is not None:
            timeout_ms = max_validation_time_ms
        else:
            # Use constant for validation timeout
            timeout_ms = FlextConstants.Validation.VALIDATION_TIMEOUT_MS
        start_time = time_module.time()

        try:
            # Exclude computed fields that are not actual model fields
            # Use model_dump with exclude_unset to avoid extra fields
            dump = model.model_dump(
                exclude={"is_initial_version", "is_modified"},
                exclude_unset=True,
            )
            # Re-validate the model from the dump
            validated_model = model.__class__.model_validate(dump)
            validation_time = (time_module.time() - start_time) * 1000

            if validation_time > timeout_ms:
                return FlextRuntime.result_fail(
                    f"Validation too slow: {validation_time:.2f}ms > {timeout_ms}ms",
                    error_code="PERFORMANCE_VALIDATION_FAILED",
                    error_data={"validation_time_ms": validation_time},
                )

            return FlextRuntime.result_ok(validated_model)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextRuntime.result_fail(
                f"Validation failed: {e}",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

    @staticmethod
    def validate_batch(
        models: FlextTypes.ObjectList,
        *validators: Callable[
            [FlextTypes.GeneralValueType],
            FlextProtocols.ResultProtocol[FlextTypes.GeneralValueType],
        ],
        fail_fast: bool = True,
    ) -> FlextProtocols.ResultProtocol[FlextTypes.ObjectList]:
        """Validate a batch of models with railway patterns.

        Args:
            models: List of models to validate
            *validators: Validation functions to apply
            fail_fast: Stop on first failure or accumulate all errors

        Returns:
            ResultProtocol: All validated models or first failure

        Example:
            ```python
            def is_adult(user: User) -> FlextResult[bool]:
                return (
                    FlextResult[bool].ok(True)
                    if user.age >= 18
                    else FlextResult[bool].fail("Not adult")
                )


            result = FlextModels.Validation.validate_batch(
                user_models,
                is_adult,
                fail_fast=False,
            )
            ```

        """
        if fail_fast:
            # Validate models one by one, stop on first failure
            valid_models: list[FlextTypes.GeneralValueType] = []
            for model in models:
                # Validate all rules for this model
                for validator in validators:
                    result = validator(model)
                    if result.is_failure:
                        base_msg = "Validation failed"
                        error_msg = (
                            f"{base_msg}: {result.error}"
                            if result.error
                            else f"{base_msg} (item validation failed)"
                        )
                        return FlextRuntime.result_fail(error_msg)

                valid_models.append(model)

            return FlextRuntime.result_ok(valid_models)
        # Accumulate all errors
        validated_models: list[FlextTypes.GeneralValueType] = []
        all_errors: list[str] = []

        for model in models:
            # Use individual validation for models since validators may return GeneralValueType
            validation_result = FlextRuntime.result_ok(model)
            for validator in validators:
                result = validator(model)
                if result.is_failure:
                    base_msg = "Validation failed"
                    error_msg = (
                        f"{base_msg}: {result.error}"
                        if result.error
                        else f"{base_msg} (model validation failed)"
                    )
                    # Create failure result using FlextRuntime
                    validation_result = FlextRuntime.result_fail(error_msg)
                    break
            if validation_result.is_success:
                validated_models.append(model)
            else:
                base_msg = "Validation failed"
                error_msg = (
                    f"{base_msg}: {validation_result.error}"
                    if validation_result.error
                    else f"{base_msg} (validation rule failed)"
                )
                all_errors.append(error_msg)

        if all_errors:
            return FlextRuntime.result_fail(
                f"Batch validation failed: {'; '.join(all_errors)}",
                error_code="BATCH_VALIDATION_FAILED",
                error_data={"error_count": len(all_errors), "errors": all_errors},
            )

        return FlextRuntime.result_ok(validated_models)

    @staticmethod
    def validate_domain_invariants(
        model: FlextTypes.GeneralValueType,
        invariants: list[
            Callable[
                [FlextTypes.GeneralValueType],
                FlextProtocols.ResultProtocol[FlextTypes.GeneralValueType],
            ]
        ],
    ) -> FlextProtocols.ResultProtocol[FlextTypes.GeneralValueType]:
        """Validate domain invariants with railway patterns.

        Args:
            model: The model to validate
            invariants: List of domain invariant validation functions

        Returns:
            ResultProtocol: Validated model or first invariant violation

        Example:
            ```python
            result = FlextModels.Validation.validate_domain_invariants(
                order_model,
                [
                    lambda o: validate_order_total(o),
                    lambda o: validate_order_items(o),
                    lambda o: validate_order_customer(o),
                ],
            )
            ```

        """
        for invariant in invariants:
            # Cast model to GeneralValueType for type safety
            result = invariant(model)
            if hasattr(result, "is_failure") and result.is_failure:
                return FlextRuntime.result_fail(
                    f"Domain invariant violation: {result.error if hasattr(result, 'error') else 'unknown'}",
                    error_code="DOMAIN_INVARIANT_VIOLATION",
                    error_data={
                        "invariant_error": result.error
                        if hasattr(result, "error")
                        else "unknown",
                    },
                )
        return FlextRuntime.result_ok(model)

    @staticmethod
    def validate_aggregate_consistency_with_rules(
        aggregate: FlextTypes.GeneralValueType,
        consistency_rules: FlextTypes.Types.ConsistencyRuleMapping,
    ) -> FlextProtocols.ResultProtocol[FlextTypes.GeneralValueType]:
        """Validate aggregate consistency with railway patterns.

        Args:
            aggregate: The aggregate to validate
            consistency_rules: Dictionary of consistency rule validators

        Returns:
            ResultProtocol: Validated aggregate or consistency violation

        Example:
            ```python
            result = FlextModels.Validation.validate_aggregate_consistency_with_rules(
                order_aggregate,
                {
                    "total_consistency": lambda a: validate_total_consistency(a),
                    "item_consistency": lambda a: validate_item_consistency(a),
                    "customer_consistency": lambda a: validate_customer_consistency(a),
                },
            )
            ```

        """
        violations: list[str] = []
        for rule_name, validator in consistency_rules.items():
            result = cast("FlextProtocols.ResultProtocol[bool]", validator(aggregate))
            if result.is_failure:
                violations.append(f"{rule_name}: {result.error}")

        if violations:
            return FlextRuntime.result_fail(
                f"Aggregate consistency violations: {'; '.join(violations)}",
                error_code="AGGREGATE_CONSISTENCY_VIOLATION",
                error_data={"violations": violations},
            )

        return FlextRuntime.result_ok(aggregate)

    @staticmethod
    def validate_event_sourcing(
        event: FlextTypes.GeneralValueType,
        event_validators: FlextTypes.Types.EventValidatorMapping,
    ) -> FlextProtocols.ResultProtocol[FlextTypes.GeneralValueType]:
        """Validate event sourcing patterns with railway patterns.

        Args:
            event: The domain event to validate
            event_validators: Dictionary of event-specific validators

        Returns:
            ResultProtocol: Validated event or validation failure

        Example:
            ```python
            result = FlextModels.Validation.validate_event_sourcing(
                order_created_event,
                {
                    "event_type": lambda e: validate_event_type(e),
                    "event_data": lambda e: validate_event_data(e),
                    "event_metadata": lambda e: validate_event_metadata(e),
                },
            )
            ```

        """
        validation_results = cast(
            "list[FlextProtocols.ResultProtocol[bool]]",
            [validator(event) for validator in event_validators.values()],
        )

        errors: list[str] = [
            result.error
            for result in validation_results
            if result.is_failure and result.error
        ]

        if errors:
            return FlextRuntime.result_fail(
                f"Event validation failed: {'; '.join(errors)}",
                error_code="EVENT_VALIDATION_FAILED",
                error_data={"event_errors": errors},
            )

        return FlextRuntime.result_ok(event)

    @staticmethod
    def validate_cqrs_patterns(
        command_or_query: FlextTypes.GeneralValueType,
        pattern_type: str,
        validators: list[
            Callable[
                [FlextTypes.GeneralValueType],
                FlextProtocols.ResultProtocol[FlextTypes.GeneralValueType],
            ]
        ],
    ) -> FlextProtocols.ResultProtocol[FlextTypes.GeneralValueType]:
        """Validate CQRS patterns with railway patterns.

        Args:
            command_or_query: The command or query to validate
            pattern_type: Type of pattern (command or query handler type)
            validators: List of pattern-specific validators

        Returns:
            ResultProtocol: Validated command/query or validation failure

        Example:
            ```python
            result = FlextModels.Validation.validate_cqrs_patterns(
                create_order_command,
                FlextConstants.Cqrs.HandlerType.COMMAND.value,
                [
                    lambda c: validate_command_structure(c),
                    lambda c: validate_command_data(c),
                    lambda c: validate_command_permissions(c),
                ],
            )
            ```

        """
        valid_pattern_types = {
            FlextConstants.Cqrs.HandlerType.COMMAND.value,
            FlextConstants.Cqrs.HandlerType.QUERY.value,
        }
        if pattern_type not in valid_pattern_types:
            return FlextRuntime.result_fail(
                f"Invalid pattern type: {pattern_type}. Must be '{FlextConstants.Cqrs.HandlerType.COMMAND.value}' or '{FlextConstants.Cqrs.HandlerType.QUERY.value}'",
                error_code="INVALID_PATTERN_TYPE",
            )

        for validator in validators:
            # Cast command_or_query to GeneralValueType for type safety
            result = validator(command_or_query)
            if result.is_failure:
                return FlextRuntime.result_fail(
                    f"CQRS {pattern_type} validation failed: {result.error}",
                    error_code=f"CQRS_{pattern_type.upper()}_VALIDATION_FAILED",
                    error_data={
                        "pattern_type": pattern_type,
                        "error": result.error,
                    },
                )

        return FlextRuntime.result_ok(command_or_query)

    @staticmethod
    def _validate_event_structure(
        event: FlextTypes.GeneralValueType,
    ) -> FlextProtocols.ResultProtocol[bool]:
        """Validate event is not None and has required attributes."""
        if event is None:
            return FlextRuntime.result_fail(
                "Domain event cannot be None",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        # Check required attributes
        required_attrs = ["event_type", "aggregate_id", "unique_id", "created_at"]
        missing_attrs = [
            attr
            for attr in required_attrs
            if not (
                hasattr(event, attr)
                or (FlextRuntime.is_dict_like(event) and attr in event)
            )
        ]
        if missing_attrs:
            return FlextRuntime.result_fail(
                f"Domain event missing required attributes: {missing_attrs}",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        return FlextRuntime.result_ok(True)

    @staticmethod
    def _validate_event_fields(
        event: FlextTypes.GeneralValueType,
    ) -> FlextProtocols.ResultProtocol[bool]:
        """Validate event field types and values."""
        # Validate event_type is non-empty string
        event_type = (
            getattr(event, "event_type", "")
            if not FlextRuntime.is_dict_like(event)
            else event.get("event_type", "")
            if isinstance(event, dict)
            else ""
        )
        if not event_type or not isinstance(event_type, str):
            return FlextRuntime.result_fail(
                "Domain event event_type must be a non-empty string",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        # Validate aggregate_id is non-empty string
        aggregate_id = (
            getattr(event, "aggregate_id", "")
            if not FlextRuntime.is_dict_like(event)
            else event.get("aggregate_id", "")
            if isinstance(event, dict)
            else ""
        )
        if not aggregate_id or not isinstance(aggregate_id, str):
            return FlextRuntime.result_fail(
                "Domain event aggregate_id must be a non-empty string",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        # Validate data is a dict
        data = (
            getattr(event, "data", None)
            if not FlextRuntime.is_dict_like(event)
            else event.get("data", None)
            if isinstance(event, dict)
            else None
        )
        if data is not None and not FlextRuntime.is_dict_like(data):
            return FlextRuntime.result_fail(
                "Domain event data must be a dictionary or None",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        return FlextRuntime.result_ok(True)

    @staticmethod
    def validate_domain_event(
        event: FlextTypes.GeneralValueType,
    ) -> FlextProtocols.ResultProtocol[bool]:
        """Enhanced domain event validation with comprehensive checks.

        Validates domain events for proper structure, required fields,
        and domain invariants. Used across all flext-ecosystem projects.

        Args:
            event: The domain event to validate

        Returns:
            ResultProtocol[bool]: Success with True if valid, failure with details

        """
        # Validate structure
        structure_result = FlextModelsValidation._validate_event_structure(event)
        if structure_result.is_failure:
            return structure_result

        # Validate fields
        fields_result = FlextModelsValidation._validate_event_fields(event)
        if fields_result.is_failure:
            return fields_result

        return FlextRuntime.result_ok(True)

    @staticmethod
    def validate_aggregate_consistency(
        aggregate: FlextTypes.GeneralValueType | FlextProtocols.HasInvariants,
    ) -> FlextProtocols.ResultProtocol[
        FlextTypes.GeneralValueType | FlextProtocols.HasInvariants
    ]:
        """Validate aggregate consistency and business invariants.

        Ensures aggregates maintain consistency boundaries and invariants
        are satisfied. Used extensively in flext-core and dependent projects.

        Args:
            aggregate: The aggregate root to validate

        Returns:
            ResultProtocol: Validated aggregate or failure with details

        """
        if aggregate is None:
            return FlextRuntime.result_fail(
                "Aggregate cannot be None",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        # Check invariants if the aggregate supports them
        if isinstance(aggregate, FlextProtocols.HasInvariants):
            try:
                _ = aggregate.check_invariants()
            except (
                AttributeError,
                TypeError,
                ValueError,
                RuntimeError,
                KeyError,
            ) as e:
                return FlextRuntime.result_fail(
                    f"Aggregate invariant violation: {e}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

        # Check for uncommitted domain events (potential consistency issue)
        if hasattr(aggregate, "domain_events"):
            events = getattr(aggregate, "domain_events", [])
            if len(events) > FlextConstants.Validation.MAX_UNCOMMITTED_EVENTS:
                max_events = FlextConstants.Validation.MAX_UNCOMMITTED_EVENTS
                event_count = len(events)
                error_msg = (
                    f"Too many uncommitted domain events: {event_count} "
                    f"(max: {max_events})"
                )
                return FlextRuntime.result_fail(
                    error_msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

        return FlextRuntime.result_ok(aggregate)

    @staticmethod
    def validate_entity_relationships(
        entity: FlextTypes.GeneralValueType,
    ) -> FlextProtocols.ResultProtocol[FlextTypes.GeneralValueType]:
        """Validate entity relationships and references.

        Ensures entity references are valid and relationships are consistent.
        Critical for maintaining data integrity across flext-ecosystem.

        Args:
            entity: The entity to validate

        Returns:
            ResultProtocol: Validated entity or failure with details

        """
        if entity is None:
            return FlextRuntime.result_fail(
                "Entity cannot be None",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        # Entity ID is validated at field level by Pydantic v2
        # No custom validation needed - use Field(pattern=r"^[a-zA-Z0-9_-]+$")

        # Validate version for optimistic locking
        if hasattr(entity, "version"):
            version = getattr(entity, "version", 0)
            if not isinstance(version, int) or version < 0:
                return FlextRuntime.result_fail(
                    "Entity version must be a non-negative integer",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

        # Validate timestamps if present
        for timestamp_field in ["created_at", "updated_at"]:
            if hasattr(entity, timestamp_field):
                timestamp = getattr(entity, timestamp_field)
                if timestamp is not None and not isinstance(timestamp, datetime):
                    return FlextRuntime.result_fail(
                        f"Entity {timestamp_field} must be a datetime or None",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

        # Return validated entity
        return FlextRuntime.result_ok(entity)


__all__ = ["FlextModelsValidation"]
