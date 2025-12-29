"""Validation utilities extracted from FlextModels.

This module contains the Validation nested class from FlextModels.
It should NOT be imported directly - use FlextModels.Validation instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from urllib.parse import urlparse

from pydantic import BaseModel

from flext_core.constants import c
from flext_core.protocols import p
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.typings import t


class FlextModelsValidation:
    """Validation utility functions."""

    @staticmethod
    def validate_business_rules(
        model: t.GeneralValueType,
        *rules: Callable[
            [t.GeneralValueType],
            r[t.GeneralValueType],
        ],
    ) -> r[t.GeneralValueType]:
        """Validate business rules with railway patterns.

        Args:
            model: The model to validate
            *rules: Business rule validation functions

        Returns:
            r[t.GeneralValueType]: Validated model or accumulated errors

        Example:
            ```python
            def check_age_business_rule(user: User) -> r[bool]:
                return (
                    r[bool].ok(True) if user.age >= 18 else r[bool].fail("Must be 18+")
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
                return r[t.GeneralValueType].fail(error_msg)

        return r[t.GeneralValueType].ok(model)

    @staticmethod
    def validate_cross_fields(
        model: t.GeneralValueType,
        field_validators: t.FieldValidatorMapping,
    ) -> r[t.GeneralValueType]:
        """Validate cross-field dependencies with railway patterns.

        Args:
            model: The model to validate
            field_validators: Field name to validator mapping

        Returns:
            r[t.GeneralValueType]: Validated model or accumulated errors

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
        # Call validators and collect results (validators return Result-like objects)
        validation_results = [
            validator(model) for validator in field_validators.values()
        ]

        # Extract errors from failed results using duck typing - use getattr for pyright
        errors: list[str] = []
        for result in validation_results:
            is_failure = getattr(result, "is_failure", False)
            error_value = getattr(result, "error", None)
            if is_failure and error_value:
                errors.append(str(error_value))

        if errors:
            return r[t.GeneralValueType].fail(
                f"Cross-field validation failed: {'; '.join(errors)}",
                error_code="CROSS_FIELD_VALIDATION_FAILED",
                error_data={"field_errors": errors},
            )

        return r[t.GeneralValueType].ok(model)

    @staticmethod
    def validate_performance(
        model: t.GeneralValueType,
        max_validation_time_ms: int | None = None,
    ) -> r[t.GeneralValueType]:
        """Validate model with performance constraints.

        Args:
            model: The model to validate
            max_validation_time_ms: Maximum validation time in milliseconds

        Returns:
            Result: Validated model or performance error

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
            timeout_ms = c.Validation.VALIDATION_TIMEOUT_MS
        start_time = time.time()

        try:
            # Only validate Pydantic models with model_dump/model_validate
            if not isinstance(model, BaseModel):
                # For non-model types, just return success
                return r[t.GeneralValueType].ok(model)

            # Exclude computed fields that are not actual model fields
            # Use model_dump with exclude_unset to avoid extra fields
            dump = model.model_dump(
                exclude={"is_initial_version", "is_modified"},
                exclude_unset=True,
            )
            # Re-validate the model from the dump
            validated_model = model.__class__.model_validate(dump)
            validation_time = (time.time() - start_time) * c.MILLISECONDS_MULTIPLIER

            if validation_time > timeout_ms:
                return r[t.GeneralValueType].fail(
                    f"Validation too slow: {validation_time:.2f}ms > {timeout_ms}ms",
                    error_code="PERFORMANCE_VALIDATION_FAILED",
                    error_data={"validation_time_ms": validation_time},
                )

            return r[t.GeneralValueType].ok(validated_model)
        except (
            AttributeError,
            TypeError,
            ValueError,
            RuntimeError,
            KeyError,
        ) as e:
            return r[t.GeneralValueType].fail(
                f"Validation failed: {e}",
                error_code=c.Errors.VALIDATION_ERROR,
            )

    @staticmethod
    def validate_batch(
        models: t.ObjectList,
        *validators: Callable[
            [t.GeneralValueType],
            r[t.GeneralValueType],
        ],
        fail_fast: bool = True,
    ) -> r[t.ObjectList]:
        """Validate a batch of models with railway patterns.

        Args:
            models: List of models to validate
            *validators: Validation functions to apply
            fail_fast: Stop on first failure or accumulate all errors

        Returns:
            Result: All validated models or first failure

        Example:
            ```python
            def is_adult(user: User) -> r[bool]:
                return r[bool].ok(True) if user.age >= 18 else r[bool].fail("Not adult")


            result = FlextModels.Validation.validate_batch(
                user_models,
                is_adult,
                fail_fast=False,
            )
            ```

        """
        if fail_fast:
            # Validate models one by one, stop on first failure
            valid_models: list[t.GeneralValueType] = []
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
                        return r[t.ObjectList].fail(error_msg)

                valid_models.append(model)

            return r[t.ObjectList].ok(valid_models)
        # Accumulate all errors
        validated_models: list[t.GeneralValueType] = []
        all_errors: list[str] = []

        for model in models:
            # Use individual validation for models since validators may return t.GeneralValueType
            validation_result: r[t.GeneralValueType] = r[t.GeneralValueType].ok(model)
            for validator in validators:
                result = validator(model)
                if result.is_failure:
                    base_msg = "Validation failed"
                    error_msg = (
                        f"{base_msg}: {result.error}"
                        if result.error
                        else f"{base_msg} (model validation failed)"
                    )
                    # Create failure result
                    validation_result = r[t.GeneralValueType].fail(error_msg)
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
            return r[t.ObjectList].fail(
                f"Batch validation failed: {'; '.join(all_errors)}",
                error_code="BATCH_VALIDATION_FAILED",
                error_data={"error_count": len(all_errors), "errors": all_errors},
            )

        return r[t.ObjectList].ok(validated_models)

    @staticmethod
    def validate_domain_invariants(
        model: t.GeneralValueType,
        invariants: list[
            Callable[
                [t.GeneralValueType],
                r[t.GeneralValueType],
            ]
        ],
    ) -> r[t.GeneralValueType]:
        """Validate domain invariants with railway patterns.

        Args:
            model: The model to validate
            invariants: List of domain invariant validation functions

        Returns:
            Result: Validated model or first invariant violation

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
            # Cast model to t.GeneralValueType for type safety
            result = invariant(model)
            is_failure = getattr(result, "is_failure", False)
            if is_failure:
                error_msg = getattr(result, "error", "unknown")
                return r[t.GeneralValueType].fail(
                    f"Domain invariant violation: {error_msg}",
                    error_code="DOMAIN_INVARIANT_VIOLATION",
                    error_data={"invariant_error": error_msg},
                )
        return r[t.GeneralValueType].ok(model)

    @staticmethod
    def validate_aggregate_consistency_with_rules(
        aggregate: t.GeneralValueType,
        consistency_rules: t.ConsistencyRuleMapping,
    ) -> r[t.GeneralValueType]:
        """Validate aggregate consistency with railway patterns.

        Args:
            aggregate: The aggregate to validate
            consistency_rules: Dictionary of consistency rule validators

        Returns:
            Result: Validated aggregate or consistency violation

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
            result = validator(aggregate)
            # Duck typing for Result-like objects - use getattr for pyright compatibility
            is_failure = getattr(result, "is_failure", False)
            if is_failure:
                error_msg = getattr(result, "error", "Unknown error")
                violations.append(f"{rule_name}: {error_msg}")

        if violations:
            return r[t.GeneralValueType].fail(
                f"Aggregate consistency violations: {'; '.join(violations)}",
                error_code="AGGREGATE_CONSISTENCY_VIOLATION",
                error_data={"violations": violations},
            )

        return r[t.GeneralValueType].ok(aggregate)

    @staticmethod
    def validate_event_sourcing(
        event: t.GeneralValueType,
        event_validators: t.EventValidatorMapping,
    ) -> r[t.GeneralValueType]:
        """Validate event sourcing patterns with railway patterns.

        Args:
            event: The domain event to validate
            event_validators: Dictionary of event-specific validators

        Returns:
            Result: Validated event or validation failure

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
        # Call validators and collect results (validators return Result-like objects)
        validation_results = [
            validator(event) for validator in event_validators.values()
        ]

        # Extract errors from failed results using duck typing - use getattr for pyright
        errors: list[str] = []
        for result in validation_results:
            is_failure = getattr(result, "is_failure", False)
            error_value = getattr(result, "error", None)
            if is_failure and error_value:
                errors.append(str(error_value))

        if errors:
            return r[t.GeneralValueType].fail(
                f"Event validation failed: {'; '.join(errors)}",
                error_code="EVENT_VALIDATION_FAILED",
                error_data={"event_errors": errors},
            )

        return r[t.GeneralValueType].ok(event)

    @staticmethod
    def validate_cqrs_patterns(
        command_or_query: t.GeneralValueType,
        pattern_type: str,
        validators: list[
            Callable[
                [t.GeneralValueType],
                r[t.GeneralValueType],
            ]
        ],
    ) -> r[t.GeneralValueType]:
        """Validate CQRS patterns with railway patterns.

        Args:
            command_or_query: The command or query to validate
            pattern_type: Type of pattern (command or query handler type)
            validators: List of pattern-specific validators

        Returns:
            Result: Validated command/query or validation failure

        Example:
            ```python
            result = FlextModels.Validation.validate_cqrs_patterns(
                create_order_command,
                c.Cqrs.HandlerType.COMMAND.value,
                [
                    lambda c: validate_command_structure(c),
                    lambda c: validate_command_data(c),
                    lambda c: validate_command_permissions(c),
                ],
            )
            ```

        """
        valid_pattern_types = {
            c.Cqrs.HandlerType.COMMAND.value,
            c.Cqrs.HandlerType.QUERY.value,
        }
        if pattern_type not in valid_pattern_types:
            return r[t.GeneralValueType].fail(
                f"Invalid pattern type: {pattern_type}. Must be '{c.Cqrs.HandlerType.COMMAND.value}' or '{c.Cqrs.HandlerType.QUERY.value}'",
                error_code="INVALID_PATTERN_TYPE",
            )

        for validator in validators:
            # Cast command_or_query to t.GeneralValueType for type safety
            result = validator(command_or_query)
            if result.is_failure:
                return r[t.GeneralValueType].fail(
                    f"CQRS {pattern_type} validation failed: {result.error}",
                    error_code=f"CQRS_{pattern_type.upper()}_VALIDATION_FAILED",
                    error_data={
                        "pattern_type": pattern_type,
                        "error": result.error,
                    },
                )

        return r[t.GeneralValueType].ok(command_or_query)

    @staticmethod
    def _validate_event_structure(
        event: t.GeneralValueType,
    ) -> r[bool]:
        """Validate event is not None and has required attributes."""
        if event is None:
            return r[bool].fail(
                "Domain event cannot be None",
                error_code=c.Errors.VALIDATION_ERROR,
            )

        # Check required attributes
        required_attrs = ["event_type", "aggregate_id", "unique_id", "created_at"]
        missing_attrs = [
            attr
            for attr in required_attrs
            if not (
                hasattr(event, attr)
                or (
                    FlextRuntime.is_dict_like(event)
                    and isinstance(event, Mapping)
                    and attr in event
                )
            )
        ]
        if missing_attrs:
            return r[bool].fail(
                f"Domain event missing required attributes: {missing_attrs}",
                error_code=c.Errors.VALIDATION_ERROR,
            )

        return r[bool].ok(True)

    @staticmethod
    def _validate_event_fields(
        event: t.GeneralValueType,
    ) -> r[bool]:
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
            return r[bool].fail(
                "Domain event event_type must be a non-empty string",
                error_code=c.Errors.VALIDATION_ERROR,
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
            return r[bool].fail(
                "Domain event aggregate_id must be a non-empty string",
                error_code=c.Errors.VALIDATION_ERROR,
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
            return r[bool].fail(
                "Domain event data must be a dictionary or None",
                error_code=c.Errors.VALIDATION_ERROR,
            )

        return r[bool].ok(True)

    @staticmethod
    def validate_domain_event(
        event: t.GeneralValueType,
    ) -> r[bool]:
        """Enhanced domain event validation with comprehensive checks.

        Validates domain events for proper structure, required fields,
        and domain invariants. Used across all flext-ecosystem projects.

        Args:
            event: The domain event to validate

        Returns:
            Result[bool]: Success with True if valid, failure with details

        """
        # Validate structure
        structure_result = FlextModelsValidation._validate_event_structure(event)
        if structure_result.is_failure:
            return structure_result

        # Validate fields
        fields_result = FlextModelsValidation._validate_event_fields(event)
        if fields_result.is_failure:
            return fields_result

        return r[bool].ok(True)

    @staticmethod
    def validate_aggregate_consistency(
        aggregate: t.GeneralValueType | p.Validation.HasInvariants,
    ) -> r[t.GeneralValueType | p.Validation.HasInvariants]:
        """Validate aggregate consistency and business invariants.

        Ensures aggregates maintain consistency boundaries and invariants
        are satisfied. Used extensively in flext-core and dependent projects.

        Args:
            aggregate: The aggregate root to validate

        Returns:
            Result: Validated aggregate or failure with details

        """
        if aggregate is None:
            return r[t.GeneralValueType | p.Validation.HasInvariants].fail(
                "Aggregate cannot be None",
                error_code=c.Errors.VALIDATION_ERROR,
            )

        # Check invariants if the aggregate supports them
        if isinstance(aggregate, p.Validation.HasInvariants):
            try:
                aggregate.check_invariants()  # Returns None, just call for side effects
            except (
                AttributeError,
                TypeError,
                ValueError,
                RuntimeError,
                KeyError,
            ) as e:
                return r[t.GeneralValueType | p.Validation.HasInvariants].fail(
                    f"Aggregate invariant violation: {e}",
                    error_code=c.Errors.VALIDATION_ERROR,
                )

        # Check for uncommitted domain events (potential consistency issue)
        if hasattr(aggregate, "domain_events"):
            events = getattr(aggregate, "domain_events", [])
            if len(events) > c.Validation.MAX_UNCOMMITTED_EVENTS:
                max_events = c.Validation.MAX_UNCOMMITTED_EVENTS
                event_count = len(events)
                error_msg = (
                    f"Too many uncommitted domain events: {event_count} "
                    f"(max: {max_events})"
                )
                return r[t.GeneralValueType | p.Validation.HasInvariants].fail(
                    error_msg,
                    error_code=c.Errors.VALIDATION_ERROR,
                )

        return r[t.GeneralValueType | p.Validation.HasInvariants].ok(aggregate)

    @staticmethod
    def validate_entity_relationships(
        entity: object,
    ) -> r[object]:
        """Validate entity relationships and references.

        Ensures entity references are valid and relationships are consistent.
        Critical for maintaining data integrity across flext-ecosystem.

        Args:
            entity: The entity to validate (accepts any object with attributes)

        Returns:
            Result[object]: Validated entity or failure with details

        """
        if entity is None:
            return r[object].fail(
                "Entity cannot be None",
                error_code=c.Errors.VALIDATION_ERROR,
            )

        # Entity ID is validated at field level by Pydantic v2
        # No custom validation needed - use Field(pattern=r"^[a-zA-Z0-9_-]+$")

        # Validate version for optimistic locking
        if hasattr(entity, "version"):
            version = getattr(entity, "version", 0)
            if not isinstance(version, int) or version < 0:
                return r[object].fail(
                    "Entity version must be a non-negative integer",
                    error_code=c.Errors.VALIDATION_ERROR,
                )

        # Validate timestamps if present
        for timestamp_field in ["created_at", "updated_at"]:
            if hasattr(entity, timestamp_field):
                timestamp = getattr(entity, timestamp_field)
                if timestamp is not None and not isinstance(timestamp, datetime):
                    return r[object].fail(
                        f"Entity {timestamp_field} must be a datetime or None",
                        error_code=c.Errors.VALIDATION_ERROR,
                    )

        # Return validated entity
        return r[object].ok(entity)

    @staticmethod
    def _validate_uri_format(
        uri: str | None,
        allowed_schemes: list[str] | None = None,
        context: str = "URI",
    ) -> r[str]:
        if uri is None:
            return r[str].fail(
                f"{context} cannot be None",
                error_code=c.Errors.VALIDATION_ERROR,
            )
        # Type narrowing: after None check, uri is guaranteed to be str
        try:
            parsed_uri = urlparse(uri)
            if not all([parsed_uri.scheme, parsed_uri.netloc]):
                return r[str].fail(
                    f"Invalid {context} format: missing scheme or netloc",
                    error_code=c.Errors.VALIDATION_ERROR,
                )
            if allowed_schemes and parsed_uri.scheme not in allowed_schemes:
                return r[str].fail(
                    f"Invalid {context} scheme: '{parsed_uri.scheme}'. Must be one of {allowed_schemes}",
                    error_code=c.Errors.VALIDATION_ERROR,
                )
            return r[str].ok(uri)
        except ValueError as e:
            return r[str].fail(
                f"Invalid {context} format: {e}",
                error_code=c.Errors.VALIDATION_ERROR,
            )

    @staticmethod
    def validate_uri(
        uri: str | None,
        allowed_schemes: list[str] | None = None,
        context: str = "URI",
    ) -> r[str]:
        return FlextModelsValidation._validate_uri_format(uri, allowed_schemes, context)

    @staticmethod
    def _validate_port_range(port: int | None, context: str = "Port") -> r[int]:
        if port is None:
            return r[int].fail(
                f"{context} cannot be None",
                error_code=c.Errors.VALIDATION_ERROR,
            )
        # Type narrowing: after None check, port is guaranteed to be int
        if not c.Network.MIN_PORT <= port <= c.Network.MAX_PORT:
            return r[int].fail(
                f"{context} must be between {c.Network.MIN_PORT} and {c.Network.MAX_PORT} (got {port})",
                error_code=c.Errors.VALIDATION_ERROR,
            )
        return r[int].ok(port)

    @staticmethod
    def validate_port_number(
        port: int | None,
        context: str = "Port",
    ) -> r[int]:
        return FlextModelsValidation._validate_port_range(port, context)

    @staticmethod
    def _validate_non_empty_string_format(
        value: str | None,
        context: str = "Field",
    ) -> r[str]:
        if value is None:
            return r[str].fail(
                f"{context} cannot be None",
                error_code=c.Errors.VALIDATION_ERROR,
            )
        # Type narrowing: after None check, value is guaranteed to be str
        if not value:
            return r[str].fail(
                f"{context} cannot be empty",
                error_code=c.Errors.VALIDATION_ERROR,
            )
        return r[str].ok(value)

    @staticmethod
    def validate_required_string(
        value: str | None,
        context: str = "Field",
    ) -> r[str]:
        return FlextModelsValidation._validate_non_empty_string_format(value, context)

    @staticmethod
    def _validate_choice_value(
        value: str,
        valid_choices: set[str],
        context: str = "Value",
        *,
        case_sensitive: bool = False,
    ) -> r[str]:
        # Type narrowing: value is already typed as str (no runtime check needed)
        if not valid_choices:
            return r[str].fail(
                "Valid choices cannot be empty",
                error_code=c.Errors.VALIDATION_ERROR,
            )

        target_value = value if case_sensitive else value.lower()
        target_choices = (
            valid_choices if case_sensitive else {c.lower() for c in valid_choices}
        )

        if target_value not in target_choices:
            return r[str].fail(
                f"Invalid {context}: '{value}'. Must be one of {list(valid_choices)}",
                error_code=c.Errors.VALIDATION_ERROR,
            )
        return r[str].ok(value)

    @staticmethod
    def validate_choice(
        value: str,
        valid_choices: set[str],
        context: str = "Value",
        *,
        case_sensitive: bool = False,
    ) -> r[str]:
        return FlextModelsValidation._validate_choice_value(
            value,
            valid_choices,
            context,
            case_sensitive=case_sensitive,
        )

    @staticmethod
    def _validate_length_range(
        value: str,
        min_length: int | None = None,
        max_length: int | None = None,
        context: str = "Value",
    ) -> r[str]:
        # Type narrowing: value is already typed as str (no runtime check needed)
        current_length = len(value)
        if min_length is not None and current_length < min_length:
            return r[str].fail(
                f"{context} length {current_length} is less than minimum {min_length}",
                error_code=c.Errors.VALIDATION_ERROR,
            )
        if max_length is not None and current_length > max_length:
            return r[str].fail(
                f"{context} length {current_length} exceeds maximum {max_length}",
                error_code=c.Errors.VALIDATION_ERROR,
            )
        return r[str].ok(value)

    @staticmethod
    def validate_length(
        value: str,
        min_length: int | None = None,
        max_length: int | None = None,
        context: str = "Value",
    ) -> r[str]:
        return FlextModelsValidation._validate_length_range(
            value,
            min_length,
            max_length,
            context,
        )

    @staticmethod
    def _validate_regex_pattern(
        value: str,
        pattern: str,
        context: str = "Value",
    ) -> r[str]:
        # Type narrowing: value is already typed as str (no runtime check needed)
        try:
            if not re.fullmatch(pattern, value):
                return r[str].fail(
                    f"{context} '{value}' does not match pattern '{pattern}'",
                    error_code=c.Errors.VALIDATION_ERROR,
                )
            return r[str].ok(value)
        except re.error as e:
            return r[str].fail(
                f"Invalid regex pattern '{pattern}': {e}",
                error_code=c.Errors.VALIDATION_ERROR,
            )

    @staticmethod
    def validate_pattern(
        value: str,
        pattern: str,
        context: str = "Value",
    ) -> r[str]:
        return FlextModelsValidation._validate_regex_pattern(value, pattern, context)


__all__ = ["FlextModelsValidation"]
