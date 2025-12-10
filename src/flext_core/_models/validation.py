"""Validation utilities extracted from FlextModels.

This module contains the Validation nested class from FlextModels.
It should NOT be imported directly - use FlextModels.Validation instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import socket
import time as time_module
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field
from pydantic.functional_validators import AfterValidator

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
        field_validators: t.Types.FieldValidatorMapping,
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
        model: BaseModel,
        max_validation_time_ms: int | None = None,
    ) -> r[BaseModel]:
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
            validation_time = (
                time_module.time() - start_time
            ) * c.MILLISECONDS_MULTIPLIER

            if validation_time > timeout_ms:
                return r[BaseModel].fail(
                    f"Validation too slow: {validation_time:.2f}ms > {timeout_ms}ms",
                    error_code="PERFORMANCE_VALIDATION_FAILED",
                    error_data={"validation_time_ms": validation_time},
                )

            return r[BaseModel].ok(validated_model)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return r[BaseModel].fail(
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
            # Use individual validation for models since validators may return GeneralValueType
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
            # Cast model to GeneralValueType for type safety
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
        consistency_rules: t.Types.ConsistencyRuleMapping,
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
        event_validators: t.Types.EventValidatorMapping,
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
            # Cast command_or_query to GeneralValueType for type safety
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

    # =========================================================================
    # VALIDATION TYPE ALIASES (Tier 1 - can import constants)
    # =========================================================================
    # Moved from typings.py (Tier 0) to here (Tier 1) because Field validators
    # require actual constant values which cannot be imported in Tier 0 modules

    class Validation:
        """Domain validation types using Pydantic Field annotations."""

        # Network validation types
        # NOTE: Field() requires literal values, so we use constants directly
        # These constants are centralized in FlextConstants for reuse
        type PortNumber = Annotated[
            int,
            Field(
                ge=c.Network.MIN_PORT,
                le=c.Network.MAX_PORT,
                description="Network port",
            ),
        ]
        type TimeoutSeconds = Annotated[
            float,
            Field(
                gt=c.ZERO,
                le=int(c.Network.DEFAULT_TIMEOUT),
                description="Timeout in seconds",
            ),
        ]
        type RetryCount = Annotated[
            int,
            Field(
                ge=c.ZERO,
                le=c.Validation.RETRY_COUNT_MAX,
                description="Retry attempts",
            ),
        ]

        # String validation types
        type NonEmptyStr = Annotated[
            str,
            Field(
                min_length=c.Reliability.RETRY_COUNT_MIN,
                description="Non-empty string",
            ),
        ]

        @staticmethod
        def validate_hostname(value: str) -> str:
            """Validate hostname by attempting DNS resolution.

            Business Rule: Validates hostname strings by attempting DNS resolution
            using socket.gethostbyname(). Ensures hostnames are resolvable before
            being used in network configurations. Raises ValueError if hostname cannot
            be resolved, preventing invalid network configurations.

            Audit Implication: Hostname validation ensures network configurations
            are valid before being used in production systems. Failed validations
            are logged with error messages for audit trail completeness. Used by
            Pydantic 2 AfterValidator for type-safe hostname validation.

            Args:
                value: Hostname string to validate

            Returns:
                Validated hostname string (same as input if valid)

            Raises:
                ValueError: If hostname cannot be resolved via DNS

            """
            try:
                _ = socket.gethostbyname(value)
                return value
            except socket.gaierror as e:
                msg = f"Cannot resolve hostname '{value}': {e}"
                raise ValueError(msg) from e

        # HostName type defined below after validate_hostname method

        type HostName = Annotated[
            str,
            Field(
                min_length=c.Reliability.RETRY_COUNT_MIN,
                max_length=c.Network.MAX_HOSTNAME_LENGTH,
                description="Valid hostname",
            ),
            AfterValidator(FlextModelsValidation.Validation.validate_hostname),
        ]


__all__ = ["FlextModelsValidation"]
