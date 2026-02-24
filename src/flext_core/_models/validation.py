"""Validation utilities extracted from FlextModels.

This module contains the Validation nested class from FlextModels.
It should NOT be imported directly - use FlextModels.Validation instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from urllib.parse import urlparse

from pydantic import BaseModel

from flext_core.constants import c
from flext_core.protocols import p
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.typings import t

type ValidationSubject = t.ScalarValue | BaseModel
type EventInput = BaseModel | Mapping[str, t.ConfigMapValue]


class FlextModelsValidation:
    """Validation utility functions."""

    @staticmethod
    def _validation_error[T](
        message: str,
        expected_type: type[T] | None = None,
    ) -> r[T]:
        return r[T].fail(
            message,
            error_code=c.Errors.VALIDATION_ERROR,
            expected_type=expected_type,
        )

    @staticmethod
    def _validation_failure_message(
        error: str | None,
        fallback_reason: str,
    ) -> str:
        if error:
            return f"Validation failed: {error}"
        return f"Validation failed ({fallback_reason})"

    @staticmethod
    def _collect_result_errors(
        validation_results: Sequence[object],
    ) -> list[str]:
        errors: list[str] = []
        for result in validation_results:
            is_failure = getattr(result, "is_failure", False)
            error_value = getattr(result, "error", None)
            if is_failure and error_value:
                errors.append(str(error_value))
        return errors

    @staticmethod
    def _validate_with_validator_mapping(
        subject: ValidationSubject,
        validators: t.FieldValidatorMap | t.EventValidatorMap,
        failure_message_prefix: str,
        failure_code: str,
        error_data_key: str,
    ) -> r[ValidationSubject]:
        validation_results = [validator(subject) for validator in validators.values()]
        errors = FlextModelsValidation._collect_result_errors(validation_results)
        if errors:
            return r[ValidationSubject].fail(
                f"{failure_message_prefix}: {'; '.join(errors)}",
                error_code=failure_code,
                error_data=t.ConfigMap(root={error_data_key: errors}),
            )
        return r[ValidationSubject].ok(subject)

    @staticmethod
    def _validate_model_with_validators(
        model: ValidationSubject,
        validators: Sequence[Callable[[ValidationSubject], r[ValidationSubject]]],
        fallback_reason: str,
    ) -> r[ValidationSubject]:
        failure_result = FlextModelsValidation._first_validation_failure_result(
            model,
            validators,
        )
        if failure_result is not None:
            error_msg = FlextModelsValidation._validation_failure_message(
                failure_result.error,
                fallback_reason,
            )
            return r[ValidationSubject].fail(error_msg)
        return r[ValidationSubject].ok(model)

    @staticmethod
    def _first_validation_failure_result(
        value: ValidationSubject,
        validators: Sequence[Callable[[ValidationSubject], r[ValidationSubject]]],
    ) -> r[ValidationSubject] | None:
        for validator in validators:
            result = validator(value)
            if result.is_failure:
                return result
        return None

    @staticmethod
    def _required_event_string_error(
        event: EventInput,
        field_name: str,
    ) -> r[bool] | None:
        field_value = FlextModelsValidation._event_get(event, field_name, "")
        if field_value and field_value.__class__ is str:
            return None
        return FlextModelsValidation._validation_error(
            f"Domain event {field_name} must be a non-empty string",
        )

    @staticmethod
    def validate_business_rules(
        model: ValidationSubject,
        *rules: Callable[
            [ValidationSubject],
            r[ValidationSubject],
        ],
    ) -> r[ValidationSubject]:
        """Validate business-rule callables for a model."""
        return FlextModelsValidation._validate_model_with_validators(
            model,
            rules,
            "validation rule failed",
        )

    @staticmethod
    def validate_cross_fields(
        model: ValidationSubject,
        field_validators: t.FieldValidatorMap,
    ) -> r[ValidationSubject]:
        """Validate cross-field dependencies with mapped validators."""
        return FlextModelsValidation._validate_with_validator_mapping(
            model,
            field_validators,
            "Cross-field validation failed",
            "CROSS_FIELD_VALIDATION_FAILED",
            "field_errors",
        )

    @staticmethod
    def validate_performance(
        model: BaseModel,
        max_validation_time_ms: int | None = None,
    ) -> r[BaseModel]:
        """Validate model execution time against a threshold."""
        timeout_ms = (
            max_validation_time_ms
            if max_validation_time_ms is not None
            else c.Validation.VALIDATION_TIMEOUT_MS
        )
        start_time = time.time()

        try:
            dump = model.model_dump(
                exclude={"is_initial_version", "is_modified"},
                exclude_unset=True,
            )
            validated_model = model.__class__.model_validate(dump)
            validation_time = (time.time() - start_time) * c.MILLISECONDS_MULTIPLIER

            if validation_time > timeout_ms:
                return r[BaseModel].fail(
                    f"Validation too slow: {validation_time:.2f}ms > {timeout_ms}ms",
                    error_code="PERFORMANCE_VALIDATION_FAILED",
                    error_data=t.ConfigMap(
                        root={"validation_time_ms": validation_time}
                    ),
                )

            return r[BaseModel].ok(validated_model)
        except (
            AttributeError,
            TypeError,
            ValueError,
            RuntimeError,
            KeyError,
        ) as e:
            return r[BaseModel].fail(
                f"Validation failed: {e}",
                error_code=c.Errors.VALIDATION_ERROR,
            )

    @staticmethod
    def validate_batch(
        models: Sequence[ValidationSubject],
        *validators: Callable[
            [ValidationSubject],
            r[ValidationSubject],
        ],
        fail_fast: bool = True,
    ) -> r[list[ValidationSubject]]:
        """Validate a model batch with optional fail-fast behavior."""
        validated_models: list[ValidationSubject] = []
        all_errors: list[str] = []
        fallback_reason = (
            "item validation failed" if fail_fast else "model validation failed"
        )

        for model in models:
            validation_result = FlextModelsValidation._validate_model_with_validators(
                model,
                validators,
                fallback_reason,
            )
            if validation_result.is_success:
                validated_models.append(model)
                continue

            if fail_fast:
                return r[list[ValidationSubject]].fail(
                    FlextModelsValidation._validation_failure_message(
                        validation_result.error,
                        "item validation failed",
                    ),
                )

            error_msg = FlextModelsValidation._validation_failure_message(
                validation_result.error,
                "validation rule failed",
            )
            all_errors.append(error_msg)

        if all_errors:
            return r[list[ValidationSubject]].fail(
                f"Batch validation failed: {'; '.join(all_errors)}",
                error_code="BATCH_VALIDATION_FAILED",
                error_data=t.ConfigMap(
                    root={"error_count": len(all_errors), "errors": all_errors}
                ),
            )

        return r[list[ValidationSubject]].ok(validated_models)

    @staticmethod
    def validate_domain_invariants(
        model: ValidationSubject,
        invariants: list[
            Callable[
                [ValidationSubject],
                r[ValidationSubject],
            ]
        ],
    ) -> r[ValidationSubject]:
        """Validate aggregate/domain invariants for a model."""
        failure_result = FlextModelsValidation._first_validation_failure_result(
            model,
            invariants,
        )
        if failure_result is not None:
            invariant_error = failure_result.error or "unknown"
            return r[ValidationSubject].fail(
                f"Domain invariant violation: {invariant_error}",
                error_code="DOMAIN_INVARIANT_VIOLATION",
                error_data=t.ConfigMap(root={"invariant_error": invariant_error}),
            )
        return r[ValidationSubject].ok(model)

    @staticmethod
    def validate_aggregate_consistency_with_rules(
        aggregate: ValidationSubject,
        consistency_rules: t.ConsistencyRuleMap,
    ) -> r[ValidationSubject]:
        """Validate aggregate consistency with named rule validators."""
        violations: list[str] = []
        for rule_name, validator in consistency_rules.items():
            result = validator(aggregate)
            is_failure = getattr(result, "is_failure", False)
            if is_failure:
                error_msg = getattr(result, "error", "Unknown error")
                violations.append(f"{rule_name}: {error_msg}")

        if violations:
            return r[ValidationSubject].fail(
                f"Aggregate consistency violations: {'; '.join(violations)}",
                error_code="AGGREGATE_CONSISTENCY_VIOLATION",
                error_data=t.ConfigMap(root={"violations": violations}),
            )

        return r[ValidationSubject].ok(aggregate)

    @staticmethod
    def validate_event_sourcing(
        event: ValidationSubject,
        event_validators: t.EventValidatorMap,
    ) -> r[ValidationSubject]:
        """Validate event-sourcing constraints for a domain event."""
        return FlextModelsValidation._validate_with_validator_mapping(
            event,
            event_validators,
            "Event validation failed",
            "EVENT_VALIDATION_FAILED",
            "event_errors",
        )

    @staticmethod
    def validate_cqrs_patterns(
        command_or_query: ValidationSubject,
        pattern_type: str,
        validators: list[
            Callable[
                [ValidationSubject],
                r[ValidationSubject],
            ]
        ],
    ) -> r[ValidationSubject]:
        """Validate CQRS command/query payloads with pattern-specific validators."""
        valid_pattern_types = {
            c.Cqrs.HandlerType.COMMAND.value,
            c.Cqrs.HandlerType.QUERY.value,
        }
        if pattern_type not in valid_pattern_types:
            return r[ValidationSubject].fail(
                f"Invalid pattern type: {pattern_type}. Must be '{c.Cqrs.HandlerType.COMMAND.value}' or '{c.Cqrs.HandlerType.QUERY.value}'",
                error_code="INVALID_PATTERN_TYPE",
            )

        failure_result = FlextModelsValidation._first_validation_failure_result(
            command_or_query,
            validators,
        )
        if failure_result is not None:
            failure_error = failure_result.error
            return r[ValidationSubject].fail(
                f"CQRS {pattern_type} validation failed: {failure_error}",
                error_code=f"CQRS_{pattern_type.upper()}_VALIDATION_FAILED",
                error_data=t.ConfigMap(
                    root={
                        "pattern_type": pattern_type,
                        "error": failure_error,
                    }
                ),
            )

        return r[ValidationSubject].ok(command_or_query)

    @staticmethod
    def _validate_event_structure(
        event: EventInput,
    ) -> r[bool]:
        """Validate event is not None and has required attributes."""
        if event is None:
            return FlextModelsValidation._validation_error(
                "Domain event cannot be None",
            )

        required_attrs = ["event_type", "aggregate_id", "unique_id", "created_at"]
        missing_attrs = [
            attr
            for attr in required_attrs
            if not FlextModelsValidation._event_has_attr(event, attr)
        ]
        if missing_attrs:
            return FlextModelsValidation._validation_error(
                f"Domain event missing required attributes: {missing_attrs}",
            )

        return r[bool].ok(value=True)

    @staticmethod
    def _validate_event_fields(
        event: EventInput,
    ) -> r[bool]:
        """Validate event field types and values."""
        for required_field in ("event_type", "aggregate_id"):
            required_field_error = FlextModelsValidation._required_event_string_error(
                event,
                required_field,
            )
            if required_field_error is not None:
                return required_field_error

        data = FlextModelsValidation._event_get(event, "data", None)
        if data is not None and not FlextRuntime.is_dict_like(data):
            return FlextModelsValidation._validation_error(
                "Domain event data must be a dictionary or None",
            )

        return r[bool].ok(value=True)

    @staticmethod
    def validate_domain_event(
        event: EventInput,
    ) -> r[bool]:
        """Enhanced domain event validation with comprehensive checks.

        Validates domain events for proper structure, required fields,
        and domain invariants. Used across all flext-ecosystem projects.

        Args:
            event: The domain event to validate

        Returns:
            Result[bool]: Success with True if valid, failure with details

        """
        for validation_result in (
            FlextModelsValidation._validate_event_structure(event),
            FlextModelsValidation._validate_event_fields(event),
        ):
            if validation_result.is_failure:
                return validation_result

        return r[bool].ok(value=True)

    @staticmethod
    def _event_mapping(
        event: EventInput,
    ) -> Mapping[str, t.ConfigMapValue]:
        if isinstance(event, BaseModel):
            return {
                str(k): FlextRuntime.normalize_to_general_value(v)
                for k, v in event.model_dump().items()
            }
        return event

    @staticmethod
    def _event_has_attr(event: EventInput, attr: str) -> bool:
        if isinstance(event, BaseModel) and hasattr(event, attr):
            return True
        return attr in FlextModelsValidation._event_mapping(event)

    @staticmethod
    def _event_get(
        event: EventInput,
        key: str,
        default: t.ConfigMapValue,
    ) -> t.ConfigMapValue:
        if isinstance(event, BaseModel):
            return getattr(event, key, default)
        return event.get(key, default)

    @staticmethod
    def validate_aggregate_consistency(
        aggregate: p.Validation.HasInvariants,
    ) -> r[p.Validation.HasInvariants]:
        """Validate aggregate consistency and business invariants.

        Ensures aggregates maintain consistency boundaries and invariants
        are satisfied. Used extensively in flext-core and dependent projects.

        Args:
            aggregate: The aggregate root to validate

        Returns:
            Result: Validated aggregate or failure with details

        """
        if aggregate is None:
            return FlextModelsValidation._validation_error(
                "Aggregate cannot be None",
            )

        try:
            aggregate.check_invariants()
        except (
            AttributeError,
            TypeError,
            ValueError,
            RuntimeError,
            KeyError,
        ) as e:
            return FlextModelsValidation._validation_error(
                f"Aggregate invariant violation: {e}",
            )

        if hasattr(aggregate, "domain_events"):
            events = getattr(aggregate, "domain_events", [])
            if len(events) > c.Validation.MAX_UNCOMMITTED_EVENTS:
                max_events = c.Validation.MAX_UNCOMMITTED_EVENTS
                event_count = len(events)
                error_msg = (
                    f"Too many uncommitted domain events: {event_count} "
                    f"(max: {max_events})"
                )
                return FlextModelsValidation._validation_error(
                    error_msg,
                )

        return r[p.Validation.HasInvariants].ok(aggregate)

    @staticmethod
    def validate_entity_relationships(
        entity: BaseModel,
    ) -> r[BaseModel]:
        """Validate entity relationships and references.

        Ensures entity references are valid and relationships are consistent.
        Critical for maintaining data integrity across flext-ecosystem.

        Args:
            entity: The entity to validate (accepts any object with attributes)

        Returns:
            Result[object]: Validated entity or failure with details

        """
        if entity is None:
            return FlextModelsValidation._validation_error(
                "Entity cannot be None",
            )

        if hasattr(entity, "version"):
            version = getattr(entity, "version", 0)
            if version.__class__ is not int or version < 0:
                return FlextModelsValidation._validation_error(
                    "Entity version must be a non-negative integer",
                )

        for timestamp_field in ["created_at", "updated_at"]:
            if hasattr(entity, timestamp_field):
                timestamp = getattr(entity, timestamp_field)
                if timestamp is not None and timestamp.__class__ is not datetime:
                    return FlextModelsValidation._validation_error(
                        f"Entity {timestamp_field} must be a datetime or None",
                    )

        return r[BaseModel].ok(entity)

    @staticmethod
    def validate_uri(
        uri: str | None,
        allowed_schemes: list[str] | None = None,
        context: str = "URI",
    ) -> r[str]:
        if uri is None:
            return FlextModelsValidation._validation_error(
                f"{context} cannot be None",
            )
        try:
            parsed_uri = urlparse(uri)
            if not all([parsed_uri.scheme, parsed_uri.netloc]):
                return FlextModelsValidation._validation_error(
                    f"Invalid {context} format: missing scheme or netloc",
                )
            if allowed_schemes and parsed_uri.scheme not in allowed_schemes:
                return FlextModelsValidation._validation_error(
                    f"Invalid {context} scheme: '{parsed_uri.scheme}'. Must be one of {allowed_schemes}",
                )
            return r[str].ok(uri)
        except ValueError as e:
            return FlextModelsValidation._validation_error(
                f"Invalid {context} format: {e}",
            )

    @staticmethod
    def validate_port_number(
        port: int | None,
        context: str = "Port",
    ) -> r[int]:
        if port is None:
            return FlextModelsValidation._validation_error(
                f"{context} cannot be None",
            )
        if not c.Network.MIN_PORT <= port <= c.Network.MAX_PORT:
            return FlextModelsValidation._validation_error(
                f"{context} must be between {c.Network.MIN_PORT} and {c.Network.MAX_PORT} (got {port})",
            )
        return r[int].ok(port)

    @staticmethod
    def validate_required_string(
        value: str | None,
        context: str = "Field",
    ) -> r[str]:
        if value is None:
            return FlextModelsValidation._validation_error(
                f"{context} cannot be None",
            )
        if not value:
            return FlextModelsValidation._validation_error(
                f"{context} cannot be empty",
            )
        return r[str].ok(value)

    @staticmethod
    def validate_choice(
        value: str,
        valid_choices: set[str] | list[str],
        context: str = "Value",
        *,
        case_sensitive: bool = False,
    ) -> r[str]:
        choices_set = (
            set(valid_choices) if valid_choices.__class__ is list else valid_choices
        )

        if not valid_choices:
            return FlextModelsValidation._validation_error(
                "Valid choices cannot be empty",
            )

        target_value = value if case_sensitive else value.lower()
        target_choices = (
            choices_set if case_sensitive else {c.lower() for c in choices_set}
        )

        if target_value not in target_choices:
            return FlextModelsValidation._validation_error(
                f"Invalid {context}: '{value}'. Must be one of {list(choices_set)}",
            )
        return r[str].ok(value)

    @staticmethod
    def validate_length(
        value: str,
        min_length: int | None = None,
        max_length: int | None = None,
        context: str = "Value",
    ) -> r[str]:
        current_length = len(value)
        if min_length is not None and current_length < min_length:
            return FlextModelsValidation._validation_error(
                f"{context} length {current_length} is less than minimum {min_length}",
            )
        if max_length is not None and current_length > max_length:
            return FlextModelsValidation._validation_error(
                f"{context} length {current_length} exceeds maximum {max_length}",
            )
        return r[str].ok(value)

    @staticmethod
    def validate_pattern(
        value: str,
        pattern: str,
        context: str = "Value",
    ) -> r[str]:
        try:
            if not re.fullmatch(pattern, value):
                return FlextModelsValidation._validation_error(
                    f"{context} '{value}' does not match pattern '{pattern}'",
                )
            return r[str].ok(value)
        except re.error as e:
            return FlextModelsValidation._validation_error(
                f"Invalid regex pattern '{pattern}': {e}",
            )


__all__ = ["FlextModelsValidation"]
