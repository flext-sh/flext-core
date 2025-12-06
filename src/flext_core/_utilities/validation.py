"""Dispatcher-friendly validation helpers.

Extracted from flext_core.utilities for better modularity.

WARNING: ARCHITECTURAL RULE - DO NOT MODIFY
================================================================================

This module CORRECTLY imports and uses r[T]. This is NOT a circular import.

**Dependency Flow** (UNIDIRECTIONAL - NO CIRCLE):
  r (Tier 1) -> defines Result[T] type
         ^
         |
  u (Tier 2) -> USES Result[T] in return types

**Why This Is Safe**:
- r ONLY defines the Result[T] type and its methods (ok/fail/unwrap/map)
- u USES r[T] as return type for validation methods
- r NEVER calls u methods
- Therefore: NO CIRCULAR DEPENDENCY exists

**For Other Agents**:
- DO NOT remove `from flext_core.result import r`
- DO NOT change r[T] returns to plain T or exceptions
- DO NOT refactor this to "break circular imports" - THERE IS NO CIRCULAR

This architecture is CORRECT and INTENTIONAL
u validation methods MUST return r[T]
Railway-oriented programming pattern requires Result types

================================================================================

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import concurrent.futures
import inspect
import json
import operator
import re
import socket
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import fields as get_dataclass_fields, is_dataclass
from datetime import datetime
from typing import ClassVar, TypeGuard, cast

import orjson

from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core._utilities.mapper import FlextUtilitiesMapper
from flext_core._utilities.validators import ValidatorSpec
from flext_core.constants import c
from flext_core.protocols import p
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.typings import t


class FlextUtilitiesValidation:
    """Unified validation patterns using railway composition.

    These helpers support dispatcher handlers and services with reusable,
    composable validators organized by domain using Protocol-based composition.

    Nested validator groups:
    - Network: URI, port, hostname validation
    - String: Pattern, length, choice validation
    - Numeric: Non-negative, positive, range validation
    - Temporal: Timestamp validation

    For data-model field validation prefer Pydantic field constraints:
    https://docs.pydantic.dev/2.12/api/fields/.
    """

    # ═══════════════════════════════════════════════════════════════════
    # NESTED PROTOCOL-BASED VALIDATOR GROUPS (Organization via Composition)
    # ═══════════════════════════════════════════════════════════════════

    class Network:
        """Network-related validation (URI, port, hostname)."""

        @staticmethod
        def validate_uri(
            uri: str | None,
            allowed_schemes: list[str] | None = None,
            context: str = "URI",
        ) -> r[str]:
            """Validate URI format."""
            return FlextUtilitiesValidation.validate_uri(uri, allowed_schemes, context)

        @staticmethod
        def validate_port_number(
            port: int | None,
            context: str = "Port",
        ) -> r[int]:
            """Validate port number (1-65535)."""
            return FlextUtilitiesValidation.validate_port_number(port, context)

        @staticmethod
        def validate_hostname(
            hostname: str,
            *,
            perform_dns_lookup: bool = True,
        ) -> r[str]:
            """Validate hostname format."""
            return FlextUtilitiesValidation.validate_hostname(
                hostname,
                perform_dns_lookup=perform_dns_lookup,
            )

    class String:
        """String-related validation (pattern, length, choice)."""

        @staticmethod
        def validate_required_string(
            value: str | None,
            context: str = "Field",
        ) -> r[str]:
            """Validate non-empty string (wraps to r)."""
            try:
                result = FlextUtilitiesValidation.validate_required_string(
                    value,
                    context,
                )
                return r[str].ok(result)
            except ValueError as e:
                return r[str].fail(str(e))

        @staticmethod
        def validate_choice(
            value: str,
            valid_choices: set[str],
            context: str = "Value",
            *,
            case_sensitive: bool = False,
        ) -> r[str]:
            """Validate value is in allowed choices."""
            return FlextUtilitiesValidation.validate_choice(
                value,
                valid_choices,
                context,
                case_sensitive=case_sensitive,
            )

        @staticmethod
        def validate_length(
            value: str,
            min_length: int | None = None,
            max_length: int | None = None,
            context: str = "Value",
        ) -> r[str]:
            """Validate string/sequence length."""
            return FlextUtilitiesValidation.validate_length(
                value,
                min_length,
                max_length,
                context,
            )

        @staticmethod
        def validate_pattern(
            value: str,
            pattern: str,
            context: str = "Value",
        ) -> r[str]:
            """Validate string matches regex pattern."""
            return FlextUtilitiesValidation.validate_pattern(value, pattern, context)

    class Numeric:
        """Numeric-related validation (positive, range, etc)."""

        @staticmethod
        def validate_non_negative(
            value: int | None,
            context: str = "Value",
        ) -> r[int]:
            """Validate value >= 0."""
            return FlextUtilitiesValidation.validate_non_negative(value, context)

        @staticmethod
        def validate_positive(
            value: int | None,
            context: str = "Value",
        ) -> r[int]:
            """Validate value > 0."""
            return FlextUtilitiesValidation.validate_positive(value, context)

        @staticmethod
        def validate_range(
            value: int,
            min_value: int | None = None,
            max_value: int | None = None,
            context: str = "Value",
        ) -> r[int]:
            """Validate numeric range."""
            return FlextUtilitiesValidation.validate_range(
                value,
                min_value,
                max_value,
                context,
            )

    # CONSOLIDATED: Use FlextUtilitiesCache for cache operations (no duplication)
    # NOTE: _normalize_component, _sort_key, _sort_dict_keys below are INTERNAL
    # recursive helpers. They are NOT duplicates of cache.py - they have different
    # logic and recursion patterns

    @staticmethod
    def _normalize_component(
        component: t.GeneralValueType,
        visited: set[int] | None = None,
    ) -> t.GeneralValueType:
        """Normalize component for consistent representation (internal recursive)."""
        # Initialize visited set if not provided (first call)
        if visited is None:
            visited = set()

        # Handle primitives and simple types without visited tracking
        if isinstance(component, (str, int, float, bool, type(None))):
            return component

        # Check for circular references using object id (only for complex types)
        component_id = id(component)
        if component_id in visited:
            # Circular reference detected - return placeholder
            return {"type": "circular_reference", "id": str(component_id)}

        # Add current component to visited set (only for complex types)
        visited.add(component_id)

        try:
            component = FlextUtilitiesValidation._handle_pydantic_model(component)
            return FlextUtilitiesValidation._normalize_by_type(component, visited)
        finally:
            # Remove from visited set when done (allow re-visiting at different depth)
            visited.discard(component_id)

    @staticmethod
    def _handle_pydantic_model(
        component: t.GeneralValueType,
    ) -> t.GeneralValueType:
        """Handle Pydantic model and dataclass conversion."""
        # Check for Pydantic model first (has model_dump method)
        model_dump_attr = getattr(component, "model_dump", None)
        if model_dump_attr is not None and callable(model_dump_attr):
            try:
                dump_result = model_dump_attr()
                if FlextUtilitiesGuards.is_type(dump_result, dict):
                    return cast("t.GeneralValueType", dump_result)
                return str(component)
            except Exception:
                return str(component)

        # Check for dataclass instance (before Sequence check to avoid treating as list)
        # GeneralValueType doesn't include type, so isinstance(component, type) is always False
        # But is_dataclass() can still be True for dataclass instances
        # Runtime check: is_dataclass() works at runtime even if type system doesn't allow type
        if is_dataclass(component):
            return FlextUtilitiesValidation._normalize_dataclass_value_instance(
                component,
            )

        # Check if already valid GeneralValueType
        return FlextUtilitiesValidation._ensure_general_value_type(component)

    @staticmethod
    def _ensure_general_value_type(
        component: t.GeneralValueType | type,
    ) -> t.GeneralValueType:
        """Ensure component is valid GeneralValueType.

        Args:
            component: Component to validate

        Returns:
            Valid GeneralValueType

        """
        if isinstance(component, (str, int, float, bool, type(None))):
            return component
        if isinstance(component, type):
            return str(component)
        if FlextUtilitiesGuards.is_type(component, "sequence_not_str_bytes"):
            return component
        if FlextUtilitiesGuards.is_type(component, "mapping"):
            return component
        return str(component)

    @staticmethod
    def _normalize_by_type(
        component: t.GeneralValueType,
        visited: set[int] | None = None,
    ) -> t.GeneralValueType:
        """Normalize component based on its type."""
        if visited is None:
            visited = set()

        # Handle primitives first to avoid Sequence matching strings
        if isinstance(component, (str, int, float, bool, type(None))):
            return component

        if FlextRuntime.is_dict_like(component):
            return FlextUtilitiesValidation._normalize_dict_like(component, visited)
        if isinstance(component, (list, tuple)) and not isinstance(
            component, (str, bytes)
        ):
            # Use _normalize_sequence which returns dict with type marker
            return FlextUtilitiesValidation._normalize_sequence(component, visited)
        # Runtime check: set is not in GeneralValueType union, but can occur at runtime
        # Type narrowing: isinstance(component, set) is always False per type system
        # But runtime check handles actual set instances
        if isinstance(component, set):
            return FlextUtilitiesValidation._normalize_set(component, visited)

        # Ensure valid GeneralValueType for primitives
        return FlextUtilitiesValidation._ensure_general_value_type(
            cast("t.GeneralValueType", component),
        )

    @staticmethod
    def _convert_items_to_dict(
        items_result: Sequence[tuple[str, t.GeneralValueType]]
        | t.Types.ConfigurationMapping,
    ) -> t.Types.ConfigurationDict:
        """Convert items() result to dict with normalization."""
        if isinstance(items_result, (list, tuple)):
            return dict(items_result)

        if FlextUtilitiesGuards.is_type(items_result, "mapping"):
            return dict(items_result)

        if not hasattr(items_result, "__iter__"):
            result_type = type(items_result)
            msg = f"items() returned non-iterable: {result_type}"
            raise TypeError(msg)

        items_iterable = cast("Iterable[tuple[str, t.GeneralValueType]]", items_result)
        items_list = list(items_iterable)
        temp_dict: t.Types.ConfigurationDict = {}
        for k, v in items_list:
            # k is already str from the cast
            # v is already GeneralValueType from the cast
            normalized_v = FlextUtilitiesValidation._normalize_component(
                v,
                visited=None,
            )
            temp_dict[k] = normalized_v
        return temp_dict

    @staticmethod
    def _extract_dict_from_component(
        component: t.Types.ConfigurationMapping | p.Foundation.HasModelDump,
        _visited: set[int] | None = None,
    ) -> t.Types.ConfigurationMapping:
        """Extract dict-like structure from component."""
        if isinstance(component, (Mapping, dict)):
            return component

        items_attr = getattr(component, "items", None)
        if items_attr is None or not callable(items_attr):
            msg = f"Cannot convert {type(component).__name__} to dict"
            raise TypeError(msg)

        items_result = items_attr()
        try:
            # Type narrowing: items_result is from dict-like object, should be iterable of tuples
            # Cast to proper type for _convert_items_to_dict
            items_typed: (
                Sequence[tuple[str, t.GeneralValueType]] | t.Types.ConfigurationMapping
            ) = cast(
                "Sequence[tuple[str, t.GeneralValueType]] | t.Types.ConfigurationMapping",
                items_result,
            )
            return FlextUtilitiesValidation._convert_items_to_dict(items_typed)
        except (TypeError, ValueError) as e:
            msg = f"Cannot convert {type(component).__name__}.items() to dict"
            raise TypeError(msg) from e

    @staticmethod
    def _convert_items_result_to_dict(
        items_result: Sequence[tuple[str, t.GeneralValueType]]
        | t.Types.ConfigurationMapping
        | Iterable[tuple[str, t.GeneralValueType]],
    ) -> t.Types.ConfigurationDict:
        """Convert items() result to dict (helper for _convert_to_mapping).

        Args:
            items_result: Result from calling items() method

        Returns:
            t.Types.ConfigurationDict: Converted dictionary

        Raises:
            TypeError: If items_result cannot be converted to dict

        """
        if isinstance(items_result, (list, tuple)):
            # items() returned list/tuple of pairs - convert to dict
            return dict(items_result)

        # items() returned something else - try to iterate
        if not hasattr(items_result, "__iter__"):
            result_type = type(items_result)
            msg = f"items() returned non-iterable: {result_type}"
            raise TypeError(msg)

        # Cast to satisfy type checker - items_result is iterable at runtime
        items_iterable = cast("Iterable[tuple[str, object]]", items_result)
        items_list = list(items_iterable)

        # Convert tuples to dict, normalizing values
        temp_dict: t.Types.ConfigurationDict = {}
        for k, v in items_list:
            if FlextUtilitiesGuards.is_type(k, str):
                # Normalize value first, then cast to GeneralValueType
                # v is object from items_list iteration, cast to GeneralValueType
                v_typed: t.GeneralValueType = cast(
                    "t.GeneralValueType",
                    v,
                )
                normalized_v = FlextUtilitiesValidation._normalize_component(
                    v_typed,
                    visited=None,
                )
                # normalized_v is already GeneralValueType from _normalize_component
                temp_dict[k] = normalized_v
        return temp_dict

    @staticmethod
    def _convert_to_mapping(
        component: t.Types.ConfigurationMapping | p.Foundation.HasModelDump,
    ) -> t.Types.ConfigurationMapping:
        """Convert object to Mapping (helper for _normalize_dict_like).

        Args:
            component: Object to convert to Mapping

        Returns:
            t.Types.ConfigurationMapping: Converted mapping

        Raises:
            TypeError: If component cannot be converted to dict

        """
        if isinstance(component, Mapping):
            return component

        if isinstance(component, dict):
            return component

        # Check if component has items() method (duck typing for dict-like objects)
        items_method = getattr(component, "items", None)
        if items_method is not None and callable(items_method):
            # Has items() method - convert to dict
            # Cast needed: items_method() returns object, but we know it's dict-like
            items_result: (
                Sequence[tuple[str, t.GeneralValueType]] | t.Types.ConfigurationMapping
            ) = cast(
                "Sequence[tuple[str, t.GeneralValueType]] | t.Types.ConfigurationMapping",
                items_method(),
            )
            try:
                return FlextUtilitiesValidation._convert_items_result_to_dict(
                    items_result,
                )
            except (TypeError, ValueError) as e:
                # Cannot convert - raise error
                msg = f"Cannot convert {type(component).__name__}.items() to dict"
                raise TypeError(msg) from e

        # Cannot convert - raise error
        msg = f"Cannot convert {type(component).__name__} to dict"
        raise TypeError(msg)

    @staticmethod
    def _normalize_dict_like(
        component: t.Types.ConfigurationMapping | p.Foundation.HasModelDump,
        visited: set[int] | None = None,
    ) -> t.Types.ConfigurationDict:
        """Normalize dict-like objects.

        Note: visited tracking is handled by _normalize_component, so we don't
        need to check/add here to avoid false circular reference detection.
        """
        if visited is None:
            visited = set()

        # Convert to Mapping for type safety
        component_dict = FlextUtilitiesValidation._convert_to_mapping(component)
        component_id = id(component_dict)

        # Normalize values in the dictionary
        normalized_dict: t.Types.ConfigurationDict = {}
        for k, v in component_dict.items():
            # Check if value is the same dict (circular reference)
            if id(v) == component_id:
                normalized_dict[str(k)] = {
                    "type": "circular_reference",
                    "id": str(component_id),
                }
            elif FlextUtilitiesGuards.is_type(
                v,
                "mapping",
            ) or FlextUtilitiesGuards.is_type(v, "sequence_not_str"):
                normalized_dict[str(k)] = FlextUtilitiesValidation._normalize_component(
                    v,
                    visited,
                )
            else:
                # Use _normalize_value for primitives
                v_normalized = FlextUtilitiesValidation._normalize_value(v)
                normalized_dict[str(k)] = v_normalized
        # Return normalized dict (empty dict if empty)
        return normalized_dict

    @staticmethod
    def _normalize_value(
        value: t.GeneralValueType,
    ) -> t.GeneralValueType:
        """Normalize a single value."""
        if isinstance(value, (str, int, float, bool, type(None))):
            # Type narrowing: these are all valid t.GeneralValueType
            return value
        if isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
            # Type narrowing: tuple is valid t.GeneralValueType
            return tuple(value)
        if isinstance(value, (dict, Mapping)):
            # Type narrowing: dict is valid t.GeneralValueType
            return dict(value) if isinstance(value, dict) else dict(value.items())
        # Fallback: convert to string (string is valid t.GeneralValueType)
        return str(value)

    @staticmethod
    def _normalize_sequence_helper(
        component: Sequence[t.GeneralValueType],
        visited: set[int] | None = None,
    ) -> list[t.GeneralValueType]:
        """Normalize sequence types (helper for internal recursion)."""
        if visited is None:
            visited = set()
        return [
            FlextUtilitiesValidation._normalize_component(item, visited)
            for item in component
        ]

    @staticmethod
    def _normalize_set_helper(
        component: set[t.GeneralValueType],
    ) -> set[t.GeneralValueType]:
        """Normalize set types (helper for internal recursion)."""
        # Normalize items and ensure they are hashable for set
        normalized_items = [
            FlextUtilitiesValidation._normalize_component(item) for item in component
        ]
        # Convert to set, ensuring hashability
        result_set: set[t.GeneralValueType] = set()
        for item in normalized_items:
            # Only add hashable items to set
            if isinstance(item, (str, int, float, bool, type(None))):
                result_set.add(item)
            elif FlextUtilitiesGuards.is_type(item, tuple):
                # Tuples are hashable if all elements are hashable
                try:
                    result_set.add(item)
                except TypeError:
                    # Non-hashable tuple - convert to string representation
                    result_set.add(str(item))
            else:
                # Non-hashable type - convert to string
                result_set.add(str(item))
        return result_set

    @staticmethod
    def _sort_key(key: str | float) -> tuple[str, str]:
        """Generate a sort key for consistent ordering."""
        key_str = str(key)
        return (key_str.casefold(), key_str)

    @staticmethod
    def _sort_dict_keys(
        data: t.GeneralValueType,
    ) -> t.GeneralValueType:
        """Sort dict keys for consistent representation (internal recursive)."""
        # Type narrowing: GeneralValueType includes Mapping[str, GeneralValueType]
        if isinstance(data, (dict, Mapping)):
            # data is Mapping[str, GeneralValueType], which is valid GeneralValueType
            data_dict: t.Types.ConfigurationMapping = (
                data if isinstance(data, dict) else dict(data.items())
            )
            return {
                str(k): FlextUtilitiesValidation._sort_dict_keys(data_dict[k])
                for k in sorted(
                    data_dict.keys(),
                    key=FlextUtilitiesValidation._sort_key,
                )
            }
        # For non-dict types, return as-is
        return data

    @staticmethod
    def validate_pipeline(
        value: str,
        validators: list[Callable[[str], r[bool]]],
    ) -> r[bool]:
        """Validate using a pipeline of validators and surface the first failure.

        Returns:
            r[bool]: ``ok(True)`` when all validators pass or a failed
            result describing the first violation.

        """
        for validator in validators:
            # Type narrowing: validators is list[Callable[[str], r[bool]]]
            # So validator is always callable per type system
            # Runtime check for safety (type: ignore needed because type system guarantees callable)
            if not callable(validator):
                # Runtime safety check (type system ensures callable)
                return r[bool].fail(
                    "Validator must be callable",
                    error_code=c.Errors.VALIDATION_ERROR,
                )
            try:
                # Execute validator - may return r[bool] or raise exception
                result = validator(value)

                # FAST FAIL: If validator returns r, check if ok(True)
                if isinstance(result, r):
                    if result.is_failure:
                        return r[bool].fail(
                            f"Validator failed: {result.error}",
                            error_code=c.Errors.VALIDATION_ERROR,
                        )
                    if result.value is not True:
                        return r[bool].fail(
                            "Validator must return r[bool].ok(True)",
                            error_code=c.Errors.VALIDATION_ERROR,
                        )
            except (
                AttributeError,
                TypeError,
                ValueError,
                RuntimeError,
                KeyError,
            ) as e:
                return r[bool].fail(
                    f"Validator failed: {e}",
                    error_code=c.Errors.VALIDATION_ERROR,
                )
        return r[bool].ok(True)

    @staticmethod
    def sort_key(value: t.GeneralValueType) -> tuple[str, str]:
        """Return a deterministic tuple for ordering normalized cache components.

        For strings, returns (casefold, original) for case-insensitive sorting.
        For other types, returns (type_category, serialized_value) for consistent sorting.
        """
        # Special handling for strings - return casefold and original
        if isinstance(value, str):
            return (value.casefold(), value)

        # Determine type category for non-strings
        if isinstance(value, (int, float)):
            type_cat = "num"
        elif FlextUtilitiesGuards.is_type(value, dict):
            type_cat = "dict"
        elif isinstance(value, (list, tuple)):
            type_cat = "seq"
        else:
            type_cat = "other"

        # Serialize value
        try:
            json_bytes = orjson.dumps(value, option=orjson.OPT_SORT_KEYS)
            serialized = json_bytes.decode(c.Utilities.DEFAULT_ENCODING)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            # Use proper logger instead of root logger
            logger = FlextRuntime.get_logger(__name__)
            logger.debug("orjson dumps failed: %s", e)
            # Use standard library json with sorted keys
            serialized = json.dumps(value, sort_keys=True, default=str)

        return (type_cat, serialized)

    @staticmethod
    def _is_dataclass_instance(
        obj: t.GeneralValueType,
    ) -> TypeGuard[t.GeneralValueType]:
        """Type guard to check if object is a dataclass instance (not class)."""
        # Check if obj is a dataclass instance
        # GeneralValueType doesn't include type, so obj is never a type
        # We only need to check if it's a dataclass
        return is_dataclass(obj)

    @staticmethod
    def _normalize_primitive_or_bytes(
        value: t.GeneralValueType,
    ) -> tuple[bool, t.GeneralValueType]:
        """Normalize primitive types and bytes.

        Returns:
            Tuple of (is_handled, normalized_value)
            - is_handled: True if value was a primitive/bytes
            - normalized_value: The normalized value (or None if not handled)

        """
        # Handle primitives (return as-is)
        if value is None or isinstance(value, (bool, int, float, str)):
            return (True, value)

        # Handle bytes (convert to dict with type marker)
        if isinstance(value, bytes):
            return (True, {"type": "bytes", "data": value.hex()})

        return (False, None)  # Not a primitive/bytes - continue dispatching

    @staticmethod
    def normalize_component(
        value: t.GeneralValueType,
    ) -> t.GeneralValueType:
        """Normalize arbitrary objects into cache-friendly deterministic structures."""
        # Handle strings specially - they should not be treated as sequences
        if FlextUtilitiesGuards.is_type(value, str):
            return value
        # Use the internal recursive method which handles cycles and returns simple structures
        return FlextUtilitiesValidation._normalize_component(value, visited=None)

    @staticmethod
    def _normalize_pydantic_value(
        value: p.Foundation.HasModelDump,
    ) -> t.GeneralValueType:
        """Normalize Pydantic model to cache-friendly structure."""
        # Fast fail: model_dump() must succeed for valid Pydantic models
        try:
            dumped: t.GeneralValueType = value.model_dump()
        except TypeError as e:
            # Fast fail: model_dump() failure indicates invalid model
            msg = (
                f"Failed to dump Pydantic value: {type(value).__name__}: "
                f"{type(e).__name__}: {e}"
            )
            raise TypeError(msg) from e
        # Use private _normalize_component to avoid infinite recursion
        normalized_dumped = FlextUtilitiesValidation._normalize_component(
            dumped,
            visited=None,
        )
        # Return as dict with type marker for cache structure
        return {"type": "pydantic", "data": normalized_dumped}

    @staticmethod
    def _normalize_dataclass_value_instance(
        value: t.GeneralValueType,
    ) -> t.GeneralValueType:
        """Normalize dataclass instance to cache-friendly structure.

        Note: This should only be called after checking is_dataclass(value) and
        ensuring it's not a type (via isinstance(value, type) check).
        """
        # Caller guarantees value is a dataclass instance via
        # _is_dataclass_instance check. Using manual field extraction
        field_dict: t.Types.ConfigurationDict = {}
        # value.__class__ is type for dataclass instances
        value_class: type = value.__class__
        for field in get_dataclass_fields(value_class):
            field_dict[field.name] = getattr(value, field.name)
        sorted_data = FlextUtilitiesValidation._sort_dict_keys(field_dict)
        # Return as dict with type marker for cache structure
        return {"type": "dataclass", "data": sorted_data}

    @staticmethod
    def _normalize_mapping(
        value: t.Types.ConfigurationMapping,
        visited: set[int] | None = None,
    ) -> t.GeneralValueType:
        """Normalize mapping to cache-friendly structure."""
        if visited is None:
            visited = set()
        sorted_items = sorted(
            value.items(),
            key=lambda x: FlextUtilitiesValidation._sort_key(x[0]),
        )
        return {
            str(k): FlextUtilitiesValidation._normalize_component(v, visited)
            for k, v in sorted_items
        }

    @staticmethod
    def _normalize_sequence(
        value: Sequence[t.GeneralValueType],
        visited: set[int] | None = None,
    ) -> t.GeneralValueType:
        """Normalize sequence to cache-friendly structure."""
        if visited is None:
            visited = set()
        sequence_items = [
            FlextUtilitiesValidation._normalize_component(item, visited)
            for item in value
        ]
        # Return as dict with type marker for cache structure
        return {"type": "sequence", "data": sequence_items}

    @staticmethod
    def _normalize_set(
        value: set[t.GeneralValueType],
        visited: set[int] | None = None,
    ) -> t.GeneralValueType:
        """Normalize set to cache-friendly structure."""
        if visited is None:
            visited = set()
        set_items = [
            FlextUtilitiesValidation._normalize_component(item, visited)
            for item in value
        ]
        set_items.sort(key=str)
        # Return as dict with type marker for cache structure
        return {"type": "set", "data": set_items}

    @staticmethod
    def _normalize_vars(
        value: t.GeneralValueType,
    ) -> t.GeneralValueType:
        """Normalize object attributes to cache-friendly structure."""
        try:
            vars_result = vars(value)
            # vars() returns t.Types.ConfigurationDict which we normalize to GeneralValueType
            # Type narrowing: vars_result is always a dict
            value_vars_dict: t.Types.ConfigurationDict = cast(
                "t.Types.ConfigurationDict",
                vars_result,
            )
            # Process vars_result - normalize all values
            normalized_vars = {
                str(key): FlextUtilitiesValidation._normalize_component(
                    val,
                    visited=None,
                )
                for key, val in sorted(
                    value_vars_dict.items(),
                    key=operator.itemgetter(0),
                )
            }
            return {"type": "vars", "data": normalized_vars}
        except TypeError:
            # vars() failed - return repr representation
            return {"type": "repr", "data": repr(value)}

    @staticmethod
    def _generate_key_from_data(
        command_type: type[t.GeneralValueType],
        sorted_data: t.GeneralValueType,
    ) -> str:
        """Generate cache key from sorted data."""
        return f"{command_type.__name__}_{hash(str(sorted_data))}"

    @staticmethod
    def _generate_key_pydantic(
        command: p.Foundation.HasModelDump,
        command_type: type[t.GeneralValueType],
    ) -> str | None:
        """Generate cache key from Pydantic model."""
        try:
            data = command.model_dump()
            sorted_data = FlextUtilitiesValidation._sort_dict_keys(data)
            return FlextUtilitiesValidation._generate_key_from_data(
                command_type,
                sorted_data,
            )
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            return None

    @staticmethod
    def _generate_key_dataclass(
        command: t.GeneralValueType,
        command_type: type[t.GeneralValueType],
    ) -> str | None:
        """Generate cache key from dataclass."""
        try:
            dataclass_data: t.Types.ConfigurationDict = {}
            # command.__class__ is type for class instances
            command_class: type = command.__class__
            for field in get_dataclass_fields(command_class):
                dataclass_data[field.name] = getattr(command, field.name)
            sorted_data = FlextUtilitiesValidation._sort_dict_keys(dataclass_data)
            return FlextUtilitiesValidation._generate_key_from_data(
                command_type,
                sorted_data,
            )
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            return None

    @staticmethod
    def _generate_key_dict(
        command: t.Types.ConfigurationMapping,
        command_type: type[t.GeneralValueType],
    ) -> str | None:
        """Generate cache key from dict."""
        try:
            sorted_data = FlextUtilitiesValidation._sort_dict_keys(command)
            return FlextUtilitiesValidation._generate_key_from_data(
                command_type,
                sorted_data,
            )
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            return None

    @staticmethod
    def generate_cache_key(
        command: t.GeneralValueType | None,
        command_type: type[t.GeneralValueType],
    ) -> str:
        """Generate a deterministic cache key for the command.

        Args:
            command: The command/query object
            command_type: The type of the command

        Returns:
            str: Deterministic cache key

        """
        # Try Pydantic model
        if isinstance(command, p.Foundation.HasModelDump):
            key = FlextUtilitiesValidation._generate_key_pydantic(command, command_type)
            if key is not None:
                return key

        # Try dataclass
        # GeneralValueType doesn't include type, so isinstance(command, type) is always False
        # But we check hasattr and is_dataclass to ensure it's a dataclass instance
        # Runtime check: hasattr ensures it's not None, is_dataclass checks type
        # Type narrowing: GeneralValueType doesn't include dataclass types, but runtime check handles instances
        if (
            command is not None
            and hasattr(command, "__dataclass_fields__")
            and is_dataclass(command)
        ):
            key = FlextUtilitiesValidation._generate_key_dataclass(
                command,
                command_type,
            )
            if key is not None:
                return key

        # Try dict
        if FlextRuntime.is_dict_like(command) and FlextUtilitiesGuards.is_type(
            command,
            "mapping",
        ):
            key = FlextUtilitiesValidation._generate_key_dict(command, command_type)
            if key is not None:
                return key

        # Last resort: string representation with hash
        command_str = "None" if command is None else str(command)
        try:
            return f"{command_type.__name__}_{hash(command_str)}"
        except TypeError:
            # If hash fails, use deterministic encoding-based hash
            encoded = command_str.encode(c.Utilities.DEFAULT_ENCODING)
            return f"{command_type.__name__}_{abs(hash(encoded))}"

    @staticmethod
    def sort_dict_keys(
        obj: t.GeneralValueType,
    ) -> t.GeneralValueType:
        """Recursively sort dictionary keys for deterministic ordering.

        Args:
            obj: Object to sort (object, list, or other)

        Returns:
            Object with sorted keys

        """
        # Type narrowing: obj can be Mapping (which is GeneralValueType)
        if isinstance(obj, (dict, Mapping)):
            # obj is Mapping[str, GeneralValueType]
            dict_obj: t.Types.ConfigurationMapping = (
                obj if isinstance(obj, dict) else dict(obj.items())
            )
            # Convert items() view to list for sorting
            items_list: list[tuple[str, t.GeneralValueType]] = list(
                dict_obj.items(),
            )
            sorted_items: list[tuple[str, t.GeneralValueType]] = sorted(
                items_list,
                key=lambda x: str(x[0]),
            )
            return {
                str(k): FlextUtilitiesValidation._sort_dict_keys(v)
                for k, v in sorted_items
            }
        # Type narrowing: obj can be Sequence (which is GeneralValueType)
        # Handle tuple first (tuple is a Sequence but needs special handling)
        if isinstance(obj, tuple):
            # obj is confirmed to be tuple, iterate directly
            return tuple(FlextUtilitiesValidation._sort_dict_keys(item) for item in obj)
        # Handle other Sequences (but not str, bytes, or tuple)
        if isinstance(obj, (list, tuple)) and not isinstance(obj, (str, bytes)):
            # obj is Sequence[GeneralValueType] - use directly
            obj_list: Sequence[t.GeneralValueType] = obj
            return [
                FlextUtilitiesValidation._sort_dict_keys(
                    item
                    if isinstance(
                        item,
                        (str, int, float, bool, type(None), Sequence, Mapping),
                    )
                    else str(item),
                )
                for item in obj_list
            ]
        return obj

    @staticmethod
    def initialize(obj: t.GeneralValueType, field_name: str) -> None:
        """Initialize validation for object.

        Simplified implementation that directly sets the validation flag.

        Args:
            obj: Object to set validation on (must support attribute assignment)
            field_name: Name of the field to set validation flag

        Note:
            The object must support attribute assignment. If setattr() fails,
            it indicates a programming error (e.g., using a frozen dataclass,
            or an object with __slots__ that doesn't include the field).

        """
        setattr(obj, field_name, True)

    @staticmethod
    def validate_required_string(
        value: str | None,
        context: str = "Field",
    ) -> str:
        """Validate that a string is not None, empty, or whitespace-only.

        This is the most commonly repeated validation pattern across flext-ldap,
        flext-ldif, flext-meltano, and other projects. Consolidation eliminates
        300+ LOC of duplication.

        Args:
            value: The string to validate (may be None or contain whitespace)
            context: Context for error message (e.g., "Password", "DN", "Username")

        Returns:
            str: Stripped value

        Raises:
            ValueError: If value is None, empty, or whitespace-only

        """
        if value is None or not value.strip():
            msg = f"{context} cannot be empty"
            raise ValueError(msg)
        return value.strip()

    @staticmethod
    def validate_choice(
        value: str,
        valid_choices: set[str],
        context: str = "Value",
        *,
        case_sensitive: bool = False,
    ) -> r[str]:
        """Validate value is in set of valid choices (enum validation).

        Common pattern in flext-ldap (scope, operation), flext-meltano (plugin type),
        and other projects. Consolidation provides consistent error messages.

        Args:
            value: The value to validate against choices
            valid_choices: Set of valid string choices
            context: Context for error message (e.g., "LDAP scope", "Operation")
            case_sensitive: Whether to perform case-sensitive comparison

        Returns:
            r[str]: Success with value (original case), or failure

        """
        # Prepare values for comparison
        check_value = value if case_sensitive else value.lower()
        check_choices = (
            valid_choices if case_sensitive else {c.lower() for c in valid_choices}
        )

        # Validate
        if check_value not in check_choices:
            choices_str = ", ".join(sorted(valid_choices))
            return r[str].fail(
                f"Invalid {context}: {value}. Must be one of {choices_str}",
            )

        return r[str].ok(value)

    @staticmethod
    def validate_length(
        value: str,
        min_length: int | None = None,
        max_length: int | None = None,
        context: str = "Value",
    ) -> r[str]:
        """Validate string length within bounds.

        This pattern is repeated 6+ times across flext-ldap (passwords, DN, etc),
        flext-ldif, flext-meltano, flext-target-ldif, and algar-oud-mig.
        Consolidation ensures consistent boundary checking.

        Args:
            value: The string to validate
            min_length: Minimum allowed length (inclusive), or None for no minimum
            max_length: Maximum allowed length (inclusive), or None for no maximum
            context: Context for error message (e.g., "Password", "DN component")

        Returns:
            r[str]: Success with value, or failure with clear bounds

        """
        if min_length is not None and len(value) < min_length:
            return r[str].fail(
                f"{context} must be at least {min_length} characters",
            )
        if max_length is not None and len(value) > max_length:
            return r[str].fail(
                f"{context} must be no more than {max_length} characters",
            )
        return r[str].ok(value)

    @staticmethod
    def validate_pattern(
        value: str,
        pattern: str,
        context: str = "Value",
    ) -> r[str]:
        r"""Validate value matches regex pattern.

        **PYDANTIC 2 ALTERNATIVE**: For new code, use Pydantic's Field(pattern=...)
        which provides compile-time validation and better error messages:

        ```python
        from pydantic import BaseModel, Field
        from typing import Annotated

        # Define reusable type
        DnString = Annotated[str, Field(pattern=c.Platform.PATTERN_DN_STRING)]


        class LdapEntry(BaseModel):
            dn: DnString  # Automatic pattern validation
            cn: Annotated[str, Field(pattern=c.Platform.PATTERN_SIMPLE_IDENTIFIER)]
        ```

        This pattern is repeated 5+ times across flext-ldap (DN, filter, attribute),
        flext-ldif (RFC compliance), flext-target-ldif, and others.
        Consolidation centralizes pattern definitions.

        Args:
            value: The string to validate
            pattern: Regex pattern (as string, will be compiled internally)
            context: Context for error message (e.g., "DN", "Attribute name")

        Returns:
            r[str]: Success with value, or failure with pattern context

        """
        try:
            compiled_pattern = re.compile(pattern)
        except re.PatternError as e:
            return r[str].fail(f"{context} pattern is invalid: {e}")
        if not compiled_pattern.match(value):
            return r[str].fail(f"{context} format is invalid: {value}")
        return r[str].ok(value)

    @staticmethod
    def validate_uri(
        uri: str | None,
        allowed_schemes: list[str] | None = None,
        context: str = "URI",
    ) -> r[str]:
        """Validate URI format and optionally check scheme.

        **PYDANTIC 2 ALTERNATIVE**: For new code, consider using Pydantic's
        built-in URL validation which is more robust and standards-compliant:

        ```python
        from pydantic import BaseModel, AnyUrl, field_validator


        class Config(BaseModel):
            server_uri: AnyUrl

            @field_validator("server_uri")
            @classmethod
            def check_scheme(cls, v: AnyUrl) -> AnyUrl:
                if v.scheme not in ["ldap", "ldaps"]:
                    raise ValueError("Must be ldap:// or ldaps://")
                return v
        ```

        Common in flext-ldap (server_uri validation), flext-meltano, and other
        projects that need to validate connection strings.

        Args:
            uri: The URI string to validate (may be None)
            allowed_schemes: List of allowed URI schemes (e.g., ["ldap", "ldaps"])
                           If None, any scheme is allowed
            context: Context for error message (e.g., "LDAP server URI")

        Returns:
            r[str]: Success with stripped URI, or failure

        """
        # Validate non-empty
        if not uri or not uri.strip():
            return r[str].fail(f"{context} cannot be empty")

        uri_stripped = uri.strip()

        # Basic URI format validation using regex
        # RFC 3986 compliant URI pattern (simplified but stricter)
        uri_pattern = re.compile(
            r"^([a-zA-Z][a-zA-Z0-9+.-]*):"  # scheme
            r"//([^/?#]+)"  # authority (required for validation)
            r"([^?#]*)"  # path
            r"(?:\?([^#]*))?"  # query (optional)
            r"(?:#(.*))?$",  # fragment (optional)
        )

        if not uri_pattern.match(uri_stripped):
            return r[str].fail(f"{context} is not a valid URI format")

        # Validate scheme if specified
        if allowed_schemes and not any(
            uri_stripped.startswith(f"{scheme}://") for scheme in allowed_schemes
        ):
            schemes_str = ", ".join(allowed_schemes)
            return r[str].fail(
                f"{context} must start with one of {schemes_str}",
            )

        return r[str].ok(uri_stripped)

    @staticmethod
    def validate_port_number(
        port: int | None,
        context: str = "Port",
    ) -> r[int]:
        """Validate port number is in valid range (1-65535).

        **PYDANTIC 2 ALTERNATIVE**: Use t.PortNumber which already
        provides validated port numbers:

        ```python
        from pydantic import BaseModel
        from flext_core import t


        class ServerConfig(BaseModel):
            port: t.PortNumber  # Automatic 1-65535 validation
            # Or with Field:
            # port: Annotated[int, Field(ge=1, le=65535)]
        ```

        Common pattern in flext-ldap, flext-meltano, and other projects that
        manage server connections.

        Args:
            port: The port number to validate (may be None)
            context: Context for error message (e.g., "LDAP port")

        Returns:
            r[int]: Success with port number, or failure

        """
        if port is None:
            return r[int].fail(f"{context} cannot be None")

        if not (c.Network.MIN_PORT <= port <= c.Network.MAX_PORT):
            error_msg = (
                f"{context} must be between {c.Network.MIN_PORT} and "
                f"{c.Network.MAX_PORT}, got {port}"
            )
            return r[int].fail(error_msg)

        return r[int].ok(port)

    @staticmethod
    def _validate_numeric_constraint[T: (int, float)](
        value: T | None,
        predicate: Callable[[T], bool],
        error_msg: str,
        context: str = "Value",
    ) -> r[T]:
        """Generic numeric validation with predicate (DRY consolidation).

        **INTERNAL METHOD**: This is a private implementation detail used
        by public validation methods. Do not call directly - use the
        specific public methods instead (validate_positive,
        validate_non_negative, etc.).

        Consolidates validate_positive, validate_non_negative, and similar
        methods following Python 3.13 generic type parameter patterns.

        Args:
            value: The numeric value to validate (may be None)
            predicate: Validation function (e.g., lambda v: v > 0)
            error_msg: Error message describing the constraint
            context: Context for error message

        Returns:
            r[T]: Success with value, or failure

        Example:
            >>> _validate_numeric_constraint(
            ...     5, lambda v: v > 0, "must be positive", "Count"
            ... )
            r[int].ok(5)

        """
        if value is None:
            return r[T].fail(f"{context} cannot be None")

        if not predicate(value):
            return r[T].fail(f"{context} {error_msg}, got {value}")

        return r[T].ok(value)

    @staticmethod
    def validate_non_negative(
        value: int | None,
        context: str = "Value",
    ) -> r[int]:
        """Validate integer is non-negative (>= 0).

        Common pattern for timeout_seconds, retry_count, size_limit, and other
        configuration values that must be non-negative.

        Args:
            value: The integer to validate (may be None)
            context: Context for error message (e.g., "Timeout seconds")

        Returns:
            r[int]: Success with value, or failure

        """
        return FlextUtilitiesValidation._validate_numeric_constraint(
            value,
            predicate=lambda v: v >= 0,
            error_msg="must be non-negative",
            context=context,
        )

    @staticmethod
    def validate_positive(
        value: int | None,
        context: str = "Value",
    ) -> r[int]:
        """Validate integer is positive (> 0).

        Useful for retry_count, max_retries, and other values requiring at least 1.

        Args:
            value: The integer to validate (may be None)
            context: Context for error message (e.g., "Max retries")

        Returns:
            r[int]: Success with value, or failure

        """
        return FlextUtilitiesValidation._validate_numeric_constraint(
            value,
            predicate=lambda v: v > 0,
            error_msg="must be positive",
            context=context,
        )

    @staticmethod
    def validate_range(
        value: int,
        min_value: int | None = None,
        max_value: int | None = None,
        context: str = "Value",
    ) -> r[int]:
        """Validate integer is within specified range.

        General-purpose range validation for any integer field.

        Args:
            value: The integer to validate
            min_value: Minimum allowed value (inclusive), or None for no minimum
            max_value: Maximum allowed value (inclusive), or None for no maximum
            context: Context for error message

        Returns:
            r[int]: Success with value, or failure

        """
        if min_value is not None and value < min_value:
            return r[int].fail(
                f"{context} must be at least {min_value}, got {value}",
            )
        if max_value is not None and value > max_value:
            return r[int].fail(
                f"{context} must be at most {max_value}, got {value}",
            )
        return r[int].ok(value)

    @staticmethod
    def validate_callable(
        value: t.GeneralValueType,
        error_message: str = "Value must be callable",
        error_code: str = c.Errors.TYPE_ERROR,
    ) -> r[bool]:
        """Validate that value is callable (generic helper for field validators).

        This generic helper consolidates duplicate callable validation logic
        across multiple Pydantic models (service.py, config.py, handler.py).

        Args:
            value: Value to validate (should be callable)
            error_message: Custom error message (default: "Value must be callable")
            error_code: Error code for validation failure

        Returns:
            r[bool]: Success with True if callable, failure otherwise

        Example:
            >>> from flext_core.utilities import u
            >>> result = u.Validation.validate_callable(lambda x: x + 1)
            >>> result.is_success
            True
            >>> result = u.Validation.validate_callable("not callable")
            >>> result.is_failure
            True

        """
        # Runtime check for callable (type system may not guarantee it)
        if not callable(value):
            return r[bool].fail(
                error_message,
                error_code=error_code,
            )
        # Return True for valid callable (not the callable itself)
        # Type narrowing: value is GeneralValueType, but runtime check ensures callable
        return r[bool].ok(True)

    @staticmethod
    def validate_timeout(
        timeout: float,
        max_timeout: float,
        error_message: str | None = None,
        error_code: str = c.Errors.VALIDATION_ERROR,
    ) -> r[float | int]:
        """Validate that timeout does not exceed maximum (generic helper).

        This generic helper consolidates duplicate timeout validation logic
        across multiple Pydantic models.

        Args:
            timeout: Timeout value to validate (in seconds)
            max_timeout: Maximum allowed timeout (in seconds)
            error_message: Custom error message (optional)
            error_code: Error code for validation failure

        Returns:
            r: Success with timeout if valid, failure if exceeds max or negative

        Example:
            >>> from flext_core.utilities import u
            >>> result = u.Validation.validate_timeout(5.0, 300.0)
            >>> result.is_success
            True
            >>> result = u.Validation.validate_timeout(500.0, 300.0)
            >>> result.is_failure
            True
            >>> result = u.Validation.validate_timeout(-1.0, 300.0)
            >>> result.is_failure
            True

        """
        # Fast fail: validate negative values
        if timeout < 0:
            msg_negative = (
                error_message or f"Timeout must be non-negative, got {timeout}"
            )
            return r[float | int].fail(msg_negative, error_code=error_code)

        # Validate maximum
        if timeout > max_timeout:
            msg_max: str = (
                error_message
                if error_message is not None
                else f"Timeout cannot exceed {max_timeout} seconds"
            )
            return r[float | int].fail(msg_max, error_code=error_code)
        return r[float | int].ok(timeout)

    @staticmethod
    def validate_http_status_codes(
        codes: list[object],
        min_code: int = c.Network.HTTP_STATUS_MIN,
        max_code: int = c.Network.HTTP_STATUS_MAX,
    ) -> r[list[int]]:
        """Validate and normalize HTTP status codes (generic helper).

        This generic helper consolidates duplicate HTTP status code validation
        logic across multiple Pydantic models (config.py).

        Args:
            codes: List of status codes (int or str) to validate
            min_code: Minimum valid HTTP status code (default: 100)
            max_code: Maximum valid HTTP status code (default: 599)

        Returns:
            r[list[int]]: Success with normalized int codes, failure otherwise

        Example:
            >>> from flext_core.utilities import u
            >>> result = u.Validation.validate_http_status_codes([
            ...     200,
            ...     "404",
            ...     500,
            ... ])
            >>> result.is_success and result.value == [200, 404, 500]
            True
            >>> result = u.Validation.validate_http_status_codes([999])
            >>> result.is_failure
            True

        """
        validated_codes: list[int] = []
        for code in codes:
            try:
                # Convert to int (handles both int and str)
                if isinstance(code, (int, str)):
                    code_int = int(str(code))
                    # Validate range
                    if not min_code <= code_int <= max_code:
                        error_msg = (
                            f"Invalid HTTP status code: {code} "
                            f"(must be {min_code}-{max_code})"
                        )
                        return r[list[int]].fail(
                            error_msg,
                            error_code=c.Errors.VALIDATION_ERROR,
                        )
                    validated_codes.append(code_int)
                else:
                    return r[list[int]].fail(
                        f"Invalid HTTP status code type: {type(code).__name__}",
                        error_code=c.Errors.TYPE_ERROR,
                    )
            except (ValueError, TypeError) as e:
                return r[list[int]].fail(
                    f"Invalid HTTP status code: {code} ({e})",
                    error_code=c.Errors.VALIDATION_ERROR,
                )

        return r[list[int]].ok(validated_codes)

    @staticmethod
    def validate_iso8601_timestamp(
        timestamp: str,
        *,
        allow_empty: bool = True,
    ) -> r[str]:
        """Validate ISO 8601 timestamp format (generic helper).

        This generic helper consolidates duplicate ISO 8601 timestamp validation
        logic across multiple Pydantic models (handler.py).

        Args:
            timestamp: Timestamp string to validate (ISO 8601 format)
            allow_empty: If True, allow empty strings (default: True)

        Returns:
            r[str]: Success with normalized timestamp, failure otherwise

        Example:
            >>> from flext_core.utilities import u
            >>> result = u.Validation.validate_iso8601_timestamp("2025-01-01T00:00:00Z")
            >>> result.is_success
            True
            >>> result = u.Validation.validate_iso8601_timestamp("invalid")
            >>> result.is_failure
            True
            >>> result = u.Validation.validate_iso8601_timestamp("", allow_empty=True)
            >>> result.is_success
            True

        """
        # Allow empty strings if configured
        if allow_empty and (not timestamp or not timestamp.strip()):
            return r[str].ok(timestamp)

        try:
            # Handle both Z suffix and explicit timezone offset
            normalized = (
                timestamp.replace("Z", "+00:00")
                if timestamp.endswith("Z")
                else timestamp
            )
            datetime.fromisoformat(normalized)
            return r[str].ok(timestamp)
        except (ValueError, TypeError) as e:
            return r[str].fail(
                f"Timestamp must be in ISO 8601 format: {e}",
                error_code=c.Errors.VALIDATION_ERROR,
            )

    @staticmethod
    def validate_hostname(
        hostname: str,
        *,
        perform_dns_lookup: bool = True,
    ) -> r[str]:
        """Validate hostname format and optionally perform DNS resolution.

        This generic helper consolidates hostname validation logic from typings.py
        and provides flexible validation with optional DNS lookup.

        Args:
            hostname: Hostname string to validate
            perform_dns_lookup: If True, perform DNS lookup to verify hostname
                resolution (default: True)

        Returns:
            r[str]: Success with hostname if valid, failure otherwise

        Example:
            >>> from flext_core.utilities import u
            >>> result = u.Validation.validate_hostname("localhost")
            >>> result.is_success
            True
            >>> result = u.Validation.validate_hostname("invalid..hostname")
            >>> result.is_failure
            True
            >>> # Skip DNS lookup for performance
            >>> result = u.Validation.validate_hostname(
            ...     "example.com", perform_dns_lookup=False
            ... )
            >>> result.is_success
            True

        """
        # Basic hostname validation (empty check)
        if not hostname or not hostname.strip():
            return r[str].fail(
                "Hostname cannot be empty",
                error_code=c.Errors.VALIDATION_ERROR,
            )

        normalized_hostname = hostname.strip()

        # Validate hostname format (RFC 1035: basic pattern)
        hostname_pattern = re.compile(
            r"^(?!-)(?!.*--)(?!.*\.$)(?!.*\.\.)[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?)*$",
        )
        if not hostname_pattern.match(normalized_hostname):
            return r[str].fail(
                f"Invalid hostname format: '{normalized_hostname}'",
                error_code=c.Errors.VALIDATION_ERROR,
            )

        # Validate hostname length (RFC 1035: max 253 characters)
        if len(normalized_hostname) > c.Network.MAX_HOSTNAME_LENGTH:
            error_msg = (
                f"Hostname '{normalized_hostname}' exceeds maximum length of "
                f"{c.Network.MAX_HOSTNAME_LENGTH} characters"
            )
            return r[str].fail(
                error_msg,
                error_code=c.Errors.VALIDATION_ERROR,
            )

        # Perform DNS lookup if requested
        if perform_dns_lookup:
            try:
                socket.gethostbyname(normalized_hostname)
            except socket.gaierror as e:
                return r[str].fail(
                    f"Cannot resolve hostname '{normalized_hostname}': {e}",
                    error_code=c.Errors.VALIDATION_ERROR,
                )
            except (OSError, ValueError) as e:
                return r[str].fail(
                    f"Invalid hostname '{normalized_hostname}': {e}",
                    error_code=c.Errors.VALIDATION_ERROR,
                )

        return r[str].ok(normalized_hostname)

    @staticmethod
    def validate_identifier(
        name: str,
        *,
        pattern: str = r"^[a-zA-Z_][a-zA-Z0-9_: ]*$",
        allow_empty: bool = False,
        strip: bool = True,
        error_message: str | None = None,
    ) -> r[str]:
        """Validate and normalize identifier/name with customizable pattern.

        This generic helper consolidates identifier validation logic from
        container.py (_validate_service_name) and provides flexible validation
        for names, identifiers, service names, etc.

        Default pattern follows Python identifier rules: start with letter/underscore,
        followed by letters, digits, underscores, colons, and spaces
        - Useful for service names with namespacing (e.g., "logger:module_name")
        - Useful for handler names with spaces (e.g., "Pre-built Handler")
        - Cannot start with digit (like Python identifiers)
        - Does not allow hyphens (unlike some systems)

        Args:
            name: Identifier/name string to validate
            pattern: Regex pattern for validation (default: alphanumeric + _: + space)
            allow_empty: If True, allow empty strings (default: False)
            strip: If True, strip whitespace before validation (default: True)
            error_message: Custom error message (optional)

        Returns:
            r[str]: Success with normalized name or failure with error

        Example:
            >>> from flext_core.utilities import u
            >>> # Service name validation
            >>> result = u.Validation.validate_identifier("logger:app")
            >>> result.is_success
            True
            >>> # Custom pattern (only alphanumeric + underscore)
            >>> result = u.Validation.validate_identifier(
            ...     "my_service", pattern=r"^[a-zA-Z0-9_]+$"
            ... )
            >>> result.is_success
            True
            >>> # Invalid characters
            >>> result = u.Validation.validate_identifier("invalid@name")
            >>> result.is_failure
            True

        """
        # Check empty string
        if not name or (not allow_empty and not name.strip()):
            return r[str].fail(
                error_message or "Identifier cannot be empty",
                error_code=c.Errors.VALIDATION_ERROR,
            )

        # Normalize (strip if requested)
        normalized_name = name.strip() if strip else name

        # Validate pattern
        if not re.match(pattern, normalized_name):
            msg: str = (
                error_message
                if error_message is not None
                else (
                    f"Identifier '{normalized_name}' contains invalid characters. "
                    f"Must match pattern: {pattern}"
                )
            )
            return r[str].fail(
                msg,
                error_code=c.Errors.VALIDATION_ERROR,
            )

        return r[str].ok(normalized_name)

    @staticmethod
    def _validate_max_attempts(
        retry_config: t.Types.ConfigurationDict | None,
    ) -> r[int]:
        """Validate max_attempts parameter."""
        if retry_config is None:
            retry_config = {}
        max_attempts_raw = retry_config.get("max_attempts", 1)
        if isinstance(max_attempts_raw, (int, str)):
            max_attempts = int(max_attempts_raw)
        else:
            max_attempts = 1
        if max_attempts < 1:
            return r[int].fail(
                "max_attempts must be >= 1",
                error_code=c.Errors.VALIDATION_ERROR,
            )
        return r[int].ok(max_attempts)

    @staticmethod
    def _validate_initial_delay(
        retry_config: t.Types.ConfigurationDict | None,
    ) -> r[float]:
        """Validate initial_delay_seconds parameter."""
        if retry_config is None:
            retry_config = {}
        initial_delay_raw = retry_config.get("initial_delay_seconds", 0.1)
        if isinstance(initial_delay_raw, (int, float, str)):
            initial_delay = float(initial_delay_raw)
        else:
            initial_delay = 0.1
        if initial_delay <= 0:
            return r[float].fail(
                "initial_delay_seconds must be > 0",
                error_code=c.Errors.VALIDATION_ERROR,
            )
        return r[float].ok(initial_delay)

    @staticmethod
    def _validate_max_delay(
        retry_config: t.Types.ConfigurationDict | None,
    ) -> r[float]:
        """Validate max_delay_seconds parameter."""
        if retry_config is None:
            retry_config = {}
        max_delay_raw = retry_config.get("max_delay_seconds", 60.0)
        if isinstance(max_delay_raw, (int, float, str)):
            max_delay = float(max_delay_raw)
        else:
            max_delay = 60.0
        if max_delay <= 0:
            return r[float].fail(
                "max_delay_seconds must be > 0",
                error_code=c.Errors.VALIDATION_ERROR,
            )
        return r[float].ok(max_delay)

    @staticmethod
    def _validate_backoff_multiplier(
        retry_config: t.Types.ConfigurationDict | None,
    ) -> r[float]:
        """Validate backoff_multiplier parameter."""
        if retry_config is None:
            return r[float].ok(
                c.DEFAULT_BACKOFF_MULTIPLIER,
            )
        backoff_multiplier = retry_config.get("backoff_multiplier")
        if backoff_multiplier is not None and isinstance(
            backoff_multiplier,
            (int, float, str),
        ):
            backoff_mult = float(backoff_multiplier)
            if backoff_mult < 1.0:
                return r[float].fail(
                    "backoff_multiplier must be >= 1.0",
                    error_code=c.Errors.VALIDATION_ERROR,
                )
            return r[float].ok(backoff_mult)
        return r[float].ok(
            c.DEFAULT_BACKOFF_MULTIPLIER,
        )

    @staticmethod
    def create_retry_config(
        retry_config: t.Types.ConfigurationDict | None,
    ) -> r[t.Config.RetryConfig]:
        """Create and validate retry configuration using railway pattern.

        Args:
            retry_config: Raw retry configuration dictionary

        Returns:
            r[t.Config.RetryConfig]: Validated retry configuration or error

        """
        if retry_config is None:
            retry_config = {}
        try:
            # Validate each parameter using railway pattern (DRY consolidation)
            result = FlextUtilitiesValidation._validate_max_attempts(retry_config)
            if result.is_failure:
                return r[t.Config.RetryConfig].fail(
                    result.error or "Max attempts validation failed",
                )

            max_attempts = result.value

            delay_result = FlextUtilitiesValidation._validate_initial_delay(
                retry_config,
            )
            if delay_result.is_failure:
                return r[t.Config.RetryConfig].fail(
                    delay_result.error or "Initial delay validation failed",
                )

            initial_delay = delay_result.value
            params_2 = (max_attempts, initial_delay)

            max_delay_result = FlextUtilitiesValidation._validate_max_delay(
                retry_config,
            )
            if max_delay_result.is_failure:
                return r[t.Config.RetryConfig].fail(
                    max_delay_result.error or "Max delay validation failed",
                )

            max_delay = max_delay_result.value
            params_3 = (*params_2, max_delay)

            backoff_result = FlextUtilitiesValidation._validate_backoff_multiplier(
                retry_config,
            )
            if backoff_result.is_failure:
                return r[t.Config.RetryConfig].fail(
                    backoff_result.error or "Backoff multiplier validation failed",
                )

            backoff_mult = backoff_result.value
            params_4 = (*params_3, backoff_mult)

            return r[t.Config.RetryConfig].ok(
                t.Config.RetryConfig(
                    max_attempts=params_4[0],
                    initial_delay_seconds=params_4[1],
                    max_delay_seconds=params_4[2],
                    exponential_backoff=bool(
                        retry_config.get("exponential_backoff"),
                    ),
                    retry_on_exceptions=(
                        cast(
                            "list[type[Exception]]",
                            [
                                exc_type
                                for exc_type in cast(
                                    "Sequence[t.GeneralValueType]",
                                    retry_config["retry_on_exceptions"],
                                )
                                if isinstance(exc_type, type)
                                and issubclass(exc_type, Exception)
                            ],
                        )
                        if "retry_on_exceptions" in retry_config
                        and retry_config["retry_on_exceptions"] is not None
                        and isinstance(
                            retry_config["retry_on_exceptions"],
                            (list, tuple),
                        )
                        else [Exception]
                    ),
                    backoff_multiplier=params_4[3],
                ),
            )

        except (ValueError, TypeError) as e:
            return r[t.Config.RetryConfig].fail(
                f"Invalid retry configuration: {e}",
                error_code=c.Errors.VALIDATION_ERROR,
            )

    @staticmethod
    def is_exception_retryable(
        exception: Exception,
        retry_on_exceptions: list[type[Exception]],
    ) -> bool:
        """Check if exception should trigger retry.

        Args:
            exception: Exception to check
            retry_on_exceptions: List of exception types that should trigger retry

        Returns:
            bool: True if exception should trigger retry

        """
        return any(isinstance(exception, exc_type) for exc_type in retry_on_exceptions)

    @staticmethod
    def format_error_message(
        exception: Exception,
        timeout_seconds: float | None = None,
    ) -> str:
        """Format error message with timeout context if applicable.

        Args:
            exception: Exception that occurred
            timeout_seconds: Timeout duration if applicable

        Returns:
            str: Formatted error message

        """
        error_msg = str(exception) or type(exception).__name__

        if (
            isinstance(exception, (TimeoutError, concurrent.futures.TimeoutError))
            and timeout_seconds
        ):
            error_msg = f"Operation timed out after {timeout_seconds} seconds"

        return error_msg

    @staticmethod
    def validate_batch_services(
        services: t.Types.ConfigurationMapping,
    ) -> r[t.Types.ConfigurationMapping]:
        """Validate batch services dictionary for container registration.

        Args:
            services: Dictionary of service names to service instances

        Returns:
            r[t.Types.ConfigurationMapping]: Validated services or validation error

        """
        # Allow empty dictionaries for batch_register flexibility

        # Validate service names
        for name in services:
            if not FlextUtilitiesGuards.is_type(name, str) or not name.strip():
                return r[t.Types.ConfigurationMapping].fail(
                    f"Invalid service name: '{name}'. Must be non-empty string",
                )

            # Check for reserved names
            if name.startswith("_"):
                return r[t.Types.ConfigurationMapping].fail(
                    f"Service name cannot start with underscore: '{name}'",
                )

        # Validate service instances
        for name, service in services.items():
            if service is None:
                return r[t.Types.ConfigurationMapping].fail(
                    f"Service '{name}' cannot be None",
                )

            # Check for callable services (should be registered as factories)
            if callable(service):
                error_msg: str = (
                    f"Service '{name}' appears to be callable. Use with_factory instead"
                )
                return r[t.Types.ConfigurationMapping].fail(
                    error_msg,
                )

        return r[t.Types.ConfigurationMapping].ok(services)

    @staticmethod
    def analyze_constructor_parameter(
        param_name: str,
        param: inspect.Parameter,
    ) -> t.GeneralValueType:
        """Analyze constructor parameter for dependency injection.

        Args:
            param_name: Parameter name
            param: inspect.Parameter object

        Returns:
            dict: Parameter analysis information

        """
        has_default = param.default != param.empty
        default_value = param.default if has_default else None

        return {
            "name": param_name,
            "has_default": has_default,
            "default_value": default_value,
            "kind": param.kind,
            "annotation": param.annotation,
        }

    @staticmethod
    def validate_dispatch_config(
        config: t.Types.ConfigurationMapping | None,
    ) -> r[t.Types.ConfigurationMapping]:
        """Validate dispatch configuration dictionary.

        Args:
            config: Dispatch configuration dictionary

        Returns:
            r[t.Types.ConfigurationMapping]: Validated configuration or validation error

        """
        if config is None:
            return r[t.Types.ConfigurationMapping].fail(
                "Configuration cannot be None",
            )
        if not FlextRuntime.is_dict_like(config):
            return r[t.Types.ConfigurationMapping].fail(
                "Configuration must be a dictionary",
            )

        # Validate metadata if present
        metadata = config.get("metadata")
        if metadata is not None and not FlextRuntime.is_dict_like(metadata):
            return r[t.Types.ConfigurationMapping].fail(
                "Metadata must be a dictionary",
            )

        # Validate correlation_id if present
        correlation_id = config.get("correlation_id")
        if correlation_id is not None and not FlextUtilitiesGuards.is_type(
            correlation_id,
            str,
        ):
            return r[t.Types.ConfigurationMapping].fail(
                "Correlation ID must be a string",
            )

        # Validate timeout_override if present
        timeout_override = config.get("timeout_override")
        if timeout_override is not None and not isinstance(
            timeout_override,
            (int, float),
        ):
            return r[t.Types.ConfigurationMapping].fail(
                "Timeout override must be a number",
            )

        # Type narrowing: config is guaranteed to be Mapping after validation above
        # Parameter type is already t.Types.ConfigurationMapping | None
        # and we've validated it's not None and is dict-like
        # Cast to correct type for type checker - config is validated as dict-like above
        return r[t.Types.ConfigurationMapping].ok(
            config,
        )

    @staticmethod
    def _validate_event_structure(
        event: p.Foundation.HasModelDump | None,
    ) -> r[bool]:
        """Validate event is not None and has required attributes."""
        if event is None:
            return r[bool].fail(
                "Domain event cannot be None",
                error_code=c.Errors.VALIDATION_ERROR,
            )

        # Check required attributes
        required_attrs = ["event_type", "aggregate_id", "unique_id", "created_at"]
        # NOTE: Cannot use u.filter() here due to circular import
        # (utilities.py imports validation.py)
        missing_attrs = [attr for attr in required_attrs if not hasattr(event, attr)]
        if missing_attrs:
            return r[bool].fail(
                f"Domain event missing required attributes: {missing_attrs}",
                error_code=c.Errors.VALIDATION_ERROR,
            )

        return r[bool].ok(True)

    @staticmethod
    def _validate_event_fields(
        event: p.Foundation.HasModelDump,
    ) -> r[bool]:
        """Validate event field types and values."""
        # Validate event_type is non-empty string
        event_type = getattr(event, "event_type", "")
        if not event_type or not FlextUtilitiesGuards.is_type(event_type, str):
            return r[bool].fail(
                "Domain event event_type must be a non-empty string",
                error_code=c.Errors.VALIDATION_ERROR,
            )

        # Validate aggregate_id is non-empty string
        aggregate_id = getattr(event, "aggregate_id", "")
        if not aggregate_id or not FlextUtilitiesGuards.is_type(aggregate_id, str):
            return r[bool].fail(
                "Domain event aggregate_id must be a non-empty string",
                error_code=c.Errors.VALIDATION_ERROR,
            )

        # Validate data is a dict
        data = getattr(event, "data", None)
        if data is not None and not FlextRuntime.is_dict_like(data):
            return r[bool].fail(
                "Domain event data must be a dictionary or None",
                error_code=c.Errors.VALIDATION_ERROR,
            )

        return r[bool].ok(True)

    @staticmethod
    def validate_domain_event(
        event: p.Foundation.HasModelDump | None,
    ) -> r[bool]:
        """Enhanced domain event validation with comprehensive checks.

        Validates domain events for proper structure, required fields,
        and domain invariants. Used across all flext-ecosystem projects.

        Args:
            event: The domain event to validate

        Returns:
            r[bool]: Success with True if valid, failure with details

        """
        # Validate structure
        structure_result = FlextUtilitiesValidation._validate_event_structure(event)
        if structure_result.is_failure:
            return structure_result

        # Validate fields (event is guaranteed to be non-None after structure validation)
        if event is None:
            return r[bool].fail(
                "Domain event cannot be None",
                error_code=c.Errors.VALIDATION_ERROR,
            )
        fields_result = FlextUtilitiesValidation._validate_event_fields(event)
        if fields_result.is_failure:
            return fields_result

        return r[bool].ok(True)

    # ═══════════════════════════════════════════════════════════════════
    # VALIDATION & GUARD HELPERS - Core validation logic
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _validate_get_desc(v: ValidatorSpec) -> str:
        """Extract validator description (helper for validate)."""
        # Try to extract description from predicate if it's a Validator (preferred)
        predicate = FlextUtilitiesMapper.get(v, "predicate", default=None)
        if predicate is not None:
            predicate_desc = FlextUtilitiesMapper.get(
                predicate,
                "description",
                default=None,
            )
            if FlextUtilitiesGuards.is_type(predicate_desc, str) and predicate_desc:
                return predicate_desc
        # Fall back to validator's own description
        desc = FlextUtilitiesMapper.get(v, "description", default="validator")
        return desc if FlextUtilitiesGuards.is_type(desc, str) else "validator"

    @staticmethod
    def _validate_check_any[T](
        value: T,
        validators: tuple[ValidatorSpec, ...],
        field_prefix: str,
    ) -> r[T]:
        """Check if any validator passes (helper for validate)."""
        for validator in validators:
            if validator(value):
                return r[T].ok(value)
        descriptions = [
            FlextUtilitiesValidation._validate_get_desc(v) for v in validators
        ]
        return r[T].fail(
            f"{field_prefix}None of the validators passed: {', '.join(descriptions)}",
        )

    @staticmethod
    def _validate_check_all[T](
        value: T,
        validators: tuple[ValidatorSpec, ...],
        field_prefix: str,
        *,
        fail_fast: bool,
        collect_errors: bool,
    ) -> r[T]:
        """Check if all validators pass (helper for validate)."""

        def validator_failed(v: ValidatorSpec) -> bool:
            """Check if validator failed."""
            return not v(value)

        failed_validators = [v for v in validators if validator_failed(v)]
        if not failed_validators:
            return r[T].ok(value)

        descriptions = [
            FlextUtilitiesValidation._validate_get_desc(v) for v in failed_validators
        ]
        if fail_fast and not collect_errors:
            first_desc = descriptions[0] if descriptions else None
            error_msg = f"{field_prefix}Validation failed: {FlextUtilitiesValidation.ResultHelpers.or_(first_desc, default='validator')}"
            return r[T].fail(error_msg)

        def format_error(d: str) -> str:
            """Format validation error message."""
            return f"{field_prefix}Validation failed: {d}"

        errors = [format_error(d) for d in descriptions]
        return r[T].fail("; ".join(errors))

    @staticmethod
    def validate[T](
        value: T,
        *validators: ValidatorSpec,
        mode: str = "all",
        fail_fast: bool = True,
        collect_errors: bool = False,
        field_name: str | None = None,
    ) -> r[T]:
        """Validate value against one or more validators.

        Business Rule: Composes validators using AND (all) or OR (any) logic.
        Validators ensure value conforms to expected type T after validation passes.
        Railway-oriented error handling ensures failures propagate correctly.

        Audit Implication: Validation failures are tracked with field context for
        audit trail completeness. Field names help identify validation failures
        in complex data structures.

        Args:
            value: The value to validate.
            *validators: One or more ValidatorSpec instances (from V namespace
                or custom validators).
            mode: Composition mode:
                - "all": ALL validators must pass (AND) - default
                - "any": AT LEAST ONE must pass (OR)
            fail_fast: Stop on first error when mode="all" (default True).
            collect_errors: Collect all errors even with fail_fast.
            field_name: Field name for error messages.

        Returns:
            r[T]: Ok(value) if validation passes, Fail with error message.

        Examples:
            # Simple validation with V namespace
            result = u.validate(
                email,
                u.V.string.non_empty,
                u.V.string.email,
            )

            # With operators
            validator = V.string.non_empty & V.string.max_length(100)
            result = u.validate(value, validator)

            # Any mode (OR)
            result = u.validate(
                value,
                V.string.email,
                V.string.url,
                mode="any",
            )

            # With field name for error context
            result = u.validate(
                config["port"],
                V.number.positive,
                V.number.in_range(1, 65535),
                field_name="config.port",
            )

        """
        if not validators:
            return r[T].ok(value)

        field_prefix = f"{field_name}: " if field_name else ""
        if mode == "any":
            return FlextUtilitiesValidation._validate_check_any(
                value,
                validators,
                field_prefix,
            )

        return FlextUtilitiesValidation._validate_check_all(
            value,
            validators,
            field_prefix,
            fail_fast=fail_fast,
            collect_errors=collect_errors,
        )

    @staticmethod
    def _guard_check_type(
        value: object,
        condition: type[object] | tuple[type[object], ...],
        context_name: str,
        error_msg: str | None,
    ) -> str | None:
        """Helper: Check type guard condition."""
        if not isinstance(value, condition):
            if error_msg is None:
                type_name = (
                    condition.__name__
                    if isinstance(condition, type)
                    else " | ".join(c.__name__ for c in condition)
                )
                return f"{context_name} must be {type_name}, got {type(value).__name__}"
            return error_msg
        return None

    @staticmethod
    def _guard_check_validator(
        value: object,
        condition: ValidatorSpec,
        context_name: str,
        error_msg: str | None,
    ) -> str | None:
        """Helper: Check ValidatorSpec condition."""
        if not condition(value):
            if error_msg is None:
                # Use Mapper.get for unified attribute access
                desc = FlextUtilitiesMapper.get(
                    condition,
                    "description",
                    default="validation",
                )
                return f"{context_name} failed {desc} check"
            return error_msg
        return None

    @staticmethod
    def _guard_check_string_shortcut(
        value: object,
        condition: str,
        context_name: str,
        error_msg: str | None,
    ) -> str | None:
        """Helper: Check string shortcut condition."""
        shortcut_result = FlextUtilitiesValidation._guard_shortcut(
            value,
            condition,
            context_name,
        )
        if shortcut_result.is_failure:
            # Use ResultHelpers.err for unified error extraction
            default_err = "Guard check failed"
            return error_msg or FlextUtilitiesValidation.ResultHelpers.err(
                shortcut_result,
                default=default_err,
            )
        return None

    @staticmethod
    def _guard_check_predicate(
        value: object,
        condition: Callable[[object], bool],
        context_name: str,
        error_msg: str | None,
    ) -> str | None:
        """Helper: Check custom predicate condition."""
        try:
            if not condition(value):
                if error_msg is None:
                    # Use Mapper.get for unified attribute access
                    func_name = FlextUtilitiesMapper.get(
                        condition,
                        "__name__",
                        default="custom",
                    )
                    return f"{context_name} failed {func_name} check"
                return error_msg
        except Exception as e:
            if error_msg is None:
                return f"{context_name} guard check raised: {e}"
            return error_msg
        return None

    @staticmethod
    def _guard_check_condition(
        value: object,
        condition: type[object]
        | tuple[type[object], ...]
        | Callable[[object], bool]
        | ValidatorSpec
        | str,
        context_name: str,
        error_msg: str | None,
    ) -> str | None:
        """Helper: Check a single guard condition, return error message if fails."""
        # Use match/case for Python 3.13+ pattern matching
        # Type guard: isinstance check
        if isinstance(condition, type):
            return FlextUtilitiesValidation._guard_check_type(
                value,
                condition,
                context_name,
                error_msg,
            )
        if isinstance(condition, tuple) and all(isinstance(c, type) for c in condition):
            return FlextUtilitiesValidation._guard_check_type(
                value,
                condition,
                context_name,
                error_msg,
            )

        # ValidatorSpec: Validator DSL (has __and__ method for composition)
        if callable(condition) and hasattr(condition, "__and__"):
            validator_condition = cast("ValidatorSpec", condition)
            return FlextUtilitiesValidation._guard_check_validator(
                value,
                validator_condition,
                context_name,
                error_msg,
            )

        # String shortcuts
        if isinstance(condition, str):
            return FlextUtilitiesValidation._guard_check_string_shortcut(
                value,
                condition,
                context_name,
                error_msg,
            )

        # Custom predicate: Callable[[T], bool]
        if callable(condition):
            return FlextUtilitiesValidation._guard_check_predicate(
                value,
                condition,
                context_name,
                error_msg,
            )

        # Unknown condition type
        return error_msg or f"{context_name} invalid guard condition type"

    @staticmethod
    def _guard_handle_failure[T](
        error_message: str,
        *,
        return_value: bool,
        default: T | None,
    ) -> r[T] | T | None:
        """Helper: Handle guard failure with return_value and default logic."""
        if return_value:
            # Use ResultHelpers.or_ for default fallback
            return FlextUtilitiesValidation.ResultHelpers.or_(default, default=None)
        if default is not None:
            return r[T].ok(default)
        return r[T].fail(error_message)

    @staticmethod
    def _guard_non_empty(value: object, error_template: str) -> r[object]:
        """Internal helper for non-empty validation."""
        # Use pattern matching for type-specific validation
        value_typed = cast("t.GeneralValueType", value)

        # String validation
        if FlextUtilitiesGuards.is_type(value, str):
            result_str = (
                r[str].ok(cast("str", value))
                if FlextUtilitiesGuards.is_type(value, "string_non_empty")
                else r[str].fail(f"{error_template} non-empty string")
            )
            # Cast to r[object] for return type compatibility
            return cast("r[object]", result_str)

        # Dict-like validation
        if FlextRuntime.is_dict_like(value_typed):
            result_dict = (
                r[t.Types.ConfigurationDict].ok(
                    cast("t.Types.ConfigurationDict", value_typed)
                )
                if FlextUtilitiesGuards.is_type(value_typed, "dict_non_empty")
                else r[t.Types.ConfigurationDict].fail(
                    f"{error_template} non-empty dict"
                )
            )
            # Cast to r[object] for return type compatibility
            return cast("r[object]", result_dict)

        # List-like validation
        if FlextRuntime.is_list_like(value_typed):
            result_list = (
                r[list[t.GeneralValueType]].ok(
                    cast("list[t.GeneralValueType]", value_typed)
                )
                if FlextUtilitiesGuards.is_type(value_typed, "list_non_empty")
                else r[list[t.GeneralValueType]].fail(
                    f"{error_template} non-empty list"
                )
            )
            # Cast to r[object] for return type compatibility
            return cast("r[object]", result_list)

        # Unknown type
        return r[object].fail(f"{error_template} non-empty (str/dict/list)")

    # Guard shortcut table: maps shortcut names to (check_fn, type_desc) tuples
    # check_fn: (value) -> bool (True = valid)
    _GUARD_SHORTCUTS: ClassVar[t.Types.StringCallableBoolStrTupleDict] = {
        # Numeric shortcuts
        "positive": (
            lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0,
            "positive number",
        ),
        "non_negative": (
            lambda v: isinstance(v, (int, float))
            and not isinstance(v, bool)
            and v >= 0,
            "non-negative number",
        ),
        # Type shortcuts
        "dict": (
            lambda v: FlextRuntime.is_dict_like(cast("t.GeneralValueType", v)),
            "dict-like",
        ),
        "list": (
            lambda v: FlextRuntime.is_list_like(cast("t.GeneralValueType", v)),
            "list-like",
        ),
        "string": (lambda v: isinstance(v, str), "string"),
        "int": (lambda v: isinstance(v, int) and not isinstance(v, bool), "int"),
        "float": (
            lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "float",
        ),
        "bool": (lambda v: isinstance(v, bool), "bool"),
    }

    @staticmethod
    def _guard_shortcut(
        value: object,
        shortcut: str,
        context: str,
    ) -> r[object]:
        """Handle string shortcuts for common guard patterns via table lookup."""
        # Use lower() instead of u.normalize to avoid dependency
        shortcut_lower = shortcut.lower()
        error_template = f"{context} must be"

        # Handle non_empty separately (complex type-specific logic)
        if shortcut_lower == "non_empty":
            return FlextUtilitiesValidation._guard_non_empty(value, error_template)

        # Table lookup for simple shortcuts
        if shortcut_lower in FlextUtilitiesValidation._GUARD_SHORTCUTS:
            check_fn, type_desc = FlextUtilitiesValidation._GUARD_SHORTCUTS[
                shortcut_lower
            ]
            if check_fn(value):
                result_ok = r[t.GeneralValueType].ok(cast("t.GeneralValueType", value))
                # Cast to r[object] for return type compatibility
                return cast("r[object]", result_ok)
            result_fail = r[t.GeneralValueType].fail(f"{error_template} {type_desc}")
            # Cast to r[object] for return type compatibility
            return cast("r[object]", result_fail)

        result_unknown = r[t.GeneralValueType].fail(
            f"{context} unknown guard shortcut: {shortcut}"
        )
        # Cast to r[object] for return type compatibility
        return cast("r[object]", result_unknown)

    @staticmethod
    def guard[T](
        value: T,
        *conditions: (
            type[T] | tuple[type[T], ...] | Callable[[T], bool] | ValidatorSpec | str
        ),
        error_message: str | None = None,
        context: str | None = None,
        default: T | None = None,
        return_value: bool = False,
    ) -> r[T] | T | None:
        """Advanced guard method unifying type guards and validations.

        Business Rule: Provides unified interface for type checking, validation,
        and custom predicates. Supports multiple validation strategies:
        - Type guards: isinstance checks (type or tuple of types)
        - Validators: ValidatorSpec instances (V.string.non_empty, etc.)
        - Custom predicates: Callable[[T], bool] functions
        - String shortcuts: "non_empty", "positive", "dict", "list", etc.

        Audit Implication: Guard failures are tracked with context for audit
        trail completeness. Error messages include validation details for
        debugging and audit purposes.

        Args:
            value: The value to guard/validate
            *conditions: One or more guard conditions:
                - type or tuple[type, ...]: isinstance check
                - ValidatorSpec: Validator DSL instance (V.string.non_empty)
                - Callable[[T], bool]: Custom predicate function
                - str: Shortcut name ("non_empty", "positive", "dict", "list")
            error_message: Custom error message (default: auto-generated)
            context: Context name for error messages (default: "Value")
            default: Default value to return on failure (if provided, returns Ok(default))
            return_value: If True, returns value directly instead of r[T]

        Returns:
            r[T] | T | None:
                - If return_value=False: p.Foundation.Result[T] (Ok(value) or Fail)
                - If return_value=True and default=None: T | None (value on success, None on failure)
                - If return_value=True and default provided: T (value on success, default on failure)

        Examples:
            # Type guard (returns Result)
            result = u.guard("hello", str)
            if result.is_success:
                value = result.value

            # Return value directly (reduces boilerplate)
            config = u.guard(data, dict, return_value=True)
            # config is dict | None

            # With default (returns value or default)
            config = u.guard(data, dict, default={}, return_value=True)
            # config is dict (always safe)

            # Multiple conditions
            result = u.guard("hello", str, "non_empty", return_value=True)

        """
        context_name = context or "Value"
        error_msg = error_message

        for condition in conditions:
            # Type narrowing: condition is one of the supported types
            condition_typed: (
                type[object]
                | tuple[type[object], ...]
                | Callable[[object], bool]
                | ValidatorSpec
                | str
            ) = cast(
                "type[object] | tuple[type[object], ...] | Callable[[object], bool] | ValidatorSpec | str",
                condition,
            )
            check_result = FlextUtilitiesValidation._guard_check_condition(
                cast("object", value),
                condition_typed,
                context_name,
                error_msg,
            )
            if check_result is not None:
                return FlextUtilitiesValidation._guard_handle_failure(
                    check_result,
                    return_value=return_value,
                    default=default,
                )

        # Return value directly if return_value=True, otherwise return Result
        if return_value:
            return value
        return r[T].ok(value)

    # ═══════════════════════════════════════════════════════════════════
    # RESULT HELPERS - Railway-Oriented Programming utilities
    # ═══════════════════════════════════════════════════════════════════
    # Extracted from FlextUtilities for modularity.
    # These methods provide mnemonic DSL patterns for working with r[T].

    class ResultHelpers:
        """Result-related helper methods (ok, fail, val, err, unwrap, etc).

        Provides mnemonic DSL patterns for railway-oriented programming.
        All methods are thin wrappers around r[T] operations.
        """

        @staticmethod
        def ok[T](value: T) -> r[T]:
            """Create success result (mnemonic: ok = success).

            Generic replacement for: p.Foundation.Result[T].ok(value)

            Args:
                value: Value to wrap

            Returns:
                r[T] with success

            Example:
                result = ResultHelpers.ok(data)
                # → r.ok(data)

            """
            return r[T].ok(value)

        @staticmethod
        def fail(error: str) -> r[object]:
            """Create failure result (mnemonic: fail = failure).

            Business Rule: Failures don't carry a value, only an error message.
            The return type r[object] allows type-safe failure results.

            Args:
                error: Error message

            Returns:
                r[object] with failure

            Example:
                result: p.Foundation.Result[Entry] = ResultHelpers.fail("Operation failed")

            """
            return r[object].fail(error)

        @staticmethod
        def err[T](
            result: p.Foundation.Result[T],
            *,
            default: str = "Unknown error",
        ) -> str:
            """Extract error message from r (mnemonic: err = error).

            Args:
                result: r to extract error from
                default: Default error message if error is None/empty

            Returns:
                Error message string

            Example:
                error_msg = ResultHelpers.err(result, default="Operation failed")

            """
            if result.is_failure:
                error_str = result.error
                if error_str:
                    return str(error_str)
            return default

        @staticmethod
        def val[T](
            result: p.Foundation.Result[T],
            *,
            default: T | None = None,
        ) -> T | None:
            """Extract value from r (mnemonic: val = value).

            Args:
                result: r to extract value from
                default: Default value if result is failure

            Returns:
                Value or default

            Example:
                data = ResultHelpers.val(result, default={})

            """
            return result.value if result.is_success else default

        @staticmethod
        def unwrap[T](
            value: T | r[T],
            *,
            default: T | None = None,
        ) -> T:
            """Unwrap r or return value (mnemonic: unwrap = extract value).

            Args:
                value: Value or r
                default: Default if result is failure

            Returns:
                Unwrapped value or default

            Example:
                data = ResultHelpers.unwrap(fields_result)

            """
            if isinstance(value, r):
                if value.is_failure:
                    return cast("T", default)
                return value.value
            return value

        @staticmethod
        def vals[T](
            items: dict[str, T] | r[dict[str, T]],
            *,
            default: list[T] | None = None,
        ) -> list[T]:
            """Extract values from dict or result (mnemonic: vals = values).

            Args:
                items: Dict or r containing dict
                default: Default if items is failure or None

            Returns:
                List of values

            Example:
                plugins = ResultHelpers.vals(plugins_result)

            """
            rh = FlextUtilitiesValidation.ResultHelpers
            items_dict = rh.val(items, default={}) if isinstance(items, r) else items
            if items_dict:
                return list(items_dict.values())
            result = rh.or_(default, default=[])
            return cast("list[T]", result)

        @staticmethod
        def or_[T](
            *values: T | None,
            default: T | None = None,
        ) -> T | None:
            """Return first non-None value (mnemonic: or_ = fallback chain).

            Args:
                *values: Values to try in order
                default: Default if all are None

            Returns:
                First non-None value or default

            Example:
                port = ResultHelpers.or_(config.get("port"), default=c.Platform.DEFAULT_HTTP_PORT)

            """
            for value in values:
                if value is not None:
                    return value
            return default

        @staticmethod
        def try_[T](
            func: Callable[[], T],
            *,
            default: T | None = None,
            catch: type[Exception] | tuple[type[Exception], ...] = Exception,
        ) -> T | None:
            """Try operation with fallback (mnemonic: try_ = safe execution).

            Args:
                func: Function to execute
                default: Default if exception occurs
                catch: Exception types to catch (default: Exception)

            Returns:
                Function result or default

            Example:
                port = ResultHelpers.try_(lambda: int(config["port"]), default=c.Platform.DEFAULT_HTTP_PORT)

            """
            try:
                return func()
            except BaseException as exc:
                if isinstance(exc, catch):
                    return default
                raise

        @staticmethod
        def from_[T](
            source: t.Types.ConfigurationMapping | object | None,
            key: str,
            *,
            as_type: type[T] | None = None,
            default: T,
        ) -> T:
            """Extract from source with type guard (mnemonic: from_ = extract).

            Args:
                source: Source data (dict/object/None)
                key: Key/attribute name
                as_type: Optional type to guard against
                default: Default value if source is None or field missing

            Returns:
                Extracted value with type guard or default

            Example:
                port = ResultHelpers.from_(config_obj, "port", as_type=int, default=c.Platform.FLEXT_API_PORT)

            """
            if source is None:
                return default
            m = FlextUtilitiesMapper
            if as_type is not None:
                taken = m.take(source, key, as_type=as_type, default=default)
                return taken if taken is not None else default
            gotten = m.get(source, key, default=default)
            return gotten if gotten is not None else default

        @staticmethod
        def req[T](
            value: T | None,
            *,
            name: str = "value",
        ) -> r[T]:
            """Require non-None value (mnemonic: req = required).

            Args:
                value: Value to check
                name: Field name for error message

            Returns:
                r[T]: Ok(value) or Fail with error

            Example:
                result = ResultHelpers.req(tap_name, name="tap_name")

            """
            if value is None or (
                FlextUtilitiesGuards.is_type(value, str) and not value
            ):
                return r[T].fail(f"{name} is required")
            return r[T].ok(value)

        @staticmethod
        def then[T, R2](
            result: p.Foundation.Result[T],
            func: Callable[[T], r[R2]],
        ) -> r[R2]:
            """Chain operations (mnemonic: then = flat_map).

            Args:
                result: Initial result
                func: Function to apply if success

            Returns:
                Chained result

            Example:
                result = ResultHelpers.then(parse_result, lambda d: ok(process(d)))

            """
            # Convert protocol to concrete type for method access
            # Protocol p.Foundation.Result doesn't have flat_map, need to use concrete r
            if result.is_success:
                value = result.value
                return func(value)
            # Return failure result unchanged - convert to r for return type
            return r[R2].fail(result.error if result.is_failure else "Unknown error")

        @staticmethod
        def if_[T](
            *,
            condition: bool = False,
            then_value: T | None = None,
            else_value: T | None = None,
        ) -> T | None:
            """Conditional value (mnemonic: if_ = if-then-else).

            Args:
                condition: Boolean condition
                then_value: Value if condition is True
                else_value: Value if condition is False

            Returns:
                then_value or else_value

            Example:
                port = ResultHelpers.if_(condition=debug, then_value=8080, else_value=80)

            """
            return then_value if condition else else_value

        @staticmethod
        def not_(*, value: bool = False) -> bool:
            """Negate boolean (mnemonic: not_ = not).

            Args:
                value: Boolean to negate

            Returns:
                Negated boolean

            Example:
                is_empty = ResultHelpers.not_(value=all_(items))

            """
            return not value

        @staticmethod
        def empty[T](
            items: list[T]
            | tuple[T, ...]
            | dict[str, T]
            | str
            | r[list[T] | dict[str, T]]
            | None,
        ) -> bool:
            """Check if collection/string is empty (mnemonic: empty = is empty).

            Args:
                items: Collection or string to check

            Returns:
                True if empty

            Example:
                if ResultHelpers.empty(items):
                    return fail("Items required")

            """
            if isinstance(items, r):
                if items.is_failure:
                    return True
                items = items.value

            if items is None:
                return True
            if isinstance(items, str):
                return not items
            return len(items) == 0

        @staticmethod
        def ends(
            value: str,
            suffix: str,
            *suffixes: str,
        ) -> bool:
            """Check if string ends with suffix(es).

            Args:
                value: String to check
                suffix: First suffix to check
                *suffixes: Additional suffixes to check

            Returns:
                True if ends with any suffix

            Example:
                if ResultHelpers.ends(filename, ".json", ".yaml"):
                    process_config()

            """
            all_suffixes: tuple[str, ...] = (suffix,) + suffixes
            if not all_suffixes:
                return False
            return any(value.endswith(s) for s in all_suffixes)

        @staticmethod
        def starts(
            value: str,
            prefix: str,
            *prefixes: str,
        ) -> bool:
            """Check if string starts with prefix(es).

            Args:
                value: String to check
                prefix: First prefix to check
                *prefixes: Additional prefixes to check

            Returns:
                True if starts with any prefix

            Example:
                if ResultHelpers.starts(name, "tap-", "target-"):
                    process_plugin()

            """
            all_prefixes: tuple[str, ...] = (prefix,) + prefixes
            return any(value.startswith(p) for p in all_prefixes)

        @staticmethod
        def in_(
            value: object,
            items: list[object]
            | tuple[object, ...]
            | set[object]
            | t.Types.ConfigurationMapping,
        ) -> bool:
            """Check if value is in items (mnemonic: in_ = membership).

            Args:
                value: Value to check
                items: Collection to check membership

            Returns:
                True if value is in items

            Example:
                if ResultHelpers.in_(role, ["admin", "user"]):
                    process_user()

            """
            return value in items

        @staticmethod
        def flat[T](
            items: list[list[T] | tuple[T, ...]]
            | list[list[T]]
            | list[tuple[T, ...]]
            | tuple[list[T], ...],
        ) -> list[T]:
            """Flatten nested lists (mnemonic: flat = flatten).

            Args:
                items: Nested list/tuple structure

            Returns:
                Flattened list

            Example:
                flat_list = ResultHelpers.flat([[1, 2], [3, 4]])
                # → [1, 2, 3, 4]

            """
            return [item for sublist in items for item in sublist]

        @staticmethod
        def all_(*values: object) -> bool:
            """Check if all values are truthy (mnemonic: all_ = all truthy).

            Args:
                *values: Values to check

            Returns:
                True if all values are truthy

            Example:
                if ResultHelpers.all_(name, email, age):
                    process_user()

            """
            return all(bool(v) for v in values)

        @staticmethod
        def any_(*values: object) -> bool:
            """Check if any value is truthy (mnemonic: any_ = any truthy).

            Args:
                *values: Values to check

            Returns:
                True if any value is truthy

            Example:
                if ResultHelpers.any_(config, env, default):
                    use_value()

            """
            return any(bool(v) for v in values)

        @staticmethod
        def count[T](
            items: list[T] | tuple[T, ...] | dict[str, T],
            predicate: Callable[[T], bool] | None = None,
        ) -> int:
            """Count items (mnemonic: count = len or filtered count).

            Args:
                items: Items to count
                predicate: Optional filter predicate

            Returns:
                Count of items (optionally filtered)

            Example:
                total = ResultHelpers.count(items)
                active = ResultHelpers.count(users, lambda u: u.is_active if hasattr(u, "is_active") else True)

            """
            if predicate is None:
                return len(items)
            # Handle dict separately - iterate over values, not items
            if isinstance(items, dict):
                items_dict: dict[str, T] = items
                # For dict, predicate receives values (T), not (key, value) pairs
                # Convert values to list for type compatibility
                values_list: list[T] = list(items_dict.values())
                count_gen = (1 for value in values_list if predicate(value))
                return sum(count_gen)
            # For list/tuple, iterate directly
            items_list: list[T] | tuple[T, ...] = (
                list(items) if isinstance(items, tuple) else items
            )
            # Type narrowing: predicate is not None, so it's Callable[[T], bool]
            # Generator produces int (1) for each item where predicate returns True
            count_gen = (1 for item in items_list if predicate(item))
            return sum(count_gen)

        @staticmethod
        def sum_(
            items: list[int] | list[float] | tuple[int, ...] | tuple[float, ...],
        ) -> float:
            """Sum items (mnemonic: sum_ = sum).

            Args:
                items: Items to sum

            Returns:
                Sum of items

            Example:
                total = ResultHelpers.sum_([1, 2, 3])

            """
            # Type narrowing: items is Iterable[int | float]
            # Cast to ensure type compatibility with builtin sum()
            items_iter: (
                list[int] | list[float] | tuple[int, ...] | tuple[float, ...]
            ) = items
            return float(sum(items_iter))

    # ═══════════════════════════════════════════════════════════════════
    # ENSURE HELPERS - Type coercion utilities
    # ═══════════════════════════════════════════════════════════════════
    # Extracted from FlextUtilities for modularity.
    # These methods provide type coercion with defaults.

    @staticmethod
    def _ensure_to_list(
        value: t.GeneralValueType
        | list[t.GeneralValueType]
        | tuple[t.GeneralValueType, ...]
        | None,
        default: list[t.GeneralValueType] | None,
    ) -> list[t.GeneralValueType]:
        """Helper: Convert value to list."""
        if value is None:
            rh = FlextUtilitiesValidation.ResultHelpers
            result = rh.or_(default, default=[])
            return cast("list[t.GeneralValueType]", result)
        match value:
            case list():
                return value
            case tuple():
                return list(value)
            case _:
                return [value]

    @staticmethod
    def _ensure_to_dict(
        value: t.GeneralValueType | t.Types.ConfigurationDict | None,
        default: t.Types.ConfigurationDict | None,
    ) -> t.Types.ConfigurationDict:
        """Helper: Convert value to dict."""
        if value is None:
            return default if default is not None else {}
        if isinstance(value, dict):
            return value
        return {"value": value}

    @staticmethod
    def ensure[T](
        value: t.GeneralValueType,
        *,
        target_type: str = "auto",
        default: T | list[T] | dict[str, T] | None = None,
    ) -> T | list[T] | dict[str, T]:
        """Unified ensure function that auto-detects or enforces target type.

        Generic replacement for: ensure_list(), ensure_dict(), ensure_str()

        Automatically detects if value should be list or dict, or enforces
        target_type. Converts single values, tuples, None to appropriate type.
        Supports string conversion via target_type="str" or "str_list".

        Args:
            value: Value to convert (single value, list, tuple, dict, or None)
            target_type: Target type - "auto" (detect), "list", "dict", "str",
                         "str_list"
            default: Default value if None

        Returns:
            Converted value based on target_type or auto-detection

        Example:
            # Auto-detect (prefers list for single values)
            items = ensure(value)

            # Force list
            items = ensure(value, target_type="list")

            # Convert to string
            str_value = ensure(value, target_type="str", default="")

        """
        m = FlextUtilitiesMapper

        # Handle string conversions first
        if target_type == "str":
            str_default = cast("str", default) if default is not None else ""
            return cast("T", m.ensure_str(value, default=str_default))
        if target_type == "str_list":
            str_list_default = (
                cast("list[str]", default) if isinstance(default, list) else None
            )
            return cast("list[T]", m.ensure(value, default=str_list_default))
        if target_type == "dict":
            dict_default_typed: t.Types.ConfigurationDict | None
            if isinstance(default, dict):
                dict_default_typed = cast("t.Types.ConfigurationDict", default)
            else:
                dict_default_typed = None
            dict_result = FlextUtilitiesValidation._ensure_to_dict(
                value,
                dict_default_typed,
            )
            return cast("T", dict_result)
        if target_type == "auto" and isinstance(value, dict):
            return cast("T", value)
        # Handle list or fallback
        list_default_fallback: list[t.GeneralValueType] | None = (
            cast("list[t.GeneralValueType]", default)
            if isinstance(default, list)
            else None
        )
        list_result = FlextUtilitiesValidation._ensure_to_list(
            value,
            list_default_fallback,
        )
        return cast("T", list_result)


__all__ = [
    "FlextUtilitiesValidation",
]
