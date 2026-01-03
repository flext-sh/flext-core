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
- DO NOT remove `# Removed: from flext_core.result import r
# Use string literals for type annotations and lazy import for runtime to break circular import`
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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar, TypeGuard

import orjson
from pydantic import (
    TypeAdapter as PydanticTypeAdapter,
    ValidationError as PydanticValidationError,
)

from flext_core._utilities.cast import FlextUtilitiesCast
from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core._utilities.mapper import FlextUtilitiesMapper
from flext_core.constants import c
from flext_core.protocols import p
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes as t

# Use centralized version from cast.py
_to_general_value_type = FlextUtilitiesCast.to_general_value_type


# Use protocol from protocols module to avoid duplication and satisfy architecture rules
_Predicate = p.Validation.Predicate


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
            if uri is None:
                return r[str].fail(f"{context} cannot be None")
            if not uri.strip():
                return r[str].fail(f"{context} cannot be empty")
            # Basic URI validation
            if "://" not in uri:
                return r[str].fail(f"{context} must contain scheme (e.g., http://)")
            scheme = uri.split("://")[0].lower()
            if allowed_schemes and scheme not in allowed_schemes:
                return r[str].fail(
                    f"{context} scheme '{scheme}' not in allowed: {allowed_schemes}",
                )
            return r[str].ok(uri)

        @staticmethod
        def validate_port_number(
            port: int | None,
            context: str = "Port",
        ) -> r[int]:
            """Validate port number (1-65535)."""
            if port is None:
                return r[int].fail(f"{context} cannot be None")
            # port is int after None check (no isinstance needed)
            if port < 1 or port > c.Network.MAX_PORT:
                return r[int].fail(
                    f"{context} must be between 1 and {c.Network.MAX_PORT}",
                )
            return r[int].ok(port)

        @staticmethod
        def validate_hostname(
            hostname: str,
        ) -> r[str]:
            """Validate hostname format."""
            return FlextUtilitiesValidation.validate_hostname_format(hostname)

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
            valid_choices: set[str] | list[str],
            context: str = "Value",
            *,
            case_sensitive: bool = False,
        ) -> r[str]:
            """Validate value is in allowed choices."""
            # Convert list to set if needed
            choices_set = (
                set(valid_choices) if isinstance(valid_choices, list) else valid_choices
            )
            return FlextUtilitiesValidation.validate_choice(
                value,
                choices_set,
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
                if isinstance(dump_result, dict):
                    # Explicit type annotation for dict[str, t.GeneralValueType]
                    dict_result: dict[str, t.GeneralValueType] = dump_result
                    return dict_result
                return str(component)
            except Exception:
                return str(component)

        # Check for dataclass instance (before Sequence check to avoid treating as list)
        # is_dataclass() returns True for both classes and instances
        # We only want instances, so exclude type objects
        if is_dataclass(component) and not isinstance(component, type):
            # Use string representation instead of trying to normalize dataclass
            return str(component)

        # Check if already valid t.GeneralValueType
        return FlextUtilitiesValidation._ensure_general_value_type(component)

    @staticmethod
    def _ensure_general_value_type(
        component: t.GeneralValueType | type,
    ) -> t.GeneralValueType:
        """Ensure component is valid t.GeneralValueType.

        Args:
            component: Component to validate

        Returns:
            Valid t.GeneralValueType

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

        # Check dict-like (Mapping is superset of ConfigurationMapping)
        if FlextRuntime.is_dict_like(component):
            # Type narrowing: is_dict_like ensures component is Mapping
            mapping_component: Mapping[str, t.GeneralValueType] = component
            return FlextUtilitiesValidation._normalize_dict_like(
                mapping_component,
                visited,
            )
        if isinstance(component, (list, tuple)):
            # Explicit type annotation for sequence
            seq_component: Sequence[t.GeneralValueType] = component
            return FlextUtilitiesValidation._normalize_sequence(seq_component, visited)
        # Runtime check: set is not in t.GeneralValueType union, but can occur at runtime
        # Type narrowing: isinstance(component, set) is always False per type system
        # But runtime check handles actual set instances
        if isinstance(component, set):
            # Explicit type annotation for set
            set_component: set[t.GeneralValueType] = component
            return FlextUtilitiesValidation._normalize_set(set_component, visited)

        # Ensure valid t.GeneralValueType for primitives
        return FlextUtilitiesValidation._ensure_general_value_type(
            component,
        )

    @staticmethod
    def _convert_items_to_dict(
        items_result: (
            Sequence[tuple[str, t.GeneralValueType]] | t.ConfigurationMapping | object
        ),
    ) -> dict[str, t.GeneralValueType]:
        """Convert items() result to dict with normalization."""
        if isinstance(items_result, (list, tuple)):
            # Iterate and build dict with type-safe unpacking
            result_dict: dict[str, t.GeneralValueType] = {}
            item: tuple[str, t.GeneralValueType] | object
            for item in items_result:
                if (
                    isinstance(item, tuple)
                    and len(item) == c.Performance.EXPECTED_TUPLE_LENGTH
                ):
                    key: str | object
                    value_raw: t.GeneralValueType
                    key, value_raw = item
                    if isinstance(key, str):
                        # Convert value to GeneralValueType
                        typed_value: t.GeneralValueType = _to_general_value_type(
                            value_raw,
                        )
                        result_dict[key] = typed_value
            return result_dict

        if isinstance(items_result, Mapping):
            # items_result is already t.ConfigurationMapping
            return FlextUtilitiesMapper.to_dict(items_result)

        # Use isinstance for type narrowing (pyrefly requires this)
        if not isinstance(items_result, Iterable):
            result_type = type(items_result)
            msg = f"items() returned non-iterable: {result_type}"
            raise TypeError(msg)

        # items_result is Iterable after isinstance check
        # Convert to list via explicit iteration
        items_list: list[tuple[str, t.GeneralValueType]] = []
        iter_item: tuple[str, t.GeneralValueType] | object
        for iter_item in items_result:
            if not isinstance(iter_item, tuple):
                continue
            # Use try/except for tuple unpacking instead of len check
            try:
                k: str | object
                v: t.GeneralValueType | object
                k, v = iter_item
            except ValueError:
                continue
            if isinstance(k, str):
                # Convert v to GeneralValueType first, then normalize
                v_typed = _to_general_value_type(v)
                normalized_v: t.GeneralValueType = (
                    FlextUtilitiesValidation._normalize_component(
                        v_typed,
                        visited=None,
                    )
                )
                items_list.append((k, normalized_v))
        return dict(items_list)

    @staticmethod
    def _extract_dict_from_component(
        component: t.ConfigurationMapping | p.HasModelDump,
        _visited: set[int] | None = None,
    ) -> t.ConfigurationMapping:
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
            # items_result is already the correct type from items() method
            return FlextUtilitiesValidation._convert_items_to_dict(items_result)
        except (TypeError, ValueError) as e:
            msg = f"Cannot convert {type(component).__name__}.items() to dict"
            raise TypeError(msg) from e

    @staticmethod
    def _convert_items_result_to_dict(
        items_result: (
            Sequence[tuple[str, t.GeneralValueType]]
            | t.ConfigurationMapping
            | Iterable[tuple[str, t.GeneralValueType]]
            | object
        ),
    ) -> dict[str, t.GeneralValueType]:
        """Convert items() result to dict (helper for _convert_to_mapping).

        Args:
            items_result: Result from calling items() method

        Returns:
            dict[str, t.GeneralValueType]: Converted dictionary

        Raises:
            TypeError: If items_result cannot be converted to dict

        """
        if isinstance(items_result, (list, tuple)):
            # Iterate and build dict with type-safe unpacking
            result_dict: dict[str, t.GeneralValueType] = {}
            item: tuple[str, t.GeneralValueType] | object
            for item in items_result:
                if (
                    isinstance(item, tuple)
                    and len(item) == c.Performance.EXPECTED_TUPLE_LENGTH
                ):
                    key: str | object
                    value_raw: t.GeneralValueType
                    key, value_raw = item
                    if isinstance(key, str):
                        # Convert value to GeneralValueType
                        typed_value: t.GeneralValueType = _to_general_value_type(
                            value_raw,
                        )
                        result_dict[key] = typed_value
            return result_dict

        if isinstance(items_result, Mapping):
            # items_result is already a Mapping - convert via dict()
            return dict(items_result)

        # Use isinstance for type narrowing (pyrefly requires this)
        if not isinstance(items_result, Iterable):
            result_type = type(items_result)
            msg = f"items() returned non-iterable: {result_type}"
            raise TypeError(msg)

        # items_result is Iterable after isinstance check
        temp_dict: dict[str, t.GeneralValueType] = {}
        item2: tuple[str, t.GeneralValueType] | object
        for item2 in items_result:
            if not isinstance(item2, tuple):
                continue
            # Use try/except for tuple unpacking instead of len check
            try:
                k: str | object
                v: t.GeneralValueType | object
                k, v = item2
            except ValueError:
                continue
            if isinstance(k, str):
                # Convert v to GeneralValueType first, then normalize
                v_typed = _to_general_value_type(v)
                normalized_v: t.GeneralValueType = (
                    FlextUtilitiesValidation._normalize_component(
                        v_typed,
                        visited=None,
                    )
                )
                temp_dict[k] = normalized_v
        return temp_dict

    @staticmethod
    def _convert_to_mapping(
        component: t.ConfigurationMapping | p.HasModelDump,
    ) -> t.ConfigurationMapping:
        """Convert object to Mapping (helper for _normalize_dict_like).

        Args:
            component: Object to convert to Mapping

        Returns:
            t.ConfigurationMapping: Converted mapping

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
            # items_method() returns dict-like items
            items_result = items_method()
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
        component: t.ConfigurationMapping | p.HasModelDump,
        visited: set[int] | None = None,
    ) -> dict[str, t.GeneralValueType]:
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
        normalized_dict: dict[str, t.GeneralValueType] = {}
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
        if isinstance(value, (list, tuple)):
            # Type narrowing: tuple is valid t.GeneralValueType
            return tuple(value)
        if FlextUtilitiesGuards.is_type(value, "mapping"):
            # Type narrowing: dict is valid t.GeneralValueType
            if isinstance(value, dict):
                # Explicit type annotation
                dict_value: dict[str, t.GeneralValueType] = value
                return dict_value
            if isinstance(value, Mapping):
                # FlextUtilitiesMapper.to_dict returns ConfigurationDict
                mapped_dict: dict[str, t.GeneralValueType] = (
                    FlextUtilitiesMapper.to_dict(value)
                )
                return mapped_dict
            # Fallback for non-mapping types
            fallback_dict: dict[str, t.GeneralValueType] = {str(value): value}
            return fallback_dict
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
        # Type narrowing: t.GeneralValueType includes Mapping[str, t.GeneralValueType]
        if FlextUtilitiesGuards.is_type(data, "mapping") and isinstance(data, Mapping):
            # data is Mapping[str, t.GeneralValueType], which is valid t.GeneralValueType
            data_dict: dict[str, t.GeneralValueType] = (
                data if isinstance(data, dict) else FlextUtilitiesMapper.to_dict(data)
            )
            sorted_result: dict[str, t.GeneralValueType] = {
                str(k): FlextUtilitiesValidation._sort_dict_keys(data_dict[k])
                for k in sorted(
                    data_dict.keys(),
                    key=FlextUtilitiesValidation._sort_key,
                )
            }
            return sorted_result
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
            try:
                # Execute validator - may return r[bool] or raise exception
                result = validator(value)

                # FAST FAIL: If validator returns r, check if ok(True)
                # Use structural typing check (hasattr) instead of isinstance for generic types
                if (
                    hasattr(result, "is_success")
                    and hasattr(result, "is_failure")
                    and hasattr(result, "value")
                ):
                    # Type narrowing: result has RuntimeResult structure
                    if result.is_failure:
                        # Use getattr for safe attribute access on object type
                        error_msg = getattr(result, "error", "Unknown error")
                        return r[bool].fail(
                            f"Validator failed: {error_msg}",
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
        # t.GeneralValueType doesn't include type, so obj is never a type
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
        value: p.HasModelDump,
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
        field_dict: dict[str, t.GeneralValueType] = {}
        # value.__class__ is type for dataclass instances
        value_class: type = value.__class__
        for field in fields(value_class):
            field_dict[field.name] = getattr(value, field.name)
        sorted_data = FlextUtilitiesValidation._sort_dict_keys(field_dict)
        # Return as dict with type marker for cache structure
        return {"type": "dataclass", "data": sorted_data}

    @staticmethod
    def _normalize_mapping(
        value: t.ConfigurationMapping,
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
            # vars() always returns a dict
            # Process vars_result - normalize all values
            normalized_vars = {
                str(key): FlextUtilitiesValidation._normalize_component(
                    val,
                    visited=None,
                )
                for key, val in sorted(
                    vars_result.items(),
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
        command: p.HasModelDump,
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
            dataclass_data: dict[str, t.GeneralValueType] = {}
            # command.__class__ is type for class instances
            command_class: type = command.__class__
            for field in fields(command_class):
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
        command: t.ConfigurationMapping,
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
        if isinstance(command, p.HasModelDump):
            key = FlextUtilitiesValidation._generate_key_pydantic(command, command_type)
            if key is not None:
                return key

        # Try dataclass instance (not class)
        # is_dataclass() returns True for both classes and instances
        # We only want instances, so exclude type objects first
        if (
            command is not None
            and not isinstance(command, type)
            and hasattr(command, "__dataclass_fields__")
            and is_dataclass(command)
        ):
            # command is a dataclass instance - convert to string representation
            return f"{command_type.__name__}_{hash(str(command))}"

        # Try dict
        if isinstance(command, Mapping):
            # command is already t.ConfigurationMapping
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
        # Type narrowing: obj can be Mapping (which is t.GeneralValueType)
        if FlextUtilitiesGuards.is_type(obj, "mapping") and isinstance(obj, Mapping):
            # obj is Mapping[str, t.GeneralValueType]
            dict_obj: dict[str, t.GeneralValueType] = (
                obj if isinstance(obj, dict) else FlextUtilitiesMapper.to_dict(obj)
            )
            # Convert items() view to list for sorting
            items_list: list[tuple[str, t.GeneralValueType]] = list(
                dict_obj.items(),
            )
            sorted_items: list[tuple[str, t.GeneralValueType]] = sorted(
                items_list,
                key=lambda x: str(x[0]),
            )
            sorted_dict: dict[str, t.GeneralValueType] = {
                str(k): FlextUtilitiesValidation._sort_dict_keys(v)
                for k, v in sorted_items
            }
            return sorted_dict
        # Type narrowing: obj can be Sequence (which is t.GeneralValueType)
        # Handle tuple first (tuple is a Sequence but needs special handling)
        if isinstance(obj, tuple):
            # obj is confirmed to be tuple, iterate directly
            # GeneralValueType includes Sequence[GeneralValueType], so tuple elements are GVT
            tuple_items: list[t.GeneralValueType] = [
                FlextUtilitiesValidation._sort_dict_keys(item)
                for item in obj
                if isinstance(
                    item,
                    (
                        str,
                        int,
                        float,
                        bool,
                        type(None),
                        Sequence,
                        Mapping,
                        datetime,
                        Path,
                    ),
                )
                or item is None
            ]
            return (*tuple_items,)
        # Handle other Sequences (but not str, bytes, or tuple)
        if isinstance(obj, (list, tuple)):
            # obj is Sequence[t.GeneralValueType] - use directly
            obj_list: Sequence[t.GeneralValueType] = obj
            sorted_list: list[t.GeneralValueType] = [
                FlextUtilitiesValidation._sort_dict_keys(
                    item2
                    if isinstance(
                        item2,
                        (str, int, float, bool, type(None), Sequence, Mapping),
                    )
                    else str(item2),
                )
                for item2 in obj_list
            ]
            return sorted_list
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
        valid_choices: set[str] | list[str],
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
        # Convert to set if list was provided
        choices_set = (
            set(valid_choices) if isinstance(valid_choices, list) else valid_choices
        )

        # Prepare values for comparison
        check_value = value if case_sensitive else value.lower()
        check_choices = (
            choices_set if case_sensitive else {c.lower() for c in choices_set}
        )

        # Validate
        if check_value not in check_choices:
            choices_str = ", ".join(sorted(choices_set))
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
            r"^([a-zA-Z][a-zA-Z0-9+.-]*):"
            r"//([^/?#]+)"
            r"([^?#]*)"
            r"(?:\?([^#]*))?"
            r"(?:#(.*))?$",
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
        value: object,
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
            >>> from flext_core._utilities.guards import FlextUtilitiesGuards
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
        # Type narrowing: value is t.GeneralValueType, but runtime check ensures callable
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
            >>> from flext_core._utilities.guards import FlextUtilitiesGuards
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
        codes: list[t.GeneralValueType],
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
            >>> from flext_core._utilities.guards import FlextUtilitiesGuards
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
            >>> from flext_core._utilities.guards import FlextUtilitiesGuards
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
            _ = datetime.fromisoformat(normalized)  # Validate timestamp format
            return r[str].ok(timestamp)
        except (ValueError, TypeError) as e:
            return r[str].fail(
                f"Timestamp must be in ISO 8601 format: {e}",
                error_code=c.Errors.VALIDATION_ERROR,
            )

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
            >>> from flext_core._utilities.guards import FlextUtilitiesGuards
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
    def is_exception_retryable(
        exception: Exception,
        retry_on_exceptions: list[type[BaseException]],
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
        services: t.ConfigurationMapping,
    ) -> r[t.ConfigurationMapping]:
        """Validate batch services dictionary for container registration.

        Args:
            services: Dictionary of service names to service instances

        Returns:
            r[t.ConfigurationMapping]: Validated services or validation error

        """
        # Allow empty dictionaries for batch_register flexibility

        # Validate service names
        for name in services:
            if not FlextUtilitiesGuards.is_type(name, str) or not name.strip():
                return r[t.ConfigurationMapping].fail(
                    f"Invalid service name: '{name}'. Must be non-empty string",
                )

            # Check for reserved names
            if name.startswith("_"):
                return r[t.ConfigurationMapping].fail(
                    f"Service name cannot start with underscore: '{name}'",
                )

        # Validate service instances
        for name, service in services.items():
            if service is None:
                return r[t.ConfigurationMapping].fail(
                    f"Service '{name}' cannot be None",
                )

            # Check for callable services (should be registered as factories)

        return r[t.ConfigurationMapping].ok(services)

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
        config: t.ConfigurationMapping | None,
    ) -> r[t.ConfigurationMapping]:
        """Validate dispatch configuration dictionary.

        Args:
            config: Dispatch configuration dictionary

        Returns:
            r[t.ConfigurationMapping]: Validated configuration or validation error

        """
        if config is None:
            return r[t.ConfigurationMapping].fail(
                "Configuration cannot be None",
            )
        if not FlextRuntime.is_dict_like(config):
            return r[t.ConfigurationMapping].fail(
                "Configuration must be a dictionary",
            )

        # Validate metadata if present
        metadata = config.get("metadata")
        if metadata is not None and not FlextRuntime.is_dict_like(metadata):
            return r[t.ConfigurationMapping].fail(
                "Metadata must be a dictionary",
            )

        # Validate correlation_id if present
        correlation_id = config.get("correlation_id")
        if correlation_id is not None and not FlextUtilitiesGuards.is_type(
            correlation_id,
            str,
        ):
            return r[t.ConfigurationMapping].fail(
                "Correlation ID must be a string",
            )

        # Validate timeout_override if present
        timeout_override = config.get("timeout_override")
        if timeout_override is not None and not isinstance(
            timeout_override,
            (int, float),
        ):
            return r[t.ConfigurationMapping].fail(
                "Timeout override must be a number",
            )

        # Type narrowing: config is guaranteed to be Mapping after validation above
        # Parameter type is already t.ConfigurationMapping | None
        # and we've validated it's not None and is dict-like
        # Cast to correct type for type checker - config is validated as dict-like above
        return r[t.ConfigurationMapping].ok(
            config,
        )

    @staticmethod
    def _validate_event_structure(
        event: p.HasModelDump | None,
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
        event: p.HasModelDump,
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
        event: p.HasModelDump | None,
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
    def _validate_get_desc(v: p.ValidatorSpec) -> str:
        """Extract validator description (helper for validate)."""
        # Try to extract description from predicate if it's a Validator (preferred)
        predicate = FlextUtilitiesMapper.get(v, "predicate")
        if predicate is not None and hasattr(predicate, "__getitem__"):
            # predicate is accessible (has attribute access)
            predicate_desc = getattr(predicate, "description", None)
            if isinstance(predicate_desc, str) and predicate_desc:
                return predicate_desc
        # Fall back to validator's own description
        desc = FlextUtilitiesMapper.get(v, "description", default="validator")
        return desc if FlextUtilitiesGuards.is_type(desc, str) else "validator"

    @staticmethod
    def _validate_check_any[T](
        value: T,
        validators: tuple[p.ValidatorSpec, ...],
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
        validators: tuple[p.ValidatorSpec, ...],
        field_prefix: str,
        *,
        fail_fast: bool,
        collect_errors: bool,
    ) -> r[T]:
        """Check if all validators pass (helper for validate)."""

        def validator_failed(v: p.ValidatorSpec) -> bool:
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
        *validators: p.ValidatorSpec,
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
            *validators: One or more p.ValidatorSpec instances (from V namespace
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

            # OR mode (at least one validator must pass)
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

    # =========================================================================
    # CONVENIENCE METHODS - Simple validation utilities
    # =========================================================================

    @staticmethod
    def check[T](value: T, *validators: p.ValidatorSpec) -> r[T]:
        """Check value against validators (all must pass)."""
        result = FlextUtilitiesValidation.validate(value, *validators, mode="all")
        return r[T].ok(value) if result.is_success else r[T].fail(result.error or "")

    @staticmethod
    def check_any[T](value: T, *validators: p.ValidatorSpec) -> r[T]:
        """Check value against validators (any must pass)."""
        result = FlextUtilitiesValidation.validate(value, *validators, mode="any")
        return r[T].ok(value) if result.is_success else r[T].fail(result.error or "")

    @staticmethod
    def entity(entity_value: p.Model | None) -> r[p.Model]:
        """Validate entity is not None and has id."""
        if entity_value is None:
            return r[p.Model].fail("Entity cannot be None")
        # entity_value is not None after check
        id_attr = getattr(entity_value, "id", None)
        if id_attr is None:
            return r[p.Model].fail("Entity must have id")
        return r[p.Model].ok(entity_value)

    @staticmethod
    def in_range(
        value: float,
        min_val: float,
        max_val: float,
        context: str = "Value",
    ) -> r[int | float]:
        """Validate value is in range [min_val, max_val]."""
        if value < min_val or value > max_val:
            return r[int | float].fail(
                f"{context} must be between {min_val} and {max_val}",
            )
        return r[int | float].ok(value)

    @staticmethod
    def matches(value: str, pattern: str, context: str = "Value") -> r[str]:
        """Validate value matches regex pattern."""
        if not re.match(pattern, value):
            return r[str].fail(f"{context} does not match pattern")
        return r[str].ok(value)

    @staticmethod
    def non_empty(value: str | None, context: str = "Value") -> r[str]:
        """Validate value is non-empty string."""
        if value is None or not value.strip():
            return r[str].fail(f"{context} cannot be empty")
        return r[str].ok(value)

    @staticmethod
    def positive(value: float, context: str = "Value") -> r[int | float]:
        """Validate value is positive (> 0)."""
        if value <= 0:
            return r[int | float].fail(f"{context} must be positive")
        return r[int | float].ok(value)

    @staticmethod
    def validate_all[T: t.GeneralValueType](
        values: list[T],
        predicate: p.Validation.Predicate,
        error: str = "Validation failed",
        *,
        fail_fast: bool = True,
    ) -> r[list[T]]:
        """Validate all values in list.

        Predicate accepts any value type for flexible validation logic.
        """
        errors: list[str] = []
        for val in values:
            result: bool = predicate(val)
            if not result:
                if fail_fast:
                    return r[list[T]].fail(error)
                errors.append(error)
        if errors:
            return r[list[T]].fail("; ".join(errors))
        return r[list[T]].ok(values)

    @staticmethod
    def validate_timestamp_format(timestamp: str) -> bool:
        """Validate timestamp is in ISO 8601 format (returns bool).

        Used by Pydantic validators that need a bool check.

        Args:
            timestamp: Timestamp string to validate

        Returns:
            True if valid ISO 8601 format, False otherwise

        """
        if not timestamp:
            return True  # Empty strings are allowed
        try:
            # Normalize Z to +00:00 for fromisoformat compatibility
            normalized = timestamp.replace("Z", "+00:00")
            datetime.fromisoformat(normalized)
            return True
        except ValueError:
            return False

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
        condition: p.ValidatorSpec,
        context_name: str,
        error_msg: str | None,
    ) -> str | None:
        """Helper: Check p.ValidatorSpec condition."""
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
            # Access error directly from result - RuntimeResult implements p.Result
            default_err = "Guard check failed"
            error_str = shortcut_result.error or default_err
            return error_msg or error_str
        return None

    @staticmethod
    def _guard_check_predicate(
        value: object,
        condition: Callable[..., object],
        context_name: str,
        error_msg: str | None,
    ) -> str | None:
        """Helper: Check custom predicate condition."""
        try:
            if not bool(condition(value)):
                if error_msg is None:
                    # Use getattr for callable attribute access (not mapper.get)
                    func_name = getattr(condition, "__name__", "custom")
                    return f"{context_name} failed {func_name} check"
                return error_msg
        except Exception as e:
            if error_msg is None:
                return f"{context_name} guard check raised: {e}"
            return error_msg
        return None

    @staticmethod
    def _guard_check_condition[T](
        value: T,
        condition: type[T]
        | tuple[type[T], ...]
        | Callable[[T], bool]
        | p.ValidatorSpec
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
        # Type narrowing: condition is tuple of types
        if isinstance(condition, tuple):
            # Check if all items are types (explicit loop for type narrowing)
            all_types = True
            c: type[T] | object
            for c in condition:
                if not isinstance(c, type):
                    all_types = False
                    break
            if all_types:
                # condition is tuple[type[T], ...] - safe to pass to _guard_check_type
                type_tuple: tuple[type[T], ...] = condition
                return FlextUtilitiesValidation._guard_check_type(
                    value,
                    type_tuple,
                    context_name,
                    error_msg,
                )

        # p.ValidatorSpec: Validator DSL (has __and__ method for composition)
        if isinstance(condition, p.ValidatorSpec):
            return FlextUtilitiesValidation._guard_check_validator(
                value,
                condition,
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
        # At this point in the type union, only Callable[[T], bool] remains
        # Pass condition directly - _guard_check_predicate accepts Callable[..., object]
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
    def _guard_non_empty(
        value: object,
        error_template: str,
    ) -> r[object]:
        """Internal helper for non-empty validation."""
        # Use pattern matching for type-specific validation
        value_typed = value

        # String validation
        if isinstance(value, str):
            if value:  # Non-empty string
                return r[object].ok(value)
            return r[object].fail(f"{error_template} non-empty string")

        # Dict-like validation
        if isinstance(value_typed, dict):
            if value_typed:  # Non-empty dict
                return r[object].ok(value_typed)
            return r[object].fail(f"{error_template} non-empty dict")

        # List-like validation
        if isinstance(value_typed, list):
            if value_typed:  # Non-empty list
                return r[object].ok(value_typed)
            return r[object].fail(f"{error_template} non-empty list")

        # Unknown type
        return r[object].fail(f"{error_template} non-empty (str/dict/list)")

    # Guard shortcut table: maps shortcut names to (check_fn, type_desc) tuples
    # check_fn: (value: object) -> bool (True = valid)
    # Note: FlextRuntime.is_dict_like and is_list_like return TypeGuard
    # but we use them here as plain bool checkers (compatible usage)
    _GUARD_SHORTCUTS: ClassVar[t.StringCallableBoolStrTupleDict] = {
        # Numeric shortcuts
        "positive": (
            lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0,
            "positive number",
        ),
        "non_negative": (
            lambda v: (
                isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0
            ),
            "non-negative number",
        ),
        # Type shortcuts - use isinstance instead of TypeGuard functions
        # to avoid argument type mismatch (object vs GeneralValueType)
        "dict": (
            lambda v: isinstance(v, Mapping) and not isinstance(v, (str, bytes)),
            "dict-like",
        ),
        "list": (
            lambda v: (
                isinstance(v, Sequence) and not isinstance(v, (str, bytes, Mapping))
            ),
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
                # Return r[object] to match function signature
                return r[object].ok(value)
            return r[object].fail(f"{error_template} {type_desc}")

        # Return r[object] to match function signature
        return r[object].fail(
            f"{context} unknown guard shortcut: {shortcut}",
        )

    @staticmethod
    def guard[T](
        value: T,
        *conditions: (
            type[T] | tuple[type[T], ...] | Callable[[T], bool] | p.ValidatorSpec | str
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
        - Validators: p.ValidatorSpec instances (V.string.non_empty, etc.)
        - Custom predicates: Callable[[T], bool] functions
        - String shortcuts: "non_empty", "positive", "dict", "list", etc.

        Audit Implication: Guard failures are tracked with context for audit
        trail completeness. Error messages include validation details for
        debugging and audit purposes.

        Args:
            value: The value to guard/validate
            *conditions: One or more guard conditions:
                - type or tuple[type, ...]: isinstance check
                - p.ValidatorSpec: Validator DSL instance (V.string.non_empty)
                - Callable[[T], bool]: Custom predicate function
                - str: Shortcut name ("non_empty", "positive", "dict", "list")
            error_message: Custom error message (default: auto-generated)
            context: Context name for error messages (default: "Value")
            default: Default value to return on failure (if provided, returns Ok(default))
            return_value: If True, returns value directly instead of r[T]

        Returns:
            r[T] | T | None:
                - If return_value=False: p.Result[T] (Ok(value) or Fail)
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
            # Type narrowing: condition is one of the supported types per annotation
            # T is bounded by object, so condition is compatible with object-based types
            check_result = FlextUtilitiesValidation._guard_check_condition(
                value,
                condition,
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

            Generic replacement for: p.Result[T].ok(value)

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
                result: p.Result[Entry] = ResultHelpers.fail("Operation failed")

            """
            return r[object].fail(error)

        @staticmethod
        def err[T](
            result: p.Result[T],
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
            result: p.Result[T],
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
            if result.is_success:
                return result.value
            return default

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
            # Handle r[dict[str, T]] or dict[str, T]
            items_dict: dict[str, T]
            if isinstance(items, r):
                # FlextResult - access value directly, empty dict on failure
                items_dict = items.value if items.is_success else {}
            else:
                items_dict = items

            if items_dict:
                return list(items_dict.values())

            return default if default is not None else []

        @staticmethod
        def vals_sequence[T](results: Sequence[p.Result[T]]) -> list[T]:
            """Extract values from sequence of results, skipping failures.

            Unlike vals() which processes dict values, this method processes
            a sequence of Result objects and returns only successful values.

            Args:
                results: Sequence of Result objects

            Returns:
                List of values from successful results

            Example:
                results = [r[int].ok(1), r[int].fail("error"), r[int].ok(3)]
                values = ResultHelpers.vals_sequence(results)
                # → [1, 3]

            """
            return [result.value for result in results if result.is_success]

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
                port = ResultHelpers.or_(
                    FlextUtilitiesMapper.get(config, "port"),
                    default=c.Platform.DEFAULT_HTTP_PORT,
                )

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
            result: p.Result[T],
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
            # Protocol p.Result doesn't have flat_map, need to use concrete r
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
        def not_(value: object) -> bool:
            """Negate boolean (mnemonic: not_ = not).

            Args:
                value: Value to negate (will be coerced to bool)

            Returns:
                Negated boolean

            Example:
                is_empty = ResultHelpers.not_(all_(items))

            """
            return not bool(value)

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
            all_suffixes: tuple[str, ...] = (suffix, *suffixes)
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
            all_prefixes: tuple[str, ...] = (prefix, *prefixes)
            return any(value.startswith(p) for p in all_prefixes)

        @staticmethod
        def in_(
            value: object,
            items: list[t.GeneralValueType]
            | tuple[object, ...]
            | set[object]
            | t.ConfigurationMapping,
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
        value: t.GeneralValueType | list[t.GeneralValueType] | None,
        default: list[t.GeneralValueType] | None,
    ) -> list[t.GeneralValueType]:
        """Helper: Convert value to list."""
        if value is None:
            return default if default is not None else []
        if isinstance(value, list):
            # Explicit type annotation
            typed_list: list[t.GeneralValueType] = value
            return typed_list
        # For all other types (including str), wrap in list
        single_item_list: list[t.GeneralValueType] = [value]
        return single_item_list

    @staticmethod
    def _ensure_to_dict(
        value: t.GeneralValueType | dict[str, t.GeneralValueType] | None,
        default: dict[str, t.GeneralValueType] | None,
    ) -> dict[str, t.GeneralValueType]:
        """Helper: Convert value to dict."""
        if value is None:
            return default if default is not None else {}
        if isinstance(value, dict):
            # Explicit type annotation
            typed_dict: dict[str, t.GeneralValueType] = value
            return typed_dict
        # Wrap non-dict value in dict
        wrapped_dict: dict[str, t.GeneralValueType] = {"value": value}
        return wrapped_dict

    @staticmethod
    def ensure(
        value: t.GeneralValueType,
        *,
        target_type: str = "auto",
        default: str
        | list[t.GeneralValueType]
        | dict[str, t.GeneralValueType]
        | None = None,
    ) -> str | list[t.GeneralValueType] | dict[str, t.GeneralValueType]:
        """Unified ensure function that auto-detects or enforces target type.

        Replacement for: ensure_list(), ensure_dict(), ensure_str()

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
        # Handle string conversions first
        if target_type == "str":
            str_default = default if isinstance(default, str) else ""
            return FlextUtilitiesMapper.ensure_str(value, default=str_default)

        if target_type == "str_list":
            # FlextUtilitiesMapper.ensure returns list[str] - coerce to GeneralValueType
            str_list_default: list[str] | None = None
            if isinstance(default, list):
                str_list_default = [str(x) for x in default]
            result: list[t.GeneralValueType] = list(
                FlextUtilitiesMapper.ensure(value, default=str_list_default),
            )
            return result

        if target_type == "dict":
            # When target_type is dict, return ConfigurationDict
            dict_default: dict[str, t.GeneralValueType] | None = (
                default if isinstance(default, dict) else None
            )
            return FlextUtilitiesValidation._ensure_to_dict(value, dict_default)

        if target_type == "auto" and isinstance(value, dict):
            return value

        # Handle list or fallback
        list_default: list[t.GeneralValueType] | None = (
            default if isinstance(default, list) else None
        )
        return FlextUtilitiesValidation._ensure_to_list(value, list_default)

    # ═══════════════════════════════════════════════════════════════════
    # TYPEADAPTER UTILITIES (Pydantic v2 dynamic validation)
    # ═══════════════════════════════════════════════════════════════════

    class TypeAdapter:
        """Pydantic TypeAdapter utilities for dynamic validation.

        Provides utilities for dynamic validation and serialization
        using Pydantic v2 TypeAdapter, enabling runtime type validation
        without requiring pre-defined model classes.
        """

        @staticmethod
        def validate[T](data: object, type_: type[T]) -> r[T]:
            """Validate data against type using TypeAdapter.

            Args:
                data: Data to validate.
                type_: Type to validate against.

            Returns:
                r[T]: Success with validated data, or failure with validation errors.

            Example:
                >>> result = u.Validation.TypeAdapter.validate(
                ...     {"name": "John", "age": 30}, UserModel
                ... )
                >>> if result.is_success:
                ...     user = result.value

            """
            adapter = PydanticTypeAdapter(type_)
            try:
                validated = adapter.validate_python(data)
                return r[T].ok(validated)
            except PydanticValidationError as e:
                error_msg = "; ".join(
                    f"{err['loc']}: {err['msg']}" for err in e.errors()
                )
                return r[T].fail(f"Validation failed: {error_msg}")

        @staticmethod
        def serialize[T](
            value: T,
            type_: type[T],
        ) -> r[Mapping[str, t.GeneralValueType]]:
            """Serialize value using TypeAdapter.

            Args:
                value: Value to serialize.
                type_: Type of the value.

            Returns:
                r[Mapping[str, GeneralValueType]]: Success with serialized data as dict,
                    or failure with serialization errors.

            Example:
                >>> result = u.Validation.TypeAdapter.serialize(user, UserModel)
                >>> if result.is_success:
                ...     data = result.value

            """
            adapter = PydanticTypeAdapter(type_)
            try:
                serialized = adapter.dump_python(value, mode="json")
                # Explicit type annotation for the result
                result_dict: dict[str, t.GeneralValueType] = (
                    serialized
                    if isinstance(serialized, dict)
                    else {"value": serialized}
                )
                return r[Mapping[str, t.GeneralValueType]].ok(result_dict)
            except Exception as e:
                return r[Mapping[str, t.GeneralValueType]].fail(
                    f"Serialization failed: {e}",
                )

        @staticmethod
        def parse_json[T](
            json_str: str,
            type_: type[T],
        ) -> r[T]:
            """Parse JSON string into typed model.

            Args:
                json_str: JSON string to parse.
                type_: Type to parse into.

            Returns:
                r[T]: Success with parsed model, or failure with parsing errors.

            Example:
                >>> result = u.Validation.TypeAdapter.parse_json(
                ...     '{"name": "John"}', UserModel
                ... )
                >>> if result.is_success:
                ...     user = result.value

            """
            adapter = PydanticTypeAdapter(type_)
            try:
                validated = adapter.validate_json(json_str)
                return r[T].ok(validated)
            except PydanticValidationError as e:
                error_msg = "; ".join(
                    f"{err['loc']}: {err['msg']}" for err in e.errors()
                )
                return r[T].fail(f"JSON parsing failed: {error_msg}")
            except Exception as e:
                return r[T].fail(f"JSON parsing failed: {e}")

    @staticmethod
    def validate_hostname_format(
        hostname: str | None,
        context: str = "Hostname",
    ) -> r[str]:
        if hostname is None:
            return r[str].fail(
                f"{context} cannot be None",
                error_code=c.Errors.VALIDATION_ERROR,
            )
        # hostname is already guaranteed to be str by type annotation
        if not hostname:
            return r[str].fail(
                f"{context} cannot be empty",
                error_code=c.Errors.VALIDATION_ERROR,
            )

        # Regex for hostname validation (RFC 1123, RFC 952):
        # - Allows letters, digits, and hyphens
        # - Must not start or end with a hyphen
        # - Each label (segment) separated by dots
        # - Max length for a label is 63 chars (not checked by this regex, but good to know)
        # - Total length max 255 chars (not checked by this regex)
        hostname_pattern = re.compile(
            r"""^(?!-)[A-Za-z0-9-]{1,63}(\.(?!-)[A-Za-z0-9-]{1,63})*(\.?)$""",
        )

        if not hostname_pattern.fullmatch(hostname):
            return r[str].fail(
                f"Invalid {context} format '{hostname}'. Must be a valid hostname (e.g., example.com)",
                error_code=c.Errors.VALIDATION_ERROR,
            )
        return r[str].ok(hostname)

    @staticmethod
    def validate_hostname(
        hostname: str,
    ) -> r[str]:
        """Validate hostname format (top-level convenience method).

        This is a convenience method that delegates to validate_hostname_format.
        Provides simpler signature for common use cases where context is not needed.

        Args:
            hostname: The hostname string to validate

        Returns:
            r[str]: Success with hostname, or failure with error message

        """
        return FlextUtilitiesValidation.validate_hostname_format(hostname)

    # ========================================================================
    # Validation Convenience Methods
    # ========================================================================

    @staticmethod
    def validate_with_validators(
        value: object,
        *validators: p.ValidatorSpec,
    ) -> r[bool]:
        """Validate value against multiple validators.

        Returns r[bool].ok(True) if all validators pass,
        r[bool].fail(message) on first failure.

        Args:
            value: Value to validate
            *validators: p.ValidatorSpec instances to apply

        Returns:
            r[bool] with success or first validation error

        Example:
            result = u.validate_with(
                email,
                V.string.non_empty,
                V.string.contains("@"),
            )
            if result.is_failure:
                return r.fail(result.error)

        """
        for validator in validators:
            try:
                if not validator(value):
                    desc = getattr(validator, "description", "validation")
                    return r[bool].fail(f"Validation failed: {desc}")
            except Exception as e:
                return r[bool].fail(f"Validator error: {e}")
        return r[bool].ok(True)

    @staticmethod
    def check_all_validators(value: object, *validators: p.ValidatorSpec) -> bool:
        """Check if value passes all validators.

        Simple boolean check without result wrapping.

        Args:
            value: Value to check
            *validators: p.ValidatorSpec instances to apply

        Returns:
            True if all validators pass, False otherwise

        Example:
            if u.check_all(email, V.string.non_empty, V.string.contains("@")):
                process_email(email)

        """
        return all(v(value) for v in validators)

    @staticmethod
    def check_any_validator(value: object, *validators: p.ValidatorSpec) -> bool:
        """Check if value passes any validator.

        Simple boolean check without result wrapping.

        Args:
            value: Value to check
            *validators: p.ValidatorSpec instances to apply

        Returns:
            True if any validator passes, False otherwise

        Example:
            if u.check_any(value, V.number.positive, V.string.numeric):
                process_numeric(value)

        """
        return any(v(value) for v in validators)

    @staticmethod
    def require_initialized[T](value: T | None, name: str) -> T:
        """Guard function that raises RuntimeError if value is None.

        Eliminates repetitive null-check boilerplate across service classes.

        Args:
            value: The value to check (may be None).
            name: Human-readable name for error messages.

        Returns:
            The value if not None.

        Raises:
            RuntimeError: If value is None.

        Example:
            @property
            def context(self) -> p.Context:
                return u.require_initialized(self._context, "Context")

        """
        if value is None:
            msg = f"{name} not initialized"
            raise RuntimeError(msg)
        return value


__all__ = [
    "FlextUtilitiesValidation",
]
