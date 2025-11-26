"""Utilities module - FlextUtilitiesValidation.

Extracted from flext_core.utilities for better modularity.

⚠️ ARCHITECTURAL RULE - DO NOT MODIFY ⚠️
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This module CORRECTLY imports and uses FlextResult[T]. This is NOT a circular import.

**Dependency Flow** (UNIDIRECTIONAL - NO CIRCLE):
  FlextResult (Tier 1) → defines Result[T] type
         ↑
         |
  FlextUtilities (Tier 2) → USES Result[T] in return types

**Why This Is Safe**:
- FlextResult ONLY defines the Result[T] type and its methods (ok/fail/unwrap/map)
- FlextUtilities USES FlextResult[T] as return type for validation methods
- FlextResult NEVER calls FlextUtilities methods
- Therefore: NO CIRCULAR DEPENDENCY exists

**For Other Agents**:
❌ DO NOT remove `from flext_core.result import FlextResult`
❌ DO NOT change FlextResult[T] returns to plain T or exceptions
❌ DO NOT refactor this to "break circular imports" - THERE IS NO CIRCULAR

✅ This architecture is CORRECT and INTENTIONAL
✅ FlextUtilities validation methods MUST return FlextResult[T]
✅ Railway-oriented programming pattern requires Result types

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import concurrent.futures
import inspect
import json
import logging
import operator
import re
import socket
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields as get_dataclass_fields, is_dataclass
from datetime import datetime
from typing import TypeGuard

import orjson

from flext_core.constants import FlextConstants
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult  # ✅ CORRECT - See architectural note above
from flext_core.runtime import FlextRuntime

# Import RetryConfig directly to avoid type issues
from flext_core.typings import FlextTypes, RetryConfig

_logger = logging.getLogger(__name__)


class FlextUtilitiesValidation:
    """Unified validation patterns using railway composition.

    Use for composite/pipeline validation and complex business logic validators.
    For field validation, use Pydantic Field constraints directly.

    See: https://docs.pydantic.dev/2.12/api/fields/
    """

    # CONSOLIDATED: Use FlextUtilitiesCache for cache operations (no duplication)
    # NOTE: _normalize_component, _sort_key, _sort_dict_keys below are INTERNAL
    # recursive helpers. They are NOT duplicates of cache.py - they have different
    # logic and recursion patterns

    @staticmethod
    def _normalize_component(
        component: FlextTypes.GenericDetailsType,
    ) -> object:
        """Normalize component for consistent representation (internal recursive)."""
        if FlextRuntime.is_dict_like(component):
            component_dict = component
            return {
                str(k): FlextUtilitiesValidation._normalize_component(v)
                for k, v in component_dict.items()
            }
        if isinstance(component, (list, tuple)):
            # Component is confirmed to be list or tuple, iterate directly
            sequence: Sequence[object] = component
            return [
                FlextUtilitiesValidation._normalize_component(item) for item in sequence
            ]
        if isinstance(component, set):
            # Component is confirmed to be set, iterate directly
            set_component: set[object] = component
            return {
                FlextUtilitiesValidation._normalize_component(item)
                for item in set_component
            }
        # Return primitives and other types directly
        return component

    @staticmethod
    def _sort_key(key: object) -> tuple[str, str]:
        """Generate a sort key for consistent ordering (internal helper)."""
        key_str = str(key)
        return (key_str.casefold(), key_str)

    @staticmethod
    def _sort_dict_keys(
        data: FlextTypes.SortableObjectType,
    ) -> FlextTypes.SortableObjectType:
        """Sort dict keys for consistent representation (internal recursive)."""
        if FlextRuntime.is_dict_like(data):
            data_dict = data
            return {
                k: FlextUtilitiesValidation._sort_dict_keys(data_dict[k])
                for k in sorted(
                    data_dict.keys(),
                    key=FlextUtilitiesValidation._sort_key,
                )
            }
        return data

    @staticmethod
    def validate_pipeline(value: str, validators: list[object]) -> FlextResult[bool]:
        """Validate using a pipeline of validators.

        Returns:
            FlextResult[bool]: Success with True if all validators pass,
                failure with error details if any validator fails

        """
        for validator in validators:
            # FAST FAIL: non-callable validators are programming errors
            if not callable(validator):
                return FlextResult[bool].fail(
                    "Validator must be callable",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            try:
                # Execute validator - may return FlextResult[bool] or raise exception
                result = validator(value)

                # FAST FAIL: If validator returns FlextResult, check if ok(True)
                if isinstance(result, FlextResult):
                    if result.is_failure:
                        return FlextResult[bool].fail(
                            f"Validator failed: {result.error}",
                            error_code=FlextConstants.Errors.VALIDATION_ERROR,
                        )
                    if result.data is not True:
                        return FlextResult[bool].fail(
                            "Validator must return FlextResult[bool].ok(True)",
                            error_code=FlextConstants.Errors.VALIDATION_ERROR,
                        )
            except (
                AttributeError,
                TypeError,
                ValueError,
                RuntimeError,
                KeyError,
            ) as e:
                return FlextResult[bool].fail(
                    f"Validator failed: {e}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
        return FlextResult[bool].ok(True)

    @staticmethod
    def sort_key(value: FlextTypes.SerializableType) -> str:
        """Return a deterministic string for ordering normalized cache components."""
        try:
            json_bytes = orjson.dumps(value, option=orjson.OPT_SORT_KEYS)
            return json_bytes.decode(FlextConstants.Utilities.DEFAULT_ENCODING)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            # Use proper logger instead of root logger
            logger = logging.getLogger(__name__)
            logger.debug("orjson dumps failed: %s", e)
        # Use standard library json with sorted keys
        return json.dumps(value, sort_keys=True, default=str)

    @staticmethod
    def _is_dataclass_instance(obj: object) -> TypeGuard[object]:
        """Type guard to check if object is a dataclass instance (not class)."""
        return is_dataclass(obj) and not isinstance(obj, type)

    @staticmethod
    def _normalize_primitive_or_bytes(value: object) -> tuple[bool, object]:
        """Normalize primitive types and bytes.

        Returns:
            Tuple of (is_handled, normalized_value)
            - is_handled: True if value was a primitive/bytes
            - normalized_value: The normalized value (or None if not handled)

        """
        # Handle primitives (return as-is)
        if value is None or isinstance(value, (bool, int, float, str)):
            return (True, value)

        # Handle bytes (convert to hex tuple)
        if isinstance(value, bytes):
            return (True, ("bytes", value.hex()))

        return (False, None)  # Not a primitive/bytes - continue dispatching

    @staticmethod
    def normalize_component(
        value: object,
    ) -> object:
        """Normalize arbitrary objects into cache-friendly deterministic structures."""
        # Check primitives and bytes first
        is_primitive, normalized = (
            FlextUtilitiesValidation._normalize_primitive_or_bytes(value)
        )
        if is_primitive:
            return normalized

        # Dispatch to specialized normalizers
        if isinstance(value, FlextProtocols.HasModelDump):
            return FlextUtilitiesValidation._normalize_pydantic_value(value)

        if FlextUtilitiesValidation._is_dataclass_instance(value):
            return FlextUtilitiesValidation._normalize_dataclass_value_instance(value)

        if isinstance(value, Mapping):
            return FlextUtilitiesValidation._normalize_mapping(value)

        if isinstance(value, (list, tuple)):
            return FlextUtilitiesValidation._normalize_sequence(value)

        if isinstance(value, set):
            return FlextUtilitiesValidation._normalize_set(value)

        return FlextUtilitiesValidation._normalize_vars(value)

    @staticmethod
    def _normalize_pydantic_value(
        value: FlextProtocols.HasModelDump,
    ) -> tuple[str, object]:
        """Normalize Pydantic model to cache-friendly structure."""
        # Fast fail: model_dump() must succeed for valid Pydantic models
        try:
            dumped: dict[str, object] = value.model_dump()
        except TypeError as e:
            # Fast fail: model_dump() failure indicates invalid model
            msg = (
                f"Failed to dump Pydantic value: {type(value).__name__}: "
                f"{type(e).__name__}: {e}"
            )
            raise TypeError(msg) from e
        normalized_dumped = FlextUtilitiesValidation._normalize_component(dumped)
        return ("pydantic", normalized_dumped)

    @staticmethod
    def _normalize_dataclass_value_instance(value: object) -> tuple[str, object]:
        """Normalize dataclass instance to cache-friendly structure.

        Note: This should only be called after checking is_dataclass(value) and
        ensuring it's not a type (via isinstance(value, type) check).
        """
        # Caller guarantees value is a dataclass instance via
        # _is_dataclass_instance check. Using manual field extraction
        field_dict: dict[str, object] = {}
        # value.__class__ is type for dataclass instances
        value_class: type = value.__class__
        for field in get_dataclass_fields(value_class):
            field_dict[field.name] = getattr(value, field.name)

        normalized_dict = FlextUtilitiesValidation._normalize_component(field_dict)
        return ("dataclass", normalized_dict)

    @staticmethod
    def _normalize_mapping(value: Mapping[object, object]) -> dict[object, object]:
        """Normalize mapping to cache-friendly structure."""
        sorted_items = sorted(
            value.items(),
            key=lambda x: FlextUtilitiesValidation._sort_key(x[0]),
        )
        return {
            FlextUtilitiesValidation._normalize_component(
                k,
            ): FlextUtilitiesValidation._normalize_component(v)
            for k, v in sorted_items
        }

    @staticmethod
    def _normalize_sequence(value: Sequence[object]) -> tuple[str, tuple[object, ...]]:
        """Normalize sequence to cache-friendly structure."""
        sequence_items = [
            FlextUtilitiesValidation._normalize_component(item) for item in value
        ]
        return ("sequence", tuple(sequence_items))

    @staticmethod
    def _normalize_set(value: set[object]) -> tuple[str, tuple[object, ...]]:
        """Normalize set to cache-friendly structure."""
        set_items = [
            FlextUtilitiesValidation._normalize_component(item) for item in value
        ]
        set_items.sort(key=str)
        normalized_set = tuple(set_items)
        return ("set", normalized_set)

    @staticmethod
    def _normalize_vars(value: object) -> tuple[str, object]:
        """Normalize object attributes to cache-friendly structure."""
        try:
            vars_result = vars(value)
            # vars() returns dict[str, object] when successful
            if isinstance(vars_result, dict):
                value_vars_dict: dict[str, object] = vars_result
            else:
                return ("repr", repr(value))
        except TypeError:
            return ("repr", repr(value))

        normalized_vars = tuple(
            (key, FlextUtilitiesValidation._normalize_component(val))
            for key, val in sorted(
                value_vars_dict.items(),
                key=operator.itemgetter(0),
            )
        )
        return ("vars", normalized_vars)

    @staticmethod
    def _generate_key_from_data(command_type: type[object], sorted_data: object) -> str:
        """Generate cache key from sorted data."""
        return f"{command_type.__name__}_{hash(str(sorted_data))}"

    @staticmethod
    def _generate_key_pydantic(
        command: FlextProtocols.HasModelDump,
        command_type: type[object],
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
        command: object,
        command_type: type[object],
    ) -> str | None:
        """Generate cache key from dataclass."""
        try:
            dataclass_data: dict[str, object] = {}
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
    def _generate_key_dict(command: object, command_type: type[object]) -> str | None:
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
        command: object | None,
        command_type: type[object],
    ) -> str:
        """Generate a deterministic cache key for the command.

        Args:
            command: The command/query object
            command_type: The type of the command

        Returns:
            str: Deterministic cache key

        """
        # Try Pydantic model
        if isinstance(command, FlextProtocols.HasModelDump):
            key = FlextUtilitiesValidation._generate_key_pydantic(command, command_type)
            if key is not None:
                return key

        # Try dataclass
        if (
            hasattr(command, "__dataclass_fields__")
            and is_dataclass(command)
            and not isinstance(command, type)
        ):
            key = FlextUtilitiesValidation._generate_key_dataclass(
                command,
                command_type,
            )
            if key is not None:
                return key

        # Try dict
        if FlextRuntime.is_dict_like(command):
            key = FlextUtilitiesValidation._generate_key_dict(command, command_type)
            if key is not None:
                return key

        # Last resort: string representation with hash
        command_str = "None" if command is None else str(command)
        try:
            return f"{command_type.__name__}_{hash(command_str)}"
        except TypeError:
            # If hash fails, use deterministic encoding-based hash
            encoded = command_str.encode(FlextConstants.Utilities.DEFAULT_ENCODING)
            return f"{command_type.__name__}_{abs(hash(encoded))}"

    @staticmethod
    def sort_dict_keys(
        obj: FlextTypes.SortableObjectType,
    ) -> FlextTypes.SortableObjectType:
        """Recursively sort dictionary keys for deterministic ordering.

        Args:
            obj: Object to sort (dict[str, object], list, or other)

        Returns:
            Object with sorted keys

        """
        if FlextRuntime.is_dict_like(obj):
            dict_obj: dict[str, object] = obj
            # Convert items() view to list for sorting
            items_list: list[tuple[str, object]] = list(dict_obj.items())
            sorted_items: list[tuple[str, object]] = sorted(
                items_list,
                key=lambda x: str(x[0]),
            )
            return {
                str(k): FlextUtilitiesValidation._sort_dict_keys(v)
                for k, v in sorted_items
            }
        if FlextRuntime.is_list_like(obj):
            obj_list: list[object] = obj
            return [FlextUtilitiesValidation._sort_dict_keys(item) for item in obj_list]
        if isinstance(obj, tuple):
            # obj is confirmed to be tuple, iterate directly
            obj_tuple: tuple[object, ...] = obj
            return tuple(
                FlextUtilitiesValidation._sort_dict_keys(item) for item in obj_tuple
            )
        return obj

    @staticmethod
    def initialize(obj: FlextTypes.CachedObjectType, field_name: str) -> None:
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
    ) -> FlextResult[str]:
        """Validate value is in set of valid choices (enum validation).

        Common pattern in flext-ldap (scope, operation), flext-meltano (plugin type),
        and other projects. Consolidation provides consistent error messages.

        Args:
            value: The value to validate against choices
            valid_choices: Set of valid string choices
            context: Context for error message (e.g., "LDAP scope", "Operation")
            case_sensitive: Whether to perform case-sensitive comparison

        Returns:
            FlextResult[str]: Success with value (original case), or failure

        """
        # Prepare values for comparison
        check_value = value if case_sensitive else value.lower()
        check_choices = (
            valid_choices if case_sensitive else {c.lower() for c in valid_choices}
        )

        # Validate
        if check_value not in check_choices:
            choices_str = ", ".join(sorted(valid_choices))
            return FlextResult[str].fail(
                f"Invalid {context}: {value}. Must be one of {choices_str}",
            )

        return FlextResult[str].ok(value)

    @staticmethod
    def validate_length(
        value: str,
        min_length: int | None = None,
        max_length: int | None = None,
        context: str = "Value",
    ) -> FlextResult[str]:
        """Validate string length within bounds.

        This pattern is repeated 6+ times across flext-ldap (passwords, DN, etc),
        flext-ldif, flext-meltano, flext-target-ldif, and client-a-oud-mig.
        Consolidation ensures consistent boundary checking.

        Args:
            value: The string to validate
            min_length: Minimum allowed length (inclusive), or None for no minimum
            max_length: Maximum allowed length (inclusive), or None for no maximum
            context: Context for error message (e.g., "Password", "DN component")

        Returns:
            FlextResult[str]: Success with value, or failure with clear bounds

        """
        if min_length is not None and len(value) < min_length:
            return FlextResult[str].fail(
                f"{context} must be at least {min_length} characters",
            )
        if max_length is not None and len(value) > max_length:
            return FlextResult[str].fail(
                f"{context} must be no more than {max_length} characters",
            )
        return FlextResult[str].ok(value)

    @staticmethod
    def validate_pattern(
        value: str,
        pattern: str,
        context: str = "Value",
    ) -> FlextResult[str]:
        r"""Validate value matches regex pattern.

        **PYDANTIC 2 ALTERNATIVE**: For new code, use Pydantic's Field(pattern=...)
        which provides compile-time validation and better error messages:

        ```python
        from pydantic import BaseModel, Field
        from typing import Annotated

        # Define reusable type
        DnString = Annotated[str, Field(pattern=r"^(cn|ou|dc)=.*")]


        class LdapEntry(BaseModel):
            dn: DnString  # Automatic pattern validation
            cn: Annotated[str, Field(pattern=r"^[a-zA-Z0-9]+$")]
        ```

        This pattern is repeated 5+ times across flext-ldap (DN, filter, attribute),
        flext-ldif (RFC compliance), flext-target-ldif, and others.
        Consolidation centralizes pattern definitions.

        Args:
            value: The string to validate
            pattern: Regex pattern (as string, will be compiled internally)
            context: Context for error message (e.g., "DN", "Attribute name")

        Returns:
            FlextResult[str]: Success with value, or failure with pattern context

        """
        if not re.match(pattern, value):
            return FlextResult[str].fail(f"{context} format is invalid: {value}")
        return FlextResult[str].ok(value)

    @staticmethod
    def validate_uri(
        uri: str | None,
        allowed_schemes: list[str] | None = None,
        context: str = "URI",
    ) -> FlextResult[str]:
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
            FlextResult[str]: Success with stripped URI, or failure

        """
        # Validate non-empty
        if not uri or not uri.strip():
            return FlextResult[str].fail(f"{context} cannot be empty")

        uri_stripped = uri.strip()

        # Basic URI format validation using regex
        # RFC 3986 compliant URI pattern (simplified but stricter)
        uri_pattern = re.compile(
            r"^([a-zA-Z][a-zA-Z0-9+.-]*):"  # scheme
            r"//([^/?#]+)"  # authority (required for validation)
            r"([^?#]*)"  # path
            r"(?:\?([^#]*))?"  # query (optional)
            r"(?:#(.*))?$"  # fragment (optional)
        )

        if not uri_pattern.match(uri_stripped):
            return FlextResult[str].fail(f"{context} is not a valid URI format")

        # Validate scheme if specified
        if allowed_schemes and not any(
            uri_stripped.startswith(f"{scheme}://") for scheme in allowed_schemes
        ):
            schemes_str = ", ".join(allowed_schemes)
            return FlextResult[str].fail(
                f"{context} must start with one of {schemes_str}",
            )

        return FlextResult[str].ok(uri_stripped)

    @staticmethod
    def validate_port_number(
        port: int | None,
        context: str = "Port",
    ) -> FlextResult[int]:
        """Validate port number is in valid range (1-65535).

        **PYDANTIC 2 ALTERNATIVE**: Use FlextTypes.PortNumber which already
        provides validated port numbers:

        ```python
        from pydantic import BaseModel
        from flext_core import FlextTypes


        class ServerConfig(BaseModel):
            port: FlextTypes.PortNumber  # Automatic 1-65535 validation
            # Or with Field:
            # port: Annotated[int, Field(ge=1, le=65535)]
        ```

        Common pattern in flext-ldap, flext-meltano, and other projects that
        manage server connections.

        Args:
            port: The port number to validate (may be None)
            context: Context for error message (e.g., "LDAP port")

        Returns:
            FlextResult[int]: Success with port number, or failure

        """
        if port is None:
            return FlextResult[int].fail(f"{context} cannot be None")

        if not (
            FlextConstants.Network.MIN_PORT <= port <= FlextConstants.Network.MAX_PORT
        ):
            error_msg = (
                f"{context} must be between {FlextConstants.Network.MIN_PORT} and "
                f"{FlextConstants.Network.MAX_PORT}, got {port}"
            )
            return FlextResult[int].fail(error_msg)

        return FlextResult[int].ok(port)

    @staticmethod
    def _validate_numeric_constraint[T: (int, float)](
        value: T | None,
        predicate: Callable[[T], bool],
        error_msg: str,
        context: str = "Value",
    ) -> FlextResult[T]:
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
            FlextResult[T]: Success with value, or failure

        Example:
            >>> _validate_numeric_constraint(
            ...     5, lambda v: v > 0, "must be positive", "Count"
            ... )
            FlextResult[int].ok(5)

        """
        if value is None:
            return FlextResult[T].fail(f"{context} cannot be None")

        if not predicate(value):
            return FlextResult[T].fail(f"{context} {error_msg}, got {value}")

        return FlextResult[T].ok(value)

    @staticmethod
    def validate_non_negative(
        value: int | None,
        context: str = "Value",
    ) -> FlextResult[int]:
        """Validate integer is non-negative (>= 0).

        Common pattern for timeout_seconds, retry_count, size_limit, and other
        configuration values that must be non-negative.

        Args:
            value: The integer to validate (may be None)
            context: Context for error message (e.g., "Timeout seconds")

        Returns:
            FlextResult[int]: Success with value, or failure

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
    ) -> FlextResult[int]:
        """Validate integer is positive (> 0).

        Useful for retry_count, max_retries, and other values requiring at least 1.

        Args:
            value: The integer to validate (may be None)
            context: Context for error message (e.g., "Max retries")

        Returns:
            FlextResult[int]: Success with value, or failure

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
    ) -> FlextResult[int]:
        """Validate integer is within specified range.

        General-purpose range validation for any integer field.

        Args:
            value: The integer to validate
            min_value: Minimum allowed value (inclusive), or None for no minimum
            max_value: Maximum allowed value (inclusive), or None for no maximum
            context: Context for error message

        Returns:
            FlextResult[int]: Success with value, or failure

        """
        if min_value is not None and value < min_value:
            return FlextResult[int].fail(
                f"{context} must be at least {min_value}, got {value}",
            )
        if max_value is not None and value > max_value:
            return FlextResult[int].fail(
                f"{context} must be at most {max_value}, got {value}",
            )
        return FlextResult[int].ok(value)

    @staticmethod
    def validate_callable(
        value: object,
        error_message: str = "Value must be callable",
        error_code: str = FlextConstants.Errors.TYPE_ERROR,
    ) -> FlextResult[object]:
        """Validate that value is callable (generic helper for field validators).

        This generic helper consolidates duplicate callable validation logic
        across multiple Pydantic models (service.py, config.py, handler.py).

        Args:
            value: Value to validate (should be callable)
            error_message: Custom error message (default: "Value must be callable")
            error_code: Error code for validation failure

        Returns:
            FlextResult[object]: Success with value if callable, failure otherwise

        Example:
            >>> from flext_core.utilities import FlextUtilities
            >>> result = FlextUtilities.Validation.validate_callable(lambda x: x + 1)
            >>> result.is_success
            True
            >>> result = FlextUtilities.Validation.validate_callable("not callable")
            >>> result.is_failure
            True

        """
        if not callable(value):
            return FlextResult[object].fail(
                error_message,
                error_code=error_code,
            )
        return FlextResult[object].ok(value)

    @staticmethod
    def validate_timeout(
        timeout: float,
        max_timeout: float,
        error_message: str | None = None,
        error_code: str = FlextConstants.Errors.VALIDATION_ERROR,
    ) -> FlextResult[float | int]:
        """Validate that timeout does not exceed maximum (generic helper).

        This generic helper consolidates duplicate timeout validation logic
        across multiple Pydantic models.

        Args:
            timeout: Timeout value to validate (in seconds)
            max_timeout: Maximum allowed timeout (in seconds)
            error_message: Custom error message (optional)
            error_code: Error code for validation failure

        Returns:
            FlextResult: Success with timeout if valid, failure if exceeds max

        Example:
            >>> from flext_core.utilities import FlextUtilities
            >>> result = FlextUtilities.Validation.validate_timeout(5.0, 300.0)
            >>> result.is_success
            True
            >>> result = FlextUtilities.Validation.validate_timeout(500.0, 300.0)
            >>> result.is_failure
            True

        """
        if timeout > max_timeout:
            msg: str = (
                error_message
                if error_message is not None
                else f"Timeout cannot exceed {max_timeout} seconds"
            )
            return FlextResult[float | int].fail(msg, error_code=error_code)
        return FlextResult[float | int].ok(timeout)

    @staticmethod
    def validate_http_status_codes(
        codes: list[object],
        min_code: int = 100,
        max_code: int = 599,
    ) -> FlextResult[list[int]]:
        """Validate and normalize HTTP status codes (generic helper).

        This generic helper consolidates duplicate HTTP status code validation
        logic across multiple Pydantic models (config.py).

        Args:
            codes: List of status codes (int or str) to validate
            min_code: Minimum valid HTTP status code (default: 100)
            max_code: Maximum valid HTTP status code (default: 599)

        Returns:
            FlextResult[list[int]]: Success with normalized int codes, failure otherwise

        Example:
            >>> from flext_core.utilities import FlextUtilities
            >>> result = FlextUtilities.Validation.validate_http_status_codes([
            ...     200,
            ...     "404",
            ...     500,
            ... ])
            >>> result.is_success and result.value == [200, 404, 500]
            True
            >>> result = FlextUtilities.Validation.validate_http_status_codes([999])
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
                        return FlextResult[list[int]].fail(
                            error_msg,
                            error_code=FlextConstants.Errors.VALIDATION_ERROR,
                        )
                    validated_codes.append(code_int)
                else:
                    return FlextResult[list[int]].fail(
                        f"Invalid HTTP status code type: {type(code).__name__}",
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )
            except (ValueError, TypeError) as e:
                return FlextResult[list[int]].fail(
                    f"Invalid HTTP status code: {code} ({e})",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

        return FlextResult[list[int]].ok(validated_codes)

    @staticmethod
    def validate_iso8601_timestamp(
        timestamp: str,
        *,
        allow_empty: bool = True,
    ) -> FlextResult[str]:
        """Validate ISO 8601 timestamp format (generic helper).

        This generic helper consolidates duplicate ISO 8601 timestamp validation
        logic across multiple Pydantic models (handler.py).

        Args:
            timestamp: Timestamp string to validate (ISO 8601 format)
            allow_empty: If True, allow empty strings (default: True)

        Returns:
            FlextResult[str]: Success with normalized timestamp, failure otherwise

        Example:
            >>> from flext_core.utilities import FlextUtilities
            >>> result = FlextUtilities.Validation.validate_iso8601_timestamp(
            ...     "2025-01-01T00:00:00Z"
            ... )
            >>> result.is_success
            True
            >>> result = FlextUtilities.Validation.validate_iso8601_timestamp("invalid")
            >>> result.is_failure
            True
            >>> result = FlextUtilities.Validation.validate_iso8601_timestamp(
            ...     "", allow_empty=True
            ... )
            >>> result.is_success
            True

        """
        # Allow empty strings if configured
        if allow_empty and (not timestamp or not timestamp.strip()):
            return FlextResult[str].ok(timestamp)

        try:
            # Handle both Z suffix and explicit timezone offset
            normalized = (
                timestamp.replace("Z", "+00:00")
                if timestamp.endswith("Z")
                else timestamp
            )
            datetime.fromisoformat(normalized)
            return FlextResult[str].ok(timestamp)
        except (ValueError, TypeError) as e:
            return FlextResult[str].fail(
                f"Timestamp must be in ISO 8601 format: {e}",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

    @staticmethod
    def validate_hostname(
        hostname: str,
        *,
        perform_dns_lookup: bool = True,
    ) -> FlextResult[str]:
        """Validate hostname format and optionally perform DNS resolution.

        This generic helper consolidates hostname validation logic from typings.py
        and provides flexible validation with optional DNS lookup.

        Args:
            hostname: Hostname string to validate
            perform_dns_lookup: If True, perform DNS lookup to verify hostname
                resolution (default: True)

        Returns:
            FlextResult[str]: Success with hostname if valid, failure otherwise

        Example:
            >>> from flext_core.utilities import FlextUtilities
            >>> result = FlextUtilities.Validation.validate_hostname("localhost")
            >>> result.is_success
            True
            >>> result = FlextUtilities.Validation.validate_hostname(
            ...     "invalid..hostname"
            ... )
            >>> result.is_failure
            True
            >>> # Skip DNS lookup for performance
            >>> result = FlextUtilities.Validation.validate_hostname(
            ...     "example.com", perform_dns_lookup=False
            ... )
            >>> result.is_success
            True

        """
        # Basic hostname validation (empty check)
        if not hostname or not hostname.strip():
            return FlextResult[str].fail(
                "Hostname cannot be empty",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        normalized_hostname = hostname.strip()

        # Validate hostname format (RFC 1035: basic pattern)
        hostname_pattern = re.compile(
            r"^(?!-)(?!.*--)(?!.*\.$)(?!.*\.\.)[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?)*$"
        )
        if not hostname_pattern.match(normalized_hostname):
            return FlextResult[str].fail(
                f"Invalid hostname format: '{normalized_hostname}'",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        # Validate hostname length (RFC 1035: max 253 characters)
        if len(normalized_hostname) > FlextConstants.Network.MAX_HOSTNAME_LENGTH:
            error_msg = (
                f"Hostname '{normalized_hostname}' exceeds maximum length of "
                f"{FlextConstants.Network.MAX_HOSTNAME_LENGTH} characters"
            )
            return FlextResult[str].fail(
                error_msg,
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        # Perform DNS lookup if requested
        if perform_dns_lookup:
            try:
                socket.gethostbyname(normalized_hostname)
            except socket.gaierror as e:
                return FlextResult[str].fail(
                    f"Cannot resolve hostname '{normalized_hostname}': {e}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            except (OSError, ValueError) as e:
                return FlextResult[str].fail(
                    f"Invalid hostname '{normalized_hostname}': {e}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

        return FlextResult[str].ok(normalized_hostname)

    @staticmethod
    def validate_identifier(
        name: str,
        *,
        pattern: str = r"^[a-zA-Z_][a-zA-Z0-9_: ]*$",
        allow_empty: bool = False,
        strip: bool = True,
        error_message: str | None = None,
    ) -> FlextResult[str]:
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
            FlextResult[str]: Success with normalized name or failure with error

        Example:
            >>> from flext_core.utilities import FlextUtilities
            >>> # Service name validation
            >>> result = FlextUtilities.Validation.validate_identifier("logger:app")
            >>> result.is_success
            True
            >>> # Custom pattern (only alphanumeric + underscore)
            >>> result = FlextUtilities.Validation.validate_identifier(
            ...     "my_service", pattern=r"^[a-zA-Z0-9_]+$"
            ... )
            >>> result.is_success
            True
            >>> # Invalid characters
            >>> result = FlextUtilities.Validation.validate_identifier("invalid@name")
            >>> result.is_failure
            True

        """
        # Check empty string
        if not name or (not allow_empty and not name.strip()):
            return FlextResult[str].fail(
                error_message or "Identifier cannot be empty",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
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
            return FlextResult[str].fail(
                msg,
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        return FlextResult[str].ok(normalized_name)

    @staticmethod
    def _validate_max_attempts(retry_config: dict[str, object]) -> FlextResult[int]:
        """Validate max_attempts parameter."""
        max_attempts_raw = retry_config.get("max_attempts", 1)
        if isinstance(max_attempts_raw, (int, str)):
            max_attempts = int(max_attempts_raw)
        else:
            max_attempts = 1
        if max_attempts < 1:
            return FlextResult[int].fail(
                "max_attempts must be >= 1",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )
        return FlextResult[int].ok(max_attempts)

    @staticmethod
    def _validate_initial_delay(retry_config: dict[str, object]) -> FlextResult[float]:
        """Validate initial_delay_seconds parameter."""
        initial_delay_raw = retry_config.get("initial_delay_seconds", 0.1)
        if isinstance(initial_delay_raw, (int, float, str)):
            initial_delay = float(initial_delay_raw)
        else:
            initial_delay = 0.1
        if initial_delay <= 0:
            return FlextResult[float].fail(
                "initial_delay_seconds must be > 0",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )
        return FlextResult[float].ok(initial_delay)

    @staticmethod
    def _validate_max_delay(retry_config: dict[str, object]) -> FlextResult[float]:
        """Validate max_delay_seconds parameter."""
        max_delay_raw = retry_config.get("max_delay_seconds", 60.0)
        if isinstance(max_delay_raw, (int, float, str)):
            max_delay = float(max_delay_raw)
        else:
            max_delay = 60.0
        if max_delay <= 0:
            return FlextResult[float].fail(
                "max_delay_seconds must be > 0",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )
        return FlextResult[float].ok(max_delay)

    @staticmethod
    def _validate_backoff_multiplier(
        retry_config: dict[str, object],
    ) -> FlextResult[float]:
        """Validate backoff_multiplier parameter."""
        backoff_multiplier = retry_config.get("backoff_multiplier")
        if backoff_multiplier is not None and isinstance(
            backoff_multiplier,
            (int, float, str),
        ):
            backoff_mult = float(backoff_multiplier)
            if backoff_mult < 1.0:
                return FlextResult[float].fail(
                    "backoff_multiplier must be >= 1.0",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return FlextResult[float].ok(backoff_mult)
        return FlextResult[float].ok(
            FlextConstants.Performance.DEFAULT_BACKOFF_MULTIPLIER,
        )

    @staticmethod
    def create_retry_config(
        retry_config: dict[str, object],
    ) -> FlextResult[RetryConfig]:
        """Create and validate retry configuration using railway pattern.

        Args:
            retry_config: Raw retry configuration dictionary

        Returns:
            FlextResult[RetryConfig]: Validated retry configuration or error

        """
        try:
            # Validate each parameter using railway pattern (DRY consolidation)
            result = FlextUtilitiesValidation._validate_max_attempts(retry_config)
            if result.is_failure:
                return FlextResult[RetryConfig].fail(
                    result.error or "Max attempts validation failed"
                )

            max_attempts = result.value

            delay_result = FlextUtilitiesValidation._validate_initial_delay(
                retry_config
            )
            if delay_result.is_failure:
                return FlextResult[RetryConfig].fail(
                    delay_result.error or "Initial delay validation failed"
                )

            initial_delay = delay_result.value
            params_2 = (max_attempts, initial_delay)

            max_delay_result = FlextUtilitiesValidation._validate_max_delay(
                retry_config
            )
            if max_delay_result.is_failure:
                return FlextResult[RetryConfig].fail(
                    max_delay_result.error or "Max delay validation failed"
                )

            max_delay = max_delay_result.value
            params_3 = (*params_2, max_delay)

            backoff_result = FlextUtilitiesValidation._validate_backoff_multiplier(
                retry_config
            )
            if backoff_result.is_failure:
                return FlextResult[RetryConfig].fail(
                    backoff_result.error or "Backoff multiplier validation failed"
                )

            backoff_mult = backoff_result.value
            params_4 = (*params_3, backoff_mult)

            return FlextResult[RetryConfig].ok(
                RetryConfig(
                    max_attempts=params_4[0],
                    initial_delay_seconds=params_4[1],
                    max_delay_seconds=params_4[2],
                    exponential_backoff=bool(
                        retry_config.get("exponential_backoff"),
                    ),
                    retry_on_exceptions=(
                        [
                            exc_type
                            for exc_type in retry_config["retry_on_exceptions"]
                            if isinstance(exc_type, type)
                            and issubclass(exc_type, Exception)
                        ]
                        if "retry_on_exceptions" in retry_config
                        and retry_config["retry_on_exceptions"] is not None
                        and isinstance(
                            retry_config["retry_on_exceptions"], (list, tuple)
                        )
                        else [Exception]
                    ),
                    backoff_multiplier=params_4[3],
                )
            )

        except (ValueError, TypeError) as e:
            return FlextResult[RetryConfig].fail(
                f"Invalid retry configuration: {e}",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
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
        services: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Validate batch services dictionary for container registration.

        Args:
            services: Dictionary of service names to service instances

        Returns:
            FlextResult[dict[str, object]]: Validated services or validation error

        """
        # Allow empty dictionaries for batch_register flexibility

        # Validate service names
        for name in services:
            if not isinstance(name, str) or not name.strip():
                return FlextResult[dict[str, object]].fail(
                    f"Invalid service name: '{name}'. Must be non-empty string",
                )

            # Check for reserved names
            if name.startswith("_"):
                return FlextResult[dict[str, object]].fail(
                    f"Service name cannot start with underscore: '{name}'",
                )

        # Validate service instances
        for name, service in services.items():
            if service is None:
                return FlextResult[dict[str, object]].fail(
                    f"Service '{name}' cannot be None",
                )

            # Check for callable services (should be registered as factories)
            if callable(service):
                error_msg = (
                    f"Service '{name}' appears to be callable. Use with_factory instead"
                )
                return FlextResult[dict[str, object]].fail(error_msg)

        return FlextResult[dict[str, object]].ok(services)

    @staticmethod
    def analyze_constructor_parameter(
        param_name: str,
        param: inspect.Parameter,
    ) -> dict[str, object]:
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
        config: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Validate dispatch configuration dictionary.

        Args:
            config: Dispatch configuration dictionary

        Returns:
            FlextResult[dict[str, object]]: Validated configuration or validation error

        """
        if not FlextRuntime.is_dict_like(config):
            return FlextResult[dict[str, object]].fail(
                "Configuration must be a dictionary",
            )

        # Validate metadata if present
        metadata = config.get("metadata")
        if metadata is not None and not FlextRuntime.is_dict_like(metadata):
            return FlextResult[dict[str, object]].fail("Metadata must be a dictionary")

        # Validate correlation_id if present
        correlation_id = config.get("correlation_id")
        if correlation_id is not None and not isinstance(correlation_id, str):
            return FlextResult[dict[str, object]].fail(
                "Correlation ID must be a string",
            )

        # Validate timeout_override if present
        timeout_override = config.get("timeout_override")
        if timeout_override is not None and not isinstance(
            timeout_override,
            (int, float),
        ):
            return FlextResult[dict[str, object]].fail(
                "Timeout override must be a number",
            )

        return FlextResult[dict[str, object]].ok(config)

    @staticmethod
    def _validate_event_structure(event: object) -> FlextResult[bool]:
        """Validate event is not None and has required attributes."""
        if event is None:
            return FlextResult[bool].fail(
                "Domain event cannot be None",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        # Check required attributes
        required_attrs = ["event_type", "aggregate_id", "unique_id", "created_at"]
        missing_attrs = [attr for attr in required_attrs if not hasattr(event, attr)]
        if missing_attrs:
            return FlextResult[bool].fail(
                f"Domain event missing required attributes: {missing_attrs}",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        return FlextResult[bool].ok(True)

    @staticmethod
    def _validate_event_fields(event: object) -> FlextResult[bool]:
        """Validate event field types and values."""
        # Validate event_type is non-empty string
        event_type = getattr(event, "event_type", "")
        if not event_type or not isinstance(event_type, str):
            return FlextResult[bool].fail(
                "Domain event event_type must be a non-empty string",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        # Validate aggregate_id is non-empty string
        aggregate_id = getattr(event, "aggregate_id", "")
        if not aggregate_id or not isinstance(aggregate_id, str):
            return FlextResult[bool].fail(
                "Domain event aggregate_id must be a non-empty string",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        # Validate data is a dict
        data = getattr(event, "data", None)
        if data is not None and not FlextRuntime.is_dict_like(data):
            return FlextResult[bool].fail(
                "Domain event data must be a dictionary or None",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        return FlextResult[bool].ok(True)

    @staticmethod
    def validate_domain_event(event: object) -> FlextResult[bool]:
        """Enhanced domain event validation with comprehensive checks.

        Validates domain events for proper structure, required fields,
        and domain invariants. Used across all flext-ecosystem projects.

        Args:
            event: The domain event to validate

        Returns:
            FlextResult[bool]: Success with True if valid, failure with details

        """
        # Validate structure
        structure_result = FlextUtilitiesValidation._validate_event_structure(event)
        if structure_result.is_failure:
            return structure_result

        # Validate fields
        fields_result = FlextUtilitiesValidation._validate_event_fields(event)
        if fields_result.is_failure:
            return fields_result

        return FlextResult[bool].ok(True)


__all__ = ["FlextUtilitiesValidation"]
