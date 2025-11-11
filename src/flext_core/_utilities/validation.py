"""Utilities module - FlextUtilitiesValidation.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import logging
import operator
import re
import socket
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields as get_dataclass_fields, is_dataclass
from datetime import datetime
from typing import TypeGuard, cast

import orjson

from flext_core._utilities.cache import FlextUtilitiesCache
from flext_core.constants import FlextConstants
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes

# Module constants
MAX_PORT_NUMBER: int = 65535
MIN_PORT_NUMBER: int = 1
MAX_HOSTNAME_LENGTH: int = 253  # RFC 1035: max 253 characters
_logger = logging.getLogger(__name__)


class FlextUtilitiesValidation:
    """Unified validation patterns using railway composition.

    Use for composite/pipeline validation and complex business logic validators.
    For field validation, use Pydantic Field constraints directly.

    See: https://docs.pydantic.dev/2.12/api/fields/
    """

    @staticmethod
    def validate_pipeline(value: str, validators: list[object]) -> FlextResult[None]:
        """Validate using a pipeline of validators."""
        for validator in validators:
            if callable(validator):
                try:
                    result: FlextResult[None] = cast(
                        "FlextResult[None]", validator(value)
                    )
                    if result.is_failure:
                        return result
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    KeyError,
                ) as e:
                    return FlextResult[None].fail(f"Validator failed: {e}")
        return FlextResult[None].ok(None)

    @staticmethod
    def clear_all_caches(obj: FlextTypes.CachedObjectType) -> FlextResult[None]:
        """Clear all caches on an object to prevent memory leaks.

        NOTE: This delegates to FlextUtilitiesCache.clear_object_cache to avoid
        code duplication.

        Args:
            obj: Object to clear caches on

        Returns:
            FlextResult indicating success or failure

        """
        return FlextUtilitiesCache.clear_object_cache(obj)

    @staticmethod
    def has_cache_attributes(obj: FlextTypes.CachedObjectType) -> bool:
        """Check if object has any cache-related attributes.

        NOTE: This delegates to FlextUtilitiesCache to avoid code duplication.

        Args:
            obj: Object to check for cache attributes

        Returns:
            True if object has cache attributes, False otherwise

        """
        return FlextUtilitiesCache.has_cache_attributes(obj)

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
        # Fallback to standard library json with sorted keys
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
        is_primitive, normalized = FlextUtilitiesValidation._normalize_primitive_or_bytes(
            value
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
        try:
            dumped: dict[str, object] = value.model_dump()
        except TypeError:
            dumped = {}
        normalized_dumped = FlextUtilitiesCache.normalize_component(dumped)
        return ("pydantic", normalized_dumped)

    @staticmethod
    def _normalize_dataclass_value_instance(value: object) -> tuple[str, object]:
        """Normalize dataclass instance to cache-friendly structure.

        Note: This should only be called after checking is_dataclass(value) and
        ensuring it's not a type (via isinstance(value, type) check).
        """
        # Caller guarantees value is a dataclass instance via _is_dataclass_instance check
        # Using manual field extraction - cast to satisfy mypy strict mode
        field_dict: dict[str, object] = {}
        # Cast value.__class__ to type since we know it's a dataclass instance
        for field in get_dataclass_fields(cast("type", value.__class__)):
            field_dict[field.name] = getattr(value, field.name)

        normalized_dict = FlextUtilitiesCache.normalize_component(field_dict)
        return ("dataclass", normalized_dict)

    @staticmethod
    def _normalize_mapping(value: Mapping[object, object]) -> dict[object, object]:
        """Normalize mapping to cache-friendly structure."""
        sorted_items = sorted(
            value.items(),
            key=lambda x: FlextUtilitiesCache.sort_key(x[0]),
        )
        return {
            FlextUtilitiesCache.normalize_component(
                k,
            ): FlextUtilitiesCache.normalize_component(v)
            for k, v in sorted_items
        }

    @staticmethod
    def _normalize_sequence(value: Sequence[object]) -> tuple[str, tuple[object, ...]]:
        """Normalize sequence to cache-friendly structure."""
        sequence_items = [
            FlextUtilitiesCache.normalize_component(item) for item in value
        ]
        return ("sequence", tuple(sequence_items))

    @staticmethod
    def _normalize_set(value: set[object]) -> tuple[str, tuple[object, ...]]:
        """Normalize set to cache-friendly structure."""
        set_items = [FlextUtilitiesCache.normalize_component(item) for item in value]
        set_items.sort(key=str)
        normalized_set = tuple(set_items)
        return ("set", normalized_set)

    @staticmethod
    def _normalize_vars(value: object) -> tuple[str, object]:
        """Normalize object attributes to cache-friendly structure."""
        try:
            value_vars_dict: dict[str, object] = cast(
                "dict[str, object]",
                vars(value),
            )
        except TypeError:
            return ("repr", repr(value))

        normalized_vars = tuple(
            (key, FlextUtilitiesCache.normalize_component(val))
            for key, val in sorted(
                value_vars_dict.items(),
                key=operator.itemgetter(0),
            )
        )
        return ("vars", normalized_vars)

    @staticmethod
    def _generate_key_from_data(
        command_type: type[object], sorted_data: object
    ) -> str:
        """Generate cache key from sorted data."""
        return f"{command_type.__name__}_{hash(str(sorted_data))}"

    @staticmethod
    def _generate_key_pydantic(
        command: FlextProtocols.HasModelDump, command_type: type[object]
    ) -> str | None:
        """Generate cache key from Pydantic model."""
        try:
            data = command.model_dump(mode="python")
            sorted_data = FlextUtilitiesCache.sort_dict_keys(data)
            return FlextUtilitiesValidation._generate_key_from_data(
                command_type, sorted_data
            )
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            return None

    @staticmethod
    def _generate_key_dataclass(command: object, command_type: type[object]) -> str | None:
        """Generate cache key from dataclass."""
        try:
            dataclass_data: dict[str, object] = {}
            for field in get_dataclass_fields(cast("type", command.__class__)):
                dataclass_data[field.name] = getattr(command, field.name)
            sorted_data = FlextUtilitiesCache.sort_dict_keys(dataclass_data)
            return FlextUtilitiesValidation._generate_key_from_data(
                command_type, sorted_data
            )
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            return None

    @staticmethod
    def _generate_key_dict(command: object, command_type: type[object]) -> str | None:
        """Generate cache key from dict."""
        try:
            sorted_data = FlextUtilitiesCache.sort_dict_keys(command)
            return FlextUtilitiesValidation._generate_key_from_data(
                command_type, sorted_data
            )
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            return None

    @staticmethod
    def _generate_key_fallback(command: object | None, command_type: type[object]) -> str:
        """Generate cache key fallback from string representation."""
        command_str = str(command) if command is not None else "None"
        try:
            return f"{command_type.__name__}_{hash(command_str)}"
        except TypeError:
            # If hash fails, use deterministic fallback with encoding
            encoded = command_str.encode(FlextConstants.Utilities.DEFAULT_ENCODING)
            return f"{command_type.__name__}_{abs(hash(encoded))}"

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
        # Cast command_type once for all uses
        typed_command_type = cast("type[object]", command_type)

        # Try Pydantic model
        if isinstance(command, FlextProtocols.HasModelDump):
            key = FlextUtilitiesValidation._generate_key_pydantic(
                command, typed_command_type
            )
            if key is not None:
                return key

        # Try dataclass
        if (
            hasattr(command, "__dataclass_fields__")
            and is_dataclass(command)
            and not isinstance(command, type)
        ):
            key = FlextUtilitiesValidation._generate_key_dataclass(
                command, typed_command_type
            )
            if key is not None:
                return key

        # Try dict
        if FlextRuntime.is_dict_like(command):
            key = FlextUtilitiesValidation._generate_key_dict(command, typed_command_type)
            if key is not None:
                return key

        # Fallback to string representation
        return FlextUtilitiesValidation._generate_key_fallback(
            command, typed_command_type
        )

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
            sorted_items: list[tuple[str, object]] = sorted(
                cast("list[tuple[str, object]]", dict_obj.items()),
                key=lambda x: str(x[0]),
            )
            return {
                str(k): FlextUtilitiesCache.sort_dict_keys(v) for k, v in sorted_items
            }
        if FlextRuntime.is_list_like(obj):
            obj_list: list[object] = obj
            return [FlextUtilitiesCache.sort_dict_keys(item) for item in obj_list]
        if isinstance(obj, tuple):
            obj_tuple: tuple[object, ...] = cast("tuple[object, ...]", obj)
            return tuple(FlextUtilitiesCache.sort_dict_keys(item) for item in obj_tuple)
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
    ) -> FlextResult[str]:
        """Validate that a string is not None, empty, or whitespace-only.

        This is the most commonly repeated validation pattern across flext-ldap,
        flext-ldif, flext-meltano, and other projects. Consolidation eliminates
        300+ LOC of duplication.

        Args:
            value: The string to validate (may be None or contain whitespace)
            context: Context for error message (e.g., "Password", "DN", "Username")

        Returns:
            FlextResult[str]: Success with stripped value, or failure with error

        """
        if value is None or not value.strip():
            return FlextResult[str].fail(f"{context} cannot be empty")
        return FlextResult[str].ok(value.strip())

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
                f"Invalid {context}: {value}. Must be one of {choices_str}"
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
        flext-ldif, flext-meltano, flext-target-ldif, and algar-oud-mig.
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
                f"{context} must be at least {min_length} characters"
            )
        if max_length is not None and len(value) > max_length:
            return FlextResult[str].fail(
                f"{context} must be no more than {max_length} characters"
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

        # Validate scheme if specified
        if allowed_schemes and not any(
            uri_stripped.startswith(f"{scheme}://") for scheme in allowed_schemes
        ):
            schemes_str = ", ".join(allowed_schemes)
            return FlextResult[str].fail(
                f"{context} must start with one of {schemes_str}"
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

        if not (MIN_PORT_NUMBER <= port <= MAX_PORT_NUMBER):
            return FlextResult[int].fail(
                f"{context} must be between {MIN_PORT_NUMBER} and {MAX_PORT_NUMBER}, got {port}"
            )

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
                f"{context} must be at least {min_value}, got {value}"
            )
        if max_value is not None and value > max_value:
            return FlextResult[int].fail(
                f"{context} must be at most {max_value}, got {value}"
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
            msg = error_message or f"Timeout cannot exceed {max_timeout} seconds"
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
            >>> result = FlextUtilities.Validation.validate_http_status_codes([200, "404", 500])
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
                        return FlextResult[list[int]].fail(
                            f"Invalid HTTP status code: {code} (must be {min_code}-{max_code})",
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
            >>> result = FlextUtilities.Validation.validate_iso8601_timestamp("2025-01-01T00:00:00Z")
            >>> result.is_success
            True
            >>> result = FlextUtilities.Validation.validate_iso8601_timestamp("invalid")
            >>> result.is_failure
            True
            >>> result = FlextUtilities.Validation.validate_iso8601_timestamp("", allow_empty=True)
            >>> result.is_success
            True

        """
        # Allow empty strings if configured
        if allow_empty and (not timestamp or not timestamp.strip()):
            return FlextResult[str].ok(timestamp)

        try:
            # Handle both Z suffix and explicit timezone offset
            normalized = timestamp.replace("Z", "+00:00") if timestamp.endswith("Z") else timestamp
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
        """Validate hostname format and optionally perform DNS resolution (generic helper).

        This generic helper consolidates hostname validation logic from typings.py
        and provides flexible validation with optional DNS lookup.

        Args:
            hostname: Hostname string to validate
            perform_dns_lookup: If True, perform DNS lookup to verify hostname resolution (default: True)

        Returns:
            FlextResult[str]: Success with hostname if valid, failure otherwise

        Example:
            >>> from flext_core.utilities import FlextUtilities
            >>> result = FlextUtilities.Validation.validate_hostname("localhost")
            >>> result.is_success
            True
            >>> result = FlextUtilities.Validation.validate_hostname("invalid..hostname")
            >>> result.is_failure
            True
            >>> # Skip DNS lookup for performance
            >>> result = FlextUtilities.Validation.validate_hostname("example.com", perform_dns_lookup=False)
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

        # Validate hostname length (RFC 1035: max 253 characters)
        if len(normalized_hostname) > MAX_HOSTNAME_LENGTH:
            return FlextResult[str].fail(
                f"Hostname '{normalized_hostname}' exceeds maximum length of {MAX_HOSTNAME_LENGTH} characters",
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
        pattern: str = r"^[a-zA-Z0-9_:\- ]+$",
        allow_empty: bool = False,
        strip: bool = True,
        error_message: str | None = None,
    ) -> FlextResult[str]:
        """Validate and normalize identifier/name with customizable pattern (generic helper).

        This generic helper consolidates identifier validation logic from container.py
        (_validate_service_name) and provides flexible validation for names, identifiers,
        service names, etc.

        Default pattern allows: alphanumeric, underscore, hyphen, colon, and space
        - Useful for service names with namespacing (e.g., "logger:module_name")
        - Useful for handler names with spaces (e.g., "Pre-built Handler")

        Args:
            name: Identifier/name string to validate
            pattern: Regex pattern for validation (default: alphanumeric + _:-  + space)
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
            ...     "my_service",
            ...     pattern=r"^[a-zA-Z0-9_]+$"
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
            msg = error_message or (
                f"Identifier '{normalized_name}' contains invalid characters. "
                f"Must match pattern: {pattern}"
            )
            return FlextResult[str].fail(
                msg,
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        return FlextResult[str].ok(normalized_name)


__all__ = ["FlextUtilitiesValidation"]
