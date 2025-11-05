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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import cast

import orjson

from flext_core._utilities.cache import FlextUtilitiesCache
from flext_core.constants import FlextConstants
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes

# Module constants
MAX_PORT_NUMBER: int = 65535
MIN_PORT_NUMBER: int = 1
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

        Args:
            obj: Object to clear caches on

        Returns:
            FlextResult indicating success or failure

        """
        try:
            # Common cache attribute names to check and clear
            cache_attributes = FlextConstants.Utilities.CACHE_ATTRIBUTE_NAMES

            cleared_count = 0
            for attr_name in cache_attributes:
                if hasattr(obj, attr_name):
                    cache_attr = getattr(obj, attr_name, None)
                    if cache_attr is not None:
                        # Clear dict[str, object]-like caches
                        if hasattr(cache_attr, "clear") and callable(
                            cache_attr.clear,
                        ):
                            cache_attr.clear()
                            cleared_count += 1
                        # Reset to None for simple cached values
                        else:
                            setattr(obj, attr_name, None)
                            cleared_count += 1

            return FlextResult[None].ok(None)
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError) as e:
            return FlextResult[None].fail(f"Failed to clear caches: {e}")

    @staticmethod
    def has_cache_attributes(obj: FlextTypes.CachedObjectType) -> bool:
        """Check if object has any cache-related attributes.

        Args:
            obj: Object to check for cache attributes

        Returns:
            True if object has cache attributes, False otherwise

        """
        cache_attributes = FlextConstants.Utilities.CACHE_ATTRIBUTE_NAMES

        return any(hasattr(obj, attr) for attr in cache_attributes)

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
    def normalize_component(
        value: object,
    ) -> object:
        """Normalize arbitrary objects into cache-friendly deterministic structures."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return value

        if isinstance(value, bytes):
            return ("bytes", value.hex())

        if isinstance(value, FlextProtocols.HasModelDump):
            return FlextUtilitiesValidation._normalize_pydantic_value(value)

        if is_dataclass(value):
            return FlextUtilitiesValidation._normalize_dataclass_value(value)

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
    def _normalize_dataclass_value(value: object) -> tuple[str, object]:
        """Normalize dataclass to cache-friendly structure."""
        if isinstance(value, type):
            return ("dataclass_class", str(value))
        dataclass_dict = asdict(value)
        normalized_dict = FlextUtilitiesCache.normalize_component(dataclass_dict)
        return ("dataclass", normalized_dict)

    @staticmethod
    def _normalize_mapping(value: Mapping[object, object]) -> dict[object, object]:
        """Normalize mapping to cache-friendly structure."""
        mapping_value = cast("Mapping[object, object]", value)
        sorted_items = sorted(
            mapping_value.items(),
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
        sequence_value = cast("Sequence[object]", value)
        sequence_items = [
            FlextUtilitiesCache.normalize_component(item) for item in sequence_value
        ]
        return ("sequence", tuple(sequence_items))

    @staticmethod
    def _normalize_set(value: set[object]) -> tuple[str, tuple[object, ...]]:
        """Normalize set to cache-friendly structure."""
        set_value = cast("set[object]", value)
        set_items = [
            FlextUtilitiesCache.normalize_component(item) for item in set_value
        ]
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
        try:
            # For Pydantic models, use model_dump with sorted keys
            if isinstance(command, FlextProtocols.HasModelDump):
                data = command.model_dump(mode="python")
                # Sort keys recursively for deterministic ordering
                sorted_data = FlextUtilitiesCache.sort_dict_keys(data)
                return f"{cast('type', command_type).__name__}_{hash(str(sorted_data))}"

            # For dataclasses, use asdict with sorted keys
            if (
                hasattr(command, "__dataclass_fields__")
                and is_dataclass(command)
                and not isinstance(command, type)
            ):
                dataclass_data = asdict(command)
                dataclass_sorted_data = FlextUtilitiesCache.sort_dict_keys(
                    dataclass_data,
                )
                return f"{cast('type', command_type).__name__}_{hash(str(dataclass_sorted_data))}"

            # For dictionaries, sort keys
            if isinstance(command, dict):
                dict_sorted_data = FlextUtilitiesCache.sort_dict_keys(
                    cast("dict[str, object]", command),
                )
                return f"{cast('type', command_type).__name__}_{hash(str(dict_sorted_data))}"

            # For other objects, use string representation
            command_str = str(command) if command is not None else "None"
            command_hash = hash(command_str)
            return f"{cast('type', command_type).__name__}_{command_hash}"

        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            # Fallback to string representation if anything fails
            command_str_fallback: str = str(command) if command is not None else "None"
            # Ensure we have a valid string for encoding
            try:
                command_hash_fallback = hash(command_str_fallback)
                return f"{cast('type', command_type).__name__}_{command_hash_fallback}"
            except TypeError:
                # If hash fails, use a deterministic fallback with proper encoding
                encoded_fallback = command_str_fallback.encode(
                    FlextConstants.Utilities.DEFAULT_ENCODING
                )
                return f"{cast('type', command_type).__name__}_{abs(hash(encoded_fallback))}"

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
        if isinstance(obj, dict):
            dict_obj: dict[str, object] = obj
            sorted_items: list[tuple[str, object]] = sorted(
                cast("list[tuple[str, object]]", dict_obj.items()),
                key=lambda x: str(x[0]),
            )
            return {
                str(k): FlextUtilitiesCache.sort_dict_keys(v) for k, v in sorted_items
            }
        if isinstance(obj, list):
            obj_list: list[object] = cast("list[object]", obj)
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


__all__ = ["FlextUtilitiesValidation"]
