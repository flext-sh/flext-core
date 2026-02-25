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
import re
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import cast

from pydantic import (
    TypeAdapter as PydanticTypeAdapter,
    ValidationError as PydanticValidationError,
)

from flext_core._utilities.cache import FlextUtilitiesCache
from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core._utilities.mapper import FlextUtilitiesMapper
from flext_core._utilities.normalize import FlextUtilitiesNormalize
from flext_core._utilities.result_helpers import (
    ResultHelpers as FlextUtilitiesResultHelpers,
)
from flext_core.constants import c
from flext_core.protocols import p
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.typings import t

PENDING_DELETION = True


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
        """Deprecated network validators.

        PENDING REMOVAL: Prefer Annotated aliases from ``t.Validation``:
        ``PortNumber``, ``UriString``, and ``HostnameStr``.
        """

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
            match valid_choices:
                case list():
                    choices_set = set(valid_choices)
                case _:
                    choices_set = valid_choices
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

    normalize_component = staticmethod(FlextUtilitiesNormalize.normalize_component)
    sort_key = staticmethod(FlextUtilitiesNormalize.sort_key)
    sort_dict_keys = staticmethod(FlextUtilitiesNormalize.sort_dict_keys)
    generate_cache_key = staticmethod(
        FlextUtilitiesCache.generate_cache_key_for_command
    )
    _normalize_component = staticmethod(FlextUtilitiesNormalize.normalize_component)
    _guard_check_type = staticmethod(FlextUtilitiesGuards._guard_check_type)
    _guard_check_validator = staticmethod(FlextUtilitiesGuards._guard_check_validator)
    _guard_check_predicate = staticmethod(FlextUtilitiesGuards._guard_check_predicate)
    _guard_check_condition = staticmethod(FlextUtilitiesGuards._guard_check_condition)
    _guard_handle_failure = staticmethod(FlextUtilitiesGuards._guard_handle_failure)

    @staticmethod
    def _guard_non_empty(
        value: t.ConfigMapValue,
        context_name: str,
        error_msg: str | None,
    ) -> str | None:
        if isinstance(value, str | list | dict) and bool(value):
            return None
        return error_msg or f"{context_name} must be non-empty"

    @staticmethod
    def _ensure_general_value_type(component: object) -> t.ConfigMapValue:
        if component is None or isinstance(
            component, str | int | float | bool | datetime
        ):
            return component
        if isinstance(component, Mapping):
            return FlextUtilitiesNormalize._normalize_object_mapping(component)
        if isinstance(component, list):
            return [
                FlextUtilitiesValidation._ensure_general_value_type(item)
                for item in component
            ]
        if isinstance(component, tuple):
            return tuple(
                FlextUtilitiesValidation._ensure_general_value_type(item)
                for item in component
            )
        if isinstance(component, set):
            return tuple(
                sorted(
                    (
                        FlextUtilitiesValidation._ensure_general_value_type(item)
                        for item in component
                    ),
                    key=str,
                )
            )
        return str(component)

    @staticmethod
    def _normalize_mapping(
        value: Mapping[str, t.ConfigMapValue],
        visited: set[int] | None = None,
    ) -> t.ConfigMapValue:
        _ = visited
        return FlextUtilitiesNormalize.sort_dict_keys(value)

    @staticmethod
    def _normalize_set_helper(
        component: set[t.ConfigMapValue],
    ) -> set[t.ConfigMapValue]:
        result: set[t.ConfigMapValue] = set()
        for item in component:
            normalized = FlextUtilitiesNormalize.normalize_component(item)
            if normalized is None or isinstance(normalized, str | int | float | bool):
                result.add(normalized)
            elif isinstance(normalized, tuple):
                try:
                    result.add(normalized)
                except TypeError:
                    result.add(str(normalized))
            else:
                result.add(str(normalized))
        return result

    @staticmethod
    def _normalize_sequence(
        value: tuple[t.ConfigMapValue, ...] | list[t.ConfigMapValue],
        visited: set[int] | None = None,
    ) -> t.ConfigMapValue:
        _ = visited
        return {
            "type": "sequence",
            "data": [
                FlextUtilitiesNormalize.normalize_component(item) for item in value
            ],
        }

    @staticmethod
    def _normalize_set(
        value: set[t.ConfigMapValue],
        visited: set[int] | None = None,
    ) -> t.ConfigMapValue:
        _ = visited
        return {
            "type": "set",
            "data": sorted(
                (FlextUtilitiesNormalize.normalize_component(item) for item in value),
                key=str,
            ),
        }

    @staticmethod
    def _normalize_by_type(
        component: t.ConfigMapValue,
        visited: set[int] | None = None,
    ) -> t.ConfigMapValue:
        if component is None or isinstance(component, str | int | float | bool):
            return component
        if isinstance(component, Mapping):
            return FlextUtilitiesValidation._normalize_mapping(component, visited)
        if isinstance(component, list | tuple):
            return FlextUtilitiesValidation._normalize_sequence(component, visited)
        if isinstance(component, set):
            return FlextUtilitiesValidation._normalize_set(component, visited)
        return FlextUtilitiesValidation._ensure_general_value_type(component)

    @staticmethod
    def _normalize_vars(value: t.ConfigMapValue) -> t.ConfigMapValue:
        try:
            vars_result = vars(value)
        except TypeError:
            return {"type": "repr", "data": repr(value)}

        normalized_vars = {
            str(key): FlextUtilitiesNormalize.normalize_component(val)
            for key, val in sorted(vars_result.items(), key=lambda item: str(item[0]))
        }
        return {"type": "vars", "data": normalized_vars}

    @staticmethod
    def _generate_key_pydantic(
        command: p.HasModelDump,
        command_type: type,
    ) -> str | None:
        try:
            data = command.model_dump()
            if isinstance(data, Mapping):
                typed_data: Mapping[str, t.ConfigMapValue] = {
                    str(k): FlextUtilitiesNormalize.normalize_component(v)
                    for k, v in data.items()
                }
                return FlextUtilitiesCache.generate_cache_key_for_command(
                    typed_data,
                    command_type,
                )
            return None
        except (AttributeError, TypeError, ValueError, RuntimeError, KeyError):
            return None

    @staticmethod
    def _handle_pydantic_model(component: t.ConfigMapValue) -> t.ConfigMapValue:
        model_dump_attr = getattr(component, "model_dump", None)
        if model_dump_attr is not None and callable(model_dump_attr):
            try:
                dump_result = model_dump_attr()
                if isinstance(dump_result, Mapping):
                    return FlextUtilitiesNormalize._normalize_object_mapping(
                        dump_result
                    )
                return str(component)
            except Exception:
                return str(component)
        return component

    @staticmethod
    def _extract_dict_from_component(
        component: Mapping[str, t.ConfigMapValue] | p.HasModelDump,
        _visited: set[int] | None = None,
    ) -> Mapping[str, t.ConfigMapValue]:
        if isinstance(component, Mapping):
            return dict(component)

        items_attr = getattr(component, "items", None)
        if items_attr is None or not callable(items_attr):
            msg = f"Cannot convert {component.__class__.__name__} to dict"
            raise TypeError(msg)

        try:
            items_result = items_attr()
        except Exception as e:
            msg = f"Cannot convert {component.__class__.__name__}.items() to dict"
            raise TypeError(msg) from e

        if isinstance(items_result, Mapping):
            return FlextUtilitiesNormalize._normalize_object_mapping(items_result)

        if isinstance(items_result, list | tuple):
            result: dict[str, t.ConfigMapValue] = {}
            for item in items_result:
                if (
                    not isinstance(item, tuple)
                    or len(item) != c.Performance.EXPECTED_TUPLE_LENGTH
                ):
                    continue
                raw_key = item[0]
                raw_value = item[1]
                if not isinstance(raw_key, str):
                    continue
                result[raw_key] = FlextUtilitiesNormalize.normalize_component(raw_value)
            return result

        msg = f"items() returned non-iterable: {items_result.__class__}"
        raise TypeError(msg)

    @staticmethod
    def _normalize_pydantic_value(value: p.HasModelDump) -> t.ConfigMapValue:
        try:
            dumped: t.ConfigMapValue = value.model_dump()
        except TypeError as e:
            msg = (
                f"Failed to dump Pydantic value: {value.__class__.__name__}: "
                f"{e.__class__.__name__}: {e}"
            )
            raise TypeError(msg) from e
        return {
            "type": "pydantic",
            "data": FlextUtilitiesNormalize.normalize_component(dumped),
        }

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
        return r[bool].ok(value=True)

    @staticmethod
    def initialize(obj: t.ConfigMapValue, field_name: str) -> None:
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
            set(valid_choices) if valid_choices.__class__ is list else valid_choices
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
        flext-ldif, flext-meltano, flext-target-ldif, and flext-oud-mig.
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
            r"^([a-zA-Z][a-zA-Z0-9+.-]*)://([^/?#]+)([^?#]*)(?:\?([^#]*))?(?:#(.*))?$"
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
        value: t.ConfigMapValue,
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
            >>> from flext_core import u
            >>> result = u.validate_callable(lambda x: x + 1)
            >>> result.is_success
            True
            >>> result = u.validate_callable("not callable")
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
        # Type narrowing: value is t.ConfigMapValue, but runtime check ensures callable
        return r[bool].ok(value=True)

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
        codes: list[t.ConfigMapValue],
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
                if code.__class__ in {int, str}:
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
                        f"Invalid HTTP status code type: {code.__class__.__name__}",
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
        return any(exception.__class__ is exc_type for exc_type in retry_on_exceptions)

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
        error_msg = str(exception) or exception.__class__.__name__

        if (
            exception.__class__ in {TimeoutError, concurrent.futures.TimeoutError}
            and timeout_seconds
        ):
            error_msg = f"Operation timed out after {timeout_seconds} seconds"

        return error_msg

    @staticmethod
    def validate_batch_services(
        services: t.ConfigMap,
    ) -> r[t.ConfigMap]:
        """Validate batch services dictionary for container registration.

        Args:
            services: Dictionary of services (name -> service instance).

        Returns:
            r[t.ConfigMap]: Success with services, or failure.

        """
        # Pydantic validation for structure (handled by ConfigMap type)
        # But we need logical validation (reserved names, etc.)

        # Check for empty batch
        if not services.root:
            return r[t.ConfigMap].fail(
                "Batch services cannot be empty",
                error_code=c.Errors.VALIDATION_ERROR,
            )

        # Validate names
        for name in services.root:
            if not FlextUtilitiesGuards.is_type(name, str) or not name.strip():
                return r[t.ConfigMap].fail(
                    f"Invalid service name: '{name}'. Must be non-empty string",
                )

            # Check for reserved names
            if name.startswith("_"):
                return r[t.ConfigMap].fail(
                    f"Service name cannot start with underscore: '{name}'",
                )

        # Validate values (cannot be None)
        for name, service in services.root.items():
            if service is None:
                return r[t.ConfigMap].fail(
                    f"Service '{name}' cannot be None",
                )

            # Check for callable services (should be registered as factories)

        return r[t.ConfigMap].ok(services)

    @staticmethod
    def validate_dispatch_config(
        config: t.ConfigMap | None,
    ) -> r[t.ConfigMap]:
        """Validate dispatch configuration dictionary.

        Args:
            config: Dispatch configuration dictionary.

        Returns:
            r[t.ConfigMap]: Success with config, or failure.

        """
        if config is None:
            return r[t.ConfigMap].fail(
                "Configuration cannot be None",
            )
        # ConfigMap guarantees dict-like structure via RootModel
        # No need for explicit is_dict_like check if type hint is respected
        # But for runtime safety with untyped inputs:
        if not hasattr(config, "get"):
            return r[t.ConfigMap].fail(
                "Configuration must be a dictionary",
            )

        # Validate metadata if present
        metadata = config.get("metadata")
        if metadata is not None and not FlextRuntime.is_dict_like(metadata):
            return r[t.ConfigMap].fail(
                "Metadata must be a dictionary",
            )

        # Validate types for known keys if present
        # Use guards for type checking
        if "correlation_id" in config:
            val = config.get("correlation_id")
            if not FlextUtilitiesGuards.is_type(val, str):
                return r[t.ConfigMap].fail(
                    "Correlation ID must be a string",
                )

        if "timeout" in config:
            val = config.get("timeout")
            if not FlextUtilitiesGuards.is_type(val, (int, float)):
                return r[t.ConfigMap].fail(
                    "Timeout override must be a number",
                )

        return r[t.ConfigMap].ok(config)

    @staticmethod
    def analyze_constructor_parameter(
        param_name: str,
        param: inspect.Parameter,
    ) -> t.ConfigMapValue:
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

        return r[bool].ok(value=True)

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

        return r[bool].ok(value=True)

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

        return r[bool].ok(value=True)

    # ═══════════════════════════════════════════════════════════════════
    # VALIDATION & GUARD HELPERS - Core validation logic
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _validate_get_desc(v: p.ValidatorSpec) -> str:
        """Extract validator description (helper for validate)."""
        # Try to extract description from predicate if it's a Validator (preferred)
        # Use raw dict/attr access to preserve object (avoid FlextUtilitiesMapper.get
        # which converts non-PayloadValue objects to strings)
        predicate: t.ConfigMapValue = None
        if hasattr(v, "predicate"):
            predicate = getattr(v, "predicate", None)
        if predicate is not None and hasattr(predicate, "description"):
            # predicate has description attribute
            predicate_desc = getattr(predicate, "description", None)
            if predicate_desc.__class__ is str and predicate_desc:
                return predicate_desc
        # Fall back to validator's own description
        desc = FlextUtilitiesMapper.get(v, "description", default="validator")
        return desc if FlextUtilitiesGuards.is_type(desc, str) else "validator"

    @staticmethod
    def _validate_check_any[T: t.ConfigMapValue](
        value: T,
        validators: tuple[p.ValidatorSpec, ...],
        field_prefix: str,
    ) -> r[T]:
        """Check if any validator passes (helper for validate)."""
        for validator in validators:
            if validator(value):
                return r.ok(value)
        descriptions = [
            FlextUtilitiesValidation._validate_get_desc(v) for v in validators
        ]
        return r.fail(
            f"{field_prefix}None of the validators passed: {', '.join(descriptions)}",
        )

    @staticmethod
    def _validate_check_all[T: t.ConfigMapValue](
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
            return r.ok(value)

        descriptions = [
            FlextUtilitiesValidation._validate_get_desc(v) for v in failed_validators
        ]
        if fail_fast and not collect_errors:
            first_desc = descriptions[0] if descriptions else None
            error_msg = f"{field_prefix}Validation failed: {first_desc or 'validator'}"
            return r.fail(error_msg)

        def format_error(d: str) -> str:
            """Format validation error message."""
            return f"{field_prefix}Validation failed: {d}"

        errors = [format_error(d) for d in descriptions]
        return r.fail("; ".join(errors))

    @staticmethod
    def validate[T: t.ConfigMapValue](
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
            return r.ok(value)

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
    def check[T: t.ConfigMapValue](value: T, *validators: p.ValidatorSpec) -> r[T]:
        """Check value against validators (all must pass)."""
        result = FlextUtilitiesValidation.validate(value, *validators, mode="all")
        return r.ok(value) if result.is_success else r.fail(result.error or "")

    @staticmethod
    def check_any[T: t.ConfigMapValue](
        value: T,
        *validators: p.ValidatorSpec,
    ) -> r[T]:
        """Check value against validators (any must pass)."""
        result = FlextUtilitiesValidation.validate(value, *validators, mode="any")
        return r.ok(value) if result.is_success else r.fail(result.error or "")

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
    def validate_all[T: t.ConfigMapValue](
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
            _ = datetime.fromisoformat(normalized)
            return True
        except ValueError:
            return False

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
        guard_result = cast(
            "Callable[..., r[T] | T | None]",
            getattr(FlextUtilitiesGuards, "guard_result"),
        )
        return guard_result(
            value,
            *conditions,
            error_message=error_message,
            context=context,
            default=default,
            return_value=return_value,
        )

    @staticmethod
    def ensure(
        value: t.ConfigMapValue,
        *,
        target_type: str = "auto",
        default: str
        | list[t.ConfigMapValue]
        | Mapping[str, t.ConfigMapValue]
        | None = None,
    ) -> str | list[t.ConfigMapValue] | Mapping[str, t.ConfigMapValue]:
        return FlextUtilitiesGuards.ensure(
            value,
            target_type=target_type,
            default=default,
        )

    ResultHelpers: type[FlextUtilitiesResultHelpers] = FlextUtilitiesResultHelpers

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
        def validate[T](data: t.ConfigMapValue, type_: type[T]) -> r[T]:
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
                return r.ok(validated)
            except PydanticValidationError as e:
                error_msg = "; ".join(
                    f"{err['loc']}: {err['msg']}" for err in e.errors()
                )
                return r.fail(f"Validation failed: {error_msg}")

        @staticmethod
        def serialize[T](
            value: T,
            type_: type[T],
        ) -> r[Mapping[str, t.ConfigMapValue]]:
            """Serialize value using TypeAdapter.

            Args:
                value: Value to serialize.
                type_: Type of the value.

            Returns:
                r[Mapping[str, PayloadValue]]: Success with serialized data as dict,
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
                result_dict: Mapping[str, t.ConfigMapValue] = (
                    serialized
                    if serialized.__class__ is dict
                    else {"value": serialized}
                )
                return r[Mapping[str, t.ConfigMapValue]].ok(result_dict)
            except Exception as e:
                return r[Mapping[str, t.ConfigMapValue]].fail(
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
                return r.ok(validated)
            except PydanticValidationError as e:
                error_msg = "; ".join(
                    f"{err['loc']}: {err['msg']}" for err in e.errors()
                )
                return r.fail(f"JSON parsing failed: {error_msg}")
            except Exception as e:
                return r.fail(f"JSON parsing failed: {e}")

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
        value: t.ConfigMapValue,
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
        return r[bool].ok(value=True)

    @staticmethod
    def check_all_validators(
        value: t.ConfigMapValue, *validators: p.ValidatorSpec
    ) -> bool:
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
    def check_any_validator(
        value: t.ConfigMapValue, *validators: p.ValidatorSpec
    ) -> bool:
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
