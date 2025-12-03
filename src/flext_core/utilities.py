"""Utility façade for validation, parsing, and reliability helpers.

**CRITICAL ARCHITECTURE**: u is a THIN FACADE - pure delegation
to _utilities classes. No other module can import from _utilities directly.
All external code MUST use u as the single access point.

This module provides enterprise-grade utility functions for common operations
throughout the FLEXT ecosystem following the single class per module principle.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import builtins
import time
import uuid
from collections.abc import Callable, Collection, Mapping
from enum import StrEnum
from itertools import starmap
from typing import Self, Union, cast, overload

from pydantic import BaseModel

from flext_core._utilities.args import FlextArgs
from flext_core._utilities.cache import FlextCache
from flext_core._utilities.collection import FlextCollection
from flext_core._utilities.configuration import FlextConfiguration
from flext_core._utilities.context import FlextContext
from flext_core._utilities.data_mapper import FlextDataMapper
from flext_core._utilities.domain import FlextDomain
from flext_core._utilities.enum import FlextEnum
from flext_core._utilities.generators import FlextGenerators
from flext_core._utilities.model import FlextModel
from flext_core._utilities.pagination import FlextUtilitiesPagination
from flext_core._utilities.reliability import FlextReliability
from flext_core._utilities.string_parser import FlextStringParser
from flext_core._utilities.text_processor import FlextTextProcessor
from flext_core._utilities.type_checker import FlextTypeChecker
from flext_core._utilities.type_guards import FlextTypeGuards
from flext_core._utilities.validation import FlextValidation
from flext_core._utilities.validators import (
    ValidatorBuilder,
    ValidatorDSL,
    ValidatorSpec,
)
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.typings import t

# Constants for magic numbers
TWO_ARG_COUNT = 2

# Type aliases to avoid conflicts with builder method names
_StrType = builtins.str
_IntType = builtins.int
_BoolType = builtins.bool
_DictType = builtins.dict
_ListType = builtins.list
_SetType = builtins.set


class FlextUtilities:  # noqa: PLR0904
    """Stable utility surface for dispatcher-friendly helpers.

    Provides enterprise-grade utility functions for common operations
    throughout the FLEXT ecosystem. This is a PURE FACADE that delegates
    to _utilities package implementations.

    Architecture: Tier 1.5 (Foundation Utilities)
    ==============================================
    - No nested class definitions (single class per module principle)
    - All attributes reference _utilities classes directly
    - External code uses u.XxxClass.method() pattern
    - No direct _utilities imports allowed outside this module

    Core Namespaces:
    - Enum: StrEnum utilities for type-safe enum handling
    - Collection: Collection conversion utilities
    - Args: Automatic args/kwargs parsing
    - Model: Pydantic model initialization utilities
    - Cache: Data normalization and cache key generation
    - Validation: Comprehensive input validation
    - Generators: ID, UUID, timestamp generation
    - TextProcessor: Text cleaning and processing
    - TypeGuards: Runtime type checking
    - Reliability: Timeout and retry patterns
    - TypeChecker: Runtime type introspection
    - Configuration: Parameter access/manipulation
    - Context: Context variable creation utilities
    - DataMapper: Data mapping and transformation utilities
    - Domain: Domain-specific utilities
    - Pagination: API pagination utilities
    - StringParser: String parsing utilities

    Usage Pattern:
        from flext_core import u
        result = u.Enum.parse(MyEnum, "value")
    """

    # ═══════════════════════════════════════════════════════════════════
    # CLASS-LEVEL ATTRIBUTES: Module References (NOT nested classes)
    # ═══════════════════════════════════════════════════════════════════
    # Each attribute points directly to _utilities class for pure delegation

    Enum = FlextEnum
    Collection = FlextCollection
    Args = FlextArgs
    Model = FlextModel
    Cache = FlextCache
    Validation = FlextValidation
    Generators = FlextGenerators
    TextProcessor = FlextTextProcessor
    TypeGuards = FlextTypeGuards
    Reliability = FlextReliability
    TypeChecker = FlextTypeChecker
    Configuration = FlextConfiguration
    Context = FlextContext
    DataMapper = FlextDataMapper
    Domain = FlextDomain
    Pagination = FlextUtilitiesPagination
    StringParser = FlextStringParser

    # ═══════════════════════════════════════════════════════════════════
    # VALIDATOR DSL: Declarative validation with operator composition
    # ═══════════════════════════════════════════════════════════════════
    # V is the namespace for validator DSL: V.string.non_empty & V.string.email
    # Validator is the builder class: Validator().string().non_empty().build()

    V = ValidatorDSL
    Validator = ValidatorBuilder

    # ═══════════════════════════════════════════════════════════════════
    # POWER METHODS: Direct utility operations with r
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _validate_get_desc(v: ValidatorSpec) -> str:
        """Extract validator description (helper for validate)."""
        desc = u.get(v, "description", default="validator")
        return desc if isinstance(desc, str) else "validator"

    @staticmethod
    def _validate_check_any[T](
        value: T, validators: tuple[ValidatorSpec, ...], field_prefix: str
    ) -> r[T]:
        """Check if any validator passes (helper for validate)."""
        for validator in validators:
            if validator(value):
                return r[T].ok(value)
        descriptions = u.map(validators, FlextUtilities._validate_get_desc)
        return r[T].fail(
            f"{field_prefix}None of the validators passed: "
            f"{u.join(descriptions, sep=', ')}"
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

        failed_validators_list = u.filter(validators, validator_failed)
        failed_validators = cast("list[ValidatorSpec]", failed_validators_list)
        if not failed_validators:
            return r[T].ok(value)

        descriptions = u.map(failed_validators, FlextUtilities._validate_get_desc)
        if fail_fast and not collect_errors:
            first_desc = u.first(descriptions) if descriptions else None
            error_msg = (
                f"{field_prefix}Validation failed: {u.or_(first_desc, 'validator')}"
            )
            return r[T].fail(error_msg)

        def format_error(d: str) -> str:
            """Format validation error message."""
            return f"{field_prefix}Validation failed: {d}"

        errors = u.map(descriptions, format_error)
        return r[T].fail(u.join(errors, sep="; "))

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
            return FlextUtilities._validate_check_any(value, validators, field_prefix)

        return FlextUtilities._validate_check_all(
            value,
            validators,
            field_prefix,
            fail_fast=fail_fast,
            collect_errors=collect_errors,
        )

    @staticmethod
    def _parse_with_default[T](
        default: T | None,
        default_factory: Callable[[], T] | None,
        error_msg: str,
    ) -> r[T]:
        """Return default or error for parse failures.

        Business Rule: Provides fallback mechanism for parse operations.
        Default values allow graceful degradation when parsing fails.
        """
        if default is not None:
            return r[T].ok(default)
        if default_factory is not None:
            return r[T].ok(default_factory())
        return r[T].fail(error_msg)

    @staticmethod
    def _parse_enum[T](
        value: str,
        target: type[T],
        *,
        case_insensitive: bool,
    ) -> r[T] | None:
        """Parse StrEnum with optional case-insensitivity. Returns None if not enum.

        Business Rule: StrEnum parsing supports case-insensitive matching for
        user-friendly input handling. Enum members are matched by value or name.
        """
        if not (isinstance(target, type) and issubclass(target, StrEnum)):
            return None
        # Type narrowing: After issubclass check, target is type[StrEnum]
        # which is type[T]
        # Business Rule: StrEnum classes expose __members__ dict with all enum members
        # Cast to type[StrEnum] to help type checker understand __members__ attribute
        enum_type: type[StrEnum] = cast("type[StrEnum]", target)
        if case_insensitive:
            # Business Rule: StrEnum classes expose __members__ dict with all enum
            # Use u.find() for unified finding with predicate
            # Use u.get() for unified attribute access (DSL pattern)
            members_dict: dict[str, object] = cast(
                "dict[str, object]", u.get(enum_type, "__members__", default={})
            )
            members_list = list(members_dict.values())

            def match_member(member: object) -> bool:
                """Match enum member by value or name."""
                # Use u.has() + getattr() for unified attribute access (DSL pattern)
                if not u.has(member, "value") or not u.has(member, "name"):
                    return False
                # Type narrowing: member has value and name attributes
                # Use getattr directly since u.get() returns object | None
                member_value = getattr(member, "value", None)
                member_name = getattr(member, "name", None)
                if member_value is None or member_name is None:
                    return False
                return bool(
                    u.normalize(member_value, value) or u.normalize(member_name, value)
                )

            found = u.find(members_list, match_member)
            if found is not None:
                # Type narrowing: found is not None, cast to T
                found_enum = cast("T", found)
                return r[T].ok(found_enum)
        result = FlextEnum.parse(target, value)
        if result.is_success:
            return r[T].ok(result.value)
        # Use u.err() for unified error extraction (DSL pattern)
        return r[T].fail(u.err(result, default="Enum parse failed"))

    @staticmethod
    def _parse_model[T](
        value: object,
        target: type[T],
        field_prefix: str,
        *,
        strict: bool,
    ) -> r[T] | None:
        """Parse Pydantic BaseModel. Returns None if not model.

        Business Rule: Pydantic model parsing supports strict and non-strict modes.
        Strict mode requires exact type matching, non-strict allows coercion.
        """
        if not (isinstance(target, type) and issubclass(target, BaseModel)):
            return None
        if not isinstance(value, Mapping):
            return r[T].fail(
                f"{field_prefix}Expected dict for model, got {type(value).__name__}"
            )
        result = FlextModel.from_dict(target, dict(value), strict=strict)
        if result.is_success:
            return r[T].ok(result.value)
        # Use u.err() for unified error extraction (DSL pattern)
        return r[T].fail(u.err(result, default="Model parse failed"))

    @staticmethod
    def _apply_transform_filters(
        data: dict[str, t.GeneralValueType],
        filter_keys: set[str] | None,
        exclude_keys: set[str] | None,
        *,
        strip_none: bool,
        strip_empty: bool,
    ) -> dict[str, t.GeneralValueType]:
        """Helper: Apply all transform filters in sequence using unified filter."""

        def combined_predicate(k: str, v: t.GeneralValueType) -> bool:
            """Combined predicate for all filter conditions."""
            return (
                (filter_keys is None or k in filter_keys)
                and (not exclude_keys or k not in exclude_keys)
                and (not strip_none or v is not None)
                and not (strip_empty and v in ("", [], {}, None))
            )

        filtered = u.filter(data, combined_predicate)
        return filtered if isinstance(filtered, dict) else data

    @staticmethod
    def _coerce_to_int(value: object, *, _strict: bool = False) -> r[int] | None:
        """Helper: Coerce value to int."""
        if isinstance(value, (str, float)):
            try:
                return r[int].ok(int(value))
            except (ValueError, TypeError):
                return None
        return None

    @staticmethod
    def _coerce_to_float(value: object, *, _strict: bool = False) -> r[float] | None:
        """Helper: Coerce value to float."""
        if isinstance(value, (str, int)):
            try:
                return r[float].ok(float(value))
            except (ValueError, TypeError):
                return None
        return None

    @staticmethod
    def _coerce_to_str(value: object) -> r[str]:
        """Helper: Coerce value to str."""
        return r[str].ok(str(value))

    @staticmethod
    def _coerce_to_bool_from_str(value: str) -> r[bool] | None:
        """Helper: Coerce string to bool using u.normalize() and u.find()."""
        normalized_val = cast("str", u.normalize(value, case="lower"))
        # Use u.find() for unified finding
        true_values = {"true", "1", "yes", "on"}
        false_values = {"false", "0", "no", "off"}

        def match_true(val: str) -> bool:
            """Match true value."""
            return val == normalized_val

        def match_false(val: str) -> bool:
            """Match false value."""
            return val == normalized_val

        if u.find(true_values, match_true):
            return r[bool].ok(True)
        if u.find(false_values, match_false):
            return r[bool].ok(False)
        return None

    @staticmethod
    def _coerce_primitive[T](value: object, target: type[T]) -> r[T] | None:
        """Coerce primitive types. Returns None if no coercion applied.

        Business Rule: Primitive type coercion supports common conversions:
        str↔int, str↔float, str↔bool. Boolean coercion recognizes common
        string representations (true/false, yes/no, on/off, 1/0).
        """
        # Use type narrowing for proper type checking
        if target is int:
            int_result = FlextUtilities._coerce_to_int(value)
            if int_result is not None:
                return cast("r[T]", int_result)
        elif target is float:
            float_result = FlextUtilities._coerce_to_float(value)
            if float_result is not None:
                return cast("r[T]", float_result)
        elif target is str:
            str_result = FlextUtilities._coerce_to_str(value)
            if str_result is not None:
                return cast("r[T]", str_result)
        elif target is bool:
            if isinstance(value, str):
                bool_result = FlextUtilities._coerce_to_bool_from_str(value)
                if bool_result is not None:
                    return cast("r[T]", bool_result)
            bool_value = bool(value)
            return cast("r[T]", r[bool].ok(bool_value))
        return None

    @staticmethod
    def _parse_handle_none[T](
        default: T | None,
        default_factory: Callable[[], T] | None,
        field_prefix: str,
    ) -> r[T]:
        """Helper: Handle None value in parse."""
        return FlextUtilities._parse_with_default(
            default, default_factory, f"{field_prefix}Cannot parse None value"
        )

    @staticmethod
    def _parse_handle_already_correct_type[T](value: T, _target: type[T]) -> r[T]:
        """Helper: Handle value already matching target type."""
        return r[T].ok(value)

    @staticmethod
    def _parse_try_enum[T](  # noqa: PLR0913
        value: object,
        target: type[T],
        *,
        case_insensitive: bool,
        default: T | None,
        default_factory: Callable[[], T] | None,
        field_prefix: str,
    ) -> r[T] | None:
        """Helper: Try enum parsing, return None if not enum."""
        enum_result = FlextUtilities._parse_enum(
            str(value), target, case_insensitive=case_insensitive
        )
        if enum_result is None:
            return None
        if enum_result.is_success:
            return enum_result
        return FlextUtilities._parse_with_default(
            default, default_factory, f"{field_prefix}{enum_result.error}"
        )

    @staticmethod
    def _parse_try_model[T](  # noqa: PLR0913
        value: object,
        target: type[T],
        field_prefix: str,
        *,
        strict: bool,
        default: T | None,
        default_factory: Callable[[], T] | None,
    ) -> r[T] | None:
        """Helper: Try model parsing, return None if not model."""
        model_result = FlextUtilities._parse_model(
            value, target, field_prefix, strict=strict
        )
        if model_result is None:
            return None
        if model_result.is_success:
            return model_result
        # Use u.err() for unified error extraction (DSL pattern)
        return FlextUtilities._parse_with_default(
            default, default_factory, u.err(model_result, default="")
        )

    @staticmethod
    def _parse_try_primitive[T](
        value: object,
        target: type[T],
        default: T | None,
        default_factory: Callable[[], T] | None,
        field_prefix: str,
    ) -> r[T] | None:
        """Helper: Try primitive coercion."""
        try:
            prim_result = FlextUtilities._coerce_primitive(value, target)
            if prim_result is not None:
                return prim_result
        except (ValueError, TypeError) as e:
            # Use u.get() for unified attribute access (DSL pattern)
            target_name = u.get(target, "__name__", default="type")
            return FlextUtilities._parse_with_default(
                default,
                default_factory,
                f"{field_prefix}Cannot coerce {type(value).__name__} to "
                f"{target_name}: {e}",
            )
        return None

    @staticmethod
    def _parse_try_direct[T](
        value: object,
        target: type[T],
        default: T | None,
        default_factory: Callable[[], T] | None,
        field_prefix: str,
    ) -> r[T]:
        """Helper: Try direct type call."""
        try:
            # Type narrowing: target is callable and accepts value
            # Use cast to help type checker understand target is callable
            target_callable = cast("Callable[[object], T]", target)
            parsed = target_callable(value)
            return r[T].ok(parsed)
        except Exception as e:
            # Use u.get() for unified attribute access (DSL pattern)
            target_name = u.get(target, "__name__", default="type")
            return FlextUtilities._parse_with_default(
                default,
                default_factory,
                f"{field_prefix}Cannot parse {type(value).__name__} "
                f"to {target_name}: {e}",
            )

    @staticmethod
    def parse[T](  # noqa: PLR0913, PLR0911
        value: object,
        target: type[T],
        *,
        strict: bool = False,
        coerce: bool = True,
        case_insensitive: bool = False,
        default: T | None = None,
        default_factory: Callable[[], T] | None = None,
        field_name: str | None = None,
    ) -> r[T]:
        """Universal type parser supporting enums, models, and primitives.

        Business Rule: Unified parsing interface for all supported types.
        Parsing order: enum → model → primitive coercion → direct type call.
        Case-insensitive enum matching improves user experience.
        Default values allow graceful degradation on parse failure.

        Audit Implication: Parse operations are tracked with field context
        for audit trail completeness. Parse failures include type information
        for debugging and audit purposes.

        Composes existing utilities for unified parsing:
        - Enums: Enum.parse() with optional case-insensitive matching
        - Models: Model.from_dict() for Pydantic BaseModel
        - Primitives: Direct type coercion (str→int, int→str, etc.)

        Args:
            value: The value to parse.
            target: Target type to parse into.
            strict: If True, disable type coercion (exact match only).
            coerce: If True (default), allow type coercion.
            case_insensitive: For enums, match case-insensitively.
            default: Default value to return on parse failure.
            default_factory: Callable to create default on failure.
            field_name: Field name for error messages.

        Returns:
            r[T]: Ok(parsed_value) or Fail with error message.

        Examples:
            # Enum parsing
            result = u.parse("ACTIVE", Status)

            # Case-insensitive enum
            result = u.parse("active", Status, case_insensitive=True)

            # Pydantic model from dict
            result = u.parse({"name": "John"}, UserModel)

            # Primitive coercion
            result = u.parse("42", int)  # Ok(42)

            # With default
            result = u.parse("invalid", int, default=0)

        """
        field_prefix = f"{field_name}: " if field_name else ""

        # Handle None value using u.when() for DSL pattern
        if value is None:
            if default is not None:
                return r[T].ok(default)
            if default_factory is not None:
                return r[T].ok(default_factory())
            return r[T].fail(field_prefix or "Value is None")

        # Already the target type
        if isinstance(value, target):
            return FlextUtilities._parse_handle_already_correct_type(value, target)

        # Try enum parsing
        enum_result = FlextUtilities._parse_try_enum(
            value,
            target,
            case_insensitive=case_insensitive,
            default=default,
            default_factory=default_factory,
            field_prefix=field_prefix,
        )
        if enum_result is not None:
            return enum_result

        # Try model parsing
        model_result = FlextUtilities._parse_try_model(
            value,
            target,
            field_prefix,
            strict=strict,
            default=default,
            default_factory=default_factory,
        )
        if model_result is not None:
            return model_result

        # Try primitive coercion
        if coerce and not strict:
            prim_result = FlextUtilities._parse_try_primitive(
                value, target, default, default_factory, field_prefix
            )
            if prim_result is not None:
                return prim_result

        # Direct type call as last resort
        return FlextUtilities._parse_try_direct(
            value, target, default, default_factory, field_prefix
        )

    @staticmethod
    def transform(  # noqa: PLR0913
        data: Mapping[str, t.GeneralValueType],
        *,
        normalize: bool = False,
        strip_none: bool = False,
        strip_empty: bool = False,
        map_keys: dict[str, str] | None = None,
        filter_keys: set[str] | None = None,
        exclude_keys: set[str] | None = None,
        to_json: bool = False,
        to_model: type[BaseModel] | None = None,
    ) -> r[dict[str, t.GeneralValueType]]:
        """Transform data with normalization, filtering, and conversion.

        Business Rule: Composes multiple transformation operations in sequence.
        Normalization ensures consistent data formatting. Key mapping supports
        schema evolution. Filtering and exclusion support data sanitization.
        JSON conversion ensures serializability. Model conversion validates structure.

        Audit Implication: Transform operations preserve data lineage through
        transformation chain. Key mappings are tracked for audit purposes.

        Composes existing utilities for unified transformation:
        - Cache.normalize_component() for normalization
        - DataMapper.map_dict_keys() for key mapping
        - DataMapper.convert_dict_to_json() for JSON conversion
        - Model.from_dict() for model parsing

        Args:
            data: Input data to transform (dict or Mapping).
            normalize: Apply Cache.normalize_component for consistent formatting.
            strip_none: Remove keys with None values.
            strip_empty: Remove keys with empty strings/lists/dicts.
            map_keys: Dict mapping old keys to new keys.
            filter_keys: Set of keys to keep.
            exclude_keys: Set of keys to remove.
            to_json: Convert output to JSON-serializable dict.
            to_model: Parse output to Pydantic model (returns model, not dict).

        Returns:
            r[dict[str, GeneralValueType]]: Transformed data or error.

        Examples:
            # Normalize and filter
            result = u.transform(
                data,
                normalize=True,
                filter_keys={"name", "email"},
            )

            # Map keys and strip None
            result = u.transform(
                data,
                map_keys={"old_name": "new_name"},
                strip_none=True,
            )

            # Convert to JSON-serializable
            result = u.transform(data, to_json=True)

            # Transform to Pydantic model
            result = u.transform(
                raw_data,
                to_model=UserModel,
                strip_none=True,
            )

        """
        return FlextUtilities._transform_apply(
            data,
            normalize=normalize,
            map_keys=map_keys,
            filter_keys=filter_keys,
            exclude_keys=exclude_keys,
            strip_none=strip_none,
            strip_empty=strip_empty,
            to_json=to_json,
            to_model=to_model,
        )

    @staticmethod
    def _transform_apply(  # noqa: PLR0913
        data: Mapping[str, t.GeneralValueType],
        *,
        normalize: bool,
        map_keys: dict[str, str] | None,
        filter_keys: set[str] | None,
        exclude_keys: set[str] | None,
        strip_none: bool,
        strip_empty: bool,
        to_json: bool,
        to_model: type[BaseModel] | None,
    ) -> r[dict[str, t.GeneralValueType]]:
        """Internal helper for transform operations."""
        try:
            result: dict[str, t.GeneralValueType] = dict(data)

            # Normalize
            if normalize:
                normalized = FlextCache.normalize_component(result)
                if isinstance(normalized, dict):
                    result = normalized

            # Map keys
            if map_keys:
                mapped = FlextDataMapper.map_dict_keys(result, map_keys)
                result = mapped.value

            # Apply filters using unified filter method
            result = FlextUtilities._apply_transform_filters(
                result,
                filter_keys,
                exclude_keys,
                strip_none=strip_none,
                strip_empty=strip_empty,
            )

            # Convert to JSON
            if to_json:
                result = FlextDataMapper.convert_dict_to_json(result)

            # Parse to model
            if to_model is not None:
                # Type narrowing: to_model is BaseModel subclass, result is dict
                # Convert result to dict for from_dict compatibility
                # (FlextModel.from_dict accepts dict)
                result_dict = dict(result)
                # Convert dict[str, GeneralValueType] to dict[str, FlexibleValue]
                # for from_dict
                result_flexible: dict[str, t.FlexibleValue] = {}
                for key, val in result_dict.items():
                    # GeneralValueType is compatible with FlexibleValue
                    # (both include base types)
                    result_flexible[key] = cast("t.FlexibleValue", val)
                model_result = FlextModel.from_dict(to_model, result_flexible)
                if model_result.is_failure:
                    # Use u.err() for unified error extraction (DSL pattern)
                    return r[dict[str, t.GeneralValueType]].fail(
                        u.err(model_result, default="Model conversion failed")
                    )
                # Return model as dict representation
                result = model_result.value.model_dump()

            return r[dict[str, t.GeneralValueType]].ok(result)

        except Exception as e:
            return r[dict[str, t.GeneralValueType]].fail(f"Transform failed: {e}")

    @staticmethod
    def pipe(
        value: object,
        *operations: Callable[[object], object],
        on_error: str = "stop",
    ) -> r[object]:
        """Functional pipeline with railway-oriented error handling.

        Business Rule: Chains operations sequentially, unwrapping r
        values automatically. Error handling modes: "stop" (fail fast) or
        "skip" (continue with previous value). Railway pattern ensures errors
        propagate correctly through the pipeline.

        Args:
            value: Initial value to process
            *operations: Functions to apply in sequence
            on_error: Error handling ("stop" or "skip")

        Returns:
            r containing final value or error

        Example:
            result = u.pipe(
                "  hello world  ",
                str.strip,
                str.upper,
                lambda s: s.replace(" ", "_"),
            )
            # → r.ok("HELLO_WORLD")

        """
        if not operations:
            return r[object].ok(value)

        current: object = value
        for i, op in enumerate(operations):
            try:
                result = op(current)

                # Unwrap r if returned
                if isinstance(result, r):
                    if result.is_failure:
                        if on_error == "stop":
                            # Use u.err() for unified error extraction (DSL pattern)
                            return r[object].fail(
                                f"Pipeline step {i} failed: "
                                f"{u.err(result, default='Unknown error')}"
                            )
                        # on_error == "skip": continue with previous value
                        continue
                    current = result.value
                else:
                    current = result

            except Exception as e:
                if on_error == "stop":
                    return r[object].fail(f"Pipeline step {i} failed: {e}")
                # on_error == "skip": continue with previous value

        return r[object].ok(current)

    @staticmethod
    def _merge_should_include(
        v: object, *, filter_none: bool, filter_empty: bool
    ) -> bool:
        """Helper: Check if value should be included in merge."""
        if filter_none and v is None:
            return False
        return not (filter_empty and v in ("", [], {}))

    @staticmethod
    def _merge_deep_merge_dicts(
        base: dict[str, object],
        overlay: dict[str, object],
        strategy: str,
        should_include: Callable[[object], bool],
    ) -> dict[str, object]:
        """Helper: Deep merge two dictionaries."""
        result = dict(base)
        for key, value in overlay.items():
            if not should_include(value):
                continue
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                base_dict: dict[str, object] = dict(
                    cast("dict[str, object]", result[key])
                )
                overlay_dict: dict[str, object] = dict(cast("dict[str, object]", value))
                result[key] = FlextUtilities._merge_deep_merge_dicts(
                    base_dict, overlay_dict, strategy, should_include
                )
            elif (
                strategy == "append"
                and key in result
                and isinstance(result[key], list)
                and isinstance(value, list)
            ):
                # Type narrowing: both are lists
                existing_list = cast("list[object]", result[key])
                new_list = cast("list[object]", value)
                result[key] = existing_list + new_list
            else:
                result[key] = value
        return result

    @staticmethod
    def merge(
        *dicts: Mapping[str, t.GeneralValueType],
        strategy: str = "deep",
        filter_none: bool = False,
        filter_empty: bool = False,
    ) -> r[dict[str, t.GeneralValueType]]:
        """Intelligent dictionary merging with conflict resolution.

        Business Rule: Merges dictionaries with configurable strategies:
        "override" (later wins), "deep" (nested merge), "append" (list concatenation).
        Filtering options support data sanitization. Deep merge preserves nested
        structures while allowing overrides.

        Args:
            *dicts: Dictionaries to merge (later ones override earlier)
            strategy: Merge strategy ("override", "deep", "append")
            filter_none: Skip None values
            filter_empty: Skip empty strings/lists/dicts

        Returns:
            r containing merged dictionary

        Example:
            result = u.merge(
                {"a": 1, "b": {"x": 1}},
                {"b": {"y": 2}, "c": 3},
                strategy="deep",
            )
            # → r.ok({"a": 1, "b": {"x": 1, "y": 2}, "c": 3})

        """
        if not dicts:
            return r[dict[str, t.GeneralValueType]].ok({})

        def should_include_fn(v: object) -> bool:
            return FlextUtilities._merge_should_include(
                v, filter_none=filter_none, filter_empty=filter_empty
            )

        try:
            merged: dict[str, object] = {}
            for d in dicts:
                # Use u.filter() for unified filtering
                filtered = cast(
                    "dict[str, object]",
                    u.filter(d, lambda _key, val: should_include_fn(val)),
                )
                if strategy == "override":
                    merged.update(filtered)
                else:  # deep or append
                    merged = FlextUtilities._merge_deep_merge_dicts(
                        merged,
                        filtered,
                        strategy,
                        should_include_fn,
                    )

            return r[dict[str, t.GeneralValueType]].ok(
                cast("dict[str, t.GeneralValueType]", merged)
            )

        except Exception as e:
            return r[dict[str, t.GeneralValueType]].fail(f"Merge failed: {e}")

    @staticmethod
    def _extract_parse_array_index(part: str) -> tuple[str, str | None]:
        """Helper: Parse array index from path part (e.g., "items[0]")."""
        if "[" in part and part.endswith("]"):
            bracket_pos = part.index("[")
            array_match = part[bracket_pos + 1 : -1]
            key_part = part[:bracket_pos]
            return key_part, array_match
        return part, None

    @staticmethod
    def _extract_get_value[T](
        current: object,
        key_part: str,
        path_context: str,
        *,
        required: bool,
        default: T | None,
    ) -> r[T | None] | None:
        """Helper: Get value from dict/object/model."""

        # Helper to create error or default result
        def error_or_default(msg: str) -> r[T | None]:
            return r[T | None].fail(msg) if required else r[T | None].ok(default)

        result: r[T | None] | None = None

        # Handle Mapping (dict)
        if isinstance(current, Mapping):
            result = (
                r[T | None].ok(cast("T | None", current[key_part]))
                if key_part in current
                else error_or_default(f"Key '{key_part}' not found at '{path_context}'")
            )
        # Handle object attribute
        # Use u.has() + u.get() for unified attribute access (DSL pattern)
        elif u.has(current, key_part):
            result = r[T | None].ok(cast("T | None", u.get(current, key_part)))
        # Handle Pydantic model
        elif u.has(current, "model_dump"):
            model_dump_attr: Callable[[], dict[str, object]] | None = cast(
                "Callable[[], dict[str, object]] | None",
                u.get(current, "model_dump", default=None),
            )
            if model_dump_attr is None:
                result = error_or_default(
                    f"Cannot access '{key_part}' at '{path_context}'"
                )
            else:
                # Type narrowing: model_dump_attr is callable
                model_dump_method = model_dump_attr
                model_dict = model_dump_method()
                result = (
                    r[T | None].ok(cast("T | None", model_dict[key_part]))
                    if key_part in model_dict
                    else error_or_default(
                        f"Key '{key_part}' not found at '{path_context}'"
                    )
                )
        # Cannot access
        else:
            result = error_or_default(f"Cannot access '{key_part}' at '{path_context}'")

        return result

    @staticmethod
    def _extract_handle_array_index[T](
        current: object,
        array_match: str,
        key_part: str,
        *,
        required: bool,
        default: T | None,
    ) -> r[T | None]:
        """Helper: Handle array indexing."""
        if not isinstance(current, (list, tuple)):
            if required:
                return r[T | None].fail(f"'{key_part}' is not a sequence")
            return r[T | None].ok(default)
        try:
            idx = int(array_match)
            return r[T | None].ok(cast("T | None", current[idx]))
        except (ValueError, IndexError):
            if required:
                return r[T | None].fail(
                    f"Invalid index '{array_match}' for '{key_part}'"
                )
            return r[T | None].ok(default)

    @staticmethod
    def extract[T](
        data: Mapping[str, object] | object,
        path: str,
        *,
        default: T | None = None,
        required: bool = False,
        separator: str = ".",
    ) -> r[T | None]:
        """Safe nested data extraction with dot notation.

        Business Rule: Extracts nested values using dot notation paths.
        Supports dict access, object attributes, and Pydantic model fields.
        Array indexing supported via "key[0]" syntax. Required mode fails
        if path not found, otherwise returns default.

        Args:
            data: Source data (dict, object with attributes, or Pydantic model)
            path: Dot-separated path (e.g., "user.profile.name")
            default: Default value if path not found
            required: Fail if path not found
            separator: Path separator (default: ".")

        Returns:
            r containing extracted value or default

        Example:
            config = {"database": {"host": "localhost", "port": 5432}}
            result = u.extract(config, "database.port")
            # → r.ok(5432)

        """
        try:
            parts = path.split(separator)
            current: object = data

            for i, part in enumerate(parts):
                if current is None:
                    if required:
                        return r[T | None].fail(
                            f"Path '{separator.join(parts[:i])}' is None"
                        )
                    return r[T | None].ok(default)

                # Handle array indexing (e.g., "items[0]")
                key_part, array_match = FlextUtilities._extract_parse_array_index(part)

                # Get value from dict, object, or Pydantic model
                path_context = separator.join(parts[:i])
                get_result = FlextUtilities._extract_get_value(
                    current,
                    key_part,
                    path_context,
                    required=required,
                    default=default,
                )
                if get_result is None:
                    continue
                if get_result.is_failure:
                    # Type narrowing: get_result is r[object], return as r[T | None]
                    return r[T | None].fail(get_result.error or "Extraction failed")
                current = get_result.value

                # Handle array index
                if array_match is not None:
                    array_result = FlextUtilities._extract_handle_array_index(
                        current,
                        array_match,
                        key_part,
                        required=required,
                        default=default,
                    )
                    if array_result.is_failure:
                        # Type narrowing: array_result is r[object]
                        # return as r[T | None]
                        return r[T | None].fail(
                            array_result.error or "Array extraction failed"
                        )
                    current = array_result.value

            return r[T | None].ok(cast("T | None", current))

        except Exception as e:
            return r[T | None].fail(f"Extract failed: {e}")

    @staticmethod
    def generate(  # noqa: PLR0913
        kind: str = "id",
        *,
        prefix: str | None = None,
        length: int | None = None,
        include_timestamp: bool = False,
        separator: str = "_",
        microseconds: bool = False,
        milliseconds: bool = False,
    ) -> str:
        """Unified ID and timestamp generation with domain-driven prefixes.

        Business Rule: Generates IDs with domain-specific prefixes for
        traceability. Supports UUID v4 for globally unique IDs and short
        IDs for compact representations. Timestamp inclusion supports
        chronological ordering.

        Args:
            kind: Type to generate:
                - "id": Generic short ID (UUID v4)
                - "uuid": Full UUID v4
                - "timestamp": ISO timestamp string
                - "correlation": Request tracing ID
                - "entity": Domain entity ID
                - "batch": Batch operation ID
                - "transaction": Transaction ID
                - "event": Domain event ID
                - "command": CQRS command ID
                - "query": CQRS query ID
            prefix: Custom prefix (overrides default)
            length: ID length for short IDs
            include_timestamp: Include timestamp in ID
            separator: Separator character
            microseconds: Include microseconds in timestamp (only for kind="timestamp")
            milliseconds: Include milliseconds in timestamp (only for kind="timestamp")

        Returns:
            Generated string (ID or timestamp)

        Example:
            id = u.generate("entity", prefix="user")
            # → "user_a1b2c3d4"
            timestamp = u.generate("timestamp")
            # → "2025-01-15T10:30:00"
            timestamp_ms = u.generate("timestamp", milliseconds=True)
            # → "2025-01-15T10:30:00.123456"

        """
        # Handle timestamp generation directly with proper typing
        if kind == "timestamp":
            dt = FlextGenerators.generate_datetime_utc()
            if microseconds:
                # Full precision with microseconds (6 decimal places)
                iso_str = dt.isoformat()
                # Ensure UTC timezone indicator
                if not iso_str.endswith("+00:00") and not iso_str.endswith("Z"):
                    iso_str = iso_str.replace("+00:00", "") + "Z"
                return iso_str
            if milliseconds:
                # Format with milliseconds (3 decimal places)
                return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            # Default: no microseconds/milliseconds (seconds precision)
            return dt.replace(microsecond=0).isoformat() + "Z"

        default_prefixes = {
            "id": "",
            "uuid": "",
            "correlation": "corr",
            "entity": "ent",
            "batch": "batch",
            "transaction": "txn",
            "event": "evt",
            "command": "cmd",
            "query": "qry",
        }

        actual_prefix = prefix if prefix is not None else default_prefixes.get(kind, "")

        # Generate base ID
        if kind == "uuid":
            base_id = str(uuid.uuid4())
        elif kind == "id":
            base_id = FlextGenerators.generate_id()
        else:
            base_id = FlextGenerators.generate_short_id(length or 8)

        # Build parts
        parts: list[str] = []
        if actual_prefix:
            parts.append(actual_prefix)
        if include_timestamp:
            ts = FlextGenerators.generate_iso_timestamp()
            parts.append(ts[:10].replace("-", ""))  # YYYYMMDD
        parts.append(base_id)

        return separator.join(parts)

    @staticmethod
    def _batch_process_single_item[T, R](
        item: T,
        idx: int,
        operation: Callable[[T], R | r[R]],
        errors: list[tuple[int, str]],
        on_error: str,
    ) -> R | r[t.Types.BatchResultDict] | None:
        """Helper: Process a single batch item.

        Returns result, error Result, or None if skipped.
        """
        try:
            result = operation(item)
            if isinstance(result, r):
                if result.is_failure:
                    # Use u.err() for unified error extraction (DSL pattern)
                    error_msg = u.err(result, default="Unknown error")
                    error_text = f"Item {idx} failed: {error_msg}"
                    if on_error == "fail":
                        return r[t.Types.BatchResultDict].fail(error_text)
                    if on_error == "collect":
                        errors.append((idx, error_msg))
                    return None  # Skip this item
                return result.value
            return result
        except Exception as e:
            error_msg = str(e)
            error_text = f"Item {idx} failed: {error_msg}"
            if on_error == "fail":
                return r[t.Types.BatchResultDict].fail(error_text)
            if on_error == "collect":
                errors.append((idx, error_msg))
            return None  # Skip this item

    @staticmethod
    def _batch_flatten_results(
        validated_results: list[t.GeneralValueType],
        *,
        flatten: bool,
    ) -> list[t.GeneralValueType]:
        """Helper: Flatten nested lists if requested using u.flat()."""
        if not flatten:
            return validated_results

        # Filter to get only list/tuple items, then use u.flat()
        def is_list_or_tuple(item: object) -> bool:
            """Check if item is list or tuple."""
            return isinstance(item, (list, tuple))

        nested = cast(
            "list[list[t.GeneralValueType] | tuple[t.GeneralValueType, ...]]",
            u.filter(validated_results, is_list_or_tuple),
        )
        if not nested:
            return validated_results
        # Use u.flat() for unified flattening
        flattened: list[t.GeneralValueType] = u.flat(nested)
        # Add non-list items
        non_list = cast(
            "list[t.GeneralValueType]",
            u.filter(validated_results, lambda x: not is_list_or_tuple(x)),
        )
        return flattened + non_list

    @staticmethod
    def batch[T, R](  # noqa: PLR0912, PLR0913, C901  # Too many branches required for batch processing
        items: list[T],
        operation: Callable[[T], R | r[R]],
        *,
        _size: int = 100,  # Reserved for future chunking support
        on_error: str = "collect",
        _parallel: bool = False,  # Reserved for future async support
        progress: Callable[[int, int], None]
        | None = None,  # Progress callback (current, total)
        progress_interval: int = 1,  # Reserved for future chunking  # noqa: ARG004
        pre_validate: Callable[[T], bool] | None = None,  # Pre-validation filter
        post_validate: Callable[[R], bool] | None = None,  # Post-validation filter
        flatten: bool = False,  # Flatten nested lists in results
    ) -> r[t.Types.BatchResultDict]:
        """Batch processing with chunking and error handling.

        Business Rule: Processes items in batch with configurable error handling.
        Supports three error modes: "skip" (continue), "fail" (stop on first error),
        "collect" (continue and collect errors). Results are returned in a structured
        TypedDict with results, errors, and counts for audit trail completeness.

        Uses unified `process()` function internally for consistent processing.

        Args:
            items: Items to process
            operation: Function to apply to each item (can return r or direct value)
            _size: Reserved for future chunking support (not yet implemented)
            on_error: Error handling mode ("skip", "fail", "collect")
            _parallel: Reserved for future async/parallel support (not yet implemented)
            progress: Optional callback(current_index, total_count) called during processing
            progress_interval: Call progress callback every N items (default: 1)
            pre_validate: Optional validator function(item) -> bool to filter items before processing
            post_validate: Optional validator function(result) -> bool to filter results after processing
            flatten: If True, flatten nested lists in results (default: False)

        Returns:
            r[t.Types.BatchResultDict]: Dict with keys:
                - results: list[GeneralValueType] - Successful operation results
                - errors: list[tuple[int, str]] - (index, error_message) for failures
                - total: int - Total items processed
                - success_count: int - Number of successful operations
                - error_count: int - Number of failed operations

        Example:
            result = u.batch(
                [1, 2, 3],
                lambda x: x * 2,
                on_error="collect",
            )
            if result.is_success:
                batch_data = result.value
                assert batch_data["results"] == [2, 4, 6]
                assert batch_data["total"] == 3

        """
        total_items = len(items)

        # Pre-filter items if pre_validate provided
        if pre_validate is not None:
            filtered_result = u.filter(items, pre_validate)
            # Type narrowing: u.filter() returns list[T] | dict[str, T] for list/dict inputs
            # For list inputs, filter returns list; for dict inputs, filter returns dict
            # Since items is list[T] | dict[str, T], filtered_result is list[T] | dict[str, T]
            if isinstance(filtered_result, list):
                filtered_items: list[T] = filtered_result
            elif isinstance(filtered_result, dict):
                # For dict, convert values to list
                filtered_items = list(filtered_result.values())
            else:
                # Fallback: should not happen, but handle gracefully
                filtered_items = list(items) if isinstance(items, list) else []
        else:
            filtered_items = items

        # Process items directly to collect errors properly for batch format
        # Note: Cannot use u.process() here because batch needs per-item error tracking
        # with index information for BatchResultDict format
        errors: list[tuple[int, str]] = []
        processed_results: list[R] = []

        for idx, item in enumerate(filtered_items):
            process_result = FlextUtilities._batch_process_single_item(
                item, idx, operation, errors, on_error
            )
            if process_result is None:
                continue  # Item skipped
            if isinstance(process_result, r):
                return process_result  # Fail mode returned error
            processed_results.append(process_result)

        # Post-validate and filter results using filter()
        validated_results_raw: list[R]
        if post_validate is not None:
            filtered_validation = u.filter(processed_results, post_validate)
            # Type narrowing: u.filter() on list[R] returns list[R]
            # processed_results is list[R], so filter returns list[R]
            if isinstance(filtered_validation, list):
                validated_results_raw = filtered_validation
            elif isinstance(filtered_validation, dict):
                # Should not happen for list input, but handle gracefully
                validated_results_raw = list(filtered_validation.values())
            else:
                # Fallback: should not happen, but handle gracefully
                validated_results_raw = processed_results
        else:
            validated_results_raw = processed_results

        # Convert to GeneralValueType for flattening using u.map
        def to_general_value(item: object) -> t.GeneralValueType:
            """Convert item to GeneralValueType."""
            return cast("t.GeneralValueType", item)

        validated_results_raw_list = cast("list[object]", validated_results_raw)
        validated_results = u.map(validated_results_raw_list, to_general_value)

        # Flatten nested lists if requested
        flattened_results = FlextUtilities._batch_flatten_results(
            validated_results, flatten=flatten
        )

        # Call progress callback if provided
        if progress is not None:
            progress(total_items, total_items)
            # Note: progress_interval is reserved for future chunking support

        batch_result: t.Types.BatchResultDict = {
            "results": flattened_results,
            "errors": errors,
            "total": total_items,
            "success_count": len(flattened_results),
            "error_count": len(errors),
        }

        return r[t.Types.BatchResultDict].ok(batch_result)

    @staticmethod
    def retry[T](  # noqa: PLR0913
        operation: Callable[[], T],
        *,
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff: str = "exponential",
        backoff_factor: float = 2.0,
        retry_on: tuple[type[Exception], ...] = (Exception,),
    ) -> r[T]:
        """Retry an operation with configurable backoff.

        Business Rule: Retries operations with exponential or linear backoff.
        Exponential backoff reduces load on failing systems. Linear backoff
        provides predictable retry intervals. Retry-on-exception filtering
        allows selective retry behavior.

        Args:
            operation: Callable to execute
            max_attempts: Maximum number of attempts
            delay: Initial delay between retries (seconds)
            backoff: Backoff strategy ("none", "linear", "exponential")
            backoff_factor: Multiplier for backoff
            retry_on: Exception types to retry on

        Returns:
            r with operation result or error

        Example:
            result = u.retry(
                fetch_data,
                max_attempts=3,
                backoff="exponential",
            )

        """
        last_error: Exception | None = None
        current_delay = delay

        for attempt in range(max_attempts):
            try:
                result = operation()
                return r[T].ok(result)
            except retry_on as e:
                last_error = e
                if attempt < max_attempts - 1:
                    time.sleep(current_delay)
                    # Calculate next delay
                    if backoff == "linear":
                        current_delay = delay * (attempt + 2)
                    elif backoff == "exponential":
                        current_delay = delay * (backoff_factor ** (attempt + 1))
                    # "none": keep same delay

        return r[T].fail(
            f"Operation failed after {max_attempts} attempts: {last_error}"
        )

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
                - If return_value=False: r[T] (Ok(value) or Fail)
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
            check_result = FlextUtilities._guard_check_condition(
                cast("object", value),
                condition_typed,
                context_name,
                error_msg,
            )
            if check_result is not None:
                return FlextUtilities._guard_handle_failure(
                    check_result, return_value=return_value, default=default
                )

        # Return value directly if return_value=True, otherwise return Result
        if return_value:
            return value
        return r[T].ok(value)

    @staticmethod
    @overload
    def get(
        data: Mapping[str, object] | object,
        key: str,
        *,
        default: str = "",
    ) -> str: ...

    @staticmethod
    @overload
    def get[T](
        data: Mapping[str, object] | object,
        key: str,
        *,
        default: list[T],
    ) -> list[T]: ...

    @staticmethod
    @overload
    def get[T](
        data: Mapping[str, object] | object,
        key: str,
        *,
        default: T | None = None,
    ) -> T | None: ...

    @staticmethod
    def get[T](
        data: Mapping[str, object] | object,
        key: str,
        *,
        default: T | None = None,
    ) -> T | None:
        """Unified get function for dict/object access with default.

        Generic replacement for: get_or_default(), get_str(), get_list()

        Automatically detects if data is dict or object and extracts value.
        Uses DSL conversion when default type indicates desired return type.

        Args:
            data: Source data (dict or object)
            key: Key/attribute name
            default: Default value if not found
                - str (e.g., "") -> returns str (generalized from get_str)
                - list[T] (e.g., []) -> returns list[T] (generalized from get_list)
                - Other -> returns T | None

        Returns:
            Extracted value or default (type inferred from default)

        Example:
            # String (generalized from get_str)
            name = u.get(data, "name", default="")

            # List (generalized from get_list)
            models = u.get(data, "models", default=[])

            # Generic
            port = u.get(config, "port", default=8080)

        """
        # Handle string default (generalized from get_str)
        # Check if default is empty string (type hint for string return)
        # Note: isinstance check needed because default can be None
        if isinstance(default, str) and not default:
            value = FlextUtilities._get_raw(data, key, default=default)
            build_ops_str: dict[str, object] = {
                "ensure": "str",
                "ensure_default": default,
            }
            build_result = u.build(value, ops=build_ops_str)
            if isinstance(build_result, str):
                return build_result  # type: ignore[return-value]  # T=str when default is str
            return default  # type: ignore[return-value]  # T=str when default is str

        # Handle list default (generalized from get_list)
        # Check if default is empty list (type hint for list return)
        if isinstance(default, list) and len(default) == 0:
            value_raw = FlextUtilities._get_raw(data, key, default=default)
            build_ops_list: dict[str, object] = {
                "ensure": "list",
                "ensure_default": default,
            }
            build_result = u.build(value_raw, ops=build_ops_list)
            if isinstance(build_result, list):
                return build_result  # type: ignore[return-value]  # T=list when default is list
            return default  # type: ignore[return-value]  # T=list when default is list

        # Generic get (original behavior)
        return FlextUtilities._get_raw(data, key, default=default)

    @staticmethod
    def _get_raw[T](
        data: Mapping[str, object] | object,
        key: str,
        *,
        default: T | None = None,
    ) -> T | None:
        """Internal helper for raw get without DSL conversion."""
        match data:
            case dict() | Mapping():
                if hasattr(data, "get"):
                    # Type narrowing: data has get method
                    data_with_get = cast("Mapping[str, object]", data)
                    result = data_with_get.get(key, default)
                    return cast("T | None", result)
                return default
            case _:
                return getattr(data, key, default)

    # Backward compatibility alias for get_or_default
    get_or_default = get

    @staticmethod
    def find[T](
        items: list[T]
        | tuple[T, ...]
        | set[T]
        | frozenset[T]
        | dict[str, T]
        | Mapping[str, T],
        predicate: Callable[[T], bool] | Callable[[str, T], bool],
        *,
        return_key: bool = False,
    ) -> T | tuple[str, T] | None:
        """Unified find function that auto-detects input type.

        Generic replacement for: find_in_dict()

        Automatically detects if input is list/tuple/dict and finds first matching item.

        Args:
            items: Input items (list, tuple, or dict)
            predicate: Function to test items
                - For lists: predicate(item) -> bool
                - For dicts: predicate(key, value) -> bool
            return_key: If True, returns (key, value) tuple for dicts

        Returns:
            Found value or (key, value) tuple or None if not found

        Example:
            # Find in list
            value = u.find([1, 2, 3], lambda x: x > 1)
            # → 2

            # Find in dict
            value = u.find({"a": 1, "b": 2}, lambda k, v: v > 1)
            # → 2

            key, value = u.find(
                {"a": 1, "b": 2}, lambda k, v: v > 1, return_key=True
            )
            # → ("b", 2)

        """
        match items:
            case list() | tuple() | set() | frozenset():
                list_predicate = cast("Callable[[T], bool]", predicate)
                for item in items:
                    if list_predicate(item):
                        return item
            case dict() | Mapping():
                dict_predicate = cast("Callable[[str, T], bool]", predicate)
                for key, value in items.items():
                    if dict_predicate(key, value):
                        return (key, value) if return_key else value
        return None

    @staticmethod
    def _filter_list[T, R](
        items_list: list[T],
        predicate: Callable[[T], bool] | Callable[[R], bool],
        mapper: Callable[[T], R] | None = None,
    ) -> list[T] | list[R]:
        """Filter a list with optional mapping.

        Business Rule: When mapper is provided, filtering operates on mapped values.
        When mapper is None, filtering operates directly on original values.

        This internal helper is called by u.filter() for list inputs. Uses direct
        list comprehension to avoid recursion (cannot call u.filter from here).
        """
        if mapper is not None:
            # Map first, then filter: T -> R, then filter on R
            mapped_raw = u.map(items_list, mapper)
            # Type narrowing: u.map() on list[T] returns list[R]
            mapped_list = mapped_raw if isinstance(mapped_raw, list) else []
            mapped_predicate = cast("Callable[[R], bool]", predicate)
            return [item for item in mapped_list if mapped_predicate(item)]
        # Without mapper: filter directly on T
        list_predicate = cast("Callable[[T], bool]", predicate)
        return [item for item in items_list if list_predicate(item)]

    @staticmethod
    def _filter_dict[T, R](
        items_dict: dict[str, T],
        predicate: Callable[[str, T], bool] | Callable[[str, R], bool],
        mapper: Callable[[str, T], R] | None = None,
    ) -> dict[str, T] | dict[str, R]:
        """Filter a dict with optional mapping (uses u.map internally)."""
        if mapper is not None:
            # Use u.map() for unified mapping, then filter with dict comprehension
            # (cannot use u.filter here - would cause recursion)
            mapped_dict_raw = u.map(items_dict, mapper)
            # Type narrowing: u.map() on dict[str, T] returns dict[str, R]
            mapped_dict = mapped_dict_raw if isinstance(mapped_dict_raw, dict) else {}
            # After mapping, predicate operates on R
            mapped_dict_predicate = cast("Callable[[str, R], bool]", predicate)
            # Use dict comprehension directly to avoid recursion
            # (cannot use u.filter here - would cause recursion)
            return {k: v for k, v in mapped_dict.items() if mapped_dict_predicate(k, v)}
        # Without mapper, predicate operates on T
        dict_predicate = cast("Callable[[str, T], bool]", predicate)
        # Use dict comprehension directly to avoid recursion
        # (cannot use u.filter here - would cause recursion)
        return {k: v for k, v in items_dict.items() if dict_predicate(k, v)}

    @staticmethod
    def _filter_single[T, R](
        single_item: T,
        predicate: Callable[[T], bool] | Callable[[R], bool],
        mapper: Callable[[T], R] | None = None,
    ) -> list[T] | list[R]:
        """Filter a single value with optional mapping."""
        if mapper is not None:
            mapped_item = mapper(single_item)
            mapped_single_predicate = cast("Callable[[R], bool]", predicate)
            if mapped_single_predicate(mapped_item):
                return [mapped_item]
            return []
        single_predicate = cast("Callable[[T], bool]", predicate)
        if single_predicate(single_item):
            return [single_item]
        return []

    @staticmethod
    def filter[T, R](
        items: T | list[T] | tuple[T, ...] | dict[str, T] | Mapping[str, T],
        predicate: Callable[[T], bool] | Callable[[str, T], bool],
        *,
        mapper: Callable[[T], R] | Callable[[str, T], R] | None = None,
    ) -> list[T] | list[R] | dict[str, T] | dict[str, R]:
        """Unified filter function that auto-detects input type.

        Generic replacement for: filter_list(), filter_dict(), filter_and_map_list()

        Automatically detects if input is list/tuple/dict and applies appropriate filtering.
        Supports optional mapping before filtering.

        Args:
            items: Input items (single value, list, tuple, or dict)
            predicate: Function to filter items
                - For lists: predicate(item) -> bool
                - For dicts: predicate(key, value) -> bool
            mapper: Optional function to map items before filtering
                - For lists: mapper(item) -> result
                - For dicts: mapper(key, value) -> result

        Returns:
            Filtered results (list or dict based on input)

        Example:
            # Filter list
            filtered = u.filter(
                [1, 2, 3],
                lambda x: x > 1,
            )

            # Filter and map list
            result = u.filter(
                values,
                lambda s: s != "",
                mapper=lambda v: str(v).strip(),
            )

            # Filter dict
            filtered = u.filter(
                {"a": 1, "b": 2},
                lambda k, v: v > 1,
            )

        """
        # Use match/case for Python 3.13+ pattern matching
        # Business Rule: Auto-detect input type and delegate to type-specific helper.
        # Each helper uses list/dict comprehension to avoid recursion.
        match items:
            case list() | tuple():
                # Cast to object first to break TypeVar scope, then to specific type
                list_items: list[object] = list(items)
                list_pred: Callable[[object], bool] = cast(
                    "Callable[[object], bool]", predicate
                )
                list_map: Callable[[object], object] | None = cast(
                    "Callable[[object], object] | None", mapper
                )
                return cast(
                    "list[T] | list[R]",
                    FlextUtilities._filter_list(list_items, list_pred, list_map),
                )
            case dict() | Mapping():
                dict_items: dict[str, object] = dict(items)
                dict_pred: Callable[[str, object], bool] = cast(
                    "Callable[[str, object], bool]", predicate
                )
                dict_map: Callable[[str, object], object] | None = cast(
                    "Callable[[str, object], object] | None", mapper
                )
                return cast(
                    "dict[str, T] | dict[str, R]",
                    FlextUtilities._filter_dict(dict_items, dict_pred, dict_map),
                )
            case _:
                single_item: object = cast("object", items)
                single_pred: Callable[[object], bool] = cast(
                    "Callable[[object], bool]", predicate
                )
                single_map: Callable[[object], object] | None = cast(
                    "Callable[[object], object] | None", mapper
                )
                return cast(
                    "list[T] | list[R]",
                    FlextUtilities._filter_single(single_item, single_pred, single_map),
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
                # Use u.get() for unified attribute access (DSL pattern)
                desc = u.get(condition, "description", default="validation")
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
        shortcut_result = FlextUtilities._guard_shortcut(value, condition, context_name)
        if shortcut_result.is_failure:
            # Use u.err() for unified error extraction (DSL pattern)
            return error_msg or u.err(shortcut_result, default="Guard check failed")
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
                    # Use u.get() for unified attribute access (DSL pattern)
                    func_name = u.get(condition, "__name__", default="custom")
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
        if isinstance(condition, type) or (
            isinstance(condition, tuple) and all(isinstance(c, type) for c in condition)
        ):
            return FlextUtilities._guard_check_type(
                value, condition, context_name, error_msg
            )

        # ValidatorSpec: Validator DSL (has __and__ method for composition)
        if callable(condition) and hasattr(condition, "__and__"):
            validator_condition = cast("ValidatorSpec", condition)
            return FlextUtilities._guard_check_validator(
                value, validator_condition, context_name, error_msg
            )

        # String shortcuts
        if isinstance(condition, str):
            return FlextUtilities._guard_check_string_shortcut(
                value, condition, context_name, error_msg
            )

        # Custom predicate: Callable[[T], bool]
        if callable(condition):
            predicate_func = cast("Callable[[object], bool]", condition)
            return FlextUtilities._guard_check_predicate(
                value,
                predicate_func,
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
            # Use u.or_() for default fallback (DSL pattern)
            return u.or_(default, None)
        if default is not None:
            return r[T].ok(default)
        return r[T].fail(error_message)

    @staticmethod
    def _guard_non_empty(value: object, error_template: str) -> r[object]:
        """Internal helper for non-empty validation."""
        # Use pattern matching for type-specific validation
        value_typed = cast("t.GeneralValueType", value)

        # String validation
        if isinstance(value, str):
            return (
                r.ok(value)
                if FlextTypeGuards.is_string_non_empty(value)
                else r.fail(f"{error_template} non-empty string")
            )

        # Dict-like validation
        if FlextRuntime.is_dict_like(value_typed):
            return (
                r.ok(value)
                if FlextTypeGuards.is_dict_non_empty(value_typed)
                else r.fail(f"{error_template} non-empty dict")
            )

        # List-like validation
        if FlextRuntime.is_list_like(value_typed):
            return (
                r.ok(value)
                if FlextTypeGuards.is_list_non_empty(value_typed)
                else r.fail(f"{error_template} non-empty list")
            )

        # Unknown type
        return r.fail(f"{error_template} non-empty (str/dict/list)")

    @staticmethod
    def _guard_numeric(value: object, shortcut: str, error_template: str) -> r[object]:
        """Internal helper for numeric validation."""
        match shortcut:
            case "positive":
                if not isinstance(value, (int, float)) or value <= 0:
                    return r.fail(f"{error_template} positive number")
            case "non_negative":
                if not isinstance(value, (int, float)) or value < 0:
                    return r.fail(f"{error_template} non-negative number")
        return r.ok(value)

    @staticmethod
    def _guard_type(value: object, shortcut: str, error_template: str) -> r[object]:
        """Internal helper for type validation."""
        match shortcut:
            case "dict":
                if not FlextRuntime.is_dict_like(cast("t.GeneralValueType", value)):
                    return r.fail(f"{error_template} dict-like")
            case "list":
                if not FlextRuntime.is_list_like(cast("t.GeneralValueType", value)):
                    return r.fail(f"{error_template} list-like")
            case "string":
                if not isinstance(value, str):
                    return r.fail(f"{error_template} string")
            case "int":
                if not isinstance(value, int):
                    return r.fail(f"{error_template} int")
        return r.ok(value)

    @staticmethod
    def _guard_shortcut(
        value: object,
        shortcut: str,
        context: str,
    ) -> r[object]:
        """Handle string shortcuts for common guard patterns."""
        shortcut_lower = cast("str", u.normalize(shortcut, case="lower"))
        error_template = f"{context} must be"

        # Use match/case for Python 3.13+ pattern matching
        match shortcut_lower:
            case "non_empty":
                return FlextUtilities._guard_non_empty(value, error_template)
            case "positive" | "non_negative":
                return FlextUtilities._guard_numeric(
                    value, shortcut_lower, error_template
                )
            case "dict" | "list" | "string" | "int":
                return FlextUtilities._guard_type(value, shortcut_lower, error_template)
            case _:
                return r.fail(f"{context} unknown guard shortcut: {shortcut}")

    @staticmethod
    def ensure_str_list(
        value: t.GeneralValueType,
        default: list[str] | None = None,
    ) -> list[str]:
        """Ensure value is a list of strings (DEPRECATED: use u.ensure() instead).

        **DEPRECATED**: Use u.ensure(value, target_type="str_list") instead.
        This function is deprecated. Use u.ensure() which automatically detects
        the target type from the default value or explicit target_type parameter.

        Args:
            value: Value to convert (list, tuple, set, or single value)
            default: Default value if None (empty list if not specified)

        Returns:
            List of strings

        Example:
            # Convert attribute values to string list
            str_list = u.ensure(attr_values, target_type="str_list")
            # Prefer: str_list = u.ensure(attr_values, target_type="str_list")

        """
        return cast(
            "list[str]",
            FlextUtilities.ensure(value, target_type="str_list", default=default),
        )

    @staticmethod
    def ensure_str(
        value: t.GeneralValueType,
        default: str = "",
    ) -> str:
        """Ensure value is a string, converting if needed.

        **DEPRECATED**: Use u.ensure(value, target_type="str") instead.
        Kept for backward compatibility.

        Args:
            value: Value to convert to string
            default: Default value if None or conversion fails

        Returns:
            String value or default

        Example:
            # Convert value to string safely
            str_value = u.ensure_str(value, default="")
            # Prefer: str_value = u.ensure(value, target_type="str", default="")

        """
        return cast(
            "str", FlextUtilities.ensure(value, target_type="str", default=default)
        )

    @staticmethod
    def normalize(
        value: str | list[str] | tuple[str, ...] | set[str] | frozenset[str],
        other: str
        | list[str]
        | tuple[str, ...]
        | set[str]
        | frozenset[str]
        | None = None,
        *,
        case: str = "lower",
    ) -> str | list[str] | set[str] | bool:
        """Unified normalize function for strings and collections.

        Generic replacement for: normalize_case(), normalize_collection_case(),
        normalize_match(), normalize_contains(), normalize_in()

        Automatically detects operation based on parameters:
        - If `other` is None: normalizes case of `value`
        - If `other` is str: compares/match/contains operations
        - If `other` is collection: membership check

        Args:
            value: String or collection of strings to normalize
            other: Optional second value for comparison operations
            case: "lower" (default) or "upper" for case normalization

        Returns:
            Normalized string/collection or bool for match/contains/in operations

        Example:
            # Normalize case
            normalized = u.normalize("Hello", case="lower")
            # → "hello"

            # Normalize collection
            normalized = u.normalize(["A", "B"], case="upper")
            # → ["A", "B"]

            # Match (auto-detected from two strings)
            matches = u.normalize("Hello", "hello")
            # → True

            # Contains (auto-detected from two strings)
            found = u.normalize("Hello World", "world")
            # → True

            # In (auto-detected from string + collection)
            found = u.normalize("hello", ["A", "B", "Hello"])
            # → True

        """
        # Auto-detect operation based on `other` parameter
        if other is None:
            return FlextUtilities._normalize_case_only(value, case)

        # String comparison/match/contains
        if isinstance(other, str) and isinstance(value, str):
            # Use _normalize_case_only helper (internal - avoid recursion)
            value_lower = cast(
                "str", FlextUtilities._normalize_case_only(value, "lower")
            )
            other_lower = cast(
                "str", FlextUtilities._normalize_case_only(other, "lower")
            )
            return value_lower == other_lower or other_lower in value_lower

        # Collection membership check
        if isinstance(other, (list, tuple, set, frozenset)) and isinstance(value, str):
            return FlextUtilities._normalize_membership_check(value, other)

        # Fallback to case normalization
        return FlextUtilities._normalize_case_only(value, case)

    @staticmethod
    def _normalize_case_only(
        value: str | list[str] | tuple[str, ...] | set[str] | frozenset[str],
        case: str,
    ) -> str | list[str] | set[str]:
        """Helper: Normalize case only (internal - don't call u.normalize to avoid recursion)."""
        if isinstance(value, str):
            return value.lower() if case == "lower" else value.upper()
        # Use u.map() for unified mapping - now supports sets/frozensets
        case_func = str.lower if case == "lower" else str.upper
        # Type narrowing: value is collection of strings
        mapped = u.map(value, case_func)
        # u.map() preserves collection type (list/tuple/set/frozenset)
        if isinstance(mapped, set):
            return mapped
        if isinstance(mapped, (list, tuple)):
            return mapped
        # Fallback: should not happen, but handle gracefully
        return list(mapped) if isinstance(mapped, (tuple, set, frozenset)) else []

    @staticmethod
    def _normalize_membership_check(
        value: str,
        other: list[str] | tuple[str, ...] | set[str] | frozenset[str],
    ) -> bool:
        """Helper: Check membership in collection."""
        # Use _normalize_case_only helper (internal - avoid recursion)
        item_lower = cast("str", FlextUtilities._normalize_case_only(value, "lower"))

        def normalize_func(x: str) -> str:
            """Normalize string to lowercase."""
            # Use _normalize_case_only helper (internal - avoid recursion)
            return cast(
                "str",
                FlextUtilities._normalize_case_only(
                    x if isinstance(x, str) else str(x), "lower"
                ),
            )

        # Use u.map() for unified mapping
        # Type narrowing: other is collection of strings
        other_collection: list[str] | tuple[str, ...] | set[str] | frozenset[str] = (
            cast("list[str] | tuple[str, ...] | set[str] | frozenset[str]", other)
        )
        normalized = cast(
            "list[str] | tuple[str, ...]", u.map(other_collection, normalize_func)
        )
        # Use u.when() for conditional collection type (DSL pattern)
        normalized_collection: Collection[str] = cast(
            "Collection[str]",
            u.when(
                condition=isinstance(other, (set, frozenset)),
                then_value=set(normalized),
                else_value=normalized,
            ),
        )
        return item_lower in normalized_collection

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
            # Use u.or_() for default fallback (DSL pattern)
            result = u.or_(default, [])
            return cast("list[t.GeneralValueType]", result)
        match value:
            case list():
                return value
            case tuple():
                return list(value)
            case _:
                # Type narrowing: value is GeneralValueType, wrap in list
                return [value]

    @staticmethod
    def _ensure_to_dict(
        value: t.GeneralValueType | dict[str, t.GeneralValueType] | None,
        default: dict[str, t.GeneralValueType] | None,
    ) -> dict[str, t.GeneralValueType]:
        """Helper: Convert value to dict."""
        if value is None:
            # Use u.or_() for default fallback (DSL pattern)
            result = u.or_(default, {})
            return cast("dict[str, t.GeneralValueType]", result)
        match value:
            case dict():
                # Type narrowing: value is dict[str, GeneralValueType]
                return cast("dict[str, t.GeneralValueType]", value)
            case _:
                # Type narrowing: value is GeneralValueType, wrap in dict
                return {"value": value}

    @staticmethod
    def ensure[T](
        value: t.GeneralValueType,
        *,
        target_type: str = "auto",
        default: T | list[T] | dict[str, T] | None = None,
    ) -> T | list[T] | dict[str, T]:
        """Unified ensure function that auto-detects or enforces target type.

        Generic replacement for: ensure_list(), ensure_dict(), ensure_str(), ensure_str_list()

        Automatically detects if value should be list or dict, or enforces target_type.
        Converts single values, tuples, None to appropriate type.
        Supports string conversion via target_type="str" or "str_list".

        Args:
            value: Value to convert (single value, list, tuple, dict, or None)
            target_type: Target type - "auto" (detect), "list", "dict", "str", "str_list"
            default: Default value if None (empty list/dict/empty string if not specified)

        Returns:
            Converted value based on target_type or auto-detection

        Example:
            # Auto-detect (prefers list for single values, dict for dict-like)
            items = u.ensure(value)
            # Works with: "single" → ["single"], {"key": "value"} → {"key": "value"}

            # Force list
            items = u.ensure(value, target_type="list")
            # Works with: "single" → ["single"], ("tuple",) → ["tuple"]

            # Force dict
            data = u.ensure(value, target_type="dict")
            # Works with: "single" → {"value": "single"}, None → {}

            # Convert to string
            str_value = u.ensure(value, target_type="str", default="")
            # Works with: "hello" → "hello", 123 → "123", None → ""

            # Convert to list of strings
            str_list = u.ensure(value, target_type="str_list", default=[])
            # Works with: ["a", "b"] → ["a", "b"], [1, 2] → ["1", "2"], "single" → ["single"]

        """
        # Handle string conversions first
        if target_type == "str":
            str_default = cast("str", default) if default is not None else ""
            return cast("T", FlextDataMapper.ensure_str(value, default=str_default))
        if target_type == "str_list":
            # Use u.when() for conditional cast (DSL pattern)
            str_list_default = u.when(
                condition=isinstance(default, list),
                then_value=cast("list[str]", default),
                else_value=None,
            )
            # Use FlextDataMapper directly for str_list (internal implementation)
            return cast(
                "list[T]",
                FlextDataMapper.ensure_str_list(value, default=str_list_default),
            )
        if target_type == "dict":
            # Type narrowing: value is object, default is dict[str, T] | None
            # Convert default to correct type for _ensure_to_dict
            dict_default_typed: dict[str, t.GeneralValueType] | None
            if isinstance(default, dict):
                dict_default_typed = cast("dict[str, t.GeneralValueType]", default)
            else:
                dict_default_typed = None
            # Cast value to GeneralValueType for _ensure_to_dict
            value_typed: t.GeneralValueType | dict[str, t.GeneralValueType] | None = (
                cast("t.GeneralValueType | dict[str, t.GeneralValueType] | None", value)
            )
            dict_result = FlextUtilities._ensure_to_dict(
                value_typed, dict_default_typed
            )
            return cast("T", dict_result)
        if target_type == "auto" and isinstance(value, dict):
            return cast("T", value)
        # Handle list or fallback
        # Use u.when() for conditional assignment (DSL pattern)
        list_default_fallback: list[t.GeneralValueType] | None = (
            cast("list[t.GeneralValueType]", default)
            if isinstance(default, list)
            else None
        )
        list_result = FlextUtilities._ensure_to_list(value, list_default_fallback)
        return cast("T", list_result)

    @staticmethod
    def _process_list_items[T, R](
        items_list: list[T],
        processor: Callable[[T], R],
        *,
        predicate: Callable[[T], bool] | None = None,
        on_error: str = "collect",
    ) -> r[list[R] | dict[str, R]]:
        """Helper: Process list items."""
        if predicate is not None:
            list_predicate = predicate
            items_list = cast(
                "list[T]",
                u.filter(items_list, list_predicate),
            )
        list_results: list[R] = []
        list_errors: list[str] = []
        list_processor = processor
        for item in items_list:
            try:
                processed = list_processor(item)
                list_results.append(processed)
            except Exception as e:
                if on_error == "fail":
                    return r[list[R] | dict[str, R]].fail(f"Processing failed: {e}")
                if on_error == "skip":
                    continue
                list_errors.append(str(e))
        if list_errors and on_error == "collect":
            return r[list[R] | dict[str, R]].fail(
                f"Processing errors: {', '.join(list_errors)}"
            )
        return r[list[R] | dict[str, R]].ok(list_results)

    @staticmethod
    def _process_dict_items[T, R](  # noqa: PLR0913
        items_dict: dict[str, T],
        processor: Callable[[str, T], R],
        *,
        predicate: Callable[[str, T], bool] | None = None,
        filter_keys: set[str] | None = None,
        exclude_keys: set[str] | None = None,
        on_error: str = "collect",
    ) -> r[list[R] | dict[str, R]]:
        """Helper: Process dict items."""
        # Apply key filtering first
        if filter_keys is not None or exclude_keys is not None:
            # Type narrowing: predicate for dict filtering (key, value) -> bool
            def key_predicate(k: str, _v: T) -> bool:
                """Filter predicate for dict keys."""
                return (filter_keys is None or k in filter_keys) and (
                    exclude_keys is None or k not in exclude_keys
                )

            items_dict = cast(
                "dict[str, T]",
                u.filter(items_dict, key_predicate),
            )
        # Apply predicate filtering
        if predicate is not None:
            dict_predicate = predicate
            items_dict = cast(
                "dict[str, T]",
                u.filter(items_dict, dict_predicate),
            )
        # Cannot use u.process here to avoid circular recursion
        # u.process calls _process_dict_items, which would call u.process again
        dict_result: dict[str, R] = {}
        dict_errors: list[str] = []
        dict_processor = processor
        for key, value in items_dict.items():
            try:
                processed = dict_processor(key, value)
                dict_result[key] = processed
            except Exception as e:
                if on_error == "fail":
                    return r[list[R] | dict[str, R]].fail(
                        f"Processing key '{key}' failed: {e}"
                    )
                if on_error == "skip":
                    continue
                dict_errors.append(str(e))
        if dict_errors and on_error == "collect":
            return r[list[R] | dict[str, R]].fail(
                f"Processing errors: {', '.join(dict_errors)}"
            )
        return r[list[R] | dict[str, R]].ok(dict_result)

    @staticmethod
    def process[T, R](  # noqa: PLR0913
        items: T | list[T] | tuple[T, ...] | dict[str, T] | Mapping[str, T],
        processor: Callable[[T], R] | Callable[[str, T], R],
        *,
        on_error: str = "collect",
        predicate: Callable[[T], bool] | Callable[[str, T], bool] | None = None,
        filter_keys: set[str] | None = None,
        exclude_keys: set[str] | None = None,
    ) -> r[list[R] | dict[str, R]]:
        """Unified process function that auto-detects input type.

        Generic replacement for: process_dict(), find_in_dict(), convert_dict_keys(),
        Manual loops, map/filter operations, dict processing

        Automatically detects if input is list/tuple/dict and applies appropriate processing.
        Supports filtering via predicate, filter_keys, exclude_keys and error handling.

        Args:
            items: Input items (single value, list, tuple, or dict)
            processor: Function to process each item
                - For lists: processor(item) -> result
                - For dicts: processor(key, value) -> result
            on_error: "collect" (continue), "fail" (stop), or "skip" (ignore errors)
            predicate: Optional filter function
                - For lists: predicate(item) -> bool
                - For dicts: predicate(key, value) -> bool
            filter_keys: Optional set of keys to process (dict only)
            exclude_keys: Optional set of keys to skip (dict only)

        Returns:
            r containing processed results (list or dict based on input)

        Example:
            # Process list
            result = u.process(
                [1, 2, 3],
                lambda x: x * 2,
                predicate=lambda x: x > 1,
            )

            # Process dict
            result = u.process(
                {"a": 1, "b": 2},
                lambda k, v: v * 2,
            )

            # Process dict with key filtering
            result = u.process(
                {"a": 1, "b": 2, "c": 3},
                lambda k, v: v * 2,
                filter_keys={"a", "b"},
            )

        """
        # Process list/tuple
        if isinstance(items, (list, tuple)):
            list_processor = cast("Callable[[T], R]", processor)
            list_predicate = (
                cast("Callable[[T], bool] | None", predicate)
                if predicate is not None
                else None
            )
            return FlextUtilities._process_list_items(
                list(items), list_processor, predicate=list_predicate, on_error=on_error
            )

        # Process dict
        if isinstance(items, (dict, Mapping)):
            dict_processor = cast("Callable[[str, T], R]", processor)
            dict_predicate = (
                cast("Callable[[str, T], bool] | None", predicate)
                if predicate is not None
                else None
            )
            return FlextUtilities._process_dict_items(
                dict(items),
                dict_processor,
                predicate=dict_predicate,
                filter_keys=filter_keys,
                exclude_keys=exclude_keys,
                on_error=on_error,
            )

        # Single value - wrap in list and process
        single_processor = cast("Callable[[T], R]", processor)
        single_predicate = (
            cast("Callable[[T], bool] | None", predicate)
            if predicate is not None
            else None
        )
        return FlextUtilities._process_list_items(
            [items], single_processor, predicate=single_predicate, on_error=on_error
        )

    @staticmethod
    @overload
    def map[T, R](
        items: r[T],
        mapper: Callable[[T], R],
        *,
        default_error: str = "Operation failed",
    ) -> r[R]: ...

    @staticmethod
    @overload
    def map[T, R](
        items: list[T] | tuple[T, ...],
        mapper: Callable[[T], R],
    ) -> list[R]: ...

    @staticmethod
    @overload
    def map[T, R](
        items: set[T] | frozenset[T],
        mapper: Callable[[T], R],
    ) -> set[R] | frozenset[R]: ...

    @staticmethod
    @overload
    def map[T, R](
        items: dict[str, T] | Mapping[str, T],
        mapper: Callable[[str, T], R],
    ) -> dict[str, R]: ...

    @staticmethod
    def map[T, R](  # noqa: PLR0911
        items: T
        | list[T]
        | tuple[T, ...]
        | set[T]
        | frozenset[T]
        | dict[str, T]
        | Mapping[str, T]
        | r[T],
        mapper: Callable[[T], R] | Callable[[str, T], R],
        *,
        default_error: str = "Operation failed",
    ) -> list[R] | set[R] | frozenset[R] | dict[str, R] | r[R]:
        """Unified map function that auto-detects input type.

        Generic replacement for: List/dict comprehensions, manual loops, map_or

        Args:
            items: Input items (list, tuple, dict, set, or r[T] result)
            mapper: Function to transform items
                - For lists: mapper(item) -> result
                - For dicts: mapper(key, value) -> result
                - For results: mapper(value) -> result
            default_error: Default error if mapping result fails (only for r[T])

        Returns:
            Mapped results (list, dict, set, or r[R] based on input type)

        Example:
            # Map list
            mapped = u.map([1, 2, 3], lambda x: x * 2)

            # Map dict values
            mapped = u.map({"a": 1, "b": 2}, lambda k, v: v * 2)

            # Map result (generalized from map_or)
            result = u.map(parse_result, lambda data: process(data))

        """
        # Handle r[T] case (generalized from map_or)
        if isinstance(items, r):
            if items.is_success:
                result_mapper = cast("Callable[[T], R]", mapper)
                return r[R].ok(result_mapper(items.value))
            return r[R].fail(u.err(items, default=default_error))

        # Handle collections (original map behavior)
        if isinstance(items, (list, tuple)):
            list_mapper = cast("Callable[[T], R]", mapper)
            # Cannot use u.map() here - would cause recursion (u.map calls itself for lists)
            # Use list comprehension directly for mapping
            return [list_mapper(item) for item in items]

        if isinstance(items, (set, frozenset)):
            set_mapper = cast("Callable[[T], R]", mapper)
            mapped_items: set[R] = {set_mapper(item) for item in items}
            # Preserve frozenset type if input was frozenset
            if isinstance(items, frozenset):
                return cast("frozenset[R]", frozenset(mapped_items))
            return mapped_items

        if isinstance(items, (dict, Mapping)):
            dict_mapper = cast("Callable[[str, T], R]", mapper)
            return {k: dict_mapper(k, v) for k, v in items.items()}

        # Single value - wrap in list and map
        # This handles the case where items is a single value (not list/tuple/dict/set)
        single_mapper = cast("Callable[[T], R]", mapper)
        return [single_mapper(items)]

    @staticmethod
    def _convert_to_int(value: t.GeneralValueType, default: int, /) -> int:
        """Internal helper for int conversion."""
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return default
        return default

    @staticmethod
    def _convert_to_float(value: t.GeneralValueType, default: float, /) -> float:
        """Internal helper for float conversion."""
        if isinstance(value, float):
            return value
        if isinstance(value, (int, str)):
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        return default

    @staticmethod
    def _convert_to_str(value: t.GeneralValueType, default: str, /) -> str:
        """Internal helper for str conversion."""
        if isinstance(value, str):
            return value
        if value is None:
            return default
        try:
            return str(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _convert_to_bool(value: t.GeneralValueType, *, default: bool) -> bool:
        """Internal helper for bool conversion using u.normalize()."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            # Use u.normalize() for unified normalization
            normalized = cast("str", u.normalize(value, case="lower"))
            return normalized in {"true", "1", "yes", "on"}
        if isinstance(value, (int, float)):
            return bool(value)
        return default

    @staticmethod
    def convert[T](
        value: t.GeneralValueType,
        target_type: type[T],
        default: T,
    ) -> T:
        """Unified type conversion with safe fallback.

        Generic replacement for: convert_to_int_safe(), manual try/except conversions

        Automatically handles common type conversions (int, str, float, bool) with
        safe fallback to default value on conversion failure.

        Args:
            value: Value to convert
            target_type: Target type (int, str, float, bool)
            default: Default value to return on conversion failure

        Returns:
            Converted value or default

        Example:
            # Convert to int
            result = u.convert("123", int, 0)
            # → 123

            # Convert to int (invalid)
            result = u.convert("invalid", int, 0)
            # → 0

            # Convert to float
            result = u.convert("3.14", float, 0.0)
            # → 3.14

        """
        # Already correct type - fast path
        if isinstance(value, target_type):
            return value

        # Route to specific converter using dict lookup for cleaner code
        converters: dict[type[object], Callable[[t.GeneralValueType, T], T]] = {
            int: lambda v, d: cast(
                "T", FlextUtilities._convert_to_int(v, cast("int", d))
            ),
            float: lambda v, d: cast(
                "T", FlextUtilities._convert_to_float(v, cast("float", d))
            ),
            str: lambda v, d: cast(
                "T", FlextUtilities._convert_to_str(v, cast("str", d))
            ),
            bool: lambda v, d: cast(
                "T", FlextUtilities._convert_to_bool(v, default=cast("bool", d))
            ),
        }

        converter = converters.get(target_type)
        if converter is not None:
            converter_func = cast("Callable[[object, T], T]", converter)
            return converter_func(value, default)

        # Fallback: try direct conversion only if target_type is a callable type
        if isinstance(target_type, type) and callable(target_type):
            try:
                # Cast value to GeneralValueType for type compatibility
                value_typed: t.GeneralValueType = cast("t.GeneralValueType", value)
                # Call target_type as constructor
                type_constructor = cast(
                    "Callable[[t.GeneralValueType], T]", target_type
                )
                converted = type_constructor(value_typed)
                return cast("T", converted)
            except (ValueError, TypeError):
                return default
        return default

    @staticmethod
    def _build_apply_ensure(current: object, ops: dict[str, object]) -> object:
        """Helper: Apply ensure operation."""
        if "ensure" not in ops:
            return current
        ensure_type = cast("str", ops["ensure"])
        ensure_default_val = ops.get("ensure_default")
        # Use dict lookup for cleaner default value selection
        default_map: dict[str, object] = {
            "str_list": [],
            "dict": {},
            "list": [],
            "str": "",
        }
        # Use u.when() for conditional default selection
        default_val = u.when(
            condition=ensure_default_val is not None,
            then_value=ensure_default_val,
            else_value=default_map.get(ensure_type, ""),
        )
        # Type narrowing: ensure_type is str, default_val is object
        ensure_type_str = cast("str", ensure_type)
        # Cast current to GeneralValueType for ensure compatibility
        current_typed: t.GeneralValueType = cast("t.GeneralValueType", current)
        ensure_result = u.ensure(
            current_typed, target_type=ensure_type_str, default=default_val
        )
        return cast("object", ensure_result)

    @staticmethod
    def _build_apply_filter(
        current: object, ops: dict[str, object], default: object
    ) -> object:
        """Helper: Apply filter operation using unified collection handling."""
        if "filter" not in ops:
            return current
        filter_pred = cast("Callable[[object], bool]", ops["filter"])
        # Use unified collection handling DSL pattern
        match current:
            case list() | tuple():
                current_seq: list[object] | tuple[object, ...] = cast(
                    "list[object] | tuple[object, ...]", current
                )
                return u.filter(current_seq, predicate=filter_pred)
            case set() | frozenset():
                current_set: set[object] | frozenset[object] = cast(
                    "set[object] | frozenset[object]", current
                )
                return u.filter(list(current_set), predicate=filter_pred)
            case dict():
                current_dict: dict[str, object] = cast("dict[str, object]", current)

                def dict_pred(_k: str, v: object) -> bool:
                    """Filter predicate for dict items."""
                    return filter_pred(v)

                return u.filter(current_dict, predicate=dict_pred)
            case _:
                return default if not filter_pred(current) else current

    @staticmethod
    def _build_apply_map(current: object, ops: dict[str, object]) -> object:
        """Helper: Apply map operation."""
        if "map" not in ops:
            return current
        map_func = cast("Callable[[object], object]", ops["map"])
        if isinstance(current, (list, tuple)):
            # Type narrowing: current is list or tuple
            current_list: list[object] | tuple[object, ...] = cast(
                "list[object] | tuple[object, ...]", current
            )
            map_result = u.map(current_list, mapper=map_func)
            return cast("object", map_result)
        if isinstance(current, (set, frozenset)):
            # Type narrowing: current is set or frozenset
            current_set: set[object] | frozenset[object] = cast(
                "set[object] | frozenset[object]", current
            )
            map_result_set = u.map(current_set, mapper=map_func)
            return cast("object", map_result_set)
        if isinstance(current, (dict, Mapping)):
            # Type narrowing: current is dict or Mapping
            current_dict: dict[str, object] | Mapping[str, object] = cast(
                "dict[str, object] | Mapping[str, object]", current
            )
            # For dict, mapper should be Callable[[str, object], object]
            dict_mapper = cast("Callable[[str, object], object]", map_func)
            map_result_dict = u.map(current_dict, mapper=dict_mapper)
            return cast("object", map_result_dict)
        map_result_single = map_func(current)
        return cast("object", map_result_single)

    @staticmethod
    def _build_apply_normalize(current: object, ops: dict[str, object]) -> object:
        """Helper: Apply normalize operation."""
        if "normalize" not in ops or not isinstance(
            current, (str, list, tuple, set, frozenset)
        ):
            return current
        normalize_case = cast("str", ops["normalize"])
        # Type narrowing: current is str or collection
        current_normalizable: (
            str | list[str] | tuple[str, ...] | set[str] | frozenset[str]
        ) = cast(
            "str | list[str] | tuple[str, ...] | set[str] | frozenset[str]", current
        )
        normalize_result = u.normalize(current_normalizable, case=normalize_case)
        return cast("object", normalize_result)

    @staticmethod
    def _build_apply_convert(current: object, ops: dict[str, object]) -> object:
        """Helper: Apply convert operation."""
        if "convert" not in ops:
            return current
        convert_type = cast("type[object]", ops["convert"])
        convert_default = ops.get("convert_default", convert_type())
        # Type narrowing: current is object, convert_type is type
        current_typed: t.GeneralValueType = cast("t.GeneralValueType", current)
        convert_result = u.convert(current_typed, convert_type, convert_default)
        return cast("object", convert_result)

    @staticmethod
    def _build_apply_transform(  # noqa: PLR0914
        current: object, ops: dict[str, object], default: object, on_error: str
    ) -> object:
        """Helper: Apply transform operation."""
        if "transform" not in ops or not isinstance(current, (dict, Mapping)):
            return current
        transform_opts_raw = ops["transform"]
        if not isinstance(transform_opts_raw, dict):
            return current
        transform_opts = cast("dict[str, object]", transform_opts_raw)
        # Type narrowing: current is dict or Mapping
        current_dict: dict[str, object] | Mapping[str, object] = cast(
            "dict[str, object] | Mapping[str, object]", current
        )
        # Extract transform options as keyword arguments
        # Type narrowing: current_dict needs to be Mapping[str, GeneralValueType] for transform
        current_dict_typed: Mapping[str, t.GeneralValueType] = cast(
            "Mapping[str, t.GeneralValueType]", current_dict
        )
        # Extract transform options with proper types
        normalize_val = transform_opts.get("normalize")
        normalize_bool = (
            cast("bool", normalize_val) if isinstance(normalize_val, bool) else False
        )
        strip_none_val = transform_opts.get("strip_none")
        strip_none_bool = (
            cast("bool", strip_none_val) if isinstance(strip_none_val, bool) else False
        )
        strip_empty_val = transform_opts.get("strip_empty")
        strip_empty_bool = (
            cast("bool", strip_empty_val)
            if isinstance(strip_empty_val, bool)
            else False
        )
        map_keys_val = transform_opts.get("map_keys")
        map_keys_dict = (
            cast("dict[str, str]", map_keys_val)
            if isinstance(map_keys_val, dict)
            else None
        )
        filter_keys_val = transform_opts.get("filter_keys")
        filter_keys_set = (
            cast("set[str]", filter_keys_val)
            if isinstance(filter_keys_val, set)
            else None
        )
        exclude_keys_val = transform_opts.get("exclude_keys")
        exclude_keys_set = (
            cast("set[str]", exclude_keys_val)
            if isinstance(exclude_keys_val, set)
            else None
        )
        to_json_val = transform_opts.get("to_json")
        to_json_bool = (
            cast("bool", to_json_val) if isinstance(to_json_val, bool) else False
        )
        to_model_val = transform_opts.get("to_model")
        to_model_typed = (
            cast("type[BaseModel] | None", to_model_val)
            if isinstance(to_model_val, type) or to_model_val is None
            else None
        )
        transform_result = u.transform(
            current_dict_typed,
            normalize=normalize_bool,
            strip_none=strip_none_bool,
            strip_empty=strip_empty_bool,
            map_keys=map_keys_dict,
            filter_keys=filter_keys_set,
            exclude_keys=exclude_keys_set,
            to_json=to_json_bool,
            to_model=to_model_typed,
        )
        if transform_result.is_success:
            return transform_result.value
        return default if on_error == "stop" else current

    @staticmethod
    def _build_apply_process(  # noqa: PLR0911
        current: object, ops: dict[str, object], default: object, on_error: str
    ) -> object:
        """Helper: Apply process operation using u.process()."""
        if "process" not in ops:
            return current
        process_func = cast("Callable[[object], object]", ops["process"])
        # Use u.process() for unified processing
        if isinstance(current, (list, tuple)):
            # Type narrowing: current is list or tuple
            current_list: list[object] | tuple[object, ...] = cast(
                "list[object] | tuple[object, ...]", current
            )
            process_result_list = u.process(
                current_list, processor=process_func, on_error=on_error
            )
            if process_result_list.is_success:
                return cast("object", process_result_list.value)
            return default if on_error == "stop" else current
        if isinstance(current, (dict, Mapping)):
            # Type narrowing: current is dict or Mapping
            current_dict: dict[str, object] | Mapping[str, object] = cast(
                "dict[str, object] | Mapping[str, object]", current
            )
            process_result_dict = u.process(
                current_dict, processor=process_func, on_error=on_error
            )
            if process_result_dict.is_success:
                return cast("object", process_result_dict.value)
            return default if on_error == "stop" else current
        # Single value processing with error handling
        try:
            process_result = process_func(current)
            return cast("object", process_result)
        except Exception:
            return default if on_error == "stop" else current

    @staticmethod
    def _build_apply_group(current: object, ops: dict[str, object]) -> object:
        """Helper: Apply group operation."""
        if "group" not in ops:
            return current
        if not isinstance(current, (list, tuple)):
            return current
        group_spec = ops["group"]
        current_list: list[object] | tuple[object, ...] = cast(
            "list[object] | tuple[object, ...]", current
        )
        # Group by field name (str) or key function (callable)
        if isinstance(group_spec, str):
            # Group by field name
            grouped: dict[object, list[object]] = {}
            for item in current_list:
                key = u.get(item, group_spec)
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(item)
            return grouped
        if callable(group_spec):
            # Group by key function
            key_func: Callable[[object], object] = cast(
                "Callable[..., object]", group_spec
            )
            grouped_by_func: dict[object, list[object]] = {}
            for item in current_list:
                item_key = key_func(item)
                # Cast key to object for dict key (dict keys can be any hashable type)
                key_obj = cast("object", item_key)
                if key_obj not in grouped_by_func:
                    grouped_by_func[key_obj] = []
                grouped_by_func[key_obj].append(item)
            return grouped_by_func
        return current

    @staticmethod
    def _build_apply_sort(current: object, ops: dict[str, object]) -> object:
        """Helper: Apply sort operation."""
        if "sort" not in ops:
            return current
        if not isinstance(current, (list, tuple)):
            return current
        sort_spec = ops["sort"]
        current_list: list[object] | tuple[object, ...] = cast(
            "list[object] | tuple[object, ...]", current
        )
        # Sort by field name (str), key function (callable), or natural sort (True)
        if isinstance(sort_spec, str):
            # Sort by field name
            def key_func(item: object) -> object:
                return u.get(item, sort_spec)

            # Type ignore: sorted accepts callable returning object for key
            sorted_list = sorted(current_list, key=key_func)  # type: ignore[arg-type]
            return (
                list(sorted_list) if isinstance(current, list) else tuple(sorted_list)
            )
        if callable(sort_spec):
            # Sort by key function
            key_func_callable: Callable[[object], object] = cast(
                "Callable[[object], object]", sort_spec
            )
            # Type ignore: sorted accepts callable returning object for key
            sorted_list = sorted(current_list, key=key_func_callable)  # type: ignore[arg-type]
            return (
                list(sorted_list) if isinstance(current, list) else tuple(sorted_list)
            )
        if sort_spec is True:
            # Natural sort - type ignore for object comparison
            sorted_list = sorted(current_list)  # type: ignore[type-var]
            return (
                list(sorted_list) if isinstance(current, list) else tuple(sorted_list)
            )
        return current

    @staticmethod
    def _build_apply_unique(current: object, ops: dict[str, object]) -> object:
        """Helper: Apply unique operation to remove duplicates."""
        if "unique" not in ops or not ops.get("unique"):
            return current
        if not isinstance(current, (list, tuple)):
            return current
        current_list: list[object] | tuple[object, ...] = cast(
            "list[object] | tuple[object, ...]", current
        )
        # Remove duplicates while preserving order
        seen: set[object] = set()
        unique_list: list[object] = []
        for item in current_list:
            # Use hashable representation for comparison
            item_hashable = (
                item
                if isinstance(item, (str, int, float, bool, type(None)))
                else str(item)
            )
            if item_hashable not in seen:
                seen.add(item_hashable)
                unique_list.append(item)
        return list(unique_list) if isinstance(current, list) else tuple(unique_list)

    @staticmethod
    def _build_apply_slice(current: object, ops: dict[str, object]) -> object:
        """Helper: Apply slice operation."""
        if "slice" not in ops:
            return current
        if not isinstance(current, (list, tuple)):
            return current
        slice_spec = ops["slice"]
        current_list: list[object] | tuple[object, ...] = cast(
            "list[object] | tuple[object, ...]", current
        )
        # Slice can be tuple[int, int] or list[int, int]
        min_slice_length = 2
        if (
            isinstance(slice_spec, (tuple, list))
            and len(slice_spec) >= min_slice_length
        ):
            start = cast("int", slice_spec[0]) if slice_spec[0] is not None else None
            end = cast("int", slice_spec[1]) if slice_spec[1] is not None else None
            sliced = current_list[start:end]
            return list(sliced) if isinstance(current, list) else tuple(sliced)
        return current

    @staticmethod
    def _build_apply_chunk(current: object, ops: dict[str, object]) -> object:
        """Helper: Apply chunk operation to split into sublists."""
        if "chunk" not in ops:
            return current
        if not isinstance(current, (list, tuple)):
            return current
        chunk_size = ops["chunk"]
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            return current
        current_list: list[object] | tuple[object, ...] = cast(
            "list[object] | tuple[object, ...]", current
        )
        # Split into chunks of specified size
        chunked: list[list[object]] = []
        for i in range(0, len(current_list), chunk_size):
            chunk = list(current_list[i : i + chunk_size])
            chunked.append(chunk)
        return chunked

    @staticmethod
    def build[T](
        value: T,
        *,
        ops: dict[str, object] | None = None,
        on_error: str = "stop",
    ) -> T | object:
        """Builder pattern for fluent operation composition using DSL.

        Generic replacement for: Chained u.ensure(), u.map(), u.filter(), u.normalize(),
        u.convert(), u.transform(), u.process() operations

        Uses DSL dict to compose operations: {"ensure": "str", "map": lambda x: x*2, ...}
        Operations are applied in order: ensure → filter → map → normalize → convert → transform → process

        Args:
            value: Initial value to process
            ops: Dict with operation keys:
                - "ensure": str target type ("str", "str_list", "list", "dict", "auto")
                - "ensure_default": default value for ensure
                - "filter": predicate function
                - "map": mapper function
                - "normalize": case ("lower" or "upper")
                - "convert": target type class
                - "convert_default": default for convert
                - "transform": dict with transform options
                - "process": processor function
                - "group": str field name or callable for grouping
                - "sort": str field name, callable key function, or True for natural sort
                - "unique": bool to remove duplicates
                - "slice": tuple[int, int] for slicing (start, end)
                - "chunk": int size for chunking into sublists
            on_error: Error handling ("stop", "skip", "collect")

        Returns:
            Processed value (type depends on operations applied)

        Example:
            # Ensure string, normalize, filter
            result = u.build(
                value,
                ops={"ensure": "str", "ensure_default": "", "normalize": "lower", "filter": lambda v: len(v) > 0},
            )

            # Map, filter, convert
            result = u.build(
                items,
                ops={"map": lambda x: x * 2, "filter": lambda x: x > 10, "convert": int, "convert_default": 0},
            )

            # Transform dict
            result = u.build(
                data,
                ops={"transform": {"normalize": True, "strip_none": True, "filter_keys": {"name", "email"}}},
            )

            # Group, sort, unique, slice, chunk
            result = u.build(
                items,
                ops={
                    "group": "category",  # Group by field
                    "sort": "name",       # Sort by field
                    "unique": True,       # Remove duplicates
                    "slice": (0, 10),    # Take first 10
                    "chunk": 5,          # Split into chunks of 5
                },
            )

        """
        if ops is None:
            return value

        ensure_default_val: object = ops.get("ensure_default")
        current: object = value

        # Apply operations in sequence
        current = FlextUtilities._build_apply_ensure(current, ops)
        current = FlextUtilities._build_apply_filter(
            current, ops, ensure_default_val or current
        )
        current = FlextUtilities._build_apply_map(current, ops)
        current = FlextUtilities._build_apply_normalize(current, ops)
        current = FlextUtilities._build_apply_convert(current, ops)
        current = FlextUtilities._build_apply_transform(
            current, ops, ensure_default_val or current, on_error
        )
        current = FlextUtilities._build_apply_process(
            current, ops, ensure_default_val or current, on_error
        )
        current = FlextUtilities._build_apply_group(current, ops)
        current = FlextUtilities._build_apply_sort(current, ops)
        current = FlextUtilities._build_apply_unique(current, ops)
        current = FlextUtilities._build_apply_slice(current, ops)
        current = FlextUtilities._build_apply_chunk(current, ops)
        return cast("object", current)

    @staticmethod
    def agg[T](
        items: list[T] | tuple[T, ...],
        field: str | Callable[[T], object],
        *,
        fn: Callable[[list[object]], object] | None = None,
    ) -> object:
        """Aggregate field values from objects (mnemonic: agg = aggregate).

        Generic replacement for: sum(getattr(...)), max(getattr(...)), manual aggregation loops

        Args:
            items: List/tuple of objects
            field: Field name (str) or extractor function (callable)
            fn: Aggregation function (default: sum)

        Returns:
            Aggregated value

        Example:
            # Sum field values
            total = u.agg(items, "synced")
            # → 15

            # Max with custom extractor
            max_val = u.agg(items, lambda r: r.total_entries, fn=max)
            # → 30

        """
        items_list: list[T] | tuple[T, ...] = cast("list[T] | tuple[T, ...]", items)
        if callable(field):
            # Type narrowing: field is Callable[[T], object]
            field_func = cast("Callable[[T], object]", field)
            extracted = cast("list[object]", u.map(items_list, field_func))
        else:
            # Use u.get for unified extraction (works for dicts and objects)
            # Type narrowing: field is str
            field_str = cast("str", field)
            extracted = cast(
                "list[object]", u.map(items_list, lambda item: u.get(item, field_str))
            )

        # Filter None values before aggregation

        def is_not_none(x: object) -> bool:
            """Check if value is not None."""
            return x is not None

        filtered = cast("list[object]", u.filter(extracted, is_not_none))
        agg_fn = fn if fn is not None else sum
        # Type narrowing: filtered contains numeric values after mapping
        filtered_numeric: list[int | float] = [
            cast("int | float", x) for x in filtered if isinstance(x, (int, float))
        ]
        if filtered_numeric:
            # Type narrowing: agg_fn is Callable[[list[int | float]], int | float]
            agg_fn_typed = cast("Callable[[list[int | float]], int | float]", agg_fn)
            agg_result = agg_fn_typed(filtered_numeric)
        else:
            agg_result = 0
        return cast("object", agg_result)

    @staticmethod
    @overload
    def fields[T](
        source: Mapping[str, object] | object,
        name_or_spec: str,
        *,
        default: T | None = None,
        required: bool = False,
        ops: dict[str, object] | None = None,
        on_error: str = "stop",
    ) -> T | None: ...

    @staticmethod
    @overload
    def fields[T](
        source: Mapping[str, object] | object,
        name_or_spec: dict[str, dict[str, object] | T | None],
        *,
        default: T | None = None,
        required: bool = False,
        ops: dict[str, object] | None = None,
        on_error: str = "stop",
    ) -> dict[str, T | None] | r[dict[str, T]]: ...

    @staticmethod
    def fields[T](  # noqa: PLR0913
        source: Mapping[str, object] | object,
        name_or_spec: str | dict[str, dict[str, object] | T | None],
        *,
        default: T | None = None,
        required: bool = False,
        ops: dict[str, object] | None = None,
        on_error: str = "stop",
    ) -> dict[str, T | None] | r[dict[str, T]] | T | None:
        """Extract and process field(s) using DSL mnemonic pattern (generalized from field).

        Generic replacement for: Repeated u.get() + u.guard() + u.build() chains

        Extracts field(s) from source (dict/object), applies optional DSL operations,
        and validates required fields. Uses mnemonic field names for clarity.

        Args:
            source: Source data (dict or object)
            name_or_spec: Field name (single) or spec dict (multiple)
            default: Default value if field not found (single field only)
            required: If True, returns None on missing (single field only)
            ops: Optional DSL operations dict (single field only)
            on_error: Error handling for multiple fields ("stop", "skip", "collect")

        Returns:
            Extracted value(s) or default/None/dict/r[dict]

        Example:
            # Single field extraction
            name = u.fields(payload, "name", default="")

            # Single field with DSL operations
            port = u.fields(
                config,
                "port",
                default=8080,
                ops={"convert": int},
            )

            # Required single field
            api_key = u.fields(env, "API_KEY", required=True)
            if api_key is None:
                return r.fail("API_KEY is required")

            # Multiple fields
            data = u.fields(
                payload,
                {"name": "", "age": 0, "email": ""},
            )

        """
        # Handle multiple fields case
        if isinstance(name_or_spec, dict):
            # For multi-field case, on_error is required
            return FlextUtilities._fields_multi(source, name_or_spec, on_error=on_error)

        # Handle single field case
        # For single field, on_error is ignored (only used for multi-field)
        name = cast("str", name_or_spec)
        value = u.get(source, name, default=default)
        if value is None and required:
            return None
        if ops is not None:
            built_value = u.build(value, ops=ops, on_error="stop")
            return cast("T | None", built_value)
        return cast("T | None", value)

    @staticmethod
    def _fields_multi[T](
        source: Mapping[str, object] | object,
        spec: dict[str, dict[str, object] | T | None],
        *,
        on_error: str = "stop",
    ) -> dict[str, T | None] | r[dict[str, T]]:
        """Extract multiple fields using DSL mnemonic specification.

        Generic replacement for: Multiple u.extract() calls, manual dict construction

        Extracts multiple fields from source using mnemonic spec format:
        {"field_name": default_value} or {"field_name": {"default": value, "ops": {...}}}

        Args:
            source: Source data (dict or object)
            spec: Field specification dict:
                - Simple: {"name": "default"} → extracts with default
                - DSL: {"port": {"default": 8080, "ops": {"convert": int}}} → extracts with ops
                - Required: {"api_key": None} → required field (None means required)
            on_error: Error handling ("stop", "skip", "collect")

        Returns:
            dict[str, T | None] with extracted values, or r[dict[str, T]] if on_error="stop" and required field missing

        Example:
            # Simple extraction
            data = u.fields(
                payload,
                {"name": "", "age": 0, "email": ""},
            )

            # With DSL operations
            config = u.fields(
                raw_config,
                {
                    "port": {"default": 8080, "ops": {"convert": int}},
                    "host": {"default": "localhost", "ops": {"ensure": "str"}},
                    "timeout": {"default": 300, "ops": {"convert": int}},
                },
            )

            # Required fields
            result = u.fields(
                env,
                {"API_KEY": None, "API_SECRET": None},
                on_error="stop",
            )
            if isinstance(result, r) and result.is_failure:
                return result

        """
        result: dict[str, T | None] = {}
        errors: list[str] = []

        for field_name, field_spec in spec.items():
            # Determine if field is required and get spec
            if isinstance(field_spec, dict):
                field_default = field_spec.get("default")
                field_ops = field_spec.get("ops")
                field_required = field_default is None and "default" not in field_spec
            else:
                field_default = field_spec
                field_ops = None
                field_required = field_spec is None

            # Extract field using single field extraction (avoid recursion)
            value = u.get(source, field_name, default=field_default)
            if value is None and field_required:
                extracted = None
            elif field_ops is not None:
                # Type narrowing: field_ops is dict[str, object]
                field_ops_dict = cast("dict[str, object]", field_ops)
                build_result = u.build(value, ops=field_ops_dict, on_error="stop")
                extracted = cast("T | None", build_result)
            else:
                extracted = cast("T | None", value)

            if extracted is None and field_required:
                error_msg = f"Required field '{field_name}' is missing"
                if on_error == "stop":
                    return r[dict[str, T]].fail(error_msg)
                if on_error == "collect":
                    errors.append(error_msg)
                    continue
                # on_error == "skip": continue without adding field
                continue

            result[field_name] = extracted

        if errors and on_error == "collect":
            return r[dict[str, T]].fail(f"Field extraction errors: {', '.join(errors)}")

        # Always return dict when no errors (on_error="stop" or "skip" with no errors)
        return result

    @staticmethod
    def construct[T](
        spec: dict[str, object | dict[str, object]],
        source: Mapping[str, object] | object | None = None,
        *,
        on_error: str = "stop",
    ) -> dict[str, object]:
        """Construct object using mnemonic specification pattern.

        Generic replacement for: Manual dict construction with u.get(), u.guard(), u.build()

        Builds object from mnemonic spec that maps target keys to source fields or DSL operations.
        Supports field mapping, default values, and DSL operations.

        Args:
            spec: Construction specification:
                - Direct: {"target_key": "source_field"} → maps source_field to target_key
                - Default: {"target_key": {"field": "source_field", "default": value}} → with default
                - DSL: {"target_key": {"field": "source_field", "ops": {...}}} → with DSL ops
                - Literal: {"target_key": {"value": literal}} → uses literal value
            source: Optional source data (if None, uses literal values from spec)
            on_error: Error handling ("stop", "skip", "collect")

        Returns:
            Constructed dict with target keys

        Example:
            # Simple mapping
            plugin_info = u.construct(
                {
                    "name": "plugin_name",
                    "type": "plugin_type",
                    "variant": "default_variant",
                },
                source=plugin_data,
            )

            # With defaults and DSL
            config = u.construct(
                {
                    "port": {"field": "port", "default": 8080, "ops": {"convert": int}},
                    "host": {"field": "host", "default": "localhost", "ops": {"ensure": "str"}},
                    "timeout": {"value": 300},  # Literal value
                },
                source=raw_config,
            )

        """
        constructed: dict[str, object] = {}

        for target_key, target_spec in spec.items():
            try:
                # Literal value
                if isinstance(target_spec, dict) and "value" in target_spec:
                    constructed[target_key] = target_spec["value"]
                    continue

                # Field mapping
                if isinstance(target_spec, str):
                    # Simple field name mapping
                    source_field = target_spec
                    field_default = None
                    field_ops = None
                elif isinstance(target_spec, dict):
                    source_field = cast("str", target_spec.get("field", target_key))
                    field_default = target_spec.get("default")
                    field_ops = target_spec.get("ops")
                else:
                    # Direct value
                    constructed[target_key] = target_spec
                    continue

                # Extract from source using u.when() for conditional extraction (DSL pattern)
                if source is None:
                    constructed[target_key] = u.when(
                        condition=field_default is not None,
                        then_value=field_default,
                        else_value=None,
                    )
                    continue

                # Extract field value
                extracted_raw = u.extract(
                    source,
                    source_field,
                    default=field_default,
                    required=False,
                )
                # Apply ops if provided using u.build()
                if field_ops is not None and extracted_raw is not None:
                    field_ops_dict = cast("dict[str, object]", field_ops)
                    build_result = u.build(extracted_raw, ops=field_ops_dict)
                    extracted = cast("object", build_result)
                else:
                    extracted = extracted_raw
                # Use u.when() for conditional assignment (DSL pattern)
                constructed[target_key] = u.when(
                    condition=extracted is not None,
                    then_value=extracted,
                    else_value=field_default,
                )

            except Exception as e:
                error_msg = f"Failed to construct '{target_key}': {e}"
                if on_error == "stop":
                    raise ValueError(error_msg) from e
                if on_error == "skip":
                    continue
                # on_error == "collect": continue but could log

        return constructed

    @staticmethod
    @overload
    def take[T](
        data_or_items: Mapping[str, object] | object,
        key_or_n: str,
        *,
        as_type: type[T] | None = None,
        default: T | None = None,
        guard: bool = True,
        from_start: bool = True,
    ) -> T | None: ...

    @staticmethod
    @overload
    def take[T](
        data_or_items: dict[str, T],
        key_or_n: int,
        *,
        as_type: type[T] | None = None,
        default: T | None = None,
        guard: bool = True,
        from_start: bool = True,
    ) -> dict[str, T]: ...

    @staticmethod
    @overload
    def take[T](
        data_or_items: list[T] | tuple[T, ...],
        key_or_n: int,
        *,
        as_type: type[T] | None = None,
        default: T | None = None,
        guard: bool = True,
        from_start: bool = True,
    ) -> list[T]: ...

    @staticmethod
    def take[T](  # noqa: PLR0913
        data_or_items: Mapping[str, object]
        | object
        | dict[str, T]
        | list[T]
        | tuple[T, ...],
        key_or_n: str | int,
        *,
        as_type: type[T] | None = None,
        default: T | None = None,
        guard: bool = True,
        from_start: bool = True,
    ) -> dict[str, T] | list[T] | T | None:
        """Unified take function (generalized from take_n).

        Generic replacement for: u.get() + isinstance() + cast() patterns, list slicing

        Automatically detects operation based on second argument type:
        - If key_or_n is str: extracts value from dict/object with type guard
        - If key_or_n is int: takes first N items from list/dict

        Args:
            data_or_items: Source data (dict/object) or items (list/dict)
            key_or_n: Key name (str) or number of items (int)
            as_type: Optional type to guard against (for extraction mode)
            default: Default value if not found or type mismatch (for extraction mode)
            guard: If True, validate type; if False, return as-is (for extraction mode)
            from_start: If True, take from start; if False, take from end (for slice mode)

        Returns:
            Extracted value with type guard or sliced items

        Example:
            # Extract value (original take behavior)
            port = u.take(config, "port", as_type=int, default=8080)
            name = u.take(obj, "name", as_type=str, default="unknown")

            # Take N items (generalized from take_n)
            keys = u.take(plugins_dict, 10)
            items = u.take(items_list, 5)

        """
        # Detect operation mode based on key_or_n type
        if isinstance(key_or_n, str):
            # Extraction mode: extract value from dict/object
            data = cast("Mapping[str, object] | object", data_or_items)
            key = key_or_n
            value = u.get(data, key, default=default)
            if value is None:
                return cast("T | None", default)
            if as_type and guard:
                guarded = u.guard(value, as_type, return_value=True, default=default)
                return cast("T | None", guarded)
            return cast("T | None", value)

        # Slice mode: take N items from list/dict
        n = key_or_n
        if isinstance(data_or_items, dict):
            items_dict = cast("dict[str, T]", data_or_items)
            # Use u.keys() for unified key extraction
            keys = u.keys(items_dict)
            selected_keys = keys[:n] if from_start else keys[-n:]
            # Cannot use u.map() here - would cause recursion (u.map calls itself for lists)
            # Use dict comprehension directly
            return {k: items_dict[k] for k in selected_keys}
        items_list_or_tuple = cast("list[T] | tuple[T, ...]", data_or_items)
        items_list = list(items_list_or_tuple)
        sliced = items_list[:n] if from_start else items_list[-n:]
        return cast("list[T]", sliced)

    @staticmethod
    def pick[T](
        data: Mapping[str, object] | object,
        *keys: str,
        as_dict: bool = True,
    ) -> dict[str, object] | list[object]:
        """Pick multiple fields at once (mnemonic: pick = select fields).

        Generic replacement for: Multiple u.get() calls

        Args:
            data: Source data (dict or object)
            *keys: Field names to pick
            as_dict: If True, return dict; if False, return list

        Returns:
            Dict with picked fields or list of values

        Example:
            fields = u.pick(data, "name", "email", "age")
            values = u.pick(data, "x", "y", "z", as_dict=False)

        """
        # Cannot use u.map() here - keys is tuple[str, ...] which u.map treats as list
        # and would cause recursion (u.map calls itself for lists)
        # Use list comprehension directly for efficiency
        if as_dict:
            return {k: u.get(data, k) for k in keys}
        # Use list comprehension for values
        return [u.get(data, k) for k in keys]

    @staticmethod
    def as_[T](
        value: object,
        target: type[T],
        *,
        default: T | None = None,
        strict: bool = False,
    ) -> T | None:
        """Type conversion with guard (mnemonic: as_ = convert to type).

        Generic replacement for: isinstance() + cast() patterns

        Args:
            value: Value to convert
            target: Target type
            default: Default if conversion fails
            strict: If True, only exact type; if False, allow coercion

        Returns:
            Converted value or default

        Example:
            port = u.as_(config.get("port"), int, default=8080)
            name = u.as_(value, str, default="")

        """
        if isinstance(value, target):
            return cast("T", value)
        if strict:
            return default
        # Try coercion using u.convert()
        try:
            # Cast value to GeneralValueType for type compatibility
            value_typed: t.GeneralValueType = cast("t.GeneralValueType", value)
            converted = (
                u.convert(value_typed, target, default)
                if default is not None
                else u.convert(value_typed, target, cast("T", None))
            )
            return converted if converted is not None else default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def or_[T](
        *values: T | None,
        default: T | None = None,
    ) -> T | None:
        """Return first non-None value (mnemonic: or_ = fallback chain).

        Generic replacement for: value1 or value2 or default patterns

        Args:
            *values: Values to try in order
            default: Default if all are None

        Returns:
            First non-None value or default

        Example:
            port = u.or_(config.get("port"), env.get("PORT"), default=8080)
            name = u.or_(user.name, user.username, default="anonymous")
            port = u.or_(config.get("port"), 8080)  # Two-arg form

        """
        # Handle all cases: or_(*values, default=...)
        for value in values:
            if value is not None:
                return value
        return default

    @staticmethod
    def when[T](
        *,
        condition: bool = False,
        then_value: T | None = None,
        else_value: T | None = None,
    ) -> T | None:
        """Conditional value (mnemonic: when = if-then-else, same as if_).

        Generic replacement for: value if condition else default patterns

        Args:
            condition: Boolean condition
            then_value: Value if condition is True
            else_value: Value if condition is False (None if not provided)

        Returns:
            then_value or else_value

        Example:
            port = u.when(condition=debug, then_value=8080, else_value=80)
            name = u.when(condition=user.is_active, then_value=user.name)

        """
        return then_value if condition else else_value

    @staticmethod
    def try_[T](
        func: Callable[[], T],
        *,
        default: T | None = None,
        catch: type[Exception] | tuple[type[Exception], ...] = Exception,
    ) -> T | None:
        """Try operation with fallback (mnemonic: try_ = safe execution).

        Generic replacement for: try/except blocks with default

        Args:
            func: Function to execute
            default: Default if exception occurs
            catch: Exception types to catch (default: Exception)

        Returns:
            Function result or default

        Example:
            port = u.try_(lambda: int(config["port"]), default=8080)
            data = u.try_(lambda: json.loads(raw), default={})

        """
        try:
            return func()
        except catch:
            return default

    @staticmethod
    def chain(
        value: object,
        *funcs: Callable[[object], object],
    ) -> object:
        """Chain operations (mnemonic: chain = pipeline).

        Business Rule: Execute a sequence of functions in order, passing each
        result to the next function. This is the functional pipeline pattern.

        Generic replacement for: func3(func2(func1(value))) patterns

        Args:
            value: Initial value (any type)
            *funcs: Functions to apply in sequence

        Returns:
            Final result after all operations

        Example:
            result = u.chain(
                data,
                lambda x: u.get(x, "items"),
                lambda x: u.filter(x, lambda i: i > 0),
                lambda x: u.map(x, lambda i: i * 2),
            )

        """
        current: object = value
        for func in funcs:
            current = func(current)
        return current

    @staticmethod
    def flow(
        value: object,
        *ops: dict[str, object] | Callable[[object], object],
    ) -> object:
        """Flow operations using DSL or functions (mnemonic: flow = fluent pipeline).

        Generic replacement for: u.build() + u.chain() combinations

        Args:
            value: Initial value
            *ops: Operations (dict DSL or callable functions)

        Returns:
            Final result

        Example:
            result = u.flow(
                data,
                {"ensure": "dict"},
                {"get": "items", "default": []},
                lambda x: u.filter(x, lambda i: i > 0),
                {"map": lambda i: i * 2},
            )

        """
        current: object = value
        for op in ops:
            if isinstance(op, dict):
                current = u.build(current, ops=op)
            elif callable(op):
                current = op(current)
        return current

    @staticmethod
    def from_[T](
        source: Mapping[str, object] | object | None,
        key: str,
        *,
        as_type: type[T] | None = None,
        default: T,
    ) -> T:
        """Extract from source with type guard (mnemonic: from_ = extract from source).

        Generic replacement for: u.take() + u.when() patterns for optional sources

        Args:
            source: Source data (dict/object/None)
            key: Key/attribute name
            as_type: Optional type to guard against
            default: Default value if source is None or field missing

        Returns:
            Extracted value with type guard or default (always returns T, never None)

        Example:
            port = u.from_(config_obj, "port", as_type=int, default=300)
            name = u.from_(obj, "name", as_type=str, default="")

        """
        if source is None:
            return default
        if as_type:
            taken = u.take(source, key, as_type=as_type, default=default)
            return cast("T", taken) if taken is not None else default
        gotten = u.get(source, key, default=default)
        return cast("T", gotten) if gotten is not None else default

    @staticmethod
    def all_(*values: object) -> bool:
        """Check if all values are truthy (mnemonic: all_ = all truthy).

        Generic replacement for: all([v1, v2, v3]) or if v1 and v2 and v3

        Args:
            *values: Values to check

        Returns:
            True if all values are truthy

        Example:
            if u.all_(name, email, age):
                process_user()

        """
        # Cannot use u.map() here - values is a tuple from *values, would cause recursion
        # Use list comprehension directly (simpler and avoids recursion)
        return all(bool(v) for v in values)

    @staticmethod
    def any_(*values: object) -> bool:
        """Check if any value is truthy (mnemonic: any_ = any truthy).

        Generic replacement for: any([v1, v2, v3]) or if v1 or v2 or v3

        Args:
            *values: Values to check

        Returns:
            True if any value is truthy

        Example:
            if u.any_(config, env, default):
                use_value()

        """
        # Cannot use u.map() here - values is a tuple from *values, would cause recursion
        # Use list comprehension directly (simpler and avoids recursion)
        return any(bool(v) for v in values)

    @staticmethod
    def none_(*values: object) -> bool:
        """Check if all values are falsy (mnemonic: none_ = none truthy).

        Generic replacement for: not any([v1, v2, v3]) or if not (v1 or v2 or v3)

        Args:
            *values: Values to check

        Returns:
            True if all values are falsy

        Example:
            if u.none_(name, email):
                return r.fail("Missing required fields")

        """
        return not any(bool(v) for v in values)

    @staticmethod
    def join(
        items: list[str] | dict[str, str] | Mapping[str, str],
        *,
        sep: str = ",",
        key_sep: str | None = None,
    ) -> str:
        """Join items into string (mnemonic: join = string join).

        Generic replacement for: ",".join(items) or manual string building

        Args:
            items: Items to join (list of strings or dict)
            sep: Separator for list items (default: ",")
            key_sep: Separator for dict key-value pairs (e.g., "=")

        Returns:
            Joined string

        Example:
            result = u.join(["a", "b", "c"], sep=",")
            result = u.join({"a": "1", "b": "2"}, sep=",", key_sep="=")

        """
        if isinstance(items, (dict, Mapping)):
            if key_sep:
                # Use u.map for unified mapping - convert items() to list first
                items_list = list(items.items())
                # Type narrowing: items_list is list of tuples
                items_tuples: list[tuple[str, str]] = cast(
                    "list[tuple[str, str]]", items_list
                )
                mapped = cast(
                    "list[str]",
                    u.map(items_tuples, lambda kv: f"{kv[0]}{key_sep}{kv[1]}"),
                )
                return sep.join(mapped)
            # Use u.vals() for unified value extraction, then u.map for mapping
            # Cast items to dict for type compatibility
            items_dict: dict[str, str] = cast("dict[str, str]", items)
            values_list: list[str] = cast("list[str]", u.vals(items_dict))
            # Type narrowing: values_list is list[str], str() is callable
            str_func_vals: Callable[[str], str] = str
            mapped = cast("list[str]", u.map(values_list, str_func_vals))
            return sep.join(mapped)
        # Use u.map for unified mapping
        # Type narrowing: items is collection of strings
        items_collection: list[str] | tuple[str, ...] | set[str] | frozenset[str] = (
            cast("list[str] | tuple[str, ...] | set[str] | frozenset[str]", items)
        )
        str_func_collection: Callable[[str], str] = str
        mapped = cast("list[str]", u.map(items_collection, str_func_collection))
        return sep.join(mapped)

    @staticmethod
    def has(
        obj: object,
        *attrs: str,
    ) -> bool:
        """Check if object has all attributes (mnemonic: has = hasattr).

        Generic replacement for: hasattr(obj, "attr") and hasattr(obj, "attr2")

        Args:
            obj: Object to check
            *attrs: Attribute names to check

        Returns:
            True if object has all attributes

        Example:
            if u.has(config, "port", "host", "timeout"):
                use_config()

        """
        return all(hasattr(obj, attr) for attr in attrs)

    @staticmethod
    def group[T, K](
        items: list[T] | tuple[T, ...],
        key: str | Callable[[T], K],
    ) -> dict[K, list[T]]:
        """Group items by key (mnemonic: group = group by).

        Generic replacement for: Manual defaultdict loops, itertools.groupby setup

        Args:
            items: Items to group
            key: Field name (str) or key function (callable)

        Returns:
            dict[key, list[items]]

        Example:
            # Group by field
            by_type = u.group(users, "role")
            # → {"admin": [user1, user2], "user": [user3]}

            # Group by function
            by_len = u.group(words, lambda w: len(w))
            # → {3: ["cat", "dog"], 5: ["house"]}

        """
        result: dict[K, list[T]] = {}
        # Use u.map for unified mapping to extract keys
        items_list: list[T] | tuple[T, ...] = cast("list[T] | tuple[T, ...]", items)
        if callable(key):
            # Type narrowing: key is Callable[[T], K]
            key_func = cast("Callable[[T], K]", key)
            keys = cast("list[K]", u.map(items_list, key_func))
        else:
            # Use u.get for unified extraction (works for dicts and objects)
            # Type narrowing: key is str
            key_str = cast("str", key)
            keys = cast("list[K]", u.map(items_list, lambda item: u.get(item, key_str)))
        # Group items by keys using zip_ for unified zip
        # Cast to object lists for type compatibility with zip_
        items_obj: list[object] | tuple[object, ...] = cast(
            "list[object] | tuple[object, ...]", items
        )
        keys_obj: list[object] = cast("list[object]", keys)
        pairs = cast("list[tuple[T, K]]", u.zip_(items_obj, keys_obj))
        for item, k in pairs:
            result.setdefault(k, []).append(item)
        return result

    @staticmethod
    def chunk[T](
        items: list[T] | tuple[T, ...],
        size: int,
    ) -> list[list[T]]:
        """Split items into chunks (mnemonic: chunk = batch).

        Generic replacement for: Manual chunking loops, list slicing patterns

        Args:
            items: Items to chunk
            size: Chunk size

        Returns:
            list[list[T]] of chunks

        Example:
            batches = u.chunk(entries, size=100)
            # → [[entry1...entry100], [entry101...entry200]]

        """
        # Cannot use u.map() here - would cause recursion (u.map calls itself for lists)
        # Use list comprehension directly for chunking
        return [list(items[i : i + size]) for i in range(0, len(items), size)]

    @staticmethod
    def zip_(*items: list[object] | tuple[object, ...]) -> list[tuple[object, ...]]:
        """Zip multiple lists (mnemonic: zip_ = zip).

        Generic replacement for: zip(list1, list2, list3) patterns

        Args:
            *items: Lists/tuples to zip

        Returns:
            list[tuple] of zipped items

        Example:
            pairs = u.zip_([1, 2], ["a", "b"])
            # → [(1, "a"), (2, "b")]

        """
        if not items:
            return []
        return list(zip(*items, strict=False))

    @staticmethod
    def count[T](
        items: list[T] | tuple[T, ...],
        predicate: Callable[[T], bool] | None = None,
    ) -> int:
        """Count items matching predicate (mnemonic: count = count).

        Generic replacement for: sum(1 for x in items if pred(x)) or len([x for x in items if pred(x)])
        Uses u.filter() for unified filtering.

        Args:
            items: Items to count
            predicate: Optional filter function (counts all if None)

        Returns:
            Count of matching items

        Example:
            total = u.count(items)
            active = u.count(users, lambda u: u.is_active)

        """
        if predicate is None:
            return len(items)
        # Use u.filter() for unified filtering, then count
        # Type narrowing: items is list[T] | tuple[T, ...], predicate is Callable[[T], bool]
        items_list: list[T] | tuple[T, ...] = cast("list[T] | tuple[T, ...]", items)
        predicate_func = cast("Callable[[T], bool]", predicate)
        filtered = u.filter(items_list, predicate_func)
        return (
            len(filtered)
            if isinstance(filtered, (list, tuple, set, frozenset))
            else 1
            if filtered
            else 0
        )

    @staticmethod
    def err[T](
        result: r[T],
        *,
        default: str = "Unknown error",
    ) -> str:
        """Extract error message from r (mnemonic: err = error).

        Generic replacement for: str(result.error) if result.error else "Unknown error"

        Args:
            result: r to extract error from
            default: Default error message if error is None/empty

        Returns:
            Error message string

        Example:
            error_msg = u.err(result, default="Operation failed")
            # → "Connection timeout" or "Operation failed"

        """
        if result.is_failure:
            error_str = result.error
            if error_str:
                return str(error_str)
        return default

    @staticmethod
    def val[T](
        result: r[T],
        *,
        default: T | None = None,
    ) -> T | None:
        """Extract value from r (mnemonic: val = value).

        Generic replacement for: result.value if result.is_success else default

        Args:
            result: r to extract value from
            default: Default value if result is failure

        Returns:
            Value or default

        Example:
            data = u.val(result, default={})
            # → result.value or {}

        """
        # Use u.when() for conditional return (DSL pattern)
        return u.when(
            condition=result.is_success,
            then_value=result.value,
            else_value=default,
        )

    @staticmethod
    def unwrap[T](
        value: T | r[T],
        *,
        default: T | None = None,
    ) -> T:
        """Unwrap r or return value (mnemonic: unwrap = extract value).

        Generic replacement for: result.value if isinstance(result, r) else result

        Args:
            value: Value or r
            default: Default if result is failure

        Returns:
            Unwrapped value or default

        Example:
            data = u.unwrap(fields_result)
            value = u.unwrap(result, default=[])

        """
        if isinstance(value, r):
            if value.is_failure:
                # Use u.or_() for default fallback (DSL pattern)
                return cast("T", u.or_(default, None))
            return value.value
        return value

    @staticmethod
    def req[T](
        value: T | None,
        *,
        name: str = "value",
    ) -> r[T]:
        """Require non-None value (mnemonic: req = required).

        Generic replacement for: if not value: return r.fail()

        Args:
            value: Value to check
            name: Field name for error message

        Returns:
            r[T]: Ok(value) or Fail with error

        Example:
            result = u.req(tap_name, name="tap_name")
            if result.is_failure:
                return result

        """
        if value is None or (isinstance(value, str) and not value):
            return r[T].fail(f"{name} is required")
        return r[T].ok(value)

    @staticmethod
    def vals[T](
        items: dict[str, T] | r[dict[str, T]],
        *,
        default: list[T] | None = None,
    ) -> list[T]:
        """Extract values from dict or result (mnemonic: vals = values).

        Generic replacement for: list(result.value.values()) or list(dict.values())

        Args:
            items: Dict or r containing dict
            default: Default if items is failure or None

        Returns:
            List of values

        Example:
            plugins = u.vals(plugins_result)
            values = u.vals(data_dict)

        """
        # Use u.val() for result unwrapping with default
        items_dict = u.val(items, default={}) if isinstance(items, r) else items
        # Handle Mapping[str, T] by converting to dict for .values() access
        if isinstance(items_dict, Mapping) and not isinstance(items_dict, dict):
            items_dict = dict(items_dict)
        # Use u.or_() for default fallback (DSL pattern)
        if items_dict:
            return list(items_dict.values())
        result = u.or_(default, [])
        return cast("list[T]", result)

    @staticmethod
    def keys[T](
        items: dict[str, T] | r[dict[str, T]],
        *,
        default: list[str] | None = None,
    ) -> list[str]:
        """Extract keys from dict or result (mnemonic: keys = dict keys).

        Generic replacement for: list(result.value.keys()) or list(dict.keys())

        Args:
            items: Dict or r containing dict
            default: Default if items is failure or None

        Returns:
            List of keys

        Example:
            plugin_names = u.keys(plugins_result)
            field_names = u.keys(data_dict)

        """
        # Use u.val() for result unwrapping with default
        items_dict = u.val(items, default={}) if isinstance(items, r) else items
        # Use u.or_() for default fallback (DSL pattern)
        keys_list = list(items_dict.keys()) if items_dict else []
        return keys_list or (default if default is not None else [])

    @staticmethod
    def mul[T: (int, float)](
        value: T,
        factor: T,
    ) -> T:
        """Multiply value by factor (mnemonic: mul = multiply).

        Generic replacement for: value * factor patterns

        Args:
            value: Value to multiply
            factor: Multiplication factor

        Returns:
            Multiplied value

        Example:
            total = u.mul(count, 3)
            price = u.mul(quantity, unit_price)

        """
        return cast("T", value * factor)

    @staticmethod
    def sum[T: (int, float)](
        items: list[T] | tuple[T, ...] | dict[str, T] | r[list[T] | dict[str, T]],
        *,
        mapper: Callable[[T], int | float]
        | Callable[[str, T], int | float]
        | None = None,
        default: T | None = None,
    ) -> T:
        """Sum numeric values from collection (mnemonic: sum = sum).

        Generic replacement for: sum(items), sum(dict.values()), sum(map(...))

        Args:
            items: Items to sum (list, tuple, dict, or result)
            mapper: Optional function to extract numeric value
                - For lists: mapper(item) -> number
                - For dicts: mapper(key, value) -> number
            default: Default value if empty (default: 0)

        Returns:
            Sum of numeric values

        Example:
            total = u.sum([1, 2, 3])
            # → 6

            total = u.sum({"a": 1, "b": 2}, mapper=lambda k, v: v)
            # → 3

            total = u.sum(u.map(items, lambda x: x.count))
            # → Sum of counts

        """
        # Use u.val() for result unwrapping with default
        if isinstance(items, r):
            items_unwrapped = u.val(items, default=[])
            if u.empty(items_unwrapped):
                # Use u.or_() for default fallback (DSL pattern)
                return cast("T", u.or_(default, 0))
            items = cast("list[T] | tuple[T, ...] | dict[str, T]", items_unwrapped)

        # Use u.agg() for unified aggregation when mapper provided
        if mapper is not None:
            # Use u.agg() with mapper for unified aggregation
            # Type narrowing: items is collection, mapper is callable
            # Wrap sum() to avoid overloaded function type issue
            def sum_wrapper(values: list[object]) -> object:
                """Wrapper for sum() to avoid overloaded function type issue."""
                return sum(cast("list[int | float]", values))

            if isinstance(items, dict):
                # For dict, extract values first
                items_values = list(items.values())
                mapper_func = cast("Callable[[object], int | float]", mapper)
                agg_result = u.agg(items_values, mapper_func, fn=sum_wrapper)
            else:
                # For list/tuple
                items_list: list[object] | tuple[object, ...] = cast(
                    "list[object] | tuple[object, ...]", items
                )
                mapper_func = cast("Callable[[object], int | float]", mapper)
                agg_result = u.agg(items_list, mapper_func, fn=sum_wrapper)
            return cast("T", agg_result)

        # Direct sum - cannot use u.agg() with lambda x: x (causes recursion)
        # Use direct sum() on values
        if isinstance(items, dict):
            values = list(items.values())
            return (
                cast("T", sum(cast("list[int | float]", values)))
                if values
                else cast("T", u.or_(default, 0))
            )
        if isinstance(items, (list, tuple)):
            return (
                cast("T", sum(cast("list[int | float]", items)))
                if items
                else cast("T", u.or_(default, 0))
            )
        # Use u.or_() for default fallback (DSL pattern)
        return cast("T", u.or_(default, 0))

    @staticmethod
    def first[T](
        items: list[T] | tuple[T, ...] | dict[str, T],
        *,
        default: T | None = None,
    ) -> T | None:
        """Get first item (mnemonic: first = first element).

        Generic replacement for: items[0] if items else None

        Args:
            items: Items to get first from
            default: Default if empty

        Returns:
            First item or default

        Example:
            first_user = u.first(users)
            first_key = u.first(data_dict)

        """
        # Use u.empty() for unified empty check
        if u.empty(items):
            return default
        # Use u.when() for conditional access (DSL pattern)
        if isinstance(items, dict):
            keys = u.keys(items)
            return u.when(
                condition=bool(keys), then_value=items[keys[0]], else_value=default
            )
        return u.when(condition=bool(items), then_value=items[0], else_value=default)

    @staticmethod
    def last[T](
        items: list[T] | tuple[T, ...] | dict[str, T],
        *,
        default: T | None = None,
    ) -> T | None:
        """Get last item (mnemonic: last = last element).

        Generic replacement for: items[-1] if items else None

        Args:
            items: Items to get last from
            default: Default if empty

        Returns:
            Last item or default

        Example:
            last_user = u.last(users)
            last_value = u.last(data_dict)

        """
        # Use u.empty() for unified empty check
        if u.empty(items):
            return default
        # Use u.when() for conditional access (DSL pattern)
        if isinstance(items, dict):
            keys = list(items.keys())
            return u.when(
                condition=bool(keys), then_value=items[keys[-1]], else_value=default
            )
        return u.when(condition=bool(items), then_value=items[-1], else_value=default)

    @staticmethod
    def at[T](
        items: list[T] | tuple[T, ...] | dict[str, T],
        index: int | str,
        *,
        default: T | None = None,
    ) -> T | None:
        """Get item at index/key (mnemonic: at = get at position).

        Generic replacement for: items[index] with safe access

        Args:
            items: Items to access
            index: Index (int) or key (str)
            default: Default if not found

        Returns:
            Item at index/key or default

        Example:
            user = u.at(users, 0)
            value = u.at(data_dict, "key")

        """
        try:
            # Use u.when() for conditional access (DSL pattern)
            if isinstance(items, dict):
                return u.when(
                    condition=isinstance(index, str),
                    then_value=items.get(cast("str", index), default),
                    else_value=default,
                )
            # Use u.when() for conditional access (DSL pattern)
            return u.when(
                condition=0 <= cast("int", index) < len(items),
                then_value=items[cast("int", index)],
                else_value=default,
            )
        except (IndexError, KeyError, TypeError):
            return default

    @staticmethod
    def flat[T](
        items: list[list[T] | tuple[T, ...]]
        | list[list[T]]
        | list[tuple[T, ...]]
        | tuple[list[T], ...],
    ) -> list[T]:
        """Flatten nested lists (mnemonic: flat = flatten).

        Generic replacement for: [item for sublist in items for item in sublist]

        Args:
            items: Nested list/tuple structure

        Returns:
            Flattened list

        Example:
            flat_list = u.flat([[1, 2], [3, 4]])
            # → [1, 2, 3, 4]

        """
        # Use u.map for unified mapping - convert each sublist to list
        # Process each sublist to extract items
        # Type narrowing: items is list[list[T] | tuple[T, ...]] | tuple[list[T], ...]
        items_typed = cast("list[list[T] | tuple[T, ...]] | tuple[list[T], ...]", items)
        list_func: Callable[[list[T] | tuple[T, ...]], list[T]] = list
        processed = cast("list[list[T]]", u.map(items_typed, list_func))
        # Flatten using list comprehension for cleaner functional style
        return [item for sublist in processed for item in sublist]

    @staticmethod
    def ok[T](
        value: T,
    ) -> r[T]:
        """Create success result (mnemonic: ok = success).

        Generic replacement for: r[T].ok(value)

        Args:
            value: Value to wrap

        Returns:
            r[T] with success

        Example:
            result = u.ok(data)
            # → r.ok(data)

        """
        return r[T].ok(value)

    @staticmethod
    def fail[T](
        error: str,
    ) -> r[T]:
        """Create failure result (mnemonic: fail = failure).

        Business Rule: Failures don't carry a value, only an error message.
        The type parameter T allows type-safe failure results that match expected return types.

        Generic replacement for: r[T].fail(error)

        Args:
            error: Error message

        Returns:
            r[T] with failure (type-safe failure matching expected return type)

        Example:
            result: r[Entry] = u.fail[Entry]("Operation failed")
            # → r[Entry].fail("Operation failed")

        """
        return r[T].fail(error)

    @staticmethod
    def then[T, R](
        result: r[T],
        func: Callable[[T], r[R]],
    ) -> r[R]:
        """Chain operations (mnemonic: then = flat_map).

        Generic replacement for: result.flat_map(func)

        Args:
            result: Initial result
            func: Function to apply if success

        Returns:
            Chained result

        Example:
            result = u.then(parse_result, lambda data: u.ok(process(data)))

        """
        return result.flat_map(func)

    @staticmethod
    def if_[T](
        *,
        condition: bool = False,
        then_value: T | None = None,
        else_value: T | None = None,
    ) -> T | None:
        """Conditional value (mnemonic: if_ = if-then-else, alias for when).

        Generic replacement for: value1 if condition else value2

        Args:
            condition: Boolean condition
            then_value: Value if condition is True
            else_value: Value if condition is False

        Returns:
            then_value or else_value

        Example:
            port = u.if_(condition=debug, then_value=8080, else_value=80)
            mode = u.if_(condition=is_prod, then_value="production", else_value="development")

        """
        return then_value if condition else else_value

    @staticmethod
    def not_(
        *,
        value: bool = False,
    ) -> bool:
        """Negate boolean (mnemonic: not_ = not).

        Generic replacement for: not value

        Args:
            value: Boolean to negate

        Returns:
            Negated boolean

        Example:
            is_empty = u.not_(value=u.all_(items))
            is_invalid = u.not_(value=is_valid)

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

        Generic replacement for: len(items) == 0 or not items

        Args:
            items: Collection or string to check

        Returns:
            True if empty

        Example:
            if u.empty(items):
                return u.fail("Items required")

        """
        # Handle r result type
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
    @overload
    def ends(
        value: str,
        suffix: str,
    ) -> bool: ...

    @staticmethod
    @overload
    def ends(
        value: str,
        suffix: str,
        *suffixes: str,
    ) -> bool: ...

    @staticmethod
    def ends(
        value: str,
        suffix: str,
        *suffixes: str,
    ) -> bool:
        """Check if string ends with suffix(es) (generalized from ends_any).

        Generic replacement for: value.endswith(suffix) or any(value.endswith(s) for s in suffixes)

        Args:
            value: String to check
            suffix: First suffix to check
            *suffixes: Additional suffixes to check

        Returns:
            True if ends with any suffix

        Example:
            # Single suffix
            if u.ends(filename, ".json"):
                process_json()

            # Multiple suffixes
            if u.ends(filename, ".json", ".yaml", ".yml"):
                process_config()

        """
        # Combine suffix and suffixes into single tuple for unified processing
        all_suffixes: tuple[str, ...] = (suffix,) + suffixes
        if not all_suffixes:
            return False
        # Use u.any_() with u.map() for unified checking

        def check_suffix(suffix_item: str) -> bool:
            """Check if value ends with suffix."""
            return value.endswith(suffix_item)

        # Type narrowing: all_suffixes is tuple[str, ...]
        mapped = cast("list[bool]", u.map(all_suffixes, check_suffix))
        return u.any_(*mapped)

    @staticmethod
    def in_(
        value: object,
        items: list[object] | tuple[object, ...] | set[object] | dict[str, object],
    ) -> bool:
        """Check if value is in items (mnemonic: in_ = membership).

        Generic replacement for: value in items

        Args:
            value: Value to check
            items: Collection to check membership

        Returns:
            True if value is in items

        Example:
            if u.in_(role, ["admin", "user"]):
                process_user()

        """
        return value in items

    @staticmethod
    @overload
    def starts(
        value: str,
        prefix: str,
    ) -> bool: ...

    @staticmethod
    @overload
    def starts(
        value: str,
        prefix: str,
        *prefixes: str,
    ) -> bool: ...

    @staticmethod
    def starts(
        value: str,
        prefix: str,
        *prefixes: str,
    ) -> bool:
        """Check if string starts with prefix(es) (generalized from starts_any).

        Generic replacement for: any(value.startswith(p) for p in prefixes)

        Args:
            value: String to check
            prefix: First prefix to check
            *prefixes: Additional prefixes to check

        Returns:
            True if starts with any prefix

        Example:
            if u.starts(name, "tap-", "target-", "dbt-"):
                process_plugin()

        """
        # Combine prefix and prefixes into single tuple for unified processing
        all_prefixes: tuple[str, ...] = (prefix,) + prefixes
        # Use u.any_() with u.map() for unified checking
        # Use partial for cleaner function composition (DSL pattern)

        def check_prefix(prefix_item: str) -> bool:
            """Check if value starts with prefix."""
            return value.startswith(prefix_item)

        # Type narrowing: all_prefixes is tuple[str, ...]
        mapped = cast("list[bool]", u.map(all_prefixes, check_prefix))
        return u.any_(*mapped)

    @staticmethod
    @overload
    def cast[R](
        value_or_result: r[R],
        *,
        default_error: str = "Operation failed",
    ) -> r[R]: ...

    @staticmethod
    @overload
    def cast[R](
        value_or_result: R,
        *,
        default_error: str = "Operation failed",
    ) -> R: ...

    @staticmethod
    def cast[R](
        value_or_result: R | r[R],
        *,
        default_error: str = "Operation failed",
    ) -> R | r[R]:
        """Cast value or result type (generalized from cast_r).

        Generic replacement for: cast(R, value), cast_r(result)

        Generic replacement for: if result.is_success: return r.ok(cast(R, result.value)) else return r.fail(result.error)

        Args:
            value_or_result: Value or result to cast
            default_error: Default error if result.error is None

        Returns:
            Casted result or failure

        Example:
            # Cast result (generalized from cast_r)
            result = u.cast[int](str_result)
            # → r[int] with casted value or failure

            # Cast value
            value = u.cast[int]("123")
            # → 123

        """
        # Handle r[R] case (generalized from cast_r)
        if isinstance(value_or_result, r):
            if value_or_result.is_success:
                # Type narrowing: value_or_result.value is R when success
                return r[R].ok(cast("R", value_or_result.value))
            return r[R].fail(u.err(value_or_result, default=default_error))
        # Handle direct value case - value_or_result is R
        return cast("R", value_or_result)

    # ═══════════════════════════════════════════════════════════════════
    # GENERIC BUILDERS - Mnemonic DSL patterns for composition
    # ═══════════════════════════════════════════════════════════════════
    # conv() → to_str(), to_str_list(), to_int(), etc.
    # norm() → norm_str(), norm_list(), norm_join(), norm_in()
    # filter() → filter_attrs(), filter_not_none(), filter_truthy()
    # map() → map_str(), map_int(), attr_to_str_list()
    # find() → find_callable(), find_attr(), find_value()

    @staticmethod
    def conv_str(value: object, *, default: str = "") -> str:
        """Convert to string (builder: conv().str()).

        Mnemonic: conv = convert, str = string
        Uses advanced DSL: ensure_str() directly to avoid recursion.

        Args:
            value: Value to convert
            default: Default if None

        Returns:
            str: Converted string

        """
        return FlextUtilities.ensure_str(
            cast("t.GeneralValueType", value), default=default
        )

    @staticmethod
    def conv_str_list(
        value: t.GeneralValueType, *, default: list[str] | None = None
    ) -> list[str]:
        """Convert to str_list (builder: conv().str_list()).

        Mnemonic: conv = convert, str_list = list[str]
        Uses advanced DSL: ensure_str_list() directly to avoid recursion.

        Args:
            value: Value to convert
            default: Default if None

        Returns:
            list[str]: Converted list

        """
        return FlextUtilities.ensure_str_list(value, default=default or [])

    @staticmethod
    def conv_int(value: object, *, default: int = 0) -> int:
        """Convert to int (builder: conv().int()).

        Mnemonic: conv = convert, int = integer
        Uses advanced DSL: convert() directly to avoid recursion.

        Args:
            value: Value to convert
            default: Default if None

        Returns:
            int: Converted integer

        """
        converted = FlextUtilities.convert(
            cast("t.GeneralValueType", value), int, default
        )
        return cast("int", converted)

    @staticmethod
    def norm_str(value: object, *, case: str | None = None, default: str = "") -> str:
        """Normalize string (builder: norm().str()).

        Mnemonic: norm = normalize, str = string
        Uses advanced DSL: ensure_str() → normalize() for fluent composition.

        Args:
            value: Value to normalize
            case: Case normalization ("lower", "upper", "title")
            default: Default if None

        Returns:
            str: Normalized string

        """
        str_value = FlextUtilities.ensure_str(
            cast("t.GeneralValueType", value), default=default
        )
        if case:
            normalized = FlextUtilities.normalize(str_value, case=case)
            return cast("str", normalized)
        return str_value

    @staticmethod
    def norm_list(
        items: list[str] | dict[str, str],
        *,
        case: str | None = None,
        filter_truthy: bool = False,
        to_set: bool = False,
    ) -> list[str] | set[str] | dict[str, str]:
        """Normalize list/dict (builder: norm().list()).

        Mnemonic: norm = normalize, list = list[str]
        Generic replacement for: u.map(items, lambda v: u.normalize(v, case=case))

        Args:
            items: Items to normalize
            case: Case normalization
            filter_truthy: Filter truthy first
            to_set: Return set instead of list

        Returns:
            Normalized list/set/dict

        """
        if filter_truthy:
            items = cast(
                "list[str] | dict[str, str]",
                FlextUtilities.filter(items, predicate=bool),
            )

        if isinstance(items, dict):
            return FlextUtilities.map(
                items, mapper=lambda _k, v: FlextUtilities.norm_str(v, case=case)
            )

        normalized = FlextUtilities.map(
            items, mapper=lambda v: FlextUtilities.norm_str(v, case=case)
        )
        if to_set:
            return set(normalized) if isinstance(normalized, list) else set()
        return normalized if isinstance(normalized, list) else []

    @staticmethod
    def norm_join(items: list[str], *, case: str | None = None, sep: str = " ") -> str:
        """Normalize and join (builder: norm().join()).

        Mnemonic: norm = normalize, join = string join
        Uses advanced DSL: mp() → norm_str() → join() for fluent composition.

        Args:
            items: Items to normalize and join
            case: Case normalization
            sep: Separator

        Returns:
            str: Normalized and joined string

        """
        if case:
            normalized = FlextUtilities.map(
                items, mapper=lambda v: FlextUtilities.norm_str(v, case=case)
            )
        else:
            normalized = items
        # Type narrowing: normalized is list[str] after map or original items
        normalized_list = (
            normalized
            if isinstance(normalized, list)
            else list(normalized)
            if isinstance(normalized, (tuple, set))
            else []
        )
        return FlextUtilities.join(normalized_list, sep=sep)

    @staticmethod
    def norm_in(
        value: str, items: list[str] | dict[str, object], *, case: str | None = None
    ) -> bool:
        """Normalized membership check (builder: norm().in_()).

        Mnemonic: norm = normalize, in_ = membership check
        Generic replacement for: u.normalize(value, case=case) in u.map(items, lambda v: u.normalize(v, case=case))

        Args:
            value: Value to check
            items: Items to check against
            case: Case normalization

        Returns:
            bool: True if normalized value in normalized items

        """
        items_list = list(items.keys()) if isinstance(items, dict) else items
        normalized_value = FlextUtilities.norm_str(value, case=case or "lower")
        normalized_items = cast(
            "list[str]", FlextUtilities.norm_list(items_list, case=case or "lower")
        )
        return normalized_value in normalized_items

    @staticmethod
    def filter_attrs(
        attrs: dict[str, object] | dict[str, list[str]],
        *,
        predicate: Callable[[str, object], bool] | None = None,
        only_list_like: bool = False,
    ) -> dict[str, list[str]]:
        """Filter attributes (builder: filt().attrs()).

        Mnemonic: filter = filter, attrs = attributes dict
        Uses advanced DSL: filt() builder internally for fluent composition.

        Args:
            attrs: Attributes to filter
            predicate: Optional predicate
            only_list_like: Keep only list-like values

        Returns:
            dict[str, list[str]]: Filtered attributes

        """
        attrs_dict = cast("dict[str, object]", attrs) if isinstance(attrs, dict) else {}
        return cast(
            "dict[str, list[str]]",
            FlextUtilities.filt(attrs_dict)
            .attrs(predicate=predicate, only_list_like=only_list_like)
            .build(),
        )

    @staticmethod
    def filter_not_none(
        items: dict[str, object | None] | list[object | None],
    ) -> dict[str, object] | list[object]:
        """Filter not None (builder: filt().not_none()).

        Mnemonic: filter = filter, not_none = remove None values
        Uses advanced DSL: filt() builder internally for fluent composition.

        Args:
            items: Items to filter

        Returns:
            Filtered items without None

        """
        return cast(
            "dict[str, object] | list[object]",
            FlextUtilities.filt(items).not_none().build(),
        )

    @staticmethod
    def filter_truthy(
        items: list[object] | dict[str, object],
    ) -> list[object] | dict[str, object]:
        """Filter truthy (builder: filt().truthy()).

        Mnemonic: filter = filter, truthy = keep only truthy values
        Uses advanced DSL: filter() directly to avoid recursion.

        Args:
            items: Items to filter

        Returns:
            Filtered items with only truthy values

        """
        filtered = FlextUtilities.filter(items, predicate=bool)
        return (
            filtered
            if isinstance(filtered, (dict, list))
            else ([] if isinstance(items, list) else {})
        )

    @staticmethod
    def map_str(
        items: list[str] | dict[str, str],
        *,
        case: str | None = None,
        join: str | None = None,
    ) -> list[str] | dict[str, str] | str:
        """Map strings (builder: map().str()).

        Mnemonic: map = map, str = string transformation
        Generic replacement for: u.map(items, mapper=lambda v: u.normalize(v, case=case))

        Args:
            items: Items to map
            case: Case normalization
            join: Join separator (returns str if provided)

        Returns:
            Mapped items or joined string

        """
        mapped: list[str] | dict[str, str] | object = items
        if case:
            if isinstance(items, dict):
                mapped = FlextUtilities.map(
                    items, mapper=lambda _k, v: FlextUtilities.normalize(v, case=case)
                )
            else:
                mapped = FlextUtilities.map(
                    items, mapper=lambda v: FlextUtilities.normalize(v, case=case)
                )
        # Type narrowing for join
        if join:
            if isinstance(mapped, dict):
                mapped_list = list(mapped.values())
            elif isinstance(mapped, list):
                mapped_list = cast("list[str]", mapped)
            else:
                mapped_list = []
            return FlextUtilities.join(mapped_list, sep=join)
        return cast("list[str] | dict[str, str]", mapped)

    @staticmethod
    def find_callable(
        checks: dict[str, Callable[..., bool]] | list[tuple[str, Callable[..., bool]]],
        *args: object,
        **kwargs: object,
    ) -> str | None:
        """Find by callable (builder: find().callable()).

        Mnemonic: find = find, callable = predicate function
        Generic replacement for: u.find(checks, predicate=lambda _k, pred: pred(*args, **kwargs))

        Args:
            checks: Dict or list of (key, callable) tuples
            *args: Args for callables
            **kwargs: Kwargs for callables

        Returns:
            First matching key or None

        """
        if isinstance(checks, dict):
            found = FlextUtilities.find(
                checks, predicate=lambda _k, pred: pred(*args, **kwargs)
            )
            return found if isinstance(found, str) else None
        # Handle list of tuples - convert to dict for find()
        if isinstance(checks, list):
            checks_dict: dict[str, Callable[..., bool]] = {
                k: v for k, v in checks if callable(v)
            }
            found = FlextUtilities.find(
                checks_dict, predicate=lambda _k, pred: pred(*args, **kwargs)
            )
            return found if isinstance(found, str) else None
        return None

    # ═══════════════════════════════════════════════════════════════════
    # GENERALIZED CONVENIENCE METHODS - Using builders internally
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def conv_str_list_truthy(
        value: object, *, default: list[str] | None = None
    ) -> list[str]:
        """Convert to str_list and filter truthy (generalized: conv().str_list().truthy().build()).

        Mnemonic: conv_str_list_truthy = convert + filter truthy
        Generic replacement for: to_str_list_truthy() in specific utilities

        Args:
            value: Value to convert
            default: Default if None

        Returns:
            list[str]: Converted and filtered list

        """
        return cast(
            "list[str]",
            FlextUtilities.conv(value).str_list(default=default or []).truthy().build(),
        )

    @staticmethod
    def conv_str_list_safe(value: object | None) -> list[str]:
        """Safe str_list conversion (generalized: conv().str_list().safe().build()).

        Mnemonic: conv_str_list_safe = convert + safe mode
        Generic replacement for: to_str_list_safe() in specific utilities

        Args:
            value: Value to convert (can be None)

        Returns:
            list[str]: Converted list or []

        """
        return cast(
            "list[str]",
            FlextUtilities.conv(value).str_list(default=[]).safe().build(),
        )

    @staticmethod
    def map_filter(
        items: Collection[object] | object,  # type: ignore[type-arg,valid-type]  # Collection accepts object
        *,
        mapper: Callable[[object], object] | None = None,
        predicate: Callable[[object], bool] | None = None,
    ) -> list[object]:
        """Map then filter items (generalized: mp().str().filter().build()).

        Mnemonic: map_filter = map + filter
        Generic replacement for: map_filter() in specific utilities

        Args:
            items: Items to process
            mapper: Transformation function
            predicate: Filter function

        Returns:
            Processed list

        """
        if not items:
            return []
        # Ensure list
        items_list = (
            list(items)
            if isinstance(items, Collection) and not isinstance(items, str)
            else [items]
        )
        # Map if mapper provided
        mapped = [mapper(item) for item in items_list] if mapper else items_list
        # Filter if predicate provided
        if predicate:
            return [item for item in mapped if predicate(item)]
        return mapped

    @staticmethod
    def find_key(
        obj: dict[str, object],
        *,
        predicate: Callable[[str, object], bool] | None = None,
    ) -> str | None:
        """Find key using predicate (generalized: fnd(obj).key(predicate).build()).

        Mnemonic: find_key = find key
        Generic replacement for: find_key() in specific utilities

        Args:
            obj: Dict to search
            predicate: (k,v) -> bool

        Returns:
            First matching key or None

        """
        if not predicate:
            return None
        return cast(
            "str | None", FlextUtilities.fnd(obj).key(predicate=predicate).build()
        )

    @staticmethod
    def find_val(
        obj: dict[str, object],
        *,
        predicate: Callable[[str, object], bool] | None = None,
    ) -> object | None:
        """Find value using predicate (generalized: fnd(obj).val(predicate).build()).

        Mnemonic: find_val = find value
        Generic replacement for: find_val() in specific utilities

        Args:
            obj: Dict to search
            predicate: (k,v) -> bool

        Returns:
            First matching value or None

        """
        if not predicate:
            return None
        return FlextUtilities.fnd(obj).val(predicate=predicate).build()

    @staticmethod
    def map_dict(
        obj: dict[str, object],
        *,
        mapper: Callable[[str, object], object] | None = None,
        key_mapper: Callable[[str], str] | None = None,
        predicate: Callable[[str, object], bool] | None = None,
    ) -> dict[str, object]:
        """Map dict (generalized: mp(obj).dict(mapper, key_mapper, predicate).build()).

        Mnemonic: map_dict = map dictionary
        Generic replacement for: map_dict() in specific utilities

        Args:
            obj: Dict to map
            mapper: (k,v) -> new_v
            key_mapper: (k) -> new_k
            predicate: (k,v) -> bool

        Returns:
            Mapped dict

        """
        return cast(
            "dict[str, object]",
            FlextUtilities.mp(obj)
            .dict(mapper=mapper, key_mapper=key_mapper, predicate=predicate)
            .build(),
        )

    @staticmethod
    def process_flatten(
        items: Collection[object] | object,  # type: ignore[type-arg,valid-type]  # Collection accepts object
        *,
        processor: Callable[[object], object] | None = None,
        on_error: str = "skip",
    ) -> list[object]:
        """Process and flatten (generalized: mp(items).flatten().build()).

        Mnemonic: process_flatten = process then flatten
        Generic replacement for: process_flatten() in specific utilities

        Args:
            items: Items to process
            processor: Processing function
            on_error: Error handling ("skip", "fail", "return")

        Returns:
            Flattened list

        """
        if not items:
            return []
        # Ensure list
        items_list = (
            list(items)
            if isinstance(items, Collection) and not isinstance(items, str)
            else [items]
        )
        # Process if processor provided
        if processor:
            processed: list[object] = []
            for item in items_list:
                try:
                    result = processor(item)
                    processed.append(result)
                except Exception:
                    if on_error == "fail":
                        raise
                    if on_error == "return":
                        return []
                    # skip: continue
            items_list = processed
        # Flatten nested lists
        flattened: list[object] = []
        for item in items_list:
            if isinstance(item, (list, tuple)):
                flattened.extend(item)
            else:
                flattened.append(item)
        return flattened

    @staticmethod
    def _process_dict_item(
        item: dict[str, object],
        *,
        processor: Callable[[str, object], tuple[str, object]] | None = None,
        predicate: Callable[[str, object], bool] | None = None,
    ) -> dict[str, object]:
        """Process single dict item (helper for reduce_dict)."""
        if processor:
            mapped_dict: dict[str, object] = {}
            for k, v in item.items():
                if not predicate or predicate(k, v):
                    new_k, new_v = processor(k, v)
                    mapped_dict[new_k] = new_v
            return mapped_dict
        return FlextUtilities.map_dict(item, predicate=predicate)

    @staticmethod
    def reduce_dict(
        items: Collection[dict[str, object]] | dict[str, object] | object,  # type: ignore[type-arg,valid-type]  # Collection accepts dict
        *,
        processor: Callable[[str, object], tuple[str, object]] | None = None,
        predicate: Callable[[str, object], bool] | None = None,
        default: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Reduce dicts (generalized: mp(items).dict().flatten() + merge).

        Mnemonic: reduce_dict = reduce dictionaries
        Generic replacement for: reduce_dict() in specific utilities

        Args:
            items: Dicts to merge
            processor: Transform (k,v) -> (new_k, new_v)
            predicate: Filter (k,v) -> bool
            default: Default dict

        Returns:
            Merged dict

        """
        if not items:
            return default or {}
        # Ensure list of dicts
        items_list: list[dict[str, object]] = []
        if isinstance(items, dict):
            items_list = [items]
        elif isinstance(items, Collection) and not isinstance(items, str):
            items_list = [item for item in items if isinstance(item, dict)]
        else:
            return default or {}
        # Process each dict
        processed_dicts = [
            FlextUtilities._process_dict_item(
                item, processor=processor, predicate=predicate
            )
            for item in items_list
        ]
        # Merge all dicts
        if not processed_dicts:
            return default or {}
        result = processed_dicts[0]
        for d in processed_dicts[1:]:
            merge_result = FlextUtilities.merge(
                cast("Mapping[str, t.GeneralValueType]", result),
                cast("Mapping[str, t.GeneralValueType]", d),
            )
            if merge_result.is_success and isinstance(merge_result.value, dict):
                result = cast("dict[str, object]", merge_result.value)
        return result

    @staticmethod
    def map_attrs_to_str_list(
        attrs: dict[str, object] | dict[str, list[str]],
        *,
        filter_list_like: bool = False,
    ) -> dict[str, list[str]]:
        """Map attributes to str_list (generalized: mp(attrs).attrs().str_list().build()).

        Mnemonic: map_attrs_to_str_list = map attributes to string list
        Generic replacement for: attr_to_str_list() in specific utilities

        Uses advanced DSL: mp() builder for fluent mapping composition.

        Args:
            attrs: Attributes to convert
            filter_list_like: Only convert list-like values

        Returns:
            dict[str, list[str]]: Converted attributes

        """

        def convert_value(_k: str, v: object) -> list[str]:
            # DSL pattern: whn() for conditional list-like check
            if filter_list_like and not FlextRuntime.is_list_like(
                cast("t.GeneralValueType", v)
            ):
                return [str(v)]
            return FlextUtilities.conv_str_list(cast("t.GeneralValueType", v))

        # DSL pattern: mp() builder for mapping
        mapped = FlextUtilities.map(attrs, mapper=convert_value)
        return cast("dict[str, list[str]]", mapped) if isinstance(mapped, dict) else {}

    @staticmethod
    def extract_str_from_obj(
        obj: object | None,
        *,
        attr: str = "value",
        default: str = "unknown",
    ) -> str:
        """Extract string from object (generalized: whn().attr().str().build()).

        Mnemonic: extract_str_from_obj = extract string from object attribute
        Generic replacement for: dn_str() and similar in specific utilities

        Uses advanced DSL: whn() → or_() → conv_str() for fluent composition.

        Args:
            obj: Object to extract from (can be None or have attribute)
            attr: Attribute name to extract
            default: Default if None or no attribute

        Returns:
            str: Extracted string or default

        """
        # DSL pattern: whn() for None check, then or_() for value extraction
        result = FlextUtilities.when(
            condition=obj is not None,
            then_value=FlextUtilities.or_(
                str(getattr(obj, attr))
                if FlextUtilities.has(obj, attr) and getattr(obj, attr) is not None
                else None,
                str(obj) if hasattr(obj, "__str__") else None,
                default=default,
            ),
            else_value=default,
        )
        return cast("str", result) if result is not None else default

    # ═══════════════════════════════════════════════════════════════════
    # BUILDER CLASSES - DSL Parametrizado com Self para encadeamento
    # ═══════════════════════════════════════════════════════════════════

    class ConvBuilder:
        """Builder para conversão de tipos (DSL: u.conv(value).str().default("").build()).

        Mnemonic: conv = convert
        Usage:
            u.conv(value).str().default("").build()
            u.conv(value).int().default(0).build()
            u.conv(value).bool().default(False).build()
            u.conv(value).str_list().default([]).build()
        """

        def __init__(self, value: object) -> None:
            """Initialize builder with value to convert."""
            self._value = value
            self._target_type: str | None = None
            self._default: object = None
            self._filter_truthy: bool = False
            self._safe_mode: bool = False

        def str(self, *, default: _StrType = "") -> Self:
            """Set target type to string."""
            self._target_type = "str"
            self._default = default
            return self

        def int(self, *, default: _IntType = 0) -> Self:
            """Set target type to int."""
            self._target_type = "int"
            self._default = default
            return self

        def bool(self, *, default: _BoolType = False) -> Self:
            """Set target type to bool."""
            self._target_type = "bool"
            self._default = default
            return self

        def str_list(self, *, default: Union[list[_StrType], None] = None) -> Self:
            """Set target type to list[str]."""
            self._target_type = "str_list"
            self._default = default or []
            return self

        def truthy(self) -> Self:
            """Chain filter truthy after conversion."""
            self._filter_truthy = True
            return self

        def safe(self) -> Self:
            """Enable safe mode (handle None gracefully)."""
            self._safe_mode = True
            return self

        def build(self) -> _StrType | _IntType | _BoolType | list[_StrType]:
            """Build and return converted value."""
            # Handle safe mode (None handling)
            if self._safe_mode and self._value is None:
                if self._target_type == "str_list":
                    return cast("list[str]", self._default or [])
                return cast("str | int | bool", self._default)

            # Convert
            result: object = None
            if self._target_type == "str":
                result = FlextUtilities.conv_str(
                    self._value, default=cast("str", self._default)
                )
            elif self._target_type == "int":
                result = FlextUtilities.conv_int(
                    self._value, default=cast("int", self._default)
                )
            elif self._target_type == "bool":
                converted = FlextUtilities.convert(
                    cast("t.GeneralValueType", self._value),
                    bool,
                    cast("bool", self._default),
                )
                result = cast("bool", converted)
            elif self._target_type == "str_list":
                if self._safe_mode and not FlextRuntime.is_list_like(
                    cast("t.GeneralValueType", self._value)
                ):
                    result = (
                        [str(self._value)]
                        if self._value is not None
                        else (self._default or [])
                    )
                else:
                    result = FlextUtilities.conv_str_list(
                        cast("t.GeneralValueType", self._value),
                        default=cast("list[str] | None", self._default),
                    )
            else:
                msg = f"Unknown target type: {self._target_type}"
                raise ValueError(msg)

            # Apply truthy filter if requested
            if self._filter_truthy and isinstance(result, list):
                filtered = FlextUtilities.filter_truthy(result)
                return (
                    cast("list[str]", filtered)
                    if isinstance(filtered, list)
                    else cast("list[str]", [])
                )

            return cast("str | int | bool | list[str]", result)

    @staticmethod
    def conv(value: object) -> ConvBuilder:
        """Create conversion builder (DSL entry point).

        Args:
            value: Value to convert

        Returns:
            ConvBuilder instance for chaining

        Example:
            result = u.conv(value).str().default("").build()

        """
        return FlextUtilities.ConvBuilder(value)

    class NormBuilder:
        """Builder para normalização (DSL: u.norm(value).str().case("lower").build()).

        Mnemonic: norm = normalize
        Usage:
            u.norm(value).str().case("lower").build()
            u.norm(items).list().case("upper").filter_truthy().build()
            u.norm(items).join().case("lower").sep(",").build()
        """

        def __init__(self, value: object) -> None:
            """Initialize builder with value to normalize."""
            self._value = value
            self._operation: str | None = None
            self._case: str | None = None
            self._default: str = ""
            self._filter_truthy: bool = False
            self._to_set: bool = False
            self._sep: str = " "

        def str(self, *, case: str | None = None, default: str = "") -> Self:
            """Set operation to normalize string."""
            self._operation = "str"
            self._case = case
            self._default = default
            return self

        def list(
            self,
            *,
            case: _StrType | None = None,
            filter_truthy: bool = False,
            to_set: bool = False,
        ) -> Self:
            """Set operation to normalize list."""
            self._operation = "list"
            self._case = case
            self._filter_truthy = filter_truthy
            self._to_set = to_set
            return self

        def join(
            self, *, case: _StrType | None = None, sep: _StrType = " "
        ) -> Self:
            """Set operation to normalize and join."""
            self._operation = "join"
            self._case = case
            self._sep = sep
            return self

        def build(
            self,
        ) -> Union[_StrType, _ListType[_StrType], _SetType[_StrType], _DictType[_StrType, _StrType]]:
            """Build and return normalized value."""
            if self._operation == "str":
                # Use ensure_str directly to avoid recursion
                str_value = FlextUtilities.ensure_str(
                    cast("t.GeneralValueType", self._value), default=self._default
                )
                if self._case:
                    normalized = FlextUtilities.normalize(str_value, case=self._case)
                    return cast("str", normalized)
                return str_value
            if self._operation == "list":
                if not isinstance(self._value, (list, dict)):
                    return [] if not self._to_set else set()
                return FlextUtilities.norm_list(
                    cast("list[str] | dict[str, str]", self._value),
                    case=self._case,
                    filter_truthy=self._filter_truthy,
                    to_set=self._to_set,
                )
            if self._operation == "join":
                if not isinstance(self._value, list):
                    return ""
                # Use norm_join directly (already uses map internally, no recursion)
                return FlextUtilities.norm_join(
                    cast("list[str]", self._value), case=self._case, sep=self._sep
                )
            msg = f"Unknown operation: {self._operation}"
            raise ValueError(msg)

    @staticmethod
    def norm(value: object) -> NormBuilder:
        """Create normalization builder (DSL entry point).

        Args:
            value: Value to normalize

        Returns:
            NormBuilder instance for chaining

        Example:
            result = u.norm(value).str().case("lower").build()

        """
        return FlextUtilities.NormBuilder(value)

    class FilterBuilder:
        """Builder para filtragem (DSL: u.filter(items).attrs().only_list_like().build()).

        Mnemonic: filter = filter
        Usage:
            u.filter(items).attrs().only_list_like().build()
            u.filter(items).not_none().build()
            u.filter(items).truthy().build()
        """

        def __init__(self, items: dict[str, object] | list[object]) -> None:
            """Initialize builder with items to filter."""
            self._items = items
            self._operation: str | None = None
            self._predicate: Callable[[str, object], bool] | None = None
            self._only_list_like: bool = False

        def attrs(
            self,
            *,
            predicate: Callable[[str, object], bool] | None = None,
            only_list_like: bool = False,
        ) -> Self:
            """Set operation to filter attributes."""
            self._operation = "attrs"
            self._predicate = predicate
            self._only_list_like = only_list_like
            return self

        def not_none(self) -> Self:
            """Set operation to filter not None."""
            self._operation = "not_none"
            return self

        def truthy(self) -> Self:
            """Set operation to filter truthy."""
            self._operation = "truthy"
            return self

        def build(self) -> dict[str, list[str]] | dict[str, object] | list[object]:
            """Build and return filtered items."""
            if self._operation == "attrs":
                if not isinstance(self._items, dict):
                    return {}
                # Use filter directly to avoid recursion
                attrs_dict = cast(
                    "dict[str, object] | dict[str, list[str]]", self._items
                )
                if self._only_list_like:
                    filtered = FlextUtilities.filter(
                        attrs_dict, predicate=lambda _k, v: FlextRuntime.is_list_like(v)
                    )
                    filtered_dict = (
                        cast("dict[str, object]", filtered)
                        if isinstance(filtered, dict)
                        else {}
                    )
                elif self._predicate:
                    filtered = FlextUtilities.filter(
                        attrs_dict, predicate=self._predicate
                    )
                    filtered_dict = (
                        cast("dict[str, object]", filtered)
                        if isinstance(filtered, dict)
                        else {}
                    )
                else:
                    filtered_dict = cast("dict[str, object]", attrs_dict)
                # Ensure all values are list[str]
                return {
                    k: (v if isinstance(v, list) else [str(v)])
                    for k, v in filtered_dict.items()
                }
            if self._operation == "not_none":
                # Use filter directly to avoid recursion
                filtered = FlextUtilities.filter(
                    cast("dict[str, object | None] | list[object | None]", self._items),
                    predicate=lambda _k, v: v is not None,
                )
                return (
                    filtered
                    if isinstance(filtered, (dict, list))
                    else ([] if isinstance(self._items, list) else {})
                )
            if self._operation == "truthy":
                # Use filter directly to avoid recursion
                filtered = FlextUtilities.filter(self._items, predicate=bool)
                return (
                    filtered
                    if isinstance(filtered, (dict, list))
                    else ([] if isinstance(self._items, list) else {})
                )
            msg = f"Unknown operation: {self._operation}"
            raise ValueError(msg)

    @staticmethod
    def filt(items: dict[str, object] | list[object]) -> FilterBuilder:
        """Create filter builder (DSL entry point).

        Mnemonic: filt = filter builder (avoids conflict with generic filter method)

        Args:
            items: Items to filter

        Returns:
            FilterBuilder instance for chaining

        Example:
            result = u.filt(items).attrs().only_list_like().build()

        """
        return FlextUtilities.FilterBuilder(items)

    class MapBuilder:
        """Builder para mapeamento (DSL: u.map(items).str().case("lower").join(",").build()).

        Mnemonic: map = map
        Usage:
            u.map(items).str().case("lower").build()
            u.map(items).str().join(",").build()
        """

        def __init__(
            self, items: list[str] | dict[str, str] | dict[str, object]
        ) -> None:
            """Initialize builder with items to map."""
            self._items = items
            self._operation: str | None = None
            self._case: str | None = None
            self._join: str | None = None
            self._filter_predicate: Callable[[object], bool] | None = None
            self._dict_mapper: Callable[[str, object], object] | None = None
            self._key_mapper: Callable[[str], str] | None = None
            self._dict_predicate: Callable[[str, object], bool] | None = None
            self._flatten: bool = False
            self._pluck_key: str | int | Callable[[object], object] | None = None

        def str(self, *, case: str | None = None, join: str | None = None) -> Self:
            """Set operation to map strings."""
            self._operation = "str"
            self._case = case
            self._join = join
            return self

        def dict(
            self,
            *,
            mapper: Callable[[_StrType, object], object] | None = None,
            key_mapper: Callable[[_StrType], _StrType] | None = None,
            predicate: Callable[[_StrType, object], bool] | None = None,
        ) -> Self:
            """Set operation to map dict."""
            self._operation = "dict"
            self._dict_mapper = mapper
            self._key_mapper = key_mapper
            self._dict_predicate = predicate
            return self

        def filter(self, predicate: Callable[[object], bool] | None = None) -> Self:
            """Chain filter after map."""
            self._filter_predicate = predicate
            return self

        def flatten(self) -> Self:
            """Chain flatten after map."""
            self._flatten = True
            return self

        def pluck(self, key: _StrType | int | Callable[[object], object]) -> Self:
            """Chain pluck after map."""
            self._operation = "pluck"
            self._pluck_key = key
            return self

        def build(  # noqa: C901  # Complex build logic required for builder pattern
            self,
        ) -> Union[
            _ListType[_StrType],
            _DictType[_StrType, _StrType],
            _DictType[_StrType, object],
            _StrType,
            _ListType[object],
        ]:
            """Build and return mapped items."""
            if self._operation == "str":
                # Type narrowing: self._items is list[str] | dict[str, str] | dict[str, object]
                # map_str accepts list[str] | dict[str, str]
                if isinstance(self._items, list):
                    items_for_map: list[str] | dict[str, str] = self._items
                elif isinstance(self._items, dict):
                    # Check if it's dict[str, str] or dict[str, object]
                    # For type safety, cast to dict[str, str] (map_str will handle conversion)
                    items_for_map = cast("dict[str, str]", self._items)
                else:
                    items_for_map = []
                mapped = FlextUtilities.map_str(
                    items_for_map, case=self._case, join=self._join
                )
                # Apply filter if requested
                if self._filter_predicate and isinstance(mapped, list):
                    return [item for item in mapped if self._filter_predicate(item)]
                return mapped
            if self._operation == "dict":
                if not isinstance(self._items, dict):
                    return {}
                # Filter first if predicate provided
                filtered_items = self._items
                if self._dict_predicate:
                    filtered = FlextUtilities.filter(
                        self._items, predicate=self._dict_predicate
                    )
                    filtered_items = (
                        cast("dict[str, object]", filtered)
                        if isinstance(filtered, dict)
                        else {}
                    )
                # Map values and keys
                dict_result: dict[str, object] = {}
                for k, v in filtered_items.items():
                    new_k = self._key_mapper(k) if self._key_mapper else k
                    new_v = self._dict_mapper(k, v) if self._dict_mapper else v
                    dict_result[new_k] = new_v
                # Apply flatten if requested
                return list(dict_result.values()) if self._flatten else dict_result
            if self._operation == "pluck":
                if not isinstance(self._items, (list, dict)):
                    return []
                items_list = (
                    list(self._items)
                    if isinstance(self._items, list)
                    else [cast("object", v) for v in cast("dict[str, object]", self._items).values()]  # type: ignore[misc]  # List comprehension returns list[object], not list[str]
                )
                pluck_result = FlextUtilities.pluck(
                    items_list,
                    key=cast("str | int | Callable[[object], object]", self._pluck_key),
                )
                return cast("list[object]", pluck_result)
            msg = f"Unknown operation: {self._operation}"
            raise ValueError(msg)

    @staticmethod
    def mp(items: list[str] | dict[str, str] | dict[str, object]) -> MapBuilder:
        """Create map builder (DSL entry point).

        Mnemonic: mp = map builder (avoids conflict with generic map method)

        Args:
            items: Items to map

        Returns:
            MapBuilder instance for chaining

        Example:
            result = u.mp(items).str().case("lower").build()

        """
        return FlextUtilities.MapBuilder(items)

    class FindBuilder:
        """Builder para busca (DSL: u.find(checks).callable(*args, **kwargs).build()).

        Mnemonic: find = find
        Usage:
            u.find(checks).callable(*args, **kwargs).build()
        """

        def __init__(
            self,
            checks: dict[str, Callable[..., bool]]
            | list[tuple[str, Callable[..., bool]]]
            | dict[str, object],
        ) -> None:
            """Initialize builder with checks to find."""
            self._checks = checks
            self._operation: str | None = None
            self._args: tuple[object, ...] = ()
            self._kwargs: dict[str, object] = {}
            self._predicate: Callable[[str, object], bool] | None = None

        def callable(self, *args: object, **kwargs: object) -> Self:
            """Set operation to find by callable."""
            self._operation = "callable"
            self._args = args
            self._kwargs = kwargs
            return self

        def key(self, predicate: Callable[[str, object], bool] | None = None) -> Self:
            """Set operation to find key."""
            self._operation = "key"
            self._predicate = predicate
            return self

        def val(self, predicate: Callable[[str, object], bool] | None = None) -> Self:
            """Set operation to find value."""
            self._operation = "val"
            self._predicate = predicate
            return self

        def build(self) -> str | object | None:
            """Build and return found key/value or None."""
            if self._operation == "callable":
                return FlextUtilities.find_callable(
                    cast(
                        "dict[str, Callable[..., bool]] | list[tuple[str, Callable[..., bool]]]",
                        self._checks,
                    ),
                    *self._args,
                    **self._kwargs,
                )
            if (
                self._operation in {"key", "val"}
                and isinstance(self._checks, dict)
                and self._predicate
            ):
                for k, v in self._checks.items():
                    if self._predicate(k, v):
                        return k if self._operation == "key" else v
            if self._operation not in {"callable", "key", "val"}:
                msg = f"Unknown operation: {self._operation}"
                raise ValueError(msg)
            return None

    @staticmethod
    def fnd(
        checks: dict[str, Callable[..., bool]]
        | list[tuple[str, Callable[..., bool]]]
        | dict[str, object],
    ) -> FindBuilder:
        """Create find builder (DSL entry point).

        Mnemonic: fnd = find builder (avoids conflict with generic find method)

        Args:
            checks: Dict or list of (key, callable) tuples

        Returns:
            FindBuilder instance for chaining

        Example:
            result = u.fnd(checks).callable(*args, **kwargs).build()

        """
        return FlextUtilities.FindBuilder(checks)

    class WhenBuilder:
        """Builder para condicionais (DSL: u.when(value).safe(func).build()).

        Mnemonic: when = conditional execution
        Usage:
            u.when(value).safe(func).build()
            u.when(value).not_none(func).build()
            u.when(value).truthy(func).build()
        """

        def __init__(self, value: object) -> None:
            """Initialize builder with value to check."""
            self._value = value
            self._operation: str | None = None
            self._func: Callable[[object], object] | None = None

        def safe(self, func: Callable[[object], object]) -> Self:
            """Set operation to safe execution."""
            self._operation = "safe"
            self._func = func
            return self

        def not_none(self, func: Callable[[object], object]) -> Self:
            """Set operation to execute if not None."""
            self._operation = "not_none"
            self._func = func
            return self

        def truthy(self, func: Callable[[object], object]) -> Self:
            """Set operation to execute if truthy."""
            self._operation = "truthy"
            self._func = func
            return self

        def build(self) -> object:
            """Build and return result or None."""
            if self._operation == "safe":
                try:
                    return self._func(self._value) if self._func else None
                except Exception:
                    return None
            if self._operation == "not_none":
                return (
                    self._func(self._value)
                    if self._value is not None and self._func
                    else None
                )
            if self._operation == "truthy":
                return self._func(self._value) if self._value and self._func else None
            msg = f"Unknown operation: {self._operation}"
            raise ValueError(msg)

    @staticmethod
    def whn(value: object) -> WhenBuilder:
        """Create when builder (DSL entry point).

        Mnemonic: whn = when builder (avoids conflict with generic when method)

        Args:
            value: Value to check

        Returns:
            WhenBuilder instance for chaining

        Example:
            result = u.whn(value).safe(func).build()

        """
        return FlextUtilities.WhenBuilder(value)

    class EnsureBuilder:
        """Builder para garantir tipos (DSL: u.ensure(value).str().default("").build()).

        Mnemonic: ensure = ensure type
        Usage:
            u.ensure(value).str().default("").build()
            u.ensure(value).str_list().default([]).build()
        """

        def __init__(self, value: object) -> None:
            """Initialize builder with value to ensure."""
            self._value = value
            self._target_type: _StrType | None = None
            self._default: object = None

        def str(self, *, default: _StrType = "") -> Self:
            """Set target type to string."""
            self._target_type = "str"
            self._default = default
            return self

        def str_list(self, *, default: list[_StrType] | None = None) -> Self:
            """Set target type to list[str]."""
            self._target_type = "str_list"
            self._default = default or []
            return self

        def build(self) -> Union[_StrType, list[_StrType]]:
            """Build and return ensured value."""
            if self._target_type == "str":
                return FlextUtilities.ensure_str(
                    cast("t.GeneralValueType", self._value),
                    default=cast("str", self._default),
                )
            if self._target_type == "str_list":
                return FlextUtilities.ensure_str_list(
                    cast("t.GeneralValueType", self._value),
                    default=cast("list[str] | None", self._default),
                )
            msg = f"Unknown target type: {self._target_type}"
            raise ValueError(msg)

    @staticmethod
    def ens(value: object) -> EnsureBuilder:
        """Create ensure builder (DSL entry point).

        Mnemonic: ens = ensure builder (avoids conflict with generic ensure method)

        Args:
            value: Value to ensure

        Returns:
            EnsureBuilder instance for chaining

        Example:
            result = u.ens(value).str().default("").build()

        """
        return FlextUtilities.EnsureBuilder(value)

    class TransformBuilder:
        """Builder para transformações (DSL: u.transform(value).dict().normalize().build()).

        Mnemonic: transform = transform
        Usage:
            u.transform(value).dict().normalize().build()
            u.transform(value).dict().map_keys({"old": "new"}).build()
        """

        def __init__(self, value: object) -> None:
            """Initialize builder with value to transform."""
            self._value = value
            self._operation: str | None = None
            self._normalize: bool = False
            self._map_keys: dict[str, str] | None = None
            self._filter_keys: set[str] | None = None
            self._exclude_keys: set[str] | None = None

        def dict(
            self,
            *,
            normalize: bool = False,
            map_keys: dict[_StrType, _StrType] | None = None,
            filter_keys: set[_StrType] | None = None,
            exclude_keys: set[_StrType] | None = None,
        ) -> Self:
            """Set operation to transform dict."""
            self._operation = "dict"
            self._normalize = normalize
            self._map_keys = map_keys
            self._filter_keys = filter_keys
            self._exclude_keys = exclude_keys
            return self

        def build(self) -> _DictType[_StrType, object]:
            """Build and return transformed value."""
            if self._operation == "dict":
                if not isinstance(self._value, dict):
                    return {}
                transform_opts: dict[str, object] = {}
                if self._normalize:
                    transform_opts["normalize"] = True
                if self._map_keys:
                    transform_opts["map_keys"] = self._map_keys
                if self._filter_keys:
                    transform_opts["filter_keys"] = self._filter_keys
                if self._exclude_keys:
                    transform_opts["exclude_keys"] = self._exclude_keys
                if transform_opts:
                    # Extract individual options for type safety
                    normalize_val = transform_opts.get("normalize", False)
                    strip_none_val = transform_opts.get("strip_none", False)
                    strip_empty_val = transform_opts.get("strip_empty", False)
                    map_keys_val = transform_opts.get("map_keys")
                    filter_keys_val = transform_opts.get("filter_keys")
                    exclude_keys_val = transform_opts.get("exclude_keys")
                    to_json_val = transform_opts.get("to_json", False)
                    to_model_val = transform_opts.get("to_model")
                    transform_result = FlextUtilities.transform(
                        cast("Mapping[str, t.GeneralValueType]", self._value),
                        normalize=cast("bool", normalize_val)
                        if isinstance(normalize_val, bool)
                        else False,
                        strip_none=cast("bool", strip_none_val)
                        if isinstance(strip_none_val, bool)
                        else False,
                        strip_empty=cast("bool", strip_empty_val)
                        if isinstance(strip_empty_val, bool)
                        else False,
                        map_keys=cast("dict[str, str] | None", map_keys_val)
                        if isinstance(map_keys_val, dict) or map_keys_val is None
                        else None,
                        filter_keys=cast("set[str] | None", filter_keys_val)
                        if isinstance(filter_keys_val, set) or filter_keys_val is None
                        else None,
                        exclude_keys=cast("set[str] | None", exclude_keys_val)
                        if isinstance(exclude_keys_val, set) or exclude_keys_val is None
                        else None,
                        to_json=cast("bool", to_json_val)
                        if isinstance(to_json_val, bool)
                        else False,
                        to_model=cast("type[BaseModel] | None", to_model_val)
                        if isinstance(to_model_val, type) or to_model_val is None
                        else None,
                    )
                    if transform_result.is_success and isinstance(
                        transform_result.value, dict
                    ):
                        return cast("dict[str, object]", transform_result.value)
                    return {}
                return cast("dict[str, object]", self._value)
            msg = f"Unknown operation: {self._operation}"
            raise ValueError(msg)

    @staticmethod
    def transform_builder(value: object) -> TransformBuilder:
        """Create transform builder (DSL entry point).

        Mnemonic: transform_builder = transform builder (avoids conflict with transform method)

        Args:
            value: Value to transform

        Returns:
            TransformBuilder instance for chaining

        Example:
            result = u.transform_builder(value).dict().normalize().build()

        """
        return FlextUtilities.TransformBuilder(value)

    class ChainBuilder:
        """Builder para encadeamento (DSL: u.chain(value).then(func1).then(func2).build()).

        Mnemonic: chain = chain operations
        Usage:
            u.chain(value).then(func1).then(func2).build()
        """

        def __init__(self, value: object) -> None:
            """Initialize builder with initial value."""
            self._value = value
            self._funcs: list[Callable[[object], object]] = []

        def then(self, func: Callable[[object], object]) -> Self:
            """Add function to chain."""
            self._funcs.append(func)
            return self

        def build(self) -> object:
            """Build and return chained result."""
            result = self._value
            for func in self._funcs:
                result = func(result)
            return result

    @staticmethod
    def chain_builder(value: object) -> ChainBuilder:
        """Create chain builder (DSL entry point).

        Mnemonic: chain_builder = chain builder (avoids conflict with chain method)

        Args:
            value: Initial value

        Returns:
            ChainBuilder instance for chaining

        Example:
            result = u.chain_builder(value).then(func1).then(func2).build()

        """
        return FlextUtilities.ChainBuilder(value)

    class ValidateBuilder:
        """Builder para validação (DSL: u.validate(value).not_none().str().build()).

        Mnemonic: validate = validate builder
        Usage:
            u.validate(value).not_none().str().build()
            u.validate(value).truthy().int().build()
        """

        def __init__(self, value: object) -> None:
            """Initialize builder with value to validate."""
            self._value = value
            self._checks: list[str] = []
            self._target_type: str | None = None
            self._default: object = None

        def not_none(self) -> Self:
            """Add not None check."""
            self._checks.append("not_none")
            return self

        def truthy(self) -> Self:
            """Add truthy check."""
            self._checks.append("truthy")
            return self

        def str(self, *, default: _StrType = "") -> Self:
            """Set target type to string."""
            self._target_type = "str"
            self._default = default
            return self

        def int(self, *, default: _IntType = 0) -> Self:
            """Set target type to int."""
            self._target_type = "int"
            self._default = default
            return self

        def build(self) -> _StrType | _IntType | object:
            """Build and return validated value."""
            # Apply checks
            for check in self._checks:
                if check == "not_none" and self._value is None:
                    return cast("str | int | object", self._default)
                if check == "truthy" and not self._value:
                    return cast("str | int | object", self._default)
            # Convert if target type specified
            if self._target_type == "str":
                return FlextUtilities.conv_str(
                    self._value, default=cast("str", self._default)
                )
            if self._target_type == "int":
                return FlextUtilities.conv_int(
                    self._value, default=cast("int", self._default)
                )
            return self._value

    @staticmethod
    def validate_builder(value: object) -> ValidateBuilder:
        """Create validate builder (DSL entry point).

        Mnemonic: validate_builder = validate builder (avoids conflict with validate method)

        Args:
            value: Value to validate

        Returns:
            ValidateBuilder instance for chaining

        Example:
            result = u.validate_builder(value).not_none().str().build()

        """
        return FlextUtilities.ValidateBuilder(value)

    @staticmethod
    def pluck(
        items: Collection[object] | object,  # type: ignore[type-arg,valid-type]  # Collection accepts object
        *,
        key: str | int | Callable[[object], object],
    ) -> list[object]:
        """Pluck values from items (generalized: mp(items).pluck(key).build()).

        Mnemonic: pluck = extract values
        Generic replacement for: pluck() in specific utilities

        Args:
            items: Items to pluck from
            key: Key/extractor (str/int/callable)

        Returns:
            List of extracted values

        """
        if not items:
            return []
        items_list = (
            list(items)
            if isinstance(items, Collection) and not isinstance(items, str)
            else [items]
        )
        if callable(key):
            return cast("list[object]", FlextUtilities.map(items_list, mapper=key))
        return [
            FlextUtilities.get(item, str(key) if isinstance(key, int) else key)
            for item in items_list
        ]

    @staticmethod
    def fold(
        items: Collection[object] | object,  # type: ignore[type-arg,valid-type]  # Collection accepts object
        *,
        initial: object,
        folder: Callable[[object, object], object],
        predicate: Callable[[object], bool] | None = None,
    ) -> object:
        """Fold items (generalized: filt(items).fold(initial, folder).build()).

        Mnemonic: fold = reduce/fold
        Generic replacement for: fold() in specific utilities

        Args:
            items: Items to fold
            initial: Initial accumulator
            folder: (acc, item) -> new_acc
            predicate: Optional filter

        Returns:
            Final accumulator

        """
        if not items:
            return initial
        items_list = (
            list(items)
            if isinstance(items, Collection) and not isinstance(items, str)
            else [items]
        )
        if predicate:
            filtered = FlextUtilities.filter(items_list, predicate=predicate)
            items_list = list(filtered) if isinstance(filtered, (list, tuple)) else []
        result = initial
        for item in items_list:
            result = folder(result, item)
        return result

    @staticmethod
    def maybe(
        value: object | None,
        *,
        default: object | None = None,
        mapper: Callable[[object], object] | None = None,
    ) -> object:
        """Maybe monad (generalized: or_(value, default) + chain(mapper)).

        Mnemonic: maybe = optional value processing
        Generic replacement for: maybe() in specific utilities

        Args:
            value: Optional value
            default: Default if None
            mapper: Optional transformation

        Returns:
            Processed value or default

        """
        result = FlextUtilities.or_(value, default=default)
        if mapper and result is not None:
            return FlextUtilities.chain(result, mapper)
        return result

    @staticmethod
    def zip_with(
        *sequences: Collection[object],  # type: ignore[type-arg,valid-type]  # Collection accepts object
        combiner: Callable[..., object] | None = None,
    ) -> list[object]:
        """Zip with combiner (generalized: zip() + map(combiner)).

        Mnemonic: zip_with = zip and combine
        Generic replacement for: zip_with() in specific utilities

        Args:
            *sequences: Sequences to zip
            combiner: Combine function (default: tuple)

        Returns:
            List of combined results

        """
        if not sequences:
            return []
        if len(sequences) == 1:
            return list(sequences[0])
        zipped = zip(*sequences, strict=False)
        if combiner:
            return list(starmap(combiner, zipped))
        return [tuple(items) for items in zipped]

    @staticmethod
    def partition(
        items: Collection[object] | object,  # type: ignore[type-arg,valid-type]  # Collection accepts object
        *,
        predicate: Callable[[object], bool],
    ) -> tuple[list[object], list[object]]:
        """Partition items (generalized: filt(items).partition(predicate).build()).

        Mnemonic: partition = split by predicate
        Generic replacement for: partition() in specific utilities

        Args:
            items: Items to partition
            predicate: Test function

        Returns:
            (true_items, false_items)

        """
        if not items:
            return ([], [])
        items_list = (
            list(items)
            if isinstance(items, Collection) and not isinstance(items, str)
            else [items]
        )
        true_items = FlextUtilities.filter(items_list, predicate=predicate)
        true_list = list(true_items) if isinstance(true_items, (list, tuple)) else []
        false_list = [item for item in items_list if not predicate(item)]
        return (true_list, false_list)


# Alias for convenience
u = FlextUtilities

__all__ = [
    "FlextUtilities",
    "u",
]
