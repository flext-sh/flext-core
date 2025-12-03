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

import time
import uuid
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import cast, overload

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
        # Value is already T, no cast needed

        if not validators:
            # Business Rule: No validators means value is accepted as-is
            # Type narrowing: value is T when no validation required
            return r[T].ok(value)  # type: ignore[arg-type]

        errors: list[str] = []
        field_prefix = f"{field_name}: " if field_name else ""

        if mode == "any":
            # OR mode: at least one must pass
            for validator in validators:
                if validator(value):
                    # Business Rule: After validator passes, value conforms to T
                    # Type narrowing: validator ensures value is T
                    return r[T].ok(value)
            # None passed
            descriptions = list(
                u.map(validators, lambda v: getattr(v, "description", "validator"))
            )
            error_msg = (
                f"{field_prefix}None of the validators passed: "
                f"{', '.join(descriptions)}"
            )
            return r[T].fail(error_msg)

        # Default: "all" mode (AND)
        for validator in validators:
            if not validator(value):
                description = getattr(validator, "description", "validator")
                error_msg = f"{field_prefix}Validation failed: {description}"
                if fail_fast and not collect_errors:
                    return r[T].fail(error_msg)
                errors.append(error_msg)

        if errors:
            return r[T].fail("; ".join(errors))

        # Business Rule: After all validators pass, value is guaranteed to be T
        # Validators ensure value conforms to T at runtime, so we can safely return it
        # Type narrowing: value parameter is already typed as T in function signature
        # Runtime guarantee: validators ensure value conforms to T's constraints
        return r[T].ok(value)

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
        # Type narrowing: After issubclass check, target is type[StrEnum] which is type[T]
        # Business Rule: StrEnum classes expose __members__ dict with all enum members
        # Cast to type[StrEnum] to help type checker understand __members__ attribute
        enum_type: type[StrEnum] = cast("type[StrEnum]", target)
        if case_insensitive:
            # Business Rule: StrEnum classes expose __members__ dict with all enum
            # Use u.find() for unified finding with predicate
            members_dict = getattr(enum_type, "__members__", {})
            members_list = list(members_dict.values())
            found = u.find(
                members_list,
                lambda m: u.normalize(m.value, value) or u.normalize(m.name, value),  # type: ignore[arg-type]
            )
            # Use u.when() for conditional return (DSL pattern)
            found_result = u.when(
                condition=found is not None,
                then_value=r[T].ok(cast("T", found)),
                else_value=None,
            )
            if found_result is not None:
                return found_result
        result = FlextEnum.parse(target, value)
        if result.is_success:
            return r[T].ok(result.value)
        return r[T].fail(result.error or "Enum parse failed")

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
        return r[T].fail(result.error or "Model parse failed")

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
        if u.find(true_values, lambda v: v == normalized_val):  # type: ignore[arg-type]
            return r[bool].ok(True)
        if u.find(false_values, lambda v: v == normalized_val):  # type: ignore[arg-type]
            return r[bool].ok(False)
        return None

    @staticmethod
    def _coerce_primitive[T](value: object, target: type[T]) -> r[T] | None:
        """Coerce primitive types. Returns None if no coercion applied.

        Business Rule: Primitive type coercion supports common conversions:
        str↔int, str↔float, str↔bool. Boolean coercion recognizes common
        string representations (true/false, yes/no, on/off, 1/0).
        """
        if target is int:
            return FlextUtilities._coerce_to_int(value)  # type: ignore[return-value]
        if target is float:
            return FlextUtilities._coerce_to_float(value)  # type: ignore[return-value]
        if target is str:
            return FlextUtilities._coerce_to_str(value)  # type: ignore[return-value]
        if target is bool:
            if isinstance(value, str):
                bool_result = FlextUtilities._coerce_to_bool_from_str(value)
                if bool_result is not None:
                    return bool_result  # type: ignore[return-value]
            return r[T].ok(bool(value))  # type: ignore[arg-type]
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
        return FlextUtilities._parse_with_default(
            default, default_factory, model_result.error or ""
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
            target_name = getattr(target, "__name__", "type")
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
            parsed = target(value)  # type: ignore[call-arg]
            return r[T].ok(parsed)
        except Exception as e:
            target_name = getattr(target, "__name__", "type")
            return FlextUtilities._parse_with_default(
                default,
                default_factory,
                f"{field_prefix}Cannot parse {type(value).__name__} "
                f"to {target_name}: {e}",
            )

    @staticmethod
    def parse[T](  # noqa: PLR0913
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
            return u.when(
                condition=default is not None,
                then_value=r[T].ok(default),
                else_value=u.when(
                    condition=default_factory is not None,
                    then_value=r[T].ok(default_factory()),
                    else_value=r[T].fail(field_prefix or "Value is None"),
                ),
            )

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
                model_result = FlextModel.from_dict(to_model, result)  # type: ignore[arg-type]
                if model_result.is_failure:
                    return r[dict[str, t.GeneralValueType]].fail(
                        model_result.error or "Model conversion failed"
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
                            return r[object].fail(
                                f"Pipeline step {i} failed: {result.error}"
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
                result[key] = list(result[key]) + list(value)  # type: ignore[call-overload]
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
                        should_include_fn,  # type: ignore[arg-type]
                    )

            return r[dict[str, t.GeneralValueType]].ok(
                merged  # type: ignore[arg-type]
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
        elif hasattr(current, key_part):
            result = r[T | None].ok(cast("T | None", getattr(current, key_part)))
        # Handle Pydantic model
        elif hasattr(current, "model_dump"):
            model_dump_attr = getattr(current, "model_dump", None)
            if model_dump_attr is None:
                result = error_or_default(
                    f"Cannot access '{key_part}' at '{path_context}'"
                )
            else:
                model_dump_method = cast(
                    "Callable[[], dict[str, object]]", model_dump_attr
                )
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
                get_result: r[T | None] | None = FlextUtilities._extract_get_value(  # type: ignore[misc]
                    current,
                    key_part,
                    path_context,
                    required=required,
                    default=default,
                )
                if get_result is None:
                    continue
                if get_result.is_failure:
                    return get_result
                current = get_result.value

                # Handle array index
                if array_match is not None:
                    array_result: r[T | None] = (
                        FlextUtilities._extract_handle_array_index(  # type: ignore[misc]
                            current,
                            array_match,
                            key_part,
                            required=required,
                            default=default,
                        )
                    )
                    if array_result.is_failure:
                        return array_result
                    current = array_result.value

            return r[T | None].ok(current)  # type: ignore[arg-type]

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
        """Helper: Process a single batch item, return result, error Result, or None if skipped."""
        try:
            result = operation(item)
            if isinstance(result, FlextResult):
                if result.is_failure:
                    error_msg = result.error or "Unknown error"
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
        nested = cast(
            "list[list[t.GeneralValueType] | tuple[t.GeneralValueType, ...]]",
            u.filter(validated_results, lambda x: isinstance(x, (list, tuple))),  # type: ignore[arg-type]
        )
        if not nested:
            return validated_results
        # Use u.flat() for unified flattening
        flattened = u.flat(nested)  # type: ignore[arg-type]
        # Add non-list items
        non_list = cast(
            "list[t.GeneralValueType]",
            u.filter(validated_results, lambda x: not isinstance(x, (list, tuple))),  # type: ignore[arg-type]
        )
        return flattened + non_list

    @staticmethod
    def batch[T, R](  # noqa: PLR0913
        items: list[T],
        operation: Callable[[T], R | r[R]],
        *,
        _size: int = 100,  # Reserved for future chunking support
        on_error: str = "collect",
        _parallel: bool = False,  # Reserved for future async support
        progress: Callable[[int, int], None]
        | None = None,  # Progress callback (current, total)
        progress_interval: int = 1,  # Reserved for future chunking support  # noqa: ARG004
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
            filtered_items = (
                cast("list[T]", filtered_result)
                if isinstance(filtered_result, list)
                else items
            )
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
        validated_results_raw = (
            u.filter(processed_results, post_validate)
            if post_validate is not None
            else processed_results
        )
        if not isinstance(validated_results_raw, list):
            validated_results_raw = list(validated_results_raw)

        # Convert to GeneralValueType for flattening using u.map
        validated_results = cast(
            "list[t.GeneralValueType]",
            u.map(validated_results_raw, lambda r: cast("t.GeneralValueType", r)),  # type: ignore[arg-type]
        )

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
            check_result = FlextUtilities._guard_check_condition(
                value,  # type: ignore[arg-type]
                cast(
                    "type[object] | tuple[type[object], ...] | Callable[[object], bool] | ValidatorSpec | str",
                    condition,
                ),
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
    ) -> str:
        """Get string value (generalized from get_str)."""

    @staticmethod
    @overload
    def get[T](
        data: Mapping[str, object] | object,
        key: str,
        *,
        default: list[T] | None = None,
    ) -> list[T]:
        """Get list value (generalized from get_list)."""

    @staticmethod
    @overload
    def get[T](
        data: Mapping[str, object] | object,
        key: str,
        *,
        default: T | None = None,
    ) -> T | None:
        """Get value with default."""

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
        if default == "":
            value = FlextUtilities._get_raw(data, key, default=default)
            return cast(
                "str", u.build(value, ops={"ensure": "str", "ensure_default": default})
            )  # type: ignore[arg-type, return-value]

        # Handle list default (generalized from get_list)
        # Check if default is empty list (type hint for list return)
        if isinstance(default, list) and len(default) == 0:
            value = FlextUtilities._get_raw(data, key, default=default)
            result = u.build(value, ops={"ensure": "list", "ensure_default": default})  # type: ignore[arg-type]
            return cast("list[T]", result if isinstance(result, list) else default)

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
                    result = data.get(key, default)  # type: ignore[union-attr]
                    return cast("T | None", result)
                return default
            case _:
                return getattr(data, key, default)

    # Backward compatibility alias for get_or_default
    get_or_default = get

    @staticmethod
    def find[T](
        items: list[T] | tuple[T, ...] | dict[str, T] | Mapping[str, T],
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
            case list() | tuple():
                list_predicate = cast("Callable[[T], bool]", predicate)
                for item in items:
                    typed_item = cast("T", item)
                    if list_predicate(typed_item):
                        return typed_item
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
        """Filter a list with optional mapping (uses u.map + u.filter internally)."""
        if mapper is not None:
            # Use u.map() for unified mapping, then u.filter() for unified filtering
            mapped = cast("list[R]", u.map(items_list, mapper))
            mapped_predicate = cast("Callable[[R], bool]", predicate)
            # Use u.filter internally (no recursion - different signature)
            return cast("list[R]", u.filter(mapped, mapped_predicate))
        list_predicate = cast("Callable[[T], bool]", predicate)
        # Use u.filter internally (no recursion - different signature)
        return cast("list[T]", u.filter(items_list, list_predicate))

    @staticmethod
    def _filter_dict[T, R](
        items_dict: dict[str, T],
        predicate: Callable[[str, T], bool] | Callable[[str, R], bool],
        mapper: Callable[[str, T], R] | None = None,
    ) -> dict[str, T] | dict[str, R]:
        """Filter a dict with optional mapping (uses u.map + u.filter internally)."""
        if mapper is not None:
            # Use u.map() for unified mapping, then u.filter() for unified filtering
            mapped_dict = cast("dict[str, R]", u.map(items_dict, mapper))
            mapped_dict_predicate = cast("Callable[[str, R], bool]", predicate)
            # Use u.filter internally (no recursion - different signature)
            return cast("dict[str, R]", u.filter(mapped_dict, mapped_dict_predicate))
        dict_predicate = cast("Callable[[str, T], bool]", predicate)
        # Use u.filter internally (no recursion - different signature)
        return cast("dict[str, T]", u.filter(items_dict, dict_predicate))

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
        match items:
            case list() | tuple():
                list_items = list(items)
                list_predicate = cast("Callable[[T], bool]", predicate)
                list_mapper = cast("Callable[[T], R] | None", mapper)
                return FlextUtilities._filter_list(
                    list_items, list_predicate, list_mapper
                )
            case dict() | Mapping():
                dict_items = dict(items)
                dict_predicate = cast("Callable[[str, T], bool]", predicate)
                dict_mapper = cast("Callable[[str, T], R] | None", mapper)
                return FlextUtilities._filter_dict(
                    dict_items, dict_predicate, dict_mapper
                )
            case _:
                single_predicate = cast("Callable[[T], bool]", predicate)
                single_mapper = cast("Callable[[T], R] | None", mapper)
                return FlextUtilities._filter_single(
                    items, single_predicate, single_mapper
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
                desc = getattr(condition, "description", "validation")
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
            return error_msg or shortcut_result.error or "Guard check failed"
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
                    func_name = getattr(condition, "__name__", "custom")
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
            return FlextUtilities._guard_check_predicate(
                value,
                condition,
                context_name,
                error_msg,  # type: ignore[arg-type]
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
        mapped = u.map(value, case_func)  # type: ignore[arg-type]
        # u.map() now preserves set type automatically
        if isinstance(mapped, set):
            return mapped  # type: ignore[return-value]
        return cast("list[str]", mapped)

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
        normalized = u.map(other, normalize_func)  # type: ignore[arg-type]
        # Use u.when() for conditional collection type (DSL pattern)
        normalized_collection = u.when(
            condition=isinstance(other, (set, frozenset)),
            then_value=set(normalized),
            else_value=normalized,
        )
        return item_lower in normalized_collection

    @staticmethod
    def _ensure_to_list[T](
        value: T | list[T] | tuple[T, ...] | None,
        default: list[T] | None,
    ) -> list[T]:
        """Helper: Convert value to list."""
        if value is None:
            # Use u.or_() for default fallback (DSL pattern)
            return u.or_(default, [])
        match value:
            case list():
                return value
            case tuple():
                return list(value)
            case _:
                return [value]  # type: ignore[list-item]

    @staticmethod
    def _ensure_to_dict[T](
        value: T | dict[str, T] | None,
        default: dict[str, T] | None,
    ) -> dict[str, T]:
        """Helper: Convert value to dict."""
        if value is None:
            # Use u.or_() for default fallback (DSL pattern)
            return u.or_(default, {})
        match value:
            case dict():
                return value  # type: ignore[return-value]
            case _:
                return {"value": value}  # type: ignore[dict-item]

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
            list_default = u.when(
                condition=isinstance(default, list),
                then_value=cast("list[str]", default),
                else_value=None,
            )
            # Use FlextDataMapper directly for str_list (internal implementation)
            return cast(
                "list[T]", FlextDataMapper.ensure_str_list(value, default=list_default)
            )
        if target_type == "dict":
            # Use u.when() for conditional assignment (DSL pattern)
            dict_default = u.when(
                condition=isinstance(default, dict),
                then_value=default,
                else_value=None,
            )
            return FlextUtilities._ensure_to_dict(value, dict_default)  # type: ignore[arg-type, return-value]
        if target_type == "auto" and isinstance(value, dict):
            return value  # type: ignore[return-value]
        # Handle list or fallback
        # Use u.when() for conditional assignment (DSL pattern)
        list_default = u.when(
            condition=isinstance(default, list),
            then_value=default,
            else_value=None,
        )
        return FlextUtilities._ensure_to_list(value, list_default)  # type: ignore[arg-type, return-value]

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
            key_predicate = lambda k, _v: (  # noqa: E731
                (filter_keys is None or k in filter_keys)
                and (exclude_keys is None or k not in exclude_keys)
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
    ) -> r[R]:
        """Map result or return failure (generalized from map_or)."""

    @staticmethod
    @overload
    def map[T, R](
        items: T
        | list[T]
        | tuple[T, ...]
        | set[T]
        | frozenset[T]
        | dict[str, T]
        | Mapping[str, T],
        mapper: Callable[[T], R] | Callable[[str, T], R],
    ) -> list[R] | set[R] | frozenset[R] | dict[str, R]:
        """Map collection items."""

    @staticmethod
    def map[T, R](
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
                return r[R].ok(mapper(items.value))  # type: ignore[arg-type]
            return r[R].fail(u.err(items, default=default_error))

        # Handle collections (original map behavior)
        if isinstance(items, (list, tuple)):
            list_mapper = cast("Callable[[T], R]", mapper)
            # Cannot use u.map() here - would cause recursion (u.map calls itself for lists)
            # Use list comprehension directly for mapping
            return [list_mapper(item) for item in items]  # type: ignore[arg-type, list-item]

        if isinstance(items, (set, frozenset)):
            set_mapper = cast("Callable[[T], R]", mapper)
            mapped_items = {set_mapper(item) for item in items}  # type: ignore[arg-type]
            # Preserve frozenset type if input was frozenset
            if isinstance(items, frozenset):
                return cast("frozenset[R]", frozenset(mapped_items))
            return cast("set[R]", mapped_items)

        if isinstance(items, (dict, Mapping)):
            dict_mapper = cast("Callable[[str, T], R]", mapper)
            return {k: dict_mapper(k, v) for k, v in items.items()}

        # Single value - wrap in list and map
        # This handles the case where items is a single value (not list/tuple/dict/set)
        # Mypy considers this unreachable because items type is union, but runtime handles it
        single_mapper = cast("Callable[[T], R]", mapper)
        return [single_mapper(items)]  # type: ignore[arg-type, unreachable]

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
            return converter(value, default)  # type: ignore[arg-type]

        # Fallback: try direct conversion
        try:
            converted = target_type(value)  # type: ignore[call-overload, assignment, arg-type, misc]
            return cast("T", converted)
        except (ValueError, TypeError):
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
        return u.ensure(current, target_type=ensure_type, default=default_val)  # type: ignore[arg-type, return-value]

    @staticmethod
    def _build_apply_filter(
        current: object, ops: dict[str, object], default: object
    ) -> object:
        """Helper: Apply filter operation."""
        if "filter" not in ops:
            return current
        filter_pred = cast("Callable[[object], bool]", ops["filter"])
        if isinstance(current, (list, tuple, set, frozenset)):
            return u.filter(current, predicate=filter_pred)  # type: ignore[arg-type, return-value]
        if isinstance(current, dict):
            return u.filter(current, predicate=lambda _k, v: filter_pred(v))  # type: ignore[arg-type, return-value]
        return default if not filter_pred(current) else current  # type: ignore[arg-type]

    @staticmethod
    def _build_apply_map(current: object, ops: dict[str, object]) -> object:
        """Helper: Apply map operation."""
        if "map" not in ops:
            return current
        map_func = cast("Callable[[object], object]", ops["map"])
        if isinstance(current, (list, tuple, set, frozenset, dict)):
            return u.map(current, mapper=map_func)  # type: ignore[arg-type, return-value]
        return map_func(current)  # type: ignore[arg-type, return-value]

    @staticmethod
    def _build_apply_normalize(current: object, ops: dict[str, object]) -> object:
        """Helper: Apply normalize operation."""
        if "normalize" not in ops or not isinstance(
            current, (str, list, tuple, set, frozenset)
        ):
            return current
        normalize_case = cast("str", ops["normalize"])
        return u.normalize(current, case=normalize_case)  # type: ignore[arg-type, return-value]

    @staticmethod
    def _build_apply_convert(current: object, ops: dict[str, object]) -> object:
        """Helper: Apply convert operation."""
        if "convert" not in ops:
            return current
        convert_type = cast("type[object]", ops["convert"])
        convert_default = ops.get("convert_default", convert_type())
        return u.convert(current, convert_type, convert_default)  # type: ignore[arg-type, return-value]

    @staticmethod
    def _build_apply_transform(
        current: object, ops: dict[str, object], default: object, on_error: str
    ) -> object:
        """Helper: Apply transform operation."""
        if "transform" not in ops or not isinstance(current, (dict, Mapping)):
            return current
        transform_opts = cast("dict[str, object]", ops["transform"])
        transform_result = u.transform(current, **transform_opts)  # type: ignore[arg-type]
        if transform_result.is_success:
            return transform_result.value
        return default if on_error == "stop" else current

    @staticmethod
    def _build_apply_process(
        current: object, ops: dict[str, object], default: object, on_error: str
    ) -> object:
        """Helper: Apply process operation using u.process()."""
        if "process" not in ops:
            return current
        process_func = cast("Callable[[object], object]", ops["process"])
        # Use u.process() for unified processing
        if isinstance(current, (list, tuple, dict, Mapping)):
            process_result = u.process(
                current, processor=process_func, on_error=on_error
            )  # type: ignore[arg-type]
            if process_result.is_success:
                return process_result.value
            return default if on_error == "stop" else current
        # Single value processing with error handling
        try:
            return process_func(current)  # type: ignore[arg-type, return-value]
        except Exception:
            return default if on_error == "stop" else current

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
        return FlextUtilities._build_apply_process(
            current, ops, ensure_default_val or current, on_error
        )  # type: ignore[return-value]

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
        if callable(field):
            extracted = cast("list[object]", u.map(items, field))  # type: ignore[arg-type]
        else:
            # Use u.get for unified extraction (works for dicts and objects)
            extracted = cast(
                "list[object]", u.map(items, lambda item: u.get(item, field))
            )  # type: ignore[arg-type]
        # Filter None values before aggregation
        filtered = cast("list[object]", u.filter(extracted, lambda x: x is not None))  # type: ignore[arg-type]
        agg_fn = fn if fn is not None else sum
        return agg_fn(filtered)  # type: ignore[arg-type, return-value]

    @staticmethod
    @overload
    def fields[T](
        source: Mapping[str, object] | object,
        name: str,
        *,
        default: T | None = None,
        required: bool = False,
        ops: dict[str, object] | None = None,
    ) -> T | None:
        """Extract single field (overload for single field)."""

    @staticmethod
    @overload
    def fields[T](
        source: Mapping[str, object] | object,
        spec: dict[str, dict[str, object] | T | None],
        *,
        on_error: str = "stop",
    ) -> dict[str, T | None] | r[dict[str, T]]:
        """Extract multiple fields (overload for multiple fields)."""

    @staticmethod
    def fields[T](
        source: Mapping[str, object] | object,
        name_or_spec: str | dict[str, dict[str, object] | T | None],
        *,
        default: T | None = None,
        required: bool = False,
        ops: dict[str, object] | None = None,
        on_error: str = "stop",
    ) -> T | None | dict[str, T | None] | r[dict[str, T]]:
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
            return FlextUtilities._fields_multi(source, name_or_spec, on_error=on_error)

        # Handle single field case
        name = cast("str", name_or_spec)
        value = u.get(source, name, default=default)
        if value is None and required:
            return None
        if ops is not None:
            return cast("T | None", u.build(value, ops=ops, on_error="stop"))
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
                extracted = cast(
                    "T | None", u.build(value, ops=field_ops, on_error="stop")
                )
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

                extracted = u.extract(
                    source,
                    source_field,
                    default=field_default,
                    required=False,
                    ops=cast("dict[str, object] | None", field_ops),
                )
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
        data: Mapping[str, object] | object,
        key: str,
        *,
        as_type: type[T] | None = None,
        default: T | None = None,
        guard: bool = True,
    ) -> T | None:
        """Extract value with type guard (overload for dict/object extraction)."""

    @staticmethod
    @overload
    def take[T](
        items: dict[str, T],
        n: int,
        *,
        from_start: bool = True,
    ) -> dict[str, T]:
        """Take first N items from dict (overload)."""

    @staticmethod
    @overload
    def take[T](
        items: list[T] | tuple[T, ...],
        n: int,
        *,
        from_start: bool = True,
    ) -> list[T]:
        """Take first N items from list/tuple (overload)."""

    @staticmethod
    def take[T](
        data_or_items: Mapping[str, object] | object | dict[str, T] | list[T] | tuple[T, ...],
        key_or_n: str | int,
        *,
        as_type: type[T] | None = None,
        default: T | None = None,
        guard: bool = True,
        from_start: bool = True,
    ) -> T | None | dict[str, T] | list[T]:
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
                return default
            if as_type and guard:
                guarded = u.guard(value, as_type, return_value=True, default=default)
                return cast("T | None", guarded)
            return cast("T | None", value)

        # Slice mode: take N items from list/dict
        items = cast("dict[str, T] | list[T] | tuple[T, ...]", data_or_items)
        n = key_or_n
        if isinstance(items, dict):
            # Use u.keys() for unified key extraction
            keys = u.keys(items)
            selected_keys = keys[:n] if from_start else keys[-n:]
            # Cannot use u.map() here - would cause recursion (u.map calls itself for lists)
            # Use dict comprehension directly
            return {k: items[k] for k in selected_keys}
        items_list = list(items)
        return items_list[:n] if from_start else items_list[-n:]

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
            converted = (
                u.convert(value, target, default)
                if default is not None
                else u.convert(value, target, cast("T", None))
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

        """
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
    def chain[T](
        value: T,
        *funcs: Callable[[object], object],
    ) -> object:
        """Chain operations (mnemonic: chain = pipeline).

        Generic replacement for: func3(func2(func1(value))) patterns

        Args:
            value: Initial value
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
                mapped = cast(
                    "list[str]",
                    u.map(items_list, lambda kv: f"{kv[0]}{key_sep}{kv[1]}"),
                )  # type: ignore[arg-type]
                return sep.join(mapped)
            # Use u.vals() for unified value extraction, then u.map for mapping
            values_list = u.vals(items)
            mapped = cast("list[str]", u.map(values_list, str))  # type: ignore[arg-type]
            return sep.join(mapped)
        # Use u.map for unified mapping
        mapped = cast("list[str]", u.map(items, str))  # type: ignore[arg-type]
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
        if callable(key):
            keys = cast("list[K]", u.map(items, key))  # type: ignore[arg-type]
        else:
            # Use u.get for unified extraction (works for dicts and objects)
            keys = cast("list[K]", u.map(items, lambda item: u.get(item, key)))  # type: ignore[arg-type]
        # Group items by keys using zip_ for unified zip
        pairs = cast("list[tuple[T, K]]", u.zip_(items, keys))
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
        filtered = u.filter(items, predicate)  # type: ignore[arg-type]
        return (
            len(filtered)
            if isinstance(filtered, (list, tuple, set, frozenset))
            else 1
            if filtered
            else 0
        )

    @staticmethod
    def err(
        result: r[object],
        *,
        default: str = "Unknown error",
    ) -> str:
        """Extract error message from r (mnemonic: err = error).

        Generic replacement for: str(result.error) if result.error else "Unknown error"

        Args:
            result: FlextResult to extract error from
            default: Default error message if error is None/empty

        Returns:
            Error message string

        Example:
            error_msg = u.err(result, default="Operation failed")
            # → "Connection timeout" or "Operation failed"

        """
        # Use u.when() for conditional return (DSL pattern)
        return u.when(
            condition=result.is_failure,
            then_value=cast(
                "str", u.ensure(result.error, target_type="str", default=default)
            ),
            else_value=default,
        )

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
                return cast("T", default) if default is not None else cast("T", None)
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
        if isinstance(items, r):
            if items.is_failure:
                return default or []
            items_dict = items.value
        else:
            items_dict = items
        return list(items_dict.values()) if items_dict else (default or [])

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
        if isinstance(items, r):
            if items.is_failure:
                return default or []
            items_dict = items.value
        else:
            items_dict = items
        return list(items_dict.keys()) if items_dict else (default or [])

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
            items = u.val(items, default=[])
            if u.empty(items):
                # Use u.or_() for default fallback (DSL pattern)
                return cast("T", u.or_(default, 0))

        # Use u.agg() for unified aggregation when mapper provided
        if mapper is not None:
            # Use u.agg() with mapper for unified aggregation
            return cast("T", u.agg(items, mapper, fn=sum))  # type: ignore[arg-type, return-value]

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
        # Use u.keys() for unified key extraction, then access first
        if isinstance(items, dict):
            keys = u.keys(items)
            return items[keys[0]] if keys else default
        return items[0] if items else default

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
        # Use negative index for unified access
        if isinstance(items, dict):
            keys = list(items.keys())
            return items[keys[-1]] if keys else default
        return items[-1] if items else default

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
            if isinstance(items, dict):
                return (
                    items.get(cast("str", index), default)
                    if isinstance(index, str)
                    else default
                )
            return (
                items[cast("int", index)]
                if 0 <= cast("int", index) < len(items)
                else default
            )
        except (IndexError, KeyError, TypeError):
            return default

    @staticmethod
    def flat[T](
        items: list[list[T]] | list[tuple[T, ...]] | tuple[list[T], ...],
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
        processed = cast(
            "list[list[T]]",
            u.map(items, list),  # type: ignore[arg-type]
        )
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

        Generic replacement for: r[T].fail(error)

        Args:
            error: Error message

        Returns:
            r[T] with failure

        Example:
            result = u.fail("Operation failed")
            # → r.fail("Operation failed")

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
    def empty(
        items: list[object] | tuple[object, ...] | dict[str, object] | str | None,
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
    ) -> bool:
        """Check if string ends with suffix (overload for single suffix)."""

    @staticmethod
    @overload
    def ends(
        value: str,
        *suffixes: str,
    ) -> bool:
        """Check if string ends with any suffix (overload for multiple suffixes)."""

    @staticmethod
    def ends(
        value: str,
        *suffixes: str,
    ) -> bool:
        """Check if string ends with suffix(es) (generalized from ends_any).

        Generic replacement for: value.endswith(suffix) or any(value.endswith(s) for s in suffixes)

        Args:
            value: String to check
            *suffixes: One or more suffixes to check

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
        if not suffixes:
            return False
        # Use u.any_() with u.map() for unified checking

        def check_suffix(suffix: str) -> bool:
            """Check if value ends with suffix."""
            return value.endswith(suffix)

        mapped = cast("list[bool]", u.map(suffixes, check_suffix))  # type: ignore[arg-type]
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
    ) -> bool:
        """Check if string starts with prefix (overload for single prefix)."""

    @staticmethod
    @overload
    def starts(
        value: str,
        *prefixes: str,
    ) -> bool:
        """Check if string starts with any prefix (overload for multiple prefixes)."""

    @staticmethod
    def starts(
        value: str,
        *prefixes: str,
    ) -> bool:
        """Check if string starts with prefix(es) (generalized from starts_any).

        Generic replacement for: any(value.startswith(p) for p in prefixes)

        Args:
            value: String to check
            *prefixes: Prefixes to check

        Returns:
            True if starts with any prefix

        Example:
            if u.starts(name, "tap-", "target-", "dbt-"):
                process_plugin()

        """

        # Use u.any_() with u.map() for unified checking
        # Use partial for cleaner function composition (DSL pattern)
        def check_prefix(prefix: str) -> bool:
            """Check if value starts with prefix."""
            return value.startswith(prefix)

        mapped = cast("list[bool]", u.map(prefixes, check_prefix))  # type: ignore[arg-type]
        return u.any_(*mapped)

    @staticmethod
    @overload
    def cast[T, R](
        result: r[T],
        *,
        default_error: str = "Operation failed",
    ) -> r[R]:
        """Cast result type (generalized from cast_r)."""

    @staticmethod
    @overload
    def cast[T, R](
        value: T,
    ) -> R:
        """Cast value type."""

    @staticmethod
    def cast[T, R](
        value_or_result: T | r[T],
        *,
        default_error: str = "Operation failed",
    ) -> R | r[R]:
        """Cast value or result type (generalized from cast_r).

        Generic replacement for: cast(R, value), cast_r(result)

        Generic replacement for: if result.is_success: return r.ok(cast(R, result.value)) else return r.fail(result.error)

        Args:
            result: Result to cast
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
        # Handle r[T] case (generalized from cast_r)
        if isinstance(value_or_result, r):
            if value_or_result.is_success:
                return r[R].ok(cast("R", value_or_result.value))
            return r[R].fail(u.err(value_or_result, default=default_error))
        # Handle direct value case
        return cast("R", value_or_result)

    # Backward compatibility aliases (DEPRECATED - use generalized functions)
    ends_any = ends  # DEPRECATED: Use u.ends() instead
    starts_any = starts  # DEPRECATED: Use u.starts() instead
    take_n = take  # DEPRECATED: Use u.take() instead
    field = fields  # DEPRECATED: Use u.fields() instead (single field extraction)
    # get_str, get_list, map_or, map_r, cast_r, ensure_str_list are deprecated - use u.get(), u.map(), u.cast(), u.ensure() instead


# Alias for convenience
u = FlextUtilities

__all__ = [
    "FlextUtilities",
    "u",
]
