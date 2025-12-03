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
from typing import cast

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
from flext_core.result import FlextResult, r
from flext_core.runtime import FlextRuntime
from flext_core.typings import t


class FlextUtilities:
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
    # POWER METHODS: Direct utility operations with FlextResult
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
            error_msg = f"{field_prefix}None of the validators passed: {', '.join(descriptions)}"
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
            # Business Rule: StrEnum classes expose __members__ dict with all enum members
            # Iterate through enum members for case-insensitive matching
            # After validation, member is guaranteed to be T (the enum type)
            members_dict = getattr(enum_type, "__members__", {})
            for member in members_dict.values():
                if u.normalize(member.value, value) or u.normalize(member.name, value):
                    # Business Rule: member is an instance of target (type[T]), so it's T
                    # Type narrowing: after issubclass check, member is StrEnum which is T
                    return r[T].ok(cast("T", member))
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
        """Helper: Coerce string to bool."""
        normalized_val = cast("str", u.normalize(value, case="lower"))
        if normalized_val in {"true", "1", "yes", "on"}:
            return r[bool].ok(True)
        if normalized_val in {"false", "0", "no", "off"}:
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

        # Handle None value
        if value is None:
            return FlextUtilities._parse_handle_none(
                default, default_factory, field_prefix
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

        Business Rule: Chains operations sequentially, unwrapping FlextResult
        values automatically. Error handling modes: "stop" (fail fast) or
        "skip" (continue with previous value). Railway pattern ensures errors
        propagate correctly through the pipeline.

        Args:
            value: Initial value to process
            *operations: Functions to apply in sequence
            on_error: Error handling ("stop" or "skip")

        Returns:
            FlextResult containing final value or error

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

                # Unwrap FlextResult if returned
                if isinstance(result, FlextResult):
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
            FlextResult containing merged dictionary

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
            FlextResult containing extracted value or default

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
        """Helper: Flatten nested lists if requested."""
        if not flatten:
            return validated_results
        flattened: list[t.GeneralValueType] = []
        for result_item in validated_results:
            if isinstance(result_item, (list, tuple)):
                flattened.extend(
                    cast("t.GeneralValueType", item) for item in result_item
                )
            else:
                flattened.append(cast("t.GeneralValueType", result_item))
        return flattened

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
            operation: Function to apply to each item (can return FlextResult or direct value)
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

        # Process items directly to collect errors properly
        errors: list[tuple[int, str]] = []
        processed_results: list[R] = []

        for idx, item in enumerate(filtered_items):
            process_result = FlextUtilities._batch_process_single_item(
                item, idx, operation, errors, on_error
            )
            if process_result is None:
                continue  # Item skipped
            if isinstance(process_result, FlextResult):
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

        # Convert to GeneralValueType for flattening
        validated_results: list[t.GeneralValueType] = [
            cast("t.GeneralValueType", r) for r in validated_results_raw
        ]

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
            FlextResult with operation result or error

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
    def get[T](
        data: Mapping[str, object] | object,
        key: str,
        *,
        default: T | None = None,
    ) -> T | None:
        """Unified get function for dict/object access with default.

        Generic replacement for: get_or_default()

        Automatically detects if data is dict or object and extracts value.

        Args:
            data: Source data (dict or object)
            key: Key/attribute name
            default: Default value if not found (None if not specified)

        Returns:
            Extracted value or default

        Example:
            name = u.get(data, "name", default="unknown")
            port = u.get(config, "port", default=8080)

        """
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
        """Filter a list with optional mapping."""
        if mapper is not None:
            # Use u.map() for unified mapping, then filter directly (no recursion)
            mapped = cast("list[R]", u.map(items_list, mapper))
            mapped_predicate = cast("Callable[[R], bool]", predicate)
            return [item for item in mapped if mapped_predicate(item)]
        list_predicate = cast("Callable[[T], bool]", predicate)
        return [item for item in items_list if list_predicate(item)]

    @staticmethod
    def _filter_dict[T, R](
        items_dict: dict[str, T],
        predicate: Callable[[str, T], bool] | Callable[[str, R], bool],
        mapper: Callable[[str, T], R] | None = None,
    ) -> dict[str, T] | dict[str, R]:
        """Filter a dict with optional mapping."""
        if mapper is not None:
            # Use u.map() for unified mapping, then filter directly (no recursion)
            mapped_dict = cast("dict[str, R]", u.map(items_dict, mapper))
            mapped_dict_predicate = cast("Callable[[str, R], bool]", predicate)
            return {k: v for k, v in mapped_dict.items() if mapped_dict_predicate(k, v)}
        dict_predicate = cast("Callable[[str, T], bool]", predicate)
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
            return default if default is not None else None
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
        """Ensure value is a list of strings, converting if needed.

        **DEPRECATED**: Use u.ensure(value, target_type="str_list") instead.
        Kept for backward compatibility.

        Args:
            value: Value to convert (list, tuple, set, or single value)
            default: Default value if None (empty list if not specified)

        Returns:
            List of strings

        Example:
            # Convert attribute values to string list
            str_list = u.ensure_str_list(attr_values)
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
            value_lower = value.lower()
            other_lower = other.lower()
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
        """Helper: Normalize case only."""
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
        item_lower = value.lower()

        def normalize_func(x: str) -> str:
            """Normalize string to lowercase."""
            return x.lower() if isinstance(x, str) else str(x).lower()

        # Use u.map() for unified mapping
        normalized = u.map(other, normalize_func)  # type: ignore[arg-type]
        normalized_collection = (
            set(normalized) if isinstance(other, (set, frozenset)) else normalized
        )
        return item_lower in normalized_collection

    @staticmethod
    def _ensure_to_list[T](
        value: T | list[T] | tuple[T, ...] | None,
        default: list[T] | None,
    ) -> list[T]:
        """Helper: Convert value to list."""
        if value is None:
            return default if default is not None else []
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
            return default if default is not None else {}
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
            list_default = (
                cast("list[str]", default) if isinstance(default, list) else None
            )
            return cast(
                "list[T]", FlextDataMapper.ensure_str_list(value, default=list_default)
            )
        if target_type == "dict":
            dict_default = default if isinstance(default, dict) else None
            return FlextUtilities._ensure_to_dict(value, dict_default)  # type: ignore[arg-type, return-value]
        if target_type == "auto" and isinstance(value, dict):
            return value  # type: ignore[return-value]
        # Handle list or fallback
        list_default = default if isinstance(default, list) else None
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
            FlextResult containing processed results (list or dict based on input)

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
    def map[T, R](
        items: T
        | list[T]
        | tuple[T, ...]
        | set[T]
        | frozenset[T]
        | dict[str, T]
        | Mapping[str, T],
        mapper: Callable[[T], R] | Callable[[str, T], R],
    ) -> list[R] | set[R] | frozenset[R] | dict[str, R]:  # type: ignore[misc]
        """Unified map function that auto-detects input type.

        Generic replacement for: List/dict comprehensions, manual loops

        Args:
            items: Input items (list, tuple, or dict)
            mapper: Function to transform items
                - For lists: mapper(item) -> result
                - For dicts: mapper(key, value) -> result

        Returns:
            Mapped results (list or dict, preserving input type)

        Example:
            # Map list
            mapped = u.map([1, 2, 3], lambda x: x * 2)

            # Map dict values
            mapped = u.map({"a": 1, "b": 2}, lambda k, v: v * 2)

        """
        if isinstance(items, (list, tuple)):
            list_mapper = cast("Callable[[T], R]", mapper)
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
        """Internal helper for bool conversion."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"true", "1", "yes", "on"}
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


# Alias for convenience
u = FlextUtilities

__all__ = [
    "FlextUtilities",
    "u",
]
