"""Utility façade for validation, parsing, and reliability helpers.

**CRITICAL ARCHITECTURE**: FlextUtilities is a THIN FACADE - pure delegation
to _utilities classes. No other module can import from _utilities directly.
All external code MUST use FlextUtilities as the single access point.

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

from flext_core._utilities.args import FlextUtilitiesArgs
from flext_core._utilities.cache import FlextUtilitiesCache
from flext_core._utilities.collection import FlextUtilitiesCollection
from flext_core._utilities.configuration import FlextUtilitiesConfiguration
from flext_core._utilities.context import FlextUtilitiesContext
from flext_core._utilities.data_mapper import FlextUtilitiesDataMapper
from flext_core._utilities.domain import FlextUtilitiesDomain
from flext_core._utilities.enum import FlextUtilitiesEnum
from flext_core._utilities.generators import FlextUtilitiesGenerators
from flext_core._utilities.model import FlextUtilitiesModel
from flext_core._utilities.pagination import FlextUtilitiesPagination
from flext_core._utilities.reliability import FlextUtilitiesReliability
from flext_core._utilities.string_parser import FlextUtilitiesStringParser
from flext_core._utilities.text_processor import FlextUtilitiesTextProcessor
from flext_core._utilities.type_checker import FlextUtilitiesTypeChecker
from flext_core._utilities.type_guards import FlextUtilitiesTypeGuards
from flext_core._utilities.validation import FlextUtilitiesValidation
from flext_core._utilities.validators import (
    ValidatorBuilder,
    ValidatorDSL,
    ValidatorSpec,
)
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextUtilities:
    """Stable utility surface for dispatcher-friendly helpers.

    Provides enterprise-grade utility functions for common operations
    throughout the FLEXT ecosystem. This is a PURE FACADE that delegates
    to _utilities package implementations.

    Architecture: Tier 1.5 (Foundation Utilities)
    ==============================================
    - No nested class definitions (single class per module principle)
    - All attributes reference _utilities classes directly
    - External code uses FlextUtilities.XxxClass.method() pattern
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
        from flext_core import FlextUtilities
        result = FlextUtilities.Enum.parse(MyEnum, "value")
    """

    # ═══════════════════════════════════════════════════════════════════
    # CLASS-LEVEL ATTRIBUTES: Module References (NOT nested classes)
    # ═══════════════════════════════════════════════════════════════════
    # Each attribute points directly to _utilities class for pure delegation

    Enum = FlextUtilitiesEnum
    Collection = FlextUtilitiesCollection
    Args = FlextUtilitiesArgs
    Model = FlextUtilitiesModel
    Cache = FlextUtilitiesCache
    Validation = FlextUtilitiesValidation
    Generators = FlextUtilitiesGenerators
    TextProcessor = FlextUtilitiesTextProcessor
    TypeGuards = FlextUtilitiesTypeGuards
    Reliability = FlextUtilitiesReliability
    TypeChecker = FlextUtilitiesTypeChecker
    Configuration = FlextUtilitiesConfiguration
    Context = FlextUtilitiesContext
    DataMapper = FlextUtilitiesDataMapper
    Domain = FlextUtilitiesDomain
    Pagination = FlextUtilitiesPagination
    StringParser = FlextUtilitiesStringParser

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
    ) -> FlextResult[T]:
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
            FlextResult[T]: Ok(value) if validation passes, Fail with error message.

        Examples:
            # Simple validation with V namespace
            result = FlextUtilities.validate(
                email,
                FlextUtilities.V.string.non_empty,
                FlextUtilities.V.string.email,
            )

            # With operators
            validator = V.string.non_empty & V.string.max_length(100)
            result = FlextUtilities.validate(value, validator)

            # Any mode (OR)
            result = FlextUtilities.validate(
                value,
                V.string.email,
                V.string.url,
                mode="any",
            )

            # With field name for error context
            result = FlextUtilities.validate(
                config["port"],
                V.number.positive,
                V.number.in_range(1, 65535),
                field_name="config.port",
            )

        """
        if not validators:
            # Business Rule: No validators means value is accepted as-is
            # Type narrowing: value is T when no validation required
            return FlextResult[T].ok(value)

        errors: list[str] = []
        field_prefix = f"{field_name}: " if field_name else ""

        if mode == "any":
            # OR mode: at least one must pass
            for validator in validators:
                if validator(value):
                    # Business Rule: After validator passes, value conforms to T
                    # Type narrowing: validator ensures value is T
                    return FlextResult[T].ok(value)
            # None passed
            descriptions = [getattr(v, "description", "validator") for v in validators]
            error_msg = f"{field_prefix}None of the validators passed: {', '.join(descriptions)}"
            return FlextResult[T].fail(error_msg)

        # Default: "all" mode (AND)
        for validator in validators:
            if not validator(value):
                description = getattr(validator, "description", "validator")
                error_msg = f"{field_prefix}Validation failed: {description}"
                if fail_fast and not collect_errors:
                    return FlextResult[T].fail(error_msg)
                errors.append(error_msg)

        if errors:
            return FlextResult[T].fail("; ".join(errors))

        # Business Rule: After all validators pass, value is guaranteed to be T
        # Validators ensure value conforms to T at runtime, so we can safely return it
        # Type narrowing: value parameter is already typed as T in function signature
        # Runtime guarantee: validators ensure value conforms to T's constraints
        # Note: Cast needed for pyrefly type checking - pyrefly cannot infer T from validator composition
        # but mypy correctly identifies this as redundant, so we use type: ignore[misc] for mypy only
        return FlextResult[T].ok(cast("T", value))  # type: ignore[misc]

    @staticmethod
    def _parse_with_default[T](
        default: T | None,
        default_factory: Callable[[], T] | None,
        error_msg: str,
    ) -> FlextResult[T]:
        """Return default or error for parse failures.

        Business Rule: Provides fallback mechanism for parse operations.
        Default values allow graceful degradation when parsing fails.
        """
        if default is not None:
            return FlextResult[T].ok(default)
        if default_factory is not None:
            return FlextResult[T].ok(default_factory())
        return FlextResult[T].fail(error_msg)

    @staticmethod
    def _parse_enum[T](
        value: str,
        target: type[T],
        *,
        case_insensitive: bool,
    ) -> FlextResult[T] | None:
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
            lower_val = value.lower()
            # Business Rule: StrEnum classes expose __members__ dict with all enum members
            # Iterate through enum members for case-insensitive matching
            # After validation, member is guaranteed to be T (the enum type)
            members_dict = getattr(enum_type, "__members__", {})
            for member in members_dict.values():
                if (
                    member.value.lower() == lower_val
                    or member.name.lower() == lower_val
                ):
                    # Business Rule: member is an instance of target (type[T]), so it's T
                    # Type narrowing: after issubclass check, member is StrEnum which is T
                    return FlextResult[T].ok(cast("T", member))
        result = FlextUtilitiesEnum.parse(target, value)
        if result.is_success:
            return FlextResult[T].ok(result.value)
        return FlextResult[T].fail(result.error or "Enum parse failed")

    @staticmethod
    def _parse_model[T](
        value: object,
        target: type[T],
        field_prefix: str,
        *,
        strict: bool,
    ) -> FlextResult[T] | None:
        """Parse Pydantic BaseModel. Returns None if not model.

        Business Rule: Pydantic model parsing supports strict and non-strict modes.
        Strict mode requires exact type matching, non-strict allows coercion.
        """
        if not (isinstance(target, type) and issubclass(target, BaseModel)):
            return None
        if not isinstance(value, Mapping):
            return FlextResult[T].fail(
                f"{field_prefix}Expected dict for model, got {type(value).__name__}"
            )
        result = FlextUtilitiesModel.from_dict(target, dict(value), strict=strict)
        if result.is_success:
            return FlextResult[T].ok(result.value)
        return FlextResult[T].fail(result.error or "Model parse failed")

    @staticmethod
    def _coerce_primitive[T](  # noqa: PLR0911
        value: object, target: type[T]
    ) -> FlextResult[T] | None:
        """Coerce primitive types. Returns None if no coercion applied.

        Business Rule: Primitive type coercion supports common conversions:
        str↔int, str↔float, str↔bool. Boolean coercion recognizes common
        string representations (true/false, yes/no, on/off, 1/0).
        """
        if target is int and isinstance(value, (str, float)):
            return FlextResult[T].ok(int(value))  # type: ignore[arg-type]
        if target is float and isinstance(value, (str, int)):
            return FlextResult[T].ok(float(value))  # type: ignore[arg-type]
        if target is str:
            return FlextResult[T].ok(str(value))  # type: ignore[arg-type]
        if target is bool:
            if isinstance(value, str):
                lower_val = value.lower()
                if lower_val in {"true", "1", "yes", "on"}:
                    return FlextResult[T].ok(True)  # type: ignore[arg-type]
                if lower_val in {"false", "0", "no", "off"}:
                    return FlextResult[T].ok(False)  # type: ignore[arg-type]
            return FlextResult[T].ok(bool(value))  # type: ignore[arg-type]
        return None

    @staticmethod
    def parse[T](  # noqa: C901, PLR0911, PLR0913
        value: object,
        target: type[T],
        *,
        strict: bool = False,
        coerce: bool = True,
        case_insensitive: bool = False,
        default: T | None = None,
        default_factory: Callable[[], T] | None = None,
        field_name: str | None = None,
    ) -> FlextResult[T]:
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
            FlextResult[T]: Ok(parsed_value) or Fail with error message.

        Examples:
            # Enum parsing
            result = FlextUtilities.parse("ACTIVE", Status)

            # Case-insensitive enum
            result = FlextUtilities.parse("active", Status, case_insensitive=True)

            # Pydantic model from dict
            result = FlextUtilities.parse({"name": "John"}, UserModel)

            # Primitive coercion
            result = FlextUtilities.parse("42", int)  # Ok(42)

            # With default
            result = FlextUtilities.parse("invalid", int, default=0)

        """
        field_prefix = f"{field_name}: " if field_name else ""

        # Handle None value
        if value is None:
            return FlextUtilities._parse_with_default(
                default, default_factory, f"{field_prefix}Cannot parse None value"
            )

        # Already the target type
        if isinstance(value, target):
            return FlextResult[T].ok(value)

        # StrEnum parsing
        enum_result = FlextUtilities._parse_enum(
            str(value), target, case_insensitive=case_insensitive
        )
        if enum_result is not None:
            if enum_result.is_success:
                return enum_result
            return FlextUtilities._parse_with_default(
                default, default_factory, f"{field_prefix}{enum_result.error}"
            )

        # Pydantic BaseModel parsing
        model_result = FlextUtilities._parse_model(
            value, target, field_prefix, strict=strict
        )
        if model_result is not None:
            if model_result.is_success:
                return model_result
            return FlextUtilities._parse_with_default(
                default, default_factory, model_result.error or ""
            )

        # Primitive coercion (str, int, float, bool)
        if coerce and not strict:
            try:
                prim_result = FlextUtilities._coerce_primitive(value, target)
                if prim_result is not None:
                    return prim_result
            except (ValueError, TypeError) as e:
                # Business Rule: target is a type, access __name__ via getattr for type safety
                target_name = getattr(target, "__name__", "type")
                return FlextUtilities._parse_with_default(
                    default,
                    default_factory,
                    f"{field_prefix}Cannot coerce {type(value).__name__} to "
                    f"{target_name}: {e}",
                )

        # Direct type call as last resort
        try:
            parsed = target(value)  # type: ignore[call-arg]
            return FlextResult[T].ok(parsed)
        except Exception as e:
            # Business Rule: target is a type, access __name__ via getattr for type safety
            target_name = getattr(target, "__name__", "type")
            return FlextUtilities._parse_with_default(
                default,
                default_factory,
                f"{field_prefix}Cannot parse {type(value).__name__} "
                f"to {target_name}: {e}",
            )

    @staticmethod
    def transform(  # noqa: PLR0913, C901
        data: Mapping[str, FlextTypes.GeneralValueType],
        *,
        normalize: bool = False,
        strip_none: bool = False,
        strip_empty: bool = False,
        map_keys: dict[str, str] | None = None,
        filter_keys: set[str] | None = None,
        exclude_keys: set[str] | None = None,
        to_json: bool = False,
        to_model: type[BaseModel] | None = None,
    ) -> FlextResult[dict[str, FlextTypes.GeneralValueType]]:
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
            FlextResult[dict[str, GeneralValueType]]: Transformed data or error.

        Examples:
            # Normalize and filter
            result = FlextUtilities.transform(
                data,
                normalize=True,
                filter_keys={"name", "email"},
            )

            # Map keys and strip None
            result = FlextUtilities.transform(
                data,
                map_keys={"old_name": "new_name"},
                strip_none=True,
            )

            # Convert to JSON-serializable
            result = FlextUtilities.transform(data, to_json=True)

            # Transform to Pydantic model
            result = FlextUtilities.transform(
                raw_data,
                to_model=UserModel,
                strip_none=True,
            )

        """
        try:
            result: dict[str, FlextTypes.GeneralValueType] = dict(data)

            # Normalize
            if normalize:
                normalized = FlextUtilitiesCache.normalize_component(result)
                if isinstance(normalized, dict):
                    result = normalized

            # Map keys
            if map_keys:
                mapped = FlextUtilitiesDataMapper.map_dict_keys(result, map_keys)
                result = mapped.value

            # Filter keys
            if filter_keys is not None:
                result = {k: v for k, v in result.items() if k in filter_keys}

            # Exclude keys
            if exclude_keys:
                result = {k: v for k, v in result.items() if k not in exclude_keys}

            # Strip None values
            if strip_none:
                result = {k: v for k, v in result.items() if v is not None}

            # Strip empty values
            if strip_empty:
                result = {
                    k: v for k, v in result.items() if v not in ("", [], {}, None)
                }

            # Convert to JSON
            if to_json:
                result = FlextUtilitiesDataMapper.convert_dict_to_json(result)

            # Parse to model
            if to_model is not None:
                model_result = FlextUtilitiesModel.from_dict(to_model, result)  # type: ignore[arg-type]
                if model_result.is_failure:
                    return FlextResult[dict[str, FlextTypes.GeneralValueType]].fail(
                        model_result.error or "Model conversion failed"
                    )
                # Return model as dict representation
                result = model_result.value.model_dump()

            return FlextResult[dict[str, FlextTypes.GeneralValueType]].ok(result)

        except Exception as e:
            return FlextResult[dict[str, FlextTypes.GeneralValueType]].fail(
                f"Transform failed: {e}"
            )

    @staticmethod
    def pipe(
        value: object,
        *operations: Callable[[object], object],
        on_error: str = "stop",
    ) -> FlextResult[object]:
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
            result = FlextUtilities.pipe(
                "  hello world  ",
                str.strip,
                str.upper,
                lambda s: s.replace(" ", "_"),
            )
            # → FlextResult.ok("HELLO_WORLD")

        """
        if not operations:
            return FlextResult[object].ok(value)

        current: object = value
        for i, op in enumerate(operations):
            try:
                result = op(current)

                # Unwrap FlextResult if returned
                if isinstance(result, FlextResult):
                    if result.is_failure:
                        if on_error == "stop":
                            return FlextResult[object].fail(
                                f"Pipeline step {i} failed: {result.error}"
                            )
                        # on_error == "skip": continue with previous value
                        continue
                    current = result.value
                else:
                    current = result

            except Exception as e:
                if on_error == "stop":
                    return FlextResult[object].fail(f"Pipeline step {i} failed: {e}")
                # on_error == "skip": continue with previous value

        return FlextResult[object].ok(current)

    @staticmethod
    def merge(  # noqa: C901
        *dicts: Mapping[str, FlextTypes.GeneralValueType],
        strategy: str = "deep",
        filter_none: bool = False,
        filter_empty: bool = False,
    ) -> FlextResult[dict[str, FlextTypes.GeneralValueType]]:
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
            result = FlextUtilities.merge(
                {"a": 1, "b": {"x": 1}},
                {"b": {"y": 2}, "c": 3},
                strategy="deep",
            )
            # → FlextResult.ok({"a": 1, "b": {"x": 1, "y": 2}, "c": 3})

        """
        if not dicts:
            return FlextResult[dict[str, FlextTypes.GeneralValueType]].ok({})

        def should_include(v: object) -> bool:
            if filter_none and v is None:
                return False
            return not (filter_empty and v in ("", [], {}))

        def deep_merge(
            base: dict[str, object], overlay: dict[str, object]
        ) -> dict[str, object]:
            result = dict(base)
            for key, value in overlay.items():
                if not should_include(value):
                    continue
                if (
                    key in result
                    and isinstance(result[key], dict)
                    and isinstance(value, dict)
                ):
                    # After isinstance checks, we know these are dicts
                    base_dict: dict[str, object] = dict(
                        cast("dict[str, object]", result[key])
                    )
                    overlay_dict: dict[str, object] = dict(
                        cast("dict[str, object]", value)
                    )
                    result[key] = deep_merge(base_dict, overlay_dict)
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

        try:
            merged: dict[str, object] = {}
            for d in dicts:
                filtered = {k: v for k, v in d.items() if should_include(v)}
                if strategy == "override":
                    merged.update(filtered)
                else:  # deep or append
                    merged = deep_merge(merged, filtered)  # type: ignore[arg-type]

            return FlextResult[dict[str, FlextTypes.GeneralValueType]].ok(
                merged  # type: ignore[arg-type]
            )

        except Exception as e:
            return FlextResult[dict[str, FlextTypes.GeneralValueType]].fail(
                f"Merge failed: {e}"
            )

    @staticmethod
    def extract[T](  # noqa: C901, PLR0912, PLR0911
        data: Mapping[str, object] | object,
        path: str,
        *,
        default: T | None = None,
        required: bool = False,
        separator: str = ".",
    ) -> FlextResult[T | None]:
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
            result = FlextUtilities.extract(config, "database.port")
            # → FlextResult.ok(5432)

        """
        try:
            parts = path.split(separator)
            current: object = data

            for i, part in enumerate(parts):
                if current is None:
                    if required:
                        return FlextResult[T | None].fail(
                            f"Path '{separator.join(parts[:i])}' is None"
                        )
                    return FlextResult[T | None].ok(default)

                # Handle array indexing (e.g., "items[0]")
                array_match = None
                key_part = part
                if "[" in part and part.endswith("]"):
                    bracket_pos = part.index("[")
                    array_match = part[bracket_pos + 1 : -1]
                    key_part = part[:bracket_pos]

                # Get value from dict, object, or Pydantic model
                if isinstance(current, Mapping):
                    if key_part not in current:
                        if required:
                            return FlextResult[T | None].fail(
                                f"Key '{key_part}' not found at '{separator.join(parts[:i])}'"
                            )
                        return FlextResult[T | None].ok(default)
                    current = current[key_part]
                elif hasattr(current, key_part):
                    current = getattr(current, key_part)
                elif hasattr(current, "model_dump"):  # Pydantic model
                    # After hasattr check, pyright still doesn't narrow the type
                    # Use cast to indicate model_dump is available
                    model_dump_method = cast(
                        "Callable[[], dict[str, object]]",
                        current.model_dump,
                    )
                    model_dict = model_dump_method()
                    if key_part not in model_dict:
                        if required:
                            return FlextResult[T | None].fail(
                                f"Key '{key_part}' not found at '{separator.join(parts[:i])}'"
                            )
                        return FlextResult[T | None].ok(default)
                    current = model_dict[key_part]
                else:
                    if required:
                        return FlextResult[T | None].fail(
                            f"Cannot access '{key_part}' at '{separator.join(parts[:i])}'"
                        )
                    return FlextResult[T | None].ok(default)

                # Handle array index
                if array_match is not None:
                    if not isinstance(current, (list, tuple)):
                        if required:
                            return FlextResult[T | None].fail(
                                f"'{key_part}' is not a sequence"
                            )
                        return FlextResult[T | None].ok(default)
                    try:
                        idx = int(array_match)
                        current = current[idx]
                    except (ValueError, IndexError):
                        if required:
                            return FlextResult[T | None].fail(
                                f"Invalid index '{array_match}' for '{key_part}'"
                            )
                        return FlextResult[T | None].ok(default)

            return FlextResult[T | None].ok(current)  # type: ignore[arg-type]

        except Exception as e:
            return FlextResult[T | None].fail(f"Extract failed: {e}")

    @staticmethod
    def generate(
        kind: str = "id",
        *,
        prefix: str | None = None,
        length: int | None = None,
        include_timestamp: bool = False,
        separator: str = "_",
    ) -> str:
        """Unified ID generation with domain-driven prefixes.

        Business Rule: Generates IDs with domain-specific prefixes for
        traceability. Supports UUID v4 for globally unique IDs and short
        IDs for compact representations. Timestamp inclusion supports
        chronological ordering.

        Args:
            kind: Type of ID to generate:
                - "id": Generic short ID
                - "uuid": Full UUID v4
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

        Returns:
            Generated ID string

        Example:
            id = FlextUtilities.generate("entity", prefix="user")
            # → "user_a1b2c3d4"

        """
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
        else:
            base_id = FlextUtilitiesGenerators.generate_short_id(length or 8)

        # Build parts
        parts: list[str] = []
        if actual_prefix:
            parts.append(actual_prefix)
        if include_timestamp:
            ts = FlextUtilitiesGenerators.generate_iso_timestamp()
            parts.append(ts[:10].replace("-", ""))  # YYYYMMDD
        parts.append(base_id)

        return separator.join(parts)

    @staticmethod
    def batch[T, R](
        items: list[T],
        operation: Callable[[T], R],
        *,
        _size: int = 100,  # Reserved for future chunking support
        on_error: str = "collect",
        _parallel: bool = False,  # Reserved for future async support
        progress: Callable[[int, int], None] | None = None,  # Progress callback (current, total)
        progress_interval: int = 1,  # Call progress callback every N items
        pre_validate: Callable[[T], bool] | None = None,  # Pre-validation filter
        post_validate: Callable[[R], bool] | None = None,  # Post-validation filter
        flatten: bool = False,  # Flatten nested lists in results
    ) -> FlextResult[FlextTypes.Types.BatchResultDict]:
        """Batch processing with chunking and error handling.

        Business Rule: Processes items in batch with configurable error handling.
        Supports three error modes: "skip" (continue), "fail" (stop on first error),
        "collect" (continue and collect errors). Results are returned in a structured
        TypedDict with results, errors, and counts for audit trail completeness.

        Audit Implication: Batch operations are tracked with complete result metadata.
        Error tracking includes item index and error message for audit purposes.
        All batch operations return structured results suitable for logging and auditing.

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
            FlextResult[FlextTypes.Types.BatchResultDict]: Dict with keys:
                - results: list[GeneralValueType] - Successful operation results
                - errors: list[tuple[int, str]] - (index, error_message) for failures
                - total: int - Total items processed
                - success_count: int - Number of successful operations
                - error_count: int - Number of failed operations

        Example:
            result = FlextUtilities.batch(
                [1, 2, 3],
                lambda x: x * 2,
                on_error="collect",
            )
            if result.is_success:
                batch_data = result.value
                assert batch_data["results"] == [2, 4, 6]
                assert batch_data["total"] == 3

        """
        results: list[R] = []
        errors: list[tuple[int, str]] = []
        total_items = len(items)

        # Call progress callback at start if provided
        if progress is not None:
            progress(0, total_items)

        for i, item in enumerate(items):
            # Pre-validation: filter items before processing
            if pre_validate is not None:
                if not pre_validate(item):
                    # Skip this item (don't process, don't count as error)
                    continue

            try:
                result = operation(item)
                
                # Handle FlextResult return type
                if isinstance(result, FlextResult):
                    if result.is_failure:
                        if on_error == "fail":
                            return FlextResult[FlextTypes.Types.BatchResultDict].fail(
                                f"Item {i} failed: {result.error}"
                            )
                        if on_error == "collect":
                            errors.append((i, result.error or "Unknown error"))
                        # skip: just continue
                    else:
                        result_value = result.value
                        # Post-validation: filter results after processing
                        if post_validate is None or post_validate(result_value):
                            results.append(result_value)
                else:
                    # Direct value return
                    # Post-validation: filter results after processing
                    if post_validate is None or post_validate(result):
                        results.append(result)
            except Exception as e:
                if on_error == "fail":
                    return FlextResult[FlextTypes.Types.BatchResultDict].fail(
                        f"Item {i} failed: {e}"
                    )
                if on_error == "collect":
                    errors.append((i, str(e)))
                # skip: just continue

            # Call progress callback at interval
            if progress is not None and (i + 1) % progress_interval == 0:
                progress(i + 1, total_items)

        # Flatten nested lists if requested
        if flatten:
            flattened_results: list[FlextTypes.GeneralValueType] = []
            for r in results:
                # Check if result is a list-like structure to flatten
                if isinstance(r, (list, tuple)):
                    flattened_results.extend(
                        cast("FlextTypes.GeneralValueType", item) for item in r
                    )
                else:
                    flattened_results.append(cast("FlextTypes.GeneralValueType", r))
            results_for_dict = flattened_results
        else:
            # Business Rule: Convert list[R] to list[GeneralValueType] for TypedDict compatibility
            # Type narrowing: results contains R values which are GeneralValueType
            results_for_dict = [
                cast("FlextTypes.GeneralValueType", r) for r in results
            ]

        # Call progress callback at end if provided
        if progress is not None:
            progress(total_items, total_items)

        batch_result: FlextTypes.Types.BatchResultDict = {
            "results": results_for_dict,
            "errors": errors,
            "total": total_items,
            "success_count": len(results_for_dict),
            "error_count": len(errors),
        }

        return FlextResult[FlextTypes.Types.BatchResultDict].ok(batch_result)

    @staticmethod
    def retry[T](  # noqa: PLR0913
        operation: Callable[[], T],
        *,
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff: str = "exponential",
        backoff_factor: float = 2.0,
        retry_on: tuple[type[Exception], ...] = (Exception,),
    ) -> FlextResult[T]:
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
            result = FlextUtilities.retry(
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
                return FlextResult[T].ok(result)
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

        return FlextResult[T].fail(
            f"Operation failed after {max_attempts} attempts: {last_error}"
        )


__all__ = ["FlextUtilities"]
