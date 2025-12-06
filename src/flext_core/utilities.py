"""FlextUtilities - Pure Facade for FLEXT Utility Classes.

This module provides a unified entry point to all FLEXT utility functionality.
All methods are delegated to specialized classes in _utilities/ submodules.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import overload

from flext_core._utilities.args import FlextUtilitiesArgs
from flext_core._utilities.cache import FlextUtilitiesCache
from flext_core._utilities.checker import FlextUtilitiesChecker
from flext_core._utilities.collection import FlextUtilitiesCollection
from flext_core._utilities.configuration import FlextUtilitiesConfiguration
from flext_core._utilities.context import FlextUtilitiesContext
from flext_core._utilities.domain import FlextUtilitiesDomain
from flext_core._utilities.enum import FlextUtilitiesEnum
from flext_core._utilities.generators import FlextUtilitiesGenerators
from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core._utilities.mapper import FlextUtilitiesMapper
from flext_core._utilities.model import FlextUtilitiesModel
from flext_core._utilities.pagination import FlextUtilitiesPagination
from flext_core._utilities.parser import FlextUtilitiesParser
from flext_core._utilities.reliability import FlextUtilitiesReliability
from flext_core._utilities.text import FlextUtilitiesText
from flext_core._utilities.validation import FlextUtilitiesValidation
from flext_core._utilities.validators import ValidatorDSL, ValidatorSpec
from flext_core.constants import c
from flext_core.result import r
from flext_core.typings import t


class FlextUtilities:
    """Unified facade for all FLEXT utility functionality.

    This class provides access to specialized utility classes through
    class attributes. All functionality is delegated to the appropriate
    specialized class.

    Usage:
        from flext_core import FlextUtilities

        # Access via facade attributes
        result = FlextUtilities.Parser.parse(value, int)
        result = FlextUtilities.Collection.map(items, fn)
        result = FlextUtilities.Mapper.get(data, "key")

        # Or use short alias
        from flext_core.utilities import u
        result = u.Parser.parse(value, int)
    """

    # === FACADE ATTRIBUTES - Real inheritance classes ===
    class Args(FlextUtilitiesArgs):
        """Args utility class - real inheritance."""

    class Cache(FlextUtilitiesCache):
        """Cache utility class - real inheritance."""

    class Checker(FlextUtilitiesChecker):
        """Checker utility class - real inheritance."""

    class Collection(FlextUtilitiesCollection):
        """Collection utility class - real inheritance."""

    class Configuration(FlextUtilitiesConfiguration):
        """Configuration utility class - real inheritance."""

    class Context(FlextUtilitiesContext):
        """Context utility class - real inheritance."""

    class Domain(FlextUtilitiesDomain):
        """Domain utility class - real inheritance."""

    class Enum(FlextUtilitiesEnum):
        """Enum utility class - real inheritance."""

    class Generators(FlextUtilitiesGenerators):
        """Generators utility class - real inheritance."""

    class Guards(FlextUtilitiesGuards):
        """Guards utility class - real inheritance."""

    class Mapper(FlextUtilitiesMapper):
        """Mapper utility class - real inheritance."""

    class Model(FlextUtilitiesModel):
        """Model utility class - real inheritance."""

    class Pagination(FlextUtilitiesPagination):
        """Pagination utility class - real inheritance."""

    class Parser(FlextUtilitiesParser):
        """Parser utility class - real inheritance."""

    class Reliability(FlextUtilitiesReliability):
        """Reliability utility class - real inheritance."""

    class Text(FlextUtilitiesText):
        """Text utility class - real inheritance."""

    class Validation(FlextUtilitiesValidation):
        """Validation utility class - real inheritance."""

    class Validators(ValidatorDSL):
        """Validators utility class - real inheritance."""

    class V(ValidatorDSL):
        """V utility class - alias for Validators - real inheritance."""

    # === UTILITY METHODS ===
    # These are convenience methods that delegate to specialized classes.
    # Access specialized classes directly: FlextUtilities.Parser.parse(), etc.

    @staticmethod
    def is_type(value: object, type_spec: str | type | tuple[type, ...]) -> bool:
        """Generic type checking function that unifies all guard checks.

        Provides a single entry point for all type checking operations,
        supporting string-based type names, direct type/class checks, and
        protocol checks. This function delegates to FlextUtilitiesGuards.is_type().

        Args:
            value: Object to check
            type_spec: Type specification as:
                - String name: "config", "str", "dict", "list", "sequence",
                  "mapping", "callable", "sized", "list_or_tuple", "sequence_not_str",
                  "string_non_empty", "dict_non_empty", "list_non_empty"
                - Type/class: str, dict, list, tuple, Sequence, Mapping, etc.
                - Tuple of types: (int, float), (str, bytes), etc.
                - Protocol: p.Configuration.Config, p.Context.Ctx, etc.

        Returns:
            bool: True if value matches the type specification

        Examples:
            >>> # String-based checks
            >>> u.is_type(obj, "config")
            >>> u.is_type(obj, "str")
            >>> u.is_type(obj, "dict")
            >>> u.is_type(obj, "string_non_empty")

            >>> # Direct type checks
            >>> u.is_type(obj, str)
            >>> u.is_type(obj, dict)
            >>> u.is_type(obj, list)

            >>> # Tuple of types checks
            >>> u.is_type(obj, (int, float))
            >>> u.is_type(obj, (str, bytes))

            >>> # Protocol checks
            >>> u.is_type(obj, p.Configuration.Config)
            >>> u.is_type(obj, p.Context.Ctx)

        """
        return FlextUtilitiesGuards.is_type(value, type_spec)

    @staticmethod
    def merge(
        base: t.Types.ConfigurationDict,
        other: t.Types.ConfigurationDict,
        *,
        strategy: str = "deep",
    ) -> r[t.Types.ConfigurationDict]:
        """Merge two dictionaries - delegates to FlextUtilitiesCollection."""
        return FlextUtilitiesCollection.merge(base, other, strategy=strategy)

    @staticmethod
    def transform(
        source: t.Types.ConfigurationDict | t.Types.ConfigurationMapping,
        *,
        normalize: bool = False,
        strip_none: bool = False,
        strip_empty: bool = False,
        map_keys: t.Types.StringDict | None = None,
        filter_keys: set[str] | None = None,
        exclude_keys: set[str] | None = None,
        to_json: bool = False,
    ) -> r[t.Types.ConfigurationDict]:
        """Transform dictionary with multiple options - delegates to FlextUtilitiesMapper.

        Args:
            source: Source dictionary to transform.
            normalize: Normalize values using cache normalization.
            strip_none: Remove keys with None values.
            strip_empty: Remove keys with empty values (empty strings, lists, dicts).
            map_keys: Dictionary mapping old keys to new keys.
            filter_keys: Set of keys to keep (all others removed).
            exclude_keys: Set of keys to remove.
            to_json: Convert to JSON-compatible values.

        Returns:
            FlextResult with transformed dictionary.

        """
        return FlextUtilitiesMapper.transform(
            source,
            normalize=normalize,
            strip_none=strip_none,
            strip_empty=strip_empty,
            map_keys=map_keys,
            filter_keys=filter_keys,
            exclude_keys=exclude_keys,
            to_json=to_json,
        )

    # Result helpers - FlextResult DSL methods
    @staticmethod
    def val[T](result: r[T], default: T) -> T:
        """Extract value from FlextResult with default fallback."""
        return result.unwrap() if result.is_success else default

    @staticmethod
    def result_val[T](result: r[T], default: T) -> T:
        """Extract value from FlextResult with default fallback (alias for val)."""
        return FlextUtilities.val(result, default)

    @staticmethod
    def vals[T](results: Sequence[r[T]]) -> list[T]:
        """Extract values from collection of results, skipping failures."""
        return [r.unwrap() for r in results if r.is_success]

    @staticmethod
    def err[T](result: r[T], default: str = "") -> str:
        """Get error message from FlextResult."""
        if result.is_failure and result.error is not None:
            return result.error
        return default

    @staticmethod
    def generate(
        kind: str | None = None,
        *,
        prefix: str | None = None,
        length: int | None = None,
        include_timestamp: bool = False,
        separator: str = "_",
    ) -> str:
        """Generate ID by kind or custom prefix - delegates to FlextUtilitiesGenerators.

        Args:
            kind: ID kind ("uuid", "correlation", "entity", "batch", "transaction",
                "event", "command", "query", "ulid", "id"). If None, generates UUID.
            prefix: Custom prefix (overrides kind prefix if provided).
            length: Custom length for generated ID (only for ulid/short IDs).
            include_timestamp: Include timestamp in ID (not currently implemented).
            separator: Separator between prefix and ID (default: "_").

        Returns:
            Generated ID string.

        Examples:
            >>> u.generate()  # UUID (36 chars)
            >>> u.generate("uuid")  # UUID (36 chars)
            >>> u.generate("correlation")  # corr_...
            >>> u.generate("entity", prefix="user")  # user_...
            >>> u.generate("ulid", length=16)  # Short ID with 16 chars

        """
        actual_prefix = FlextUtilities._determine_prefix(kind, prefix)

        # Handle UUID/ULID/id special cases
        if FlextUtilities._should_generate_uuid(kind, actual_prefix):
            return FlextUtilitiesGenerators.generate_id()

        if kind == "ulid":
            ulid_length = length if length is not None else 8
            return FlextUtilitiesGenerators.Random.generate_short_id(ulid_length)

        if kind == "id" and actual_prefix is None:
            return FlextUtilitiesGenerators.generate_id()

        # Generate prefixed ID
        if actual_prefix is not None:
            return FlextUtilities._generate_prefixed_id(
                actual_prefix,
                length,
                include_timestamp=include_timestamp,
                separator=separator,
            )

        # Fallback: UUID
        return FlextUtilitiesGenerators.generate_id()

    @staticmethod
    def _determine_prefix(kind: str | None, prefix: str | None) -> str | None:
        """Determine actual prefix from kind or custom prefix.

        Args:
            kind: ID kind string.
            prefix: Custom prefix (overrides kind).

        Returns:
            Actual prefix string or None.

        """
        if prefix is not None:
            return prefix

        if kind is None:
            return None

        kind_prefix_map: t.Types.StringDict = {
            "correlation": "corr",
            "entity": "ent",
            "batch": c.Cqrs.ProcessingMode.BATCH,
            "transaction": "txn",
            "event": "evt",
            "command": "cmd",
            "query": "qry",
        }
        return kind_prefix_map.get(kind)

    @staticmethod
    def _should_generate_uuid(kind: str | None, actual_prefix: str | None) -> bool:
        """Check if UUID generation should be used.

        Args:
            kind: ID kind string.
            actual_prefix: Determined prefix or None.

        Returns:
            True if UUID should be generated.

        """
        return kind == "uuid" or (kind is None and actual_prefix is None)

    @staticmethod
    def _generate_prefixed_id(
        prefix: str,
        length: int | None,
        *,
        include_timestamp: bool,
        separator: str = "_",
    ) -> str:
        """Generate prefixed ID with optional timestamp and custom separator/length.

        Args:
            prefix: Prefix string.
            length: Custom length for ID part.
            include_timestamp: Whether to include timestamp.
            separator: Separator between prefix and ID.

        Returns:
            Generated prefixed ID string.

        """
        parts: list[t.GeneralValueType] = []

        if include_timestamp:
            timestamp = int(datetime.now(UTC).timestamp())
            parts.append(timestamp)

        # Custom separator or length requires manual construction
        if separator != "_" or length is not None:
            uuid_length = length if length is not None else 8
            uuid_part = str(FlextUtilitiesGenerators.generate_id())[:uuid_length]
            if parts:
                middle = str(separator).join(str(p) for p in parts)
                return f"{prefix}{separator}{middle}{separator}{uuid_part}"
            return f"{prefix}{separator}{uuid_part}"

        # Standard prefixed ID generation
        if parts:
            if length is not None:
                return FlextUtilitiesGenerators.generate_prefixed_id(
                    prefix,
                    *parts,
                    length=length,
                )
            return FlextUtilitiesGenerators.generate_prefixed_id(prefix, *parts)

        if length is not None:
            return FlextUtilitiesGenerators.generate_prefixed_id(prefix, length=length)

        return FlextUtilitiesGenerators.generate_prefixed_id(prefix)

    @staticmethod
    def extract[T](
        data: t.Types.ConfigurationMapping | object,
        path: str,
        *,
        default: T | None = None,
        required: bool = False,
        separator: str = ".",
    ) -> r[T | None]:
        """Extract nested value from data structure - delegates to FlextUtilitiesMapper.

        Args:
            data: Source data (dict, object with attributes, or Pydantic model).
            path: Dot-separated path (e.g., "user.profile.name").
            default: Default value if path not found.
            required: Fail if path not found.
            separator: Path separator (default: ".").

        Returns:
            FlextResult containing extracted value or default.

        Example:
            >>> result = u.extract({"user": {"name": "John"}}, "user.name")
            >>> value = result.unwrap()  # "John"

        """
        return FlextUtilitiesMapper.extract(
            data,
            path,
            default=default,
            required=required,
            separator=separator,
        )

    @staticmethod
    def has(obj: object, key: str) -> bool:
        """Check if object has attribute/key."""
        if isinstance(obj, dict):
            return key in obj
        return hasattr(obj, key)

    # =========================================================================
    # UNIVERSAL CHECK METHODS - Highly generalized and parametrizable
    # =========================================================================

    @staticmethod
    def chk(
        value: object,
        *,
        eq: object | None = None,
        ne: object | None = None,
        gt: float | None = None,
        gte: float | None = None,
        lt: float | None = None,
        lte: float | None = None,
        is_: type[object] | None = None,
        not_: type[object] | None = None,
        in_: Sequence[object] | None = None,
        not_in: Sequence[object] | None = None,
        none: bool | None = None,
        empty: bool | None = None,
        match: str | None = None,
        contains: str | object | None = None,
        starts: str | None = None,
        ends: str | None = None,
    ) -> bool:
        """Universal check - single method for ALL validation scenarios.

        Args:
            value: Value to check
            eq: Check value == eq
            ne: Check value != ne
            gt/gte/lt/lte: Numeric comparisons (works with len for sequences)
            is_: Check isinstance(value, is_)
            not_: Check not isinstance(value, not_)
            in_: Check value in in_
            not_in: Check value not in not_in
            none: Check value is None (True) or is not None (False)
            empty: Check if empty (True) or not empty (False)
            match: Check regex pattern match (strings)
            contains: Check if value contains item
            starts/ends: Check string prefix/suffix

        Returns:
            True if ALL conditions pass, False otherwise.

        Examples:
            u.chk(x, gt=0, lt=100)             # 0 < x < 100
            u.chk(s, empty=False, match="[0-9]+")  # non-empty and has digits
            u.chk(lst, gte=1, lte=10)          # 1 <= len(lst) <= 10
            u.chk(v, is_=str, none=False)      # is string and not None

        """
        # None checks
        if none is True and value is not None:
            return False
        if none is False and value is None:
            return False

        # Type checks
        if is_ is not None and not isinstance(value, is_):
            return False
        if not_ is not None and isinstance(value, not_):
            return False

        # Equality checks
        if eq is not None and value != eq:
            return False
        if ne is not None and value == ne:
            return False

        # Membership checks
        if in_ is not None and value not in in_:
            return False
        if not_in is not None and value in not_in:
            return False

        # Length/numeric checks - use len() for sequences, direct for numbers
        check_val: int | float
        if isinstance(value, (int, float)):
            check_val = value
        elif hasattr(value, "__len__"):
            check_val = len(value)
        else:
            check_val = 0

        if gt is not None and check_val <= gt:
            return False
        if gte is not None and check_val < gte:
            return False
        if lt is not None and check_val >= lt:
            return False
        if lte is not None and check_val > lte:
            return False

        # Empty checks (after len is computed)
        if empty is True and check_val != 0:
            return False
        if empty is False and check_val == 0:
            return False

        # String-specific checks
        if isinstance(value, str):
            if match is not None and not re.search(match, value):
                return False
            if starts is not None and not value.startswith(starts):
                return False
            if ends is not None and not value.endswith(ends):
                return False
            if (
                contains is not None
                and isinstance(contains, str)
                and contains not in value
            ):
                return False
        elif contains is not None:
            # Generic containment for sequences/dicts
            if isinstance(value, dict):
                if contains not in value:
                    return False
            elif hasattr(value, "__contains__") and contains not in value:
                return False

        return True

    @staticmethod
    def normalize(text: str, pattern: str = r"\s+", replacement: str = " ") -> str:
        """Normalize whitespace - delegates to FlextUtilitiesParser."""
        parser = FlextUtilitiesParser()
        result = parser.normalize_whitespace(
            text,
            pattern=pattern,
            replacement=replacement,
        )
        normalized = result.unwrap() if result.is_success else text
        return normalized if FlextUtilities.is_type(normalized, str) else text

    # Power methods - convenience delegates
    @staticmethod
    def pipe(
        value: object,
        *operations: Callable[[object], object],
        on_error: str = "stop",
    ) -> r[object]:
        """Functional pipeline - delegates to FlextUtilitiesReliability.pipe."""
        return FlextUtilitiesReliability.pipe(value, *operations, on_error=on_error)

    @staticmethod
    def batch[T, R](
        items: list[T],
        operation: Callable[[T], R | r[R]],
        *,
        _size: int = c.DEFAULT_BATCH_SIZE,
        on_error: str = "collect",
        _parallel: bool = False,
        progress: Callable[[int, int], None] | None = None,
        _progress_interval: int = 1,
        pre_validate: Callable[[T], bool] | None = None,
        flatten: bool = False,
    ) -> r[t.Types.BatchResultDict]:
        """Process items in batches - delegates to FlextUtilitiesCollection.batch."""
        return FlextUtilitiesCollection.batch(
            items,
            operation,
            _size=_size,
            on_error=on_error,
            _parallel=_parallel,
            progress=progress,
            _progress_interval=_progress_interval,
            pre_validate=pre_validate,
            flatten=flatten,
        )

    @staticmethod
    def retry[TResult](
        operation: Callable[[], r[TResult] | TResult],
        max_attempts: int | None = None,
        delay: float | None = None,
        delay_seconds: float | None = None,
        backoff_multiplier: float | None = None,
        retry_on: tuple[type[Exception], ...] | None = None,
    ) -> r[TResult]:
        """Execute operation with retry logic - delegates to FlextUtilitiesReliability.retry."""
        return FlextUtilitiesReliability.retry(
            operation,
            max_attempts=max_attempts,
            delay=delay,
            delay_seconds=delay_seconds,
            backoff_multiplier=backoff_multiplier,
            retry_on=retry_on,
        )

    @staticmethod
    @overload
    def get(
        data: t.Types.ConfigurationMapping | object,
        key: str,
        *,
        default: str = "",
    ) -> str: ...

    @staticmethod
    @overload
    def get[T](
        data: t.Types.ConfigurationMapping | object,
        key: str,
        *,
        default: list[T],
    ) -> list[T]: ...

    @staticmethod
    @overload
    def get[T](
        data: t.Types.ConfigurationMapping | object,
        key: str,
        *,
        default: T | None = None,
    ) -> T | None: ...

    @staticmethod
    def get[T](
        data: t.Types.ConfigurationMapping | object,
        key: str,
        *,
        default: T | None = None,
    ) -> T | None:
        """Get value from dict/object - delegates to FlextUtilitiesMapper.get."""
        return FlextUtilitiesMapper.get(data, key, default=default)

    @staticmethod
    def generate_id() -> str:
        """Generate unique ID - delegates to FlextUtilitiesGenerators.generate_id."""
        return FlextUtilitiesGenerators.generate_id()

    @staticmethod
    def generate_short_id(length: int = 8) -> str:
        """Generate short random ID - delegates to FlextUtilitiesGenerators.generate_short_id."""
        return FlextUtilitiesGenerators.generate_short_id(length)


u = FlextUtilities  # Runtime alias (not TypeAlias to avoid PYI042)

__all__ = [
    "FlextUtilities",
    "ValidatorSpec",  # Export for flext-ldif and other projects
    "u",
]
