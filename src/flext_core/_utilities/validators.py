"""Validator DSL with operator composition for declarative validation.

Provides ValidatorSpec protocol, Validator base class with operator overloads,
ValidatorDSL (V namespace), and ValidatorBuilder for fluent API composition.

This module is part of u power methods infrastructure.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self, cast

from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core.constants import c
from flext_core.protocols import p

# ============================================================================
# VALIDATOR SPEC PROTOCOL - Core interface for all validators
# ============================================================================
# ValidatorSpec is now defined in flext_core.protocols (p.ValidatorSpec)
# Re-export for backward compatibility
ValidatorSpec = p.ValidatorSpec


# ============================================================================
# VALIDATOR BASE CLASS - Concrete implementation with operators
# ============================================================================


@dataclass(frozen=True)
class Validator:
    """Base validator with operator overloads for composition.

    Wraps a predicate function and provides __and__, __or__, __invert__ operators
    for declarative composition of validation rules.

    Example:
        non_empty = Validator(lambda s: bool(s and str(s).strip()))
        max_len = Validator(lambda s: len(str(s)) <= c.Validation.MAX_RETRY_STATUS_CODES)
        combined = non_empty & max_len  # Both must pass

    """

    predicate: Callable[[object], bool]
    description: str = ""

    def __call__(self, value: object) -> bool:
        """Execute validation predicate."""
        return self.predicate(value)

    def __and__(self, other: ValidatorSpec | Validator) -> Validator:
        """Compose with AND - both validators must pass."""
        return Validator(
            predicate=lambda v: self(v) and other(v),
            description=f"({self.description} AND {getattr(other, 'description', 'validator')})",
        )

    def __or__(self, other: ValidatorSpec | Validator) -> Validator:
        """Compose with OR - at least one validator must pass."""
        return Validator(
            predicate=lambda v: self(v) or other(v),
            description=f"({self.description} OR {getattr(other, 'description', 'validator')})",
        )

    def __invert__(self) -> Validator:
        """Negate validator - passes when original fails."""
        return Validator(
            predicate=lambda v: not self(v),
            description=f"NOT {self.description}",
        )


# ============================================================================
# STRING VALIDATORS - V.string namespace
# ============================================================================


class StringValidators:
    """String validation utilities for V.string namespace.

    Provides composable validators for string validation including
    non_empty, length constraints, pattern matching, and format checks.

    Example:
        V.string.non_empty & V.string.max_length(100)

    """

    # Class-level validators (no args needed)
    non_empty: Validator = Validator(
        predicate=FlextUtilitiesGuards.is_string_non_empty,
        description="string.non_empty",
    )

    @staticmethod
    def min_length(n: int) -> Validator:
        """Validate string has at least n characters."""
        return Validator(
            predicate=lambda v: isinstance(v, str) and len(v) >= n,
            description=f"string.min_length({n})",
        )

    @staticmethod
    def max_length(n: int) -> Validator:
        """Validate string has at most n characters."""
        return Validator(
            predicate=lambda v: isinstance(v, str) and len(v) <= n,
            description=f"string.max_length({n})",
        )

    @staticmethod
    def length(min_len: int, max_len: int) -> Validator:
        """Validate string length is within range [min_len, max_len]."""
        return Validator(
            predicate=lambda v: isinstance(v, str) and min_len <= len(v) <= max_len,
            description=f"string.length({min_len}, {max_len})",
        )

    @staticmethod
    def matches(pattern: str) -> Validator:
        """Validate string matches regex pattern."""
        compiled = re.compile(pattern)
        return Validator(
            predicate=lambda v: isinstance(v, str) and bool(compiled.match(v)),
            description=f"string.matches({pattern!r})",
        )

    @staticmethod
    def contains(substring: str) -> Validator:
        """Validate string contains substring."""
        return Validator(
            predicate=lambda v: isinstance(v, str) and substring in v,
            description=f"string.contains({substring!r})",
        )

    @staticmethod
    def starts_with(prefix: str) -> Validator:
        """Validate string starts with prefix."""
        return Validator(
            predicate=lambda v: isinstance(v, str) and v.startswith(prefix),
            description=f"string.starts_with({prefix!r})",
        )

    @staticmethod
    def ends_with(suffix: str) -> Validator:
        """Validate string ends with suffix."""
        return Validator(
            predicate=lambda v: isinstance(v, str) and v.endswith(suffix),
            description=f"string.ends_with({suffix!r})",
        )

    # Common format validators
    email: Validator = Validator(
        predicate=lambda v: isinstance(v, str)
        and bool(re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", v)),
        description="string.email",
    )

    url: Validator = Validator(
        predicate=lambda v: isinstance(v, str)
        and bool(re.match(r"^https?://[^\s/$.?#].[^\s]*$", v, re.IGNORECASE)),
        description="string.url",
    )

    alphanumeric: Validator = Validator(
        predicate=lambda v: isinstance(v, str) and v.isalnum(),
        description="string.alphanumeric",
    )

    numeric: Validator = Validator(
        predicate=lambda v: isinstance(v, str) and v.isdigit(),
        description="string.numeric",
    )

    alpha: Validator = Validator(
        predicate=lambda v: isinstance(v, str) and v.isalpha(),
        description="string.alpha",
    )


# ============================================================================
# NUMBER VALIDATORS - V.number namespace
# ============================================================================


class NumberValidators:
    """Number validation utilities for V.number namespace.

    Provides composable validators for numeric validation including
    sign checks, range constraints, and type checks.

    Example:
        V.number.positive & V.number.less_than(100)

    """

    positive: Validator = Validator(
        predicate=lambda v: isinstance(v, (int, float)) and v > 0,
        description="number.positive",
    )

    negative: Validator = Validator(
        predicate=lambda v: isinstance(v, (int, float)) and v < 0,
        description="number.negative",
    )

    zero: Validator = Validator(
        predicate=lambda v: isinstance(v, (int, float)) and v == 0,
        description="number.zero",
    )

    non_negative: Validator = Validator(
        predicate=lambda v: isinstance(v, (int, float)) and v >= 0,
        description="number.non_negative",
    )

    non_positive: Validator = Validator(
        predicate=lambda v: isinstance(v, (int, float)) and v <= 0,
        description="number.non_positive",
    )

    integer: Validator = Validator(
        predicate=lambda v: isinstance(v, int)
        or (isinstance(v, float) and v.is_integer()),
        description="number.integer",
    )

    @staticmethod
    def greater_than(n: float) -> Validator:
        """Validate number is greater than n."""
        return Validator(
            predicate=lambda v: isinstance(v, (int, float)) and v > n,
            description=f"number.greater_than({n})",
        )

    @staticmethod
    def less_than(n: float) -> Validator:
        """Validate number is less than n."""
        return Validator(
            predicate=lambda v: isinstance(v, (int, float)) and v < n,
            description=f"number.less_than({n})",
        )

    @staticmethod
    def in_range(min_val: float, max_val: float) -> Validator:
        """Validate number is in range [min_val, max_val]."""
        return Validator(
            predicate=lambda v: isinstance(v, (int, float)) and min_val <= v <= max_val,
            description=f"number.in_range({min_val}, {max_val})",
        )

    @staticmethod
    def equals(n: float) -> Validator:
        """Validate number equals n."""
        return Validator(
            predicate=lambda v: isinstance(v, (int, float)) and v == n,
            description=f"number.equals({n})",
        )

    @staticmethod
    def multiple_of(n: float) -> Validator:
        """Validate number is multiple of n."""
        return Validator(
            predicate=lambda v: isinstance(v, (int, float)) and v % n == 0,
            description=f"number.multiple_of({n})",
        )


# ============================================================================
# COLLECTION VALIDATORS - V.collection namespace
# ============================================================================


class CollectionValidators:
    """Collection validation utilities for V.collection namespace.

    Provides composable validators for lists, dicts, and sequences
    including non_empty, length, and content validation.

    Example:
        V.collection.non_empty & V.collection.max_length(10)

    """

    non_empty: Validator = Validator(
        predicate=lambda v: isinstance(v, (list, tuple, dict, set))
        and len(
            cast(
                "list[object] | tuple[object, ...] | dict[object, object] | set[object]",
                v,
            ),
        )
        > 0,
        description="collection.non_empty",
    )

    is_list: Validator = Validator(
        predicate=lambda v: isinstance(v, list),
        description="collection.is_list",
    )

    is_dict: Validator = Validator(
        predicate=lambda v: isinstance(v, dict),
        description="collection.is_dict",
    )

    is_set: Validator = Validator(
        predicate=lambda v: isinstance(v, set),
        description="collection.is_set",
    )

    is_tuple: Validator = Validator(
        predicate=lambda v: isinstance(v, tuple),
        description="collection.is_tuple",
    )

    @staticmethod
    def min_length(n: int) -> Validator:
        """Validate collection has at least n items."""
        return Validator(
            predicate=lambda v: isinstance(
                v,
                (list, tuple, dict, set),
            )
            and len(
                cast(
                    "list[object] | tuple[object, ...] | dict[object, object] | set[object]",
                    v,
                ),
            )
            >= n,
            description=f"collection.min_length({n})",
        )

    @staticmethod
    def max_length(n: int) -> Validator:
        """Validate collection has at most n items."""
        return Validator(
            predicate=lambda v: isinstance(
                v,
                (list, tuple, dict, set),
            )
            and len(
                cast(
                    "list[object] | tuple[object, ...] | dict[object, object] | set[object]",
                    v,
                ),
            )
            <= n,
            description=f"collection.max_length({n})",
        )

    @staticmethod
    def length(n: int) -> Validator:
        """Validate collection has exactly n items."""
        return Validator(
            predicate=lambda v: isinstance(
                v,
                (list, tuple, dict, set),
            )
            and len(v) == n,
            description=f"collection.length({n})",
        )

    @staticmethod
    def contains(item: object) -> Validator:
        """Validate collection contains item."""
        return Validator(
            predicate=lambda v: isinstance(v, (list, tuple, set)) and item in v,
            description=f"collection.contains({item!r})",
        )

    @staticmethod
    def all_match(validator: ValidatorSpec | Validator) -> Validator:
        """Validate all items in collection match validator."""
        return Validator(
            predicate=lambda v: isinstance(v, (list, tuple, set))
            and all(
                validator(item)
                for item in cast("list[object] | tuple[object, ...] | set[object]", v)
            ),
            description=f"collection.all_match({getattr(validator, 'description', 'validator')})",
        )

    @staticmethod
    def any_match(validator: ValidatorSpec) -> Validator:
        """Validate at least one item in collection matches validator."""
        return Validator(
            predicate=lambda v: isinstance(v, (list, tuple, set))
            and any(
                validator(item)
                for item in cast("list[object] | tuple[object, ...] | set[object]", v)
            ),
            description=f"collection.any_match({getattr(validator, 'description', 'validator')})",
        )


# ============================================================================
# DICT VALIDATORS - V.dict namespace
# ============================================================================


class DictValidators:
    """Dict-specific validation utilities for V.dict namespace.

    Provides composable validators for dictionary validation including
    key presence, key types, and value validation.

    Example:
        V.dict.has_keys("host", "port") & V.dict.non_empty

    """

    non_empty: Validator = Validator(
        predicate=lambda v: isinstance(v, dict) and len(v) > 0,
        description="dict.non_empty",
    )

    @staticmethod
    def has_keys(*keys: str) -> Validator:
        """Validate dict has all specified keys."""
        return Validator(
            predicate=lambda v: isinstance(v, dict) and all(k in v for k in keys),
            description=f"dict.has_keys({', '.join(repr(k) for k in keys)})",
        )

    @staticmethod
    def has_key(key: str) -> Validator:
        """Validate dict has specified key."""
        return Validator(
            predicate=lambda v: isinstance(v, dict) and key in v,
            description=f"dict.has_key({key!r})",
        )

    @staticmethod
    def key_matches(key: str, validator: ValidatorSpec) -> Validator:
        """Validate dict key value matches validator."""
        # Type annotation: validator is ValidatorSpec (Protocol), which is callable
        # Pyright needs explicit cast to help with type inference in lambda
        # ValidatorSpec implements __call__(value: object) -> bool
        validator_callable: Callable[[object], bool] = cast(
            "Callable[[object], bool]", validator
        )
        return Validator(
            predicate=lambda v: (
                isinstance(v, dict)
                and key in v
                and validator_callable(cast("object", v[key]))
            ),
            description=f"dict.key_matches({key!r}, {getattr(validator, 'description', 'validator')})",
        )

    @staticmethod
    def all_keys_match(validator: ValidatorSpec) -> Validator:
        """Validate all dict keys match validator."""
        return Validator(
            predicate=lambda v: isinstance(v, dict)
            and all(validator(k) for k in cast("dict[object, object]", v)),
            description=f"dict.all_keys_match({getattr(validator, 'description', 'validator')})",
        )

    @staticmethod
    def all_values_match(validator: ValidatorSpec) -> Validator:
        """Validate all dict values match validator."""
        return Validator(
            predicate=lambda v: isinstance(v, dict)
            and all(validator(val) for val in cast("dict[object, object]", v).values()),
            description=f"dict.all_values_match({getattr(validator, 'description', 'validator')})",
        )


# ============================================================================
# VALIDATOR DSL - Unified V namespace
# ============================================================================


class ValidatorDSL:
    """Unified validator DSL namespace (V).

    Provides access to all validator categories through a single namespace
    with operator support for declarative validation composition.

    Usage:
        from flext_core import u
        V = u.V

        # String validation
        validator = V.string.non_empty & V.string.max_length(100)

        # Number validation
        validator = V.number.positive & V.number.less_than(1000)

        # Collection validation
        validator = V.collection.non_empty & V.collection.max_length(10)

        # Dict validation
        validator = V.dict.has_keys("host", "port")

        # Complex composition
        validator = (
            V.string.non_empty
            & V.string.min_length(3)
            & (V.string.email | V.string.url)
        )
    """

    string = StringValidators
    number = NumberValidators
    collection = CollectionValidators
    dict = DictValidators

    @staticmethod
    def custom(
        predicate: Callable[[object], bool],
        description: str = "custom",
    ) -> Validator:
        """Create a custom validator from a predicate function."""
        return Validator(predicate=predicate, description=description)

    @staticmethod
    def always_true() -> Validator:
        """Validator that always passes."""
        return Validator(predicate=lambda _: True, description="always_true")

    @staticmethod
    def always_false() -> Validator:
        """Validator that always fails."""
        return Validator(predicate=lambda _: False, description="always_false")

    @staticmethod
    def is_none() -> Validator:
        """Validator that checks if value is None."""
        return Validator(predicate=lambda v: v is None, description="is_none")

    @staticmethod
    def is_not_none() -> Validator:
        """Validator that checks if value is not None."""
        return Validator(predicate=lambda v: v is not None, description="is_not_none")

    @staticmethod
    def is_type(expected_type: type) -> Validator:
        """Validator that checks if value is of expected type."""
        return Validator(
            predicate=lambda v: FlextUtilitiesGuards.is_type(v, expected_type),
            description=f"is_type({expected_type.__name__})",
        )


# ============================================================================
# VALIDATOR BUILDER - Fluent API for building validators
# ============================================================================


class ValidatorBuilder:
    """Fluent builder for composing validators.

    Provides a chainable API for building complex validators
    with clear, readable syntax.

    Usage:
        validator = (
            ValidatorBuilder()
            .string()
            .non_empty()
            .min_length(3)
            .max_length(100)
            .build()
        )

        # Or with u
        validator = (
            u.Validator()
            .string()
            .non_empty()
            .matches(r'^[a-z]+$')
            .build()
        )
    """

    def __init__(self) -> None:
        """Initialize builder with empty validator list."""
        self._validators: list[Validator] = []
        self._mode: str = "string"  # Current mode: string, number, collection, dict

    def _add(self, validator: Validator) -> Self:
        """Add validator to the chain."""
        self._validators.append(validator)
        return self

    # Mode selectors
    def string(self) -> Self:
        """Switch to string validation mode."""
        self._mode = "string"
        return self

    def number(self) -> Self:
        """Switch to number validation mode."""
        self._mode = "number"
        return self

    def collection(self) -> Self:
        """Switch to collection validation mode."""
        self._mode = c.Mixins.OPERATION_COLLECTION
        return self

    def dict(self) -> Self:
        """Switch to dict validation mode."""
        self._mode = "dict"
        return self

    # String validators
    def non_empty(self) -> Self:
        """Add non_empty validator (string or collection based on mode)."""
        if self._mode == "string":
            return self._add(StringValidators.non_empty)
        if self._mode == c.Mixins.OPERATION_COLLECTION:
            return self._add(CollectionValidators.non_empty)
        if self._mode == "dict":
            return self._add(DictValidators.non_empty)
        return self

    def min_length(self, n: int) -> Self:
        """Add min_length validator."""
        if self._mode == "string":
            return self._add(StringValidators.min_length(n))
        if self._mode == c.Mixins.OPERATION_COLLECTION:
            return self._add(CollectionValidators.min_length(n))
        return self

    def max_length(self, n: int) -> Self:
        """Add max_length validator."""
        if self._mode == "string":
            return self._add(StringValidators.max_length(n))
        if self._mode == c.Mixins.OPERATION_COLLECTION:
            return self._add(CollectionValidators.max_length(n))
        return self

    def matches(self, pattern: str) -> Self:
        """Add regex matches validator (string mode)."""
        return self._add(StringValidators.matches(pattern))

    def contains(self, value: str | object) -> Self:
        """Add contains validator."""
        if self._mode == "string" and isinstance(value, str):
            return self._add(StringValidators.contains(value))
        if self._mode == c.Mixins.OPERATION_COLLECTION:
            return self._add(CollectionValidators.contains(value))
        return self

    def starts_with(self, prefix: str) -> Self:
        """Add starts_with validator (string mode)."""
        return self._add(StringValidators.starts_with(prefix))

    def ends_with(self, suffix: str) -> Self:
        """Add ends_with validator (string mode)."""
        return self._add(StringValidators.ends_with(suffix))

    def email(self) -> Self:
        """Add email format validator."""
        return self._add(StringValidators.email)

    def url(self) -> Self:
        """Add URL format validator."""
        return self._add(StringValidators.url)

    # Number validators
    def positive(self) -> Self:
        """Add positive number validator."""
        return self._add(NumberValidators.positive)

    def negative(self) -> Self:
        """Add negative number validator."""
        return self._add(NumberValidators.negative)

    def in_range(self, min_val: float, max_val: float) -> Self:
        """Add in_range validator."""
        return self._add(NumberValidators.in_range(min_val, max_val))

    def greater_than(self, n: float) -> Self:
        """Add greater_than validator."""
        return self._add(NumberValidators.greater_than(n))

    def less_than(self, n: float) -> Self:
        """Add less_than validator."""
        return self._add(NumberValidators.less_than(n))

    # Dict validators
    def has_keys(self, *keys: str) -> Self:
        """Add has_keys validator (dict mode)."""
        return self._add(DictValidators.has_keys(*keys))

    def key_matches(self, key: str, validator: ValidatorSpec) -> Self:
        """Add key_matches validator (dict mode)."""
        return self._add(DictValidators.key_matches(key, validator))

    # Custom validator
    def custom(
        self,
        predicate: Callable[[object], bool],
        description: str = "custom",
    ) -> Self:
        """Add custom validator."""
        return self._add(Validator(predicate=predicate, description=description))

    def build(self) -> Validator:
        """Build final validator by ANDing all validators."""
        if not self._validators:
            return ValidatorDSL.always_true()

        result = self._validators[0]
        for validator in self._validators[1:]:
            result &= validator

        return result


__all__ = [
    "CollectionValidators",
    "DictValidators",
    "NumberValidators",
    "StringValidators",
    "Validator",
    "ValidatorBuilder",
    "ValidatorDSL",
    "ValidatorSpec",
]
