"""Comprehensive edge case tests for FlextUtilities power methods.

Tests all 9 power methods with 200+ edge cases using flext_tests infrastructure:
- validate() - Declarative validation with DSL/Builder
- parse() - Universal type parsing
- transform() - Data transformation
- pipe() - Functional pipeline
- merge() - Deep dictionary merging
- extract() - Path-based extraction
- generate() - Unified ID generation
- batch() - Batch processing
- retry() - Retry with backoff

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import ClassVar, cast

import pytest
from pydantic import BaseModel

from flext_core import FlextResult, FlextTypes, FlextUtilities
from flext_tests.matchers import FlextTestsMatchers

# Alias for V namespace
V = FlextUtilities.V


# =============================================================================
# TEST FIXTURES - Models and Enums for testing
# =============================================================================


class Status(StrEnum):
    """Test enum for parse() tests."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"


class UserModel(BaseModel):
    """Test Pydantic model for parse/transform tests."""

    name: str
    email: str
    age: int = 0


class AddressModel(BaseModel):
    """Nested model for complex tests."""

    city: str
    country: str = "US"


# =============================================================================
# CENTRALIZED TEST SCENARIOS - Using ClassVar pattern from test_utilities.py
# =============================================================================


class PowerMethodScenarios:
    """Centralized test scenarios for power methods."""

    # -------------------------------------------------------------------------
    # validate() scenarios
    # -------------------------------------------------------------------------

    VALIDATE_SINGLE_PASS: ClassVar[
        list[tuple[str, object, Callable[[object], bool]]]
    ] = [
        ("non_empty_string", "hello", V.string.non_empty),
        ("valid_email", "test@example.com", V.string.email),
        ("positive_number", 42, V.number.positive),
        ("non_empty_list", [1, 2, 3], V.collection.non_empty),
        ("valid_dict", {"a": 1}, V.dict.non_empty),
    ]

    VALIDATE_SINGLE_FAIL: ClassVar[
        list[tuple[str, object, Callable[[object], bool], str]]
    ] = [
        ("empty_string", "", V.string.non_empty, "non_empty"),
        ("whitespace_only", "   ", V.string.non_empty, "non_empty"),
        ("invalid_email", "not-email", V.string.email, "email"),
        ("negative_number", -5, V.number.positive, "positive"),
        ("empty_list", [], V.collection.non_empty, "non_empty"),
    ]

    VALIDATE_MODE_ANY: ClassVar[list[tuple[str, object, bool]]] = [
        ("email_passes", "test@example.com", True),
        ("url_passes", "https://example.com", True),
        ("neither_passes", "just-text", False),
    ]

    # -------------------------------------------------------------------------
    # V.string validators scenarios
    # -------------------------------------------------------------------------

    STRING_NON_EMPTY: ClassVar[list[tuple[str, object, bool]]] = [
        ("valid_string", "hello", True),
        ("empty_string", "", False),
        ("whitespace_only", "   ", False),
        ("non_string_int", 123, False),
        ("non_string_none", None, False),
    ]

    STRING_LENGTH: ClassVar[list[tuple[str, str, int, int, bool]]] = [
        ("within_range", "hello", 3, 10, True),
        ("at_min", "hello", 5, 10, True),
        ("at_max", "hello", 1, 5, True),
        ("below_min", "hi", 5, 10, False),
        ("above_max", "hello world", 1, 5, False),
    ]

    STRING_PATTERNS: ClassVar[list[tuple[str, str, str, bool]]] = [
        ("lowercase_match", "hello", r"^[a-z]+$", True),
        ("uppercase_fail", "Hello", r"^[a-z]+$", False),
        ("alphanumeric", "abc123", r"^[a-z0-9]+$", True),
    ]

    STRING_FORMAT: ClassVar[list[tuple[str, str, str, bool]]] = [
        ("valid_email", "test@example.com", "email", True),
        ("invalid_email", "not-email", "email", False),
        ("valid_url_http", "http://example.com", "url", True),
        ("valid_url_https", "https://example.com/path", "url", True),
        ("invalid_url", "not-a-url", "url", False),
    ]

    # -------------------------------------------------------------------------
    # V.number validators scenarios
    # -------------------------------------------------------------------------

    NUMBER_SIGN: ClassVar[list[tuple[str, object, str, bool]]] = [
        ("positive_int", 42, "positive", True),
        ("positive_float", math.pi, "positive", True),
        ("zero_positive", 0, "positive", False),
        ("negative_positive", -5, "positive", False),
        ("negative_int", -10, "negative", True),
        ("zero_negative", 0, "negative", False),
        ("positive_negative", 5, "negative", False),
        ("zero_check", 0, "zero", True),
        ("non_zero_check", 5, "zero", False),
    ]

    NUMBER_RANGE: ClassVar[list[tuple[str, float, float, float, bool]]] = [
        ("within_range", 50, 0, 100, True),
        ("at_min", 0, 0, 100, True),
        ("at_max", 100, 0, 100, True),
        ("below_min", -1, 0, 100, False),
        ("above_max", 101, 0, 100, False),
    ]

    # -------------------------------------------------------------------------
    # V.collection validators scenarios
    # -------------------------------------------------------------------------

    COLLECTION_NON_EMPTY: ClassVar[list[tuple[str, object, bool]]] = [
        ("non_empty_list", [1, 2, 3], True),
        ("empty_list", [], False),
        ("non_empty_dict", {"a": 1}, True),
        ("empty_dict", {}, False),
        ("non_empty_set", {1, 2}, True),
        ("empty_set", set(), False),
        ("non_empty_tuple", (1, 2), True),
        ("empty_tuple", (), False),
    ]

    # -------------------------------------------------------------------------
    # parse() scenarios
    # -------------------------------------------------------------------------

    PARSE_PRIMITIVES: ClassVar[list[tuple[str, object, type, object]]] = [
        ("str_to_int", "42", int, 42),
        ("int_to_str", 42, str, "42"),
        ("str_to_float", "1.5", float, 1.5),
        ("int_to_float", 42, float, 42.0),
        ("str_true", "true", bool, True),
        ("str_false", "false", bool, False),
        ("str_yes", "yes", bool, True),
        ("str_no", "no", bool, False),
    ]

    PARSE_ENUM: ClassVar[list[tuple[str, str, bool, object]]] = [
        ("exact_match", "active", False, Status.ACTIVE),
        ("case_insensitive_upper", "ACTIVE", True, Status.ACTIVE),
        ("case_insensitive_mixed", "Active", True, Status.ACTIVE),
    ]

    # -------------------------------------------------------------------------
    # transform() scenarios
    # -------------------------------------------------------------------------

    TRANSFORM_STRIP: ClassVar[
        list[
            tuple[
                str,
                Mapping[str, FlextTypes.GeneralValueType],
                bool,
                bool,
                Mapping[str, FlextTypes.GeneralValueType],
            ]
        ]
    ] = [
        ("strip_none", {"a": 1, "b": None, "c": 3}, True, False, {"a": 1, "c": 3}),
        ("strip_empty", {"a": 1, "b": "", "c": []}, False, True, {"a": 1}),
        ("strip_both", {"a": 1, "b": None, "c": ""}, True, True, {"a": 1}),
        ("no_strip", {"a": 1, "b": None}, False, False, {"a": 1, "b": None}),
    ]

    TRANSFORM_KEYS: ClassVar[
        list[
            tuple[
                str,
                Mapping[str, FlextTypes.GeneralValueType],
                Mapping[str, str],
                Mapping[str, FlextTypes.GeneralValueType],
            ]
        ]
    ] = [
        ("rename_single", {"old": 1}, {"old": "new"}, {"new": 1}),
        ("rename_multiple", {"a": 1, "b": 2}, {"a": "x", "b": "y"}, {"x": 1, "y": 2}),
        ("partial_rename", {"a": 1, "b": 2}, {"a": "x"}, {"x": 1, "b": 2}),
    ]

    # -------------------------------------------------------------------------
    # merge() scenarios
    # -------------------------------------------------------------------------

    MERGE_STRATEGY: ClassVar[
        list[
            tuple[
                str,
                Mapping[str, FlextTypes.GeneralValueType],
                Mapping[str, FlextTypes.GeneralValueType],
                str,
                Mapping[str, FlextTypes.GeneralValueType],
            ]
        ]
    ] = [
        ("override_flat", {"a": 1}, {"a": 2}, "override", {"a": 2}),
        (
            "deep_nested",
            {"a": {"x": 1}},
            {"a": {"y": 2}},
            "deep",
            {"a": {"x": 1, "y": 2}},
        ),
        ("append_lists", {"a": [1]}, {"a": [2]}, "append", {"a": [1, 2]}),
    ]

    # -------------------------------------------------------------------------
    # extract() scenarios
    # -------------------------------------------------------------------------

    EXTRACT_PATH: ClassVar[
        list[
            tuple[
                str,
                Mapping[str, FlextTypes.GeneralValueType],
                str,
                FlextTypes.GeneralValueType,
            ]
        ]
    ] = [
        ("simple_key", {"name": "John"}, "name", "John"),
        ("nested_path", {"user": {"name": "John"}}, "user.name", "John"),
        ("deep_nested", {"a": {"b": {"c": 1}}}, "a.b.c", 1),
    ]

    EXTRACT_ARRAY: ClassVar[
        list[
            tuple[
                str,
                Mapping[str, FlextTypes.GeneralValueType],
                str,
                FlextTypes.GeneralValueType,
            ]
        ]
    ] = [
        ("first_item", {"items": [1, 2, 3]}, "items[0]", 1),
        ("last_item", {"items": [1, 2, 3]}, "items[-1]", 3),
        ("nested_array", {"data": {"items": ["a", "b"]}}, "data.items[0]", "a"),
    ]

    # -------------------------------------------------------------------------
    # generate() scenarios
    # -------------------------------------------------------------------------

    GENERATE_KINDS: ClassVar[list[tuple[str, str, str]]] = [
        ("id_default", "id", ""),
        ("uuid_full", "uuid", ""),
        ("correlation", "correlation", "corr"),
        ("entity", "entity", "ent"),
        ("batch", "batch", "batch"),
        ("transaction", "transaction", "txn"),
        ("event", "event", "evt"),
        ("command", "command", "cmd"),
        ("query", "query", "qry"),
    ]


# =============================================================================
# TEST CLASS: validate() POWER METHOD
# =============================================================================


class TestValidateMethod:
    """Tests for FlextUtilities.validate() power method."""

    def test_validate_no_validators_returns_ok(self) -> None:
        """Empty validators list returns Ok with original value."""
        result = FlextUtilities.validate("hello")
        FlextTestsMatchers.assert_success(result)
        assert result.value == "hello"

    @pytest.mark.parametrize(
        ("description", "value", "validator"),
        PowerMethodScenarios.VALIDATE_SINGLE_PASS,
    )
    def test_validate_single_pass(
        self,
        description: str,
        value: object,
        validator: Callable[[object], bool],
    ) -> None:
        """Single passing validator returns Ok."""
        # Business Rule: validate accepts ValidatorSpec, but Callable[[object], bool] is compatible
        # Implication: Cast is needed for type checker, but runtime accepts callable validators
        result = FlextUtilities.validate(value, validator)  # type: ignore[arg-type]
        FlextTestsMatchers.assert_success(result, f"{description} should pass")
        assert result.value == value

    @pytest.mark.parametrize(
        ("description", "value", "validator", "error_contains"),
        PowerMethodScenarios.VALIDATE_SINGLE_FAIL,
    )
    def test_validate_single_fail(
        self,
        description: str,
        value: object,
        validator: Callable[[object], bool],
        error_contains: str,
    ) -> None:
        """Single failing validator returns Fail."""
        # Business Rule: validate accepts ValidatorSpec, but Callable[[object], bool] is compatible
        # Implication: Cast is needed for type checker, but runtime accepts callable validators
        result = FlextUtilities.validate(value, validator)  # type: ignore[arg-type]
        error = FlextTestsMatchers.assert_failure(result)
        assert error_contains in error, (
            f"{description}: expected '{error_contains}' in error"
        )

    def test_validate_multiple_all_pass(self) -> None:
        """All validators pass returns Ok."""
        result = FlextUtilities.validate(
            "hello@test.com",
            V.string.non_empty,
            V.string.email,
        )
        FlextTestsMatchers.assert_success(result)

    def test_validate_multiple_first_fails(self) -> None:
        """First validator fails returns Fail (fail_fast)."""
        result = FlextUtilities.validate(
            "",
            V.string.non_empty,
            V.string.email,
        )
        error = FlextTestsMatchers.assert_failure(result)
        assert "non_empty" in error

    def test_validate_collect_errors(self) -> None:
        """collect_errors=True gathers all errors."""
        result = FlextUtilities.validate(
            "",
            V.string.non_empty,
            V.string.min_length(5),
            fail_fast=True,
            collect_errors=True,
        )
        error = FlextTestsMatchers.assert_failure(result)
        assert "non_empty" in error
        assert "min_length" in error

    @pytest.mark.parametrize(
        ("description", "value", "expected_success"),
        PowerMethodScenarios.VALIDATE_MODE_ANY,
    )
    def test_validate_mode_any(
        self,
        description: str,
        value: object,
        expected_success: bool,
    ) -> None:
        """mode='any' validation tests."""
        result = FlextUtilities.validate(
            value,
            V.string.email,
            V.string.url,
            mode="any",
        )
        if expected_success:
            FlextTestsMatchers.assert_success(result, f"{description} should pass")
        else:
            FlextTestsMatchers.assert_failure(result)

    def test_validate_with_field_name(self) -> None:
        """field_name adds context to error message."""
        result = FlextUtilities.validate(
            "",
            V.string.non_empty,
            field_name="user.email",
        )
        error = FlextTestsMatchers.assert_failure(result)
        assert "user.email:" in error

    def test_validate_and_operator(self) -> None:
        """AND operator composition."""
        validator = V.string.non_empty & V.string.max_length(10)
        result = FlextUtilities.validate("hello", validator)
        FlextTestsMatchers.assert_success(result)

    def test_validate_or_operator(self) -> None:
        """OR operator composition."""
        validator = V.string.email | V.string.url
        result = FlextUtilities.validate("test@example.com", validator)
        FlextTestsMatchers.assert_success(result)

    def test_validate_not_operator(self) -> None:
        """NOT operator negates validator."""
        validator = ~V.string.non_empty
        result = FlextUtilities.validate("", validator)
        FlextTestsMatchers.assert_success(result)

    def test_validate_complex_expression(self) -> None:
        """Complex expression with multiple operators."""
        validator = (
            V.string.non_empty
            & V.string.min_length(3)
            & (V.string.email | V.string.url)
        )
        result = FlextUtilities.validate("test@example.com", validator)
        FlextTestsMatchers.assert_success(result)


# =============================================================================
# TEST CLASS: V.string VALIDATORS
# =============================================================================


class TestStringValidators:
    """Tests for V.string validators using parametrized scenarios."""

    @pytest.mark.parametrize(
        ("description", "value", "expected"),
        PowerMethodScenarios.STRING_NON_EMPTY,
    )
    def test_string_non_empty(
        self,
        description: str,
        value: object,
        expected: bool,
    ) -> None:
        """Test string.non_empty validator."""
        result = V.string.non_empty(value)
        assert result is expected, f"{description}: expected {expected}"

    @pytest.mark.parametrize(
        ("description", "value", "min_len", "max_len", "expected"),
        PowerMethodScenarios.STRING_LENGTH,
    )
    def test_string_length(
        self,
        description: str,
        value: str,
        min_len: int,
        max_len: int,
        expected: bool,
    ) -> None:
        """Test string.length validator."""
        result = V.string.length(min_len, max_len)(value)
        assert result is expected, f"{description}: expected {expected}"

    @pytest.mark.parametrize(
        ("description", "value", "pattern", "expected"),
        PowerMethodScenarios.STRING_PATTERNS,
    )
    def test_string_matches(
        self,
        description: str,
        value: str,
        pattern: str,
        expected: bool,
    ) -> None:
        """Test string.matches validator."""
        result = V.string.matches(pattern)(value)
        assert result is expected, f"{description}: expected {expected}"

    @pytest.mark.parametrize(
        ("description", "value", "format_type", "expected"),
        PowerMethodScenarios.STRING_FORMAT,
    )
    def test_string_format(
        self,
        description: str,
        value: str,
        format_type: str,
        expected: bool,
    ) -> None:
        """Test string format validators."""
        validator = getattr(V.string, format_type)
        result = validator(value)
        assert result is expected, f"{description}: expected {expected}"

    def test_string_contains(self) -> None:
        """Test string.contains validator."""
        assert V.string.contains("ell")("hello") is True
        assert V.string.contains("xyz")("hello") is False

    def test_string_starts_with(self) -> None:
        """Test string.starts_with validator."""
        assert V.string.starts_with("hel")("hello") is True
        assert V.string.starts_with("xyz")("hello") is False

    def test_string_ends_with(self) -> None:
        """Test string.ends_with validator."""
        assert V.string.ends_with("lo")("hello") is True
        assert V.string.ends_with("xyz")("hello") is False


# =============================================================================
# TEST CLASS: V.number VALIDATORS
# =============================================================================


class TestNumberValidators:
    """Tests for V.number validators using parametrized scenarios."""

    @pytest.mark.parametrize(
        ("description", "value", "validator_name", "expected"),
        PowerMethodScenarios.NUMBER_SIGN,
    )
    def test_number_sign(
        self,
        description: str,
        value: object,
        validator_name: str,
        expected: bool,
    ) -> None:
        """Test number sign validators."""
        validator = getattr(V.number, validator_name)
        result = validator(value)
        assert result is expected, f"{description}: expected {expected}"

    @pytest.mark.parametrize(
        ("description", "value", "min_val", "max_val", "expected"),
        PowerMethodScenarios.NUMBER_RANGE,
    )
    def test_number_in_range(
        self,
        description: str,
        value: float,
        min_val: float,
        max_val: float,
        expected: bool,
    ) -> None:
        """Test number.in_range validator."""
        result = V.number.in_range(min_val, max_val)(value)
        assert result is expected, f"{description}: expected {expected}"

    def test_number_integer(self) -> None:
        """Test number.integer validator."""
        assert V.number.integer(42) is True
        assert V.number.integer(42.0) is True
        assert V.number.integer(42.5) is False

    def test_number_greater_than(self) -> None:
        """Test number.greater_than validator."""
        assert V.number.greater_than(5)(10) is True
        assert V.number.greater_than(5)(5) is False
        assert V.number.greater_than(5)(3) is False

    def test_number_less_than(self) -> None:
        """Test number.less_than validator."""
        assert V.number.less_than(10)(5) is True
        assert V.number.less_than(10)(10) is False
        assert V.number.less_than(10)(15) is False


# =============================================================================
# TEST CLASS: V.collection VALIDATORS
# =============================================================================


class TestCollectionValidators:
    """Tests for V.collection validators using parametrized scenarios."""

    @pytest.mark.parametrize(
        ("description", "value", "expected"),
        PowerMethodScenarios.COLLECTION_NON_EMPTY,
    )
    def test_collection_non_empty(
        self,
        description: str,
        value: object,
        expected: bool,
    ) -> None:
        """Test collection.non_empty validator."""
        result = V.collection.non_empty(value)
        assert result is expected, f"{description}: expected {expected}"

    def test_collection_length(self) -> None:
        """Test collection.length validator."""
        assert V.collection.length(3)([1, 2, 3]) is True
        assert V.collection.length(3)([1, 2]) is False

    def test_collection_min_length(self) -> None:
        """Test collection.min_length validator."""
        assert V.collection.min_length(2)([1, 2, 3]) is True
        assert V.collection.min_length(5)([1, 2]) is False

    def test_collection_max_length(self) -> None:
        """Test collection.max_length validator."""
        assert V.collection.max_length(5)([1, 2, 3]) is True
        assert V.collection.max_length(2)([1, 2, 3]) is False

    def test_collection_contains(self) -> None:
        """Test collection.contains validator."""
        assert V.collection.contains(2)([1, 2, 3]) is True
        assert V.collection.contains(5)([1, 2, 3]) is False

    def test_collection_all_match(self) -> None:
        """Test collection.all_match validator."""
        assert V.collection.all_match(V.number.positive)([1, 2, 3]) is True
        assert V.collection.all_match(V.number.positive)([1, -2, 3]) is False

    def test_collection_any_match(self) -> None:
        """Test collection.any_match validator."""
        assert V.collection.any_match(V.number.negative)([1, -2, 3]) is True
        assert V.collection.any_match(V.number.negative)([1, 2, 3]) is False


# =============================================================================
# TEST CLASS: V.dict VALIDATORS
# =============================================================================


class TestDictValidators:
    """Tests for V.dict validators."""

    def test_dict_non_empty(self) -> None:
        """Test dict.non_empty validator."""
        assert V.dict.non_empty({"a": 1}) is True
        assert V.dict.non_empty({}) is False

    def test_dict_has_key(self) -> None:
        """Test dict.has_key validator."""
        assert V.dict.has_key("a")({"a": 1, "b": 2}) is True
        assert V.dict.has_key("c")({"a": 1, "b": 2}) is False

    def test_dict_has_keys(self) -> None:
        """Test dict.has_keys validator."""
        assert V.dict.has_keys("a", "b")({"a": 1, "b": 2}) is True
        assert V.dict.has_keys("a", "c")({"a": 1, "b": 2}) is False

    def test_dict_key_matches(self) -> None:
        """Test dict.key_matches validator."""
        assert V.dict.key_matches("age", V.number.positive)({"age": 25}) is True
        assert V.dict.key_matches("age", V.number.positive)({"age": -5}) is False

    def test_dict_all_values_match(self) -> None:
        """Test dict.all_values_match validator."""
        assert V.dict.all_values_match(V.number.positive)({"a": 1, "b": 2}) is True
        assert V.dict.all_values_match(V.number.positive)({"a": 1, "b": -2}) is False


# =============================================================================
# TEST CLASS: parse() POWER METHOD
# =============================================================================


class TestParseMethod:
    """Tests for FlextUtilities.parse() power method."""

    @pytest.mark.parametrize(
        ("description", "value", "target_type", "expected"),
        PowerMethodScenarios.PARSE_PRIMITIVES,
    )
    def test_parse_primitives(
        self,
        description: str,
        value: object,
        target_type: type,
        expected: object,
    ) -> None:
        """Test parsing primitive types."""
        result: FlextResult[object] = FlextUtilities.parse(value, target_type)
        parsed = FlextTestsMatchers.assert_success(result, f"{description} failed")
        assert parsed == expected, f"{description}: got {parsed}, expected {expected}"

    @pytest.mark.parametrize(
        ("description", "value", "case_insensitive", "expected_member"),
        PowerMethodScenarios.PARSE_ENUM,
    )
    def test_parse_enum(
        self,
        description: str,
        value: str,
        case_insensitive: bool,
        expected_member: Status,
    ) -> None:
        """Test parsing enum values."""
        result = FlextUtilities.parse(
            value,
            Status,
            case_insensitive=case_insensitive,
        )
        parsed = FlextTestsMatchers.assert_success(result, f"{description} failed")
        assert parsed == expected_member, f"{description}"

    def test_parse_enum_case_insensitive(self) -> None:
        """Test case-insensitive enum parsing."""
        result = FlextUtilities.parse("active", Status, case_insensitive=True)
        parsed = FlextTestsMatchers.assert_success(result)
        assert parsed == Status.ACTIVE

    def test_parse_enum_invalid_fails(self) -> None:
        """Test that invalid enum value fails."""
        result = FlextUtilities.parse("invalid", Status)
        FlextTestsMatchers.assert_failure(result)

    def test_parse_with_default(self) -> None:
        """Test parsing with default value on failure."""
        result = FlextUtilities.parse("not_a_number", int, default=42)
        parsed = FlextTestsMatchers.assert_success(result)
        assert parsed == 42

    def test_parse_with_default_factory(self) -> None:
        """Test parsing with default factory on failure."""
        result = FlextUtilities.parse("invalid", int, default_factory=lambda: 99)
        parsed = FlextTestsMatchers.assert_success(result)
        assert parsed == 99

    def test_parse_str_to_int_coerce(self) -> None:
        """Test string to int coercion."""
        result = FlextUtilities.parse("123", int, coerce=True)
        parsed = FlextTestsMatchers.assert_success(result)
        assert parsed == 123
        assert isinstance(parsed, int)

    def test_parse_int_to_str_coerce(self) -> None:
        """Test int to string coercion."""
        result = FlextUtilities.parse(456, str, coerce=True)
        parsed = FlextTestsMatchers.assert_success(result)
        assert parsed == "456"
        assert isinstance(parsed, str)

    def test_parse_float_to_int_coerce(self) -> None:
        """Test float to int coercion."""
        result = FlextUtilities.parse(math.pi, int, coerce=True)
        parsed = FlextTestsMatchers.assert_success(result)
        assert parsed == 3
        assert isinstance(parsed, int)

    def test_parse_strict_mode_allows_direct_coercion(self) -> None:
        """Test strict mode still allows direct target(value) coercion."""
        # strict=True only skips primitive coercion helpers like _coerce_primitive
        # but int("123") works directly via target(value) fallback
        result = FlextUtilities.parse("123", int, strict=True)
        parsed = FlextTestsMatchers.assert_success(result)
        assert parsed == 123

    def test_parse_strict_mode_passes_on_exact_type(self) -> None:
        """Test strict mode accepts exact type match."""
        result = FlextUtilities.parse(123, int, strict=True)
        parsed = FlextTestsMatchers.assert_success(result)
        assert parsed == 123

    def test_parse_bool_values(self) -> None:
        """Test parsing boolean values."""
        result_true = FlextUtilities.parse(True, bool)
        result_false = FlextUtilities.parse(False, bool)
        assert FlextTestsMatchers.assert_success(result_true) is True
        assert FlextTestsMatchers.assert_success(result_false) is False

    def test_parse_none_with_default(self) -> None:
        """Test parsing None with default value."""
        result = FlextUtilities.parse(None, int, default=0)
        parsed = FlextTestsMatchers.assert_success(result)
        assert parsed == 0


# =============================================================================
# TEST CLASS: transform() POWER METHOD
# =============================================================================


class TestTransformMethod:
    """Tests for FlextUtilities.transform() power method."""

    @pytest.mark.parametrize(
        ("description", "input_dict", "strip_none", "strip_empty", "expected"),
        PowerMethodScenarios.TRANSFORM_STRIP,
    )
    def test_transform_strip_options(
        self,
        description: str,
        input_dict: Mapping[str, FlextTypes.GeneralValueType],
        strip_none: bool,
        strip_empty: bool,
        expected: Mapping[str, FlextTypes.GeneralValueType],
    ) -> None:
        """Test transform with strip_none and strip_empty options."""
        result = FlextUtilities.transform(
            input_dict,
            strip_none=strip_none,
            strip_empty=strip_empty,
        )
        transformed = FlextTestsMatchers.assert_success(result, f"{description} failed")
        assert transformed == expected, f"{description}"

    @pytest.mark.parametrize(
        ("description", "input_dict", "key_map", "expected"),
        PowerMethodScenarios.TRANSFORM_KEYS,
    )
    def test_transform_map_keys(
        self,
        description: str,
        input_dict: Mapping[str, FlextTypes.GeneralValueType],
        key_map: Mapping[str, str],
        expected: Mapping[str, FlextTypes.GeneralValueType],
    ) -> None:
        """Test transform with map_keys option."""
        # Business Rule: transform accepts dict[str, str] for map_keys, but Mapping is compatible
        # Implication: Convert Mapping to dict for type compatibility
        result = FlextUtilities.transform(
            input_dict, map_keys=dict(key_map) if key_map else None
        )
        transformed = FlextTestsMatchers.assert_success(result, f"{description} failed")
        assert transformed == expected, f"{description}"

    def test_transform_normalize(self) -> None:
        """Test transform with normalize option."""
        input_data: Mapping[str, FlextTypes.GeneralValueType] = {
            "Name": "  John  ",
            "Age": 25,
        }
        result = FlextUtilities.transform(input_data, normalize=True)
        transformed = FlextTestsMatchers.assert_success(result)
        assert isinstance(transformed, dict)

    def test_transform_to_json(self) -> None:
        """Test transform with to_json option."""
        input_data: Mapping[str, FlextTypes.GeneralValueType] = {
            "name": "John",
            "active": True,
        }
        result = FlextUtilities.transform(input_data, to_json=True)
        transformed = FlextTestsMatchers.assert_success(result)
        # Should be JSON-serializable dict (converts non-JSON values to JSON types)
        assert isinstance(transformed, dict)
        assert transformed == {"name": "John", "active": True}

    def test_transform_strip_empty(self) -> None:
        """Test transform with strip_empty option."""
        input_data: Mapping[str, FlextTypes.GeneralValueType] = {
            "name": "John",
            "empty": "",
            "items": [],
        }
        result = FlextUtilities.transform(input_data, strip_empty=True)
        transformed = FlextTestsMatchers.assert_success(result)
        assert "name" in transformed
        assert "empty" not in transformed

    def test_transform_filter_keys_set(self) -> None:
        """Test transform with filter_keys as set."""
        input_data: Mapping[str, FlextTypes.GeneralValueType] = {
            "name": "John",
            "age": 25,
            "email": "john@test.com",
        }
        result = FlextUtilities.transform(input_data, filter_keys={"name", "age"})
        transformed = FlextTestsMatchers.assert_success(result)
        assert "name" in transformed
        assert "age" in transformed
        assert "email" not in transformed

    def test_transform_exclude_keys(self) -> None:
        """Test transform with exclude_keys option."""
        input_data: Mapping[str, FlextTypes.GeneralValueType] = {
            "name": "John",
            "password": "secret",
            "age": 25,
        }
        result = FlextUtilities.transform(input_data, exclude_keys={"password"})
        transformed = FlextTestsMatchers.assert_success(result)
        assert "name" in transformed
        assert "password" not in transformed

    def test_transform_empty_dict(self) -> None:
        """Test transform with empty dict."""
        result = FlextUtilities.transform({}, normalize=True)
        transformed = FlextTestsMatchers.assert_success(result)
        assert transformed == {}

    def test_transform_nested_dict(self) -> None:
        """Test transform with nested dictionary."""
        input_data = {"user": {"name": "John", "age": None}}
        result = FlextUtilities.transform(input_data, strip_none=True)
        transformed = FlextTestsMatchers.assert_success(result)
        assert "user" in transformed

    def test_transform_combined_options(self) -> None:
        """Test transform with multiple options combined."""
        input_data = {"old_key": "value", "remove": None}
        result = FlextUtilities.transform(
            input_data,
            strip_none=True,
            map_keys={"old_key": "new_key"},
        )
        transformed = FlextTestsMatchers.assert_success(result)
        assert "new_key" in transformed
        assert "remove" not in transformed


# =============================================================================
# TEST CLASS: pipe() POWER METHOD
# =============================================================================


class TestPipeMethod:
    """Tests for FlextUtilities.pipe() power method."""

    # -------------------------------------------------------------------------
    # Basic pipe operations
    # -------------------------------------------------------------------------

    def test_pipe_empty_operations_returns_value(self) -> None:
        """Empty operations returns original value."""
        result = FlextUtilities.pipe("hello")
        value = FlextTestsMatchers.assert_success(result)
        assert value == "hello"

    def test_pipe_single_operation(self) -> None:
        """Single operation transforms value."""
        result = FlextUtilities.pipe(
            "hello", cast("Callable[[object], object]", str.upper)
        )
        value = FlextTestsMatchers.assert_success(result)
        assert value == "HELLO"

    def test_pipe_multiple_operations(self) -> None:
        """Multiple operations chain correctly."""
        result = FlextUtilities.pipe(
            "  hello  ",
            cast("Callable[[object], object]", str.strip),
            cast("Callable[[object], object]", str.upper),
            lambda s: f"[{s}]",
        )
        value = FlextTestsMatchers.assert_success(result)
        assert value == "[HELLO]"

    def test_pipe_with_lambdas(self) -> None:
        """Lambdas work in pipeline."""
        result = FlextUtilities.pipe(
            5,
            lambda x: cast("int", x) * 2,
            lambda x: cast("int", x) + 3,
            lambda x: cast("int", x) * cast("int", x),
        )
        value = FlextTestsMatchers.assert_success(result)
        # (5 * 2) + 3 = 13, 13 * 13 = 169
        assert value == 169

    def test_pipe_type_transformation(self) -> None:
        """Operations can change types through chain."""
        result = FlextUtilities.pipe(
            "42",
            cast("Callable[[object], object]", int),
            lambda x: cast("int", x) * 2,
            cast("Callable[[object], object]", str),
            lambda s: f"Result: {s}",
        )
        value = FlextTestsMatchers.assert_success(result)
        assert value == "Result: 84"

    # -------------------------------------------------------------------------
    # Error handling in pipe
    # -------------------------------------------------------------------------

    def test_pipe_operation_raises_stops(self) -> None:
        """Exception in operation returns Fail (on_error=stop)."""
        result = FlextUtilities.pipe(
            "not-a-number",
            cast("Callable[[object], object]", int),  # Will raise ValueError
            lambda x: cast("int", x) * 2,
        )
        error = FlextTestsMatchers.assert_failure(result)
        assert (
            "int" in error.lower()
            or "invalid" in error.lower()
            or "error" in error.lower()
        )

    def test_pipe_on_error_skip_continues(self) -> None:
        """on_error=skip continues with previous value."""
        # When an operation fails with skip, it uses previous value
        result = FlextUtilities.pipe(
            "test",
            cast("Callable[[object], object]", str.upper),
            cast(
                "Callable[[object], object]", int
            ),  # Will fail - "TEST" is not a number
            lambda x: f"final: {x}",
            on_error="skip",
        )
        # With skip mode, continues with previous value "TEST"
        value = FlextTestsMatchers.assert_success(result)
        assert "TEST" in str(value)

    def test_pipe_first_operation_fails(self) -> None:
        """First operation failing returns Fail."""
        result = FlextUtilities.pipe(
            "abc",
            cast("Callable[[object], object]", int),  # Fails immediately
        )
        FlextTestsMatchers.assert_failure(result)

    def test_pipe_middle_operation_fails(self) -> None:
        """Middle operation failing stops chain."""
        result = FlextUtilities.pipe(
            10,
            lambda x: cast("int", x) + 5,
            lambda x: 1 / (cast("int", x) - 15),  # Division by zero when x=15
            lambda x: cast("float", x) * 100,
        )
        FlextTestsMatchers.assert_failure(result)

    # -------------------------------------------------------------------------
    # FlextResult handling in pipe
    # -------------------------------------------------------------------------

    def test_pipe_with_flext_result_success_unwrapped(self) -> None:
        """FlextResult.ok values are auto-unwrapped."""
        result = FlextUtilities.pipe(
            "test",
            lambda s: FlextResult[str].ok(cast("str", s).upper()),
            lambda s: f"[{s}]",
        )
        value = FlextTestsMatchers.assert_success(result)
        assert value == "[TEST]"

    def test_pipe_with_flext_result_failure_propagates(self) -> None:
        """FlextResult.fail propagates through pipeline."""
        result = FlextUtilities.pipe(
            "test",
            lambda s: FlextResult[str].fail("validation error"),
            lambda s: cast("str", s).upper(),  # Should not execute
        )
        error = FlextTestsMatchers.assert_failure(result)
        assert "validation" in error.lower()

    def test_pipe_mixed_results_and_plain_values(self) -> None:
        """Mix of FlextResult and plain returns works."""
        result = FlextUtilities.pipe(
            5,
            lambda x: cast("int", x) * 2,  # Plain: 10
            lambda x: FlextResult[int].ok(cast("int", x) + 5),  # Result: ok(15)
            lambda x: cast("int", x) * 3,  # Plain: 45
        )
        value = FlextTestsMatchers.assert_success(result)
        assert value == 45

    # -------------------------------------------------------------------------
    # Edge cases
    # -------------------------------------------------------------------------

    def test_pipe_none_value(self) -> None:
        """None value can be piped."""
        result = FlextUtilities.pipe(
            None,
            lambda x: x is None,
            lambda b: "yes" if b else "no",
        )
        value = FlextTestsMatchers.assert_success(result)
        assert value == "yes"

    def test_pipe_empty_string(self) -> None:
        """Empty string piped correctly."""
        result = FlextUtilities.pipe(
            "",
            cast("Callable[[object], object]", len),
            lambda n: cast("int", n) == 0,
        )
        value = FlextTestsMatchers.assert_success(result)
        assert value is True

    def test_pipe_dict_transformation(self) -> None:
        """Dict can be transformed through pipe."""
        result = FlextUtilities.pipe(
            {"name": "john"},
            lambda d: {
                **cast("dict[str, str]", d),
                "upper_name": cast("dict[str, str]", d)["name"].upper(),
            },
            cast("Callable[[object], object]", operator.itemgetter("upper_name")),
        )
        value = FlextTestsMatchers.assert_success(result)
        assert value == "JOHN"

    def test_pipe_list_operations(self) -> None:
        """List operations in pipe."""
        result = FlextUtilities.pipe(
            [3, 1, 4, 1, 5],
            cast("Callable[[object], object]", sorted),
            cast("Callable[[object], object]", operator.itemgetter(slice(3))),
            cast("Callable[[object], object]", sum),
        )
        value = FlextTestsMatchers.assert_success(result)
        assert value == 5  # sorted: [1,1,3,4,5], first 3: [1,1,3], sum: 5


# =============================================================================
# TEST CLASS: merge() POWER METHOD
# =============================================================================


class TestMergeMethod:
    """Tests for FlextUtilities.merge() power method."""

    # -------------------------------------------------------------------------
    # Basic merge operations
    # -------------------------------------------------------------------------

    def test_merge_empty_returns_empty(self) -> None:
        """No dicts returns empty dict."""
        result = FlextUtilities.merge()
        value = FlextTestsMatchers.assert_success(result)
        assert value == {}

    def test_merge_single_dict_returns_copy(self) -> None:
        """Single dict returns copy of it."""
        original: dict[str, FlextTypes.GeneralValueType] = {"a": 1, "b": 2}
        result = FlextUtilities.merge(original)
        value = FlextTestsMatchers.assert_success(result)
        assert value == {"a": 1, "b": 2}
        # Verify it's a copy, not same reference
        assert value is not original

    def test_merge_two_dicts_override(self) -> None:
        """Second dict overrides first (shallow)."""
        d1: dict[str, FlextTypes.GeneralValueType] = {"a": 1, "b": 2}
        d2: dict[str, FlextTypes.GeneralValueType] = {"b": 3, "c": 4}
        result = FlextUtilities.merge(d1, d2, strategy="override")
        value = FlextTestsMatchers.assert_success(result)
        assert value == {"a": 1, "b": 3, "c": 4}

    @pytest.mark.parametrize(
        ("description", "d1", "d2", "strategy", "expected"),
        PowerMethodScenarios.MERGE_STRATEGY,
    )
    def test_merge_strategies(
        self,
        description: str,
        d1: Mapping[str, FlextTypes.GeneralValueType],
        d2: Mapping[str, FlextTypes.GeneralValueType],
        strategy: str,
        expected: Mapping[str, FlextTypes.GeneralValueType],
    ) -> None:
        """Test different merge strategies."""
        result = FlextUtilities.merge(d1, d2, strategy=strategy)
        value = FlextTestsMatchers.assert_success(result, f"{description} failed")
        assert value == expected, f"{description}"

    # -------------------------------------------------------------------------
    # Deep merge tests
    # -------------------------------------------------------------------------

    def test_merge_deep_nested_dicts(self) -> None:
        """Deep strategy merges nested dicts."""
        d1: Mapping[str, FlextTypes.GeneralValueType] = {"a": {"x": 1, "y": 2}, "b": 3}
        d2: Mapping[str, FlextTypes.GeneralValueType] = {
            "a": {"y": 20, "z": 30},
            "c": 4,
        }
        result = FlextUtilities.merge(d1, d2, strategy="deep")
        value = FlextTestsMatchers.assert_success(result)
        assert value == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3, "c": 4}

    def test_merge_deep_three_levels(self) -> None:
        """Deep merge works with deeply nested structures."""
        d1 = {"l1": {"l2": {"l3": {"a": 1}}}}
        d2 = {"l1": {"l2": {"l3": {"b": 2}}}}
        result = FlextUtilities.merge(d1, d2, strategy="deep")
        value = FlextTestsMatchers.assert_success(result)
        assert value == {"l1": {"l2": {"l3": {"a": 1, "b": 2}}}}

    def test_merge_deep_non_dict_override(self) -> None:
        """Deep merge: non-dict replaces dict."""
        d1: dict[str, FlextTypes.GeneralValueType] = {"a": {"nested": True}}
        d2: dict[str, FlextTypes.GeneralValueType] = {"a": "string_value"}
        result = FlextUtilities.merge(d1, d2, strategy="deep")
        value = FlextTestsMatchers.assert_success(result)
        assert value == {"a": "string_value"}

    # -------------------------------------------------------------------------
    # Append strategy tests
    # -------------------------------------------------------------------------

    def test_merge_append_lists(self) -> None:
        """Append strategy concatenates lists."""
        d1: dict[str, FlextTypes.GeneralValueType] = {"items": [1, 2]}
        d2: dict[str, FlextTypes.GeneralValueType] = {"items": [3, 4]}
        result = FlextUtilities.merge(d1, d2, strategy="append")
        value = FlextTestsMatchers.assert_success(result)
        assert value == {"items": [1, 2, 3, 4]}

    def test_merge_append_non_list_overrides(self) -> None:
        """Append strategy: non-lists override."""
        d1: dict[str, FlextTypes.GeneralValueType] = {"a": 1}
        d2: dict[str, FlextTypes.GeneralValueType] = {"a": 2}
        result = FlextUtilities.merge(d1, d2, strategy="append")
        value = FlextTestsMatchers.assert_success(result)
        assert value == {"a": 2}

    # -------------------------------------------------------------------------
    # Filter options tests
    # -------------------------------------------------------------------------

    def test_merge_filter_none(self) -> None:
        """filter_none skips None values."""
        d1: dict[str, FlextTypes.GeneralValueType] = {"a": 1}
        d2: dict[str, FlextTypes.GeneralValueType] = {"a": None, "b": 2}
        result = FlextUtilities.merge(d1, d2, filter_none=True)
        value = FlextTestsMatchers.assert_success(result)
        assert value.get("a") == 1  # Not overridden by None
        assert value.get("b") == 2

    def test_merge_filter_empty(self) -> None:
        """filter_empty skips empty values."""
        d1: dict[str, FlextTypes.GeneralValueType] = {"a": "value", "b": [1, 2]}
        d2: dict[str, FlextTypes.GeneralValueType] = {"a": "", "b": [], "c": {}}
        result = FlextUtilities.merge(d1, d2, filter_empty=True)
        value = FlextTestsMatchers.assert_success(result)
        assert value.get("a") == "value"  # Not overridden by ""
        assert value.get("b") == [1, 2]  # Not overridden by []
        assert "c" not in value or value.get("c") == {}

    def test_merge_filter_both(self) -> None:
        """Both filters can be combined."""
        d1: dict[str, FlextTypes.GeneralValueType] = {"a": 1, "b": "text"}
        d2: dict[str, FlextTypes.GeneralValueType] = {"a": None, "b": ""}
        result = FlextUtilities.merge(d1, d2, filter_none=True, filter_empty=True)
        value = FlextTestsMatchers.assert_success(result)
        assert value == {"a": 1, "b": "text"}

    # -------------------------------------------------------------------------
    # Multiple dicts merge
    # -------------------------------------------------------------------------

    def test_merge_three_dicts(self) -> None:
        """Three dicts merge correctly."""
        d1: dict[str, FlextTypes.GeneralValueType] = {"a": 1}
        d2: dict[str, FlextTypes.GeneralValueType] = {"b": 2}
        d3: dict[str, FlextTypes.GeneralValueType] = {"c": 3, "a": 10}
        result = FlextUtilities.merge(d1, d2, d3)
        value = FlextTestsMatchers.assert_success(result)
        assert value == {"a": 10, "b": 2, "c": 3}

    def test_merge_many_dicts(self) -> None:
        """Many dicts merge in order."""
        dicts: list[dict[str, FlextTypes.GeneralValueType]] = [
            {"key": 1},
            {"key": 2},
            {"key": 3},
            {"key": 4},
            {"key": 5},
        ]
        result = FlextUtilities.merge(*dicts)
        value = FlextTestsMatchers.assert_success(result)
        assert value == {"key": 5}  # Last one wins

    # -------------------------------------------------------------------------
    # Edge cases
    # -------------------------------------------------------------------------

    def test_merge_empty_dicts(self) -> None:
        """Empty dicts merge to empty."""
        result = FlextUtilities.merge({}, {}, {})
        value = FlextTestsMatchers.assert_success(result)
        assert value == {}

    def test_merge_preserves_types(self) -> None:
        """Merge preserves various value types."""
        d1: dict[str, FlextTypes.GeneralValueType] = {
            "int": 42,
            "float": math.pi,
            "str": "hello",
            "bool": True,
            "list": [1, 2, 3],
            "dict": {"nested": True},
        }
        result = FlextUtilities.merge(d1)
        value = FlextTestsMatchers.assert_success(result)
        assert value["int"] == 42
        assert value["float"] == math.pi
        assert value["str"] == "hello"
        assert value["bool"] is True
        assert value["list"] == [1, 2, 3]
        assert value["dict"] == {"nested": True}


# =============================================================================
# TEST CLASS: extract() POWER METHOD
# =============================================================================


class TestExtractMethod:
    """Tests for FlextUtilities.extract() power method."""

    # -------------------------------------------------------------------------
    # Basic extraction
    # -------------------------------------------------------------------------

    def test_extract_simple_key(self) -> None:
        """Extract simple top-level key."""
        data = {"name": "John", "age": 30}
        result = FlextUtilities.extract(data, "name")
        value = FlextTestsMatchers.assert_success(result)
        assert value == "John"

    def test_extract_nested_path(self) -> None:
        """Extract nested value with dot notation."""
        data = {"user": {"name": "John", "email": "john@test.com"}}
        result = FlextUtilities.extract(data, "user.name")
        value = FlextTestsMatchers.assert_success(result)
        assert value == "John"

    def test_extract_deep_nested(self) -> None:
        """Extract deeply nested value."""
        data = {"level1": {"level2": {"level3": {"value": 42}}}}
        result = FlextUtilities.extract(data, "level1.level2.level3.value")
        value = FlextTestsMatchers.assert_success(result)
        assert value == 42

    @pytest.mark.parametrize(
        ("description", "data", "path", "expected"),
        PowerMethodScenarios.EXTRACT_PATH,
    )
    def test_extract_paths(
        self,
        description: str,
        data: Mapping[str, FlextTypes.GeneralValueType],
        path: str,
        expected: FlextTypes.GeneralValueType,
    ) -> None:
        """Test path extraction scenarios."""
        result = FlextUtilities.extract(data, path)
        value = FlextTestsMatchers.assert_success(result, f"{description} failed")
        assert value == expected, f"{description}"

    # -------------------------------------------------------------------------
    # Array indexing
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("description", "data", "path", "expected"),
        PowerMethodScenarios.EXTRACT_ARRAY,
    )
    def test_extract_array_index(
        self,
        description: str,
        data: Mapping[str, FlextTypes.GeneralValueType],
        path: str,
        expected: FlextTypes.GeneralValueType,
    ) -> None:
        """Test array index extraction."""
        result = FlextUtilities.extract(data, path)
        value = FlextTestsMatchers.assert_success(result, f"{description} failed")
        assert value == expected, f"{description}"

    def test_extract_array_first_element(self) -> None:
        """Extract first element from array."""
        data = {"items": [10, 20, 30]}
        result = FlextUtilities.extract(data, "items[0]")
        value = FlextTestsMatchers.assert_success(result)
        assert value == 10

    def test_extract_array_last_element(self) -> None:
        """Extract last element from array."""
        data = {"items": [10, 20, 30]}
        result = FlextUtilities.extract(data, "items[-1]")
        value = FlextTestsMatchers.assert_success(result)
        assert value == 30

    def test_extract_nested_array(self) -> None:
        """Extract from nested array (consecutive indexes need proper path)."""
        data = {"matrix": {"row0": [1, 2], "row1": [3, 4]}}
        # Extract first element from row0
        result = FlextUtilities.extract(data, "matrix.row0[0]")
        value = FlextTestsMatchers.assert_success(result)
        assert value == 1

    # -------------------------------------------------------------------------
    # Default values
    # -------------------------------------------------------------------------

    def test_extract_missing_key_returns_default(self) -> None:
        """Missing key returns default value."""
        data = {"a": 1}
        result = FlextUtilities.extract(data, "missing", default="fallback")
        value = FlextTestsMatchers.assert_success(result)
        assert value == "fallback"

    def test_extract_missing_nested_returns_sentinel(self) -> None:
        """Missing nested key returns sentinel (non-None default)."""
        data = {"user": {"name": "John"}}
        # FlextResult cannot have None as success value, so use sentinel
        result = FlextUtilities.extract(data, "user.missing.deep", default="NOT_FOUND")
        value = FlextTestsMatchers.assert_success(result)
        assert value == "NOT_FOUND"

    def test_extract_missing_with_none_default_fails(self) -> None:
        """Missing key with None default returns failure (FlextResult constraint)."""
        data = {"a": 1}
        # FlextResult cannot have None as success value
        result = FlextUtilities.extract(data, "missing")
        # This should fail because default=None can't be wrapped in FlextResult.ok()
        FlextTestsMatchers.assert_failure(result)

    # -------------------------------------------------------------------------
    # Required option
    # -------------------------------------------------------------------------

    def test_extract_required_success(self) -> None:
        """Required extraction succeeds when path exists."""
        data = {"key": "value"}
        result = FlextUtilities.extract(data, "key", required=True)
        value = FlextTestsMatchers.assert_success(result)
        assert value == "value"

    def test_extract_required_fails_when_missing(self) -> None:
        """Required extraction fails when path missing."""
        data = {"a": 1}
        result = FlextUtilities.extract(data, "missing", required=True)
        error = FlextTestsMatchers.assert_failure(result)
        assert (
            "missing" in error.lower()
            or "not found" in error.lower()
            or "path" in error.lower()
        )

    def test_extract_required_nested_fails(self) -> None:
        """Required extraction fails for missing nested path."""
        data = {"user": {"name": "John"}}
        result = FlextUtilities.extract(data, "user.email", required=True)
        FlextTestsMatchers.assert_failure(result)

    # -------------------------------------------------------------------------
    # Custom separator
    # -------------------------------------------------------------------------

    def test_extract_custom_separator_slash(self) -> None:
        """Custom separator: forward slash."""
        data = {"user": {"profile": {"name": "John"}}}
        result = FlextUtilities.extract(data, "user/profile/name", separator="/")
        value = FlextTestsMatchers.assert_success(result)
        assert value == "John"

    def test_extract_custom_separator_arrow(self) -> None:
        """Custom separator: arrow."""
        data = {"a": {"b": {"c": 100}}}
        result = FlextUtilities.extract(data, "a->b->c", separator="->")
        value = FlextTestsMatchers.assert_success(result)
        assert value == 100

    # -------------------------------------------------------------------------
    # Edge cases
    # -------------------------------------------------------------------------

    def test_extract_from_empty_dict(self) -> None:
        """Extract from empty dict returns default."""
        data: dict[str, FlextTypes.GeneralValueType] = {}
        result = FlextUtilities.extract(data, "any.path", default="default")
        value = FlextTestsMatchers.assert_success(result)
        assert value == "default"

    def test_extract_none_in_path(self) -> None:
        """Handle None value in path."""
        data: dict[str, FlextTypes.GeneralValueType] = {"a": None}
        result = FlextUtilities.extract(data, "a.b", default="fallback")
        value = FlextTestsMatchers.assert_success(result)
        assert value == "fallback"

    def test_extract_preserves_types(self) -> None:
        """Extract preserves various value types."""
        data = {
            "int_val": 42,
            "str_val": "hello",
            "bool_val": True,
            "list_val": [1, 2, 3],
        }
        assert (
            FlextTestsMatchers.assert_success(FlextUtilities.extract(data, "int_val"))
            == 42
        )
        assert (
            FlextTestsMatchers.assert_success(FlextUtilities.extract(data, "str_val"))
            == "hello"
        )
        assert (
            FlextTestsMatchers.assert_success(FlextUtilities.extract(data, "bool_val"))
            is True
        )
        assert FlextTestsMatchers.assert_success(
            FlextUtilities.extract(data, "list_val")
        ) == [1, 2, 3]


# =============================================================================
# TEST CLASS: generate() POWER METHOD
# =============================================================================


class TestGenerateMethod:
    """Tests for FlextUtilities.generate() power method."""

    # -------------------------------------------------------------------------
    # Basic generation by kind
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("description", "kind", "expected_prefix"),
        PowerMethodScenarios.GENERATE_KINDS,
    )
    def test_generate_kinds(
        self,
        description: str,
        kind: str,
        expected_prefix: str,
    ) -> None:
        """Test different ID generation kinds."""
        result = FlextUtilities.generate(kind)
        assert isinstance(result, str)
        if expected_prefix:
            assert result.startswith(f"{expected_prefix}_"), (
                f"{description}: expected prefix '{expected_prefix}_'"
            )

    def test_generate_default_id(self) -> None:
        """Default generation produces short ID."""
        result = FlextUtilities.generate()
        assert isinstance(result, str)
        assert len(result) == 8  # Default short ID length

    def test_generate_uuid(self) -> None:
        """UUID generation produces valid UUID format."""
        result = FlextUtilities.generate("uuid")
        assert isinstance(result, str)
        # UUID4 format: 8-4-4-4-12 hex chars
        assert len(result) == 36
        assert result.count("-") == 4

    def test_generate_correlation_id(self) -> None:
        """Correlation ID has 'corr' prefix."""
        result = FlextUtilities.generate("correlation")
        assert result.startswith("corr_")

    def test_generate_entity_id(self) -> None:
        """Entity ID has 'ent' prefix."""
        result = FlextUtilities.generate("entity")
        assert result.startswith("ent_")

    def test_generate_batch_id(self) -> None:
        """Batch ID has 'batch' prefix."""
        result = FlextUtilities.generate("batch")
        assert result.startswith("batch_")

    def test_generate_transaction_id(self) -> None:
        """Transaction ID has 'txn' prefix."""
        result = FlextUtilities.generate("transaction")
        assert result.startswith("txn_")

    def test_generate_event_id(self) -> None:
        """Event ID has 'evt' prefix."""
        result = FlextUtilities.generate("event")
        assert result.startswith("evt_")

    def test_generate_command_id(self) -> None:
        """Command ID has 'cmd' prefix."""
        result = FlextUtilities.generate("command")
        assert result.startswith("cmd_")

    def test_generate_query_id(self) -> None:
        """Query ID has 'qry' prefix."""
        result = FlextUtilities.generate("query")
        assert result.startswith("qry_")

    # -------------------------------------------------------------------------
    # Custom prefix
    # -------------------------------------------------------------------------

    def test_generate_custom_prefix(self) -> None:
        """Custom prefix overrides default."""
        result = FlextUtilities.generate("entity", prefix="user")
        assert result.startswith("user_")

    def test_generate_custom_prefix_empty(self) -> None:
        """Empty custom prefix produces no prefix."""
        result = FlextUtilities.generate("entity", prefix="")
        # With empty prefix, should just be the ID
        assert "_" not in result or not result.startswith("ent_")

    # -------------------------------------------------------------------------
    # Custom length
    # -------------------------------------------------------------------------

    def test_generate_custom_length_short(self) -> None:
        """Custom short length."""
        result = FlextUtilities.generate("id", length=4)
        assert len(result) == 4

    def test_generate_custom_length_long(self) -> None:
        """Custom long length."""
        result = FlextUtilities.generate("id", length=16)
        assert len(result) == 16

    def test_generate_with_prefix_and_length(self) -> None:
        """Prefix + custom length."""
        result = FlextUtilities.generate("entity", length=6)
        parts = result.split("_")
        assert parts[0] == "ent"
        assert len(parts[1]) == 6

    # -------------------------------------------------------------------------
    # Timestamp option
    # -------------------------------------------------------------------------

    def test_generate_with_timestamp(self) -> None:
        """Include timestamp in ID."""
        result = FlextUtilities.generate("batch", include_timestamp=True)
        parts = result.split("_")
        assert parts[0] == "batch"
        # Should have timestamp component
        assert len(parts) >= 2

    # -------------------------------------------------------------------------
    # Custom separator
    # -------------------------------------------------------------------------

    def test_generate_custom_separator_dash(self) -> None:
        """Use dash separator."""
        result = FlextUtilities.generate("entity", separator="-")
        assert "-" in result
        assert result.startswith("ent-")

    def test_generate_custom_separator_dot(self) -> None:
        """Use dot separator."""
        result = FlextUtilities.generate("entity", separator=".")
        assert "." in result
        assert result.startswith("ent.")

    # -------------------------------------------------------------------------
    # Uniqueness tests
    # -------------------------------------------------------------------------

    def test_generate_uniqueness(self) -> None:
        """Generated IDs are unique."""
        ids = [FlextUtilities.generate() for _ in range(100)]
        assert len(ids) == len(set(ids))  # All unique

    def test_generate_uuid_uniqueness(self) -> None:
        """Generated UUIDs are unique."""
        uuids = [FlextUtilities.generate("uuid") for _ in range(100)]
        assert len(uuids) == len(set(uuids))

    def test_generate_entity_uniqueness(self) -> None:
        """Generated entity IDs are unique."""
        entity_ids = [FlextUtilities.generate("entity") for _ in range(100)]
        assert len(entity_ids) == len(set(entity_ids))


class TestBatchMethod:
    """Tests for FlextUtilities.batch() power method."""

    # =========================================================================
    # Basic batch processing
    # =========================================================================

    def test_batch_empty_items_returns_empty_results(self) -> None:
        """Batch processing empty items returns empty results.

        Business Rule: batch() returns a dict with keys 'results', 'errors',
        'total', 'success_count', 'error_count'.
        """
        # Business Rule: batch() processes items with operation function
        # Empty list returns empty results dict
        result = FlextUtilities.batch(
            [], lambda x: int(x) * 2 if isinstance(x, (int, float)) else 0
        )
        FlextTestsMatchers.assert_success(result)
        batch_result = result.unwrap()
        assert batch_result["results"] == []
        assert batch_result["total"] == 0

    def test_batch_single_item(self) -> None:
        """Batch processing single item works correctly."""
        result = FlextUtilities.batch([5], lambda x: x * 2)
        FlextTestsMatchers.assert_success(result)
        batch_result = result.unwrap()
        assert batch_result["results"] == [10]
        assert batch_result["total"] == 1
        assert batch_result["success_count"] == 1
        assert batch_result["error_count"] == 0

    def test_batch_multiple_items(self) -> None:
        """Batch processing multiple items transforms all."""
        result = FlextUtilities.batch([1, 2, 3, 4, 5], lambda x: x * 2)
        FlextTestsMatchers.assert_success(result)
        batch_result = result.unwrap()
        assert batch_result["results"] == [2, 4, 6, 8, 10]
        assert batch_result["total"] == 5
        assert batch_result["success_count"] == 5

    def test_batch_with_custom_size(self) -> None:
        """Batch processing with custom chunk size.

        Note: size= parameter is reserved (_size) but not yet implemented.
        Test uses default chunking behavior.
        """
        items = list(range(10))
        result = FlextUtilities.batch(items, lambda x: x + 1)
        FlextTestsMatchers.assert_success(result)
        batch_result = result.unwrap()
        assert batch_result["results"] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert batch_result["total"] == 10

    def test_batch_with_index_operation(self) -> None:
        """Batch operation receives item only (no index).

        Note: Index-based operations are not implemented.
        This test validates basic string transformation.
        """
        result = FlextUtilities.batch(
            ["a", "b", "c"],
            lambda item: item.upper(),
        )
        FlextTestsMatchers.assert_success(result)
        batch_result = result.unwrap()
        assert batch_result["results"] == ["A", "B", "C"]

    # =========================================================================
    # Error handling modes
    # =========================================================================

    def test_batch_on_error_collect_continues(self) -> None:
        """On error collect mode continues and collects errors."""

        def operation(x: int) -> int:
            if x == 3:
                msg = "Error on 3"
                raise ValueError(msg)
            return x * 2

        result = FlextUtilities.batch([1, 2, 3, 4, 5], operation, on_error="collect")
        FlextTestsMatchers.assert_success(result)
        batch_result = result.unwrap()
        assert batch_result["results"] == [2, 4, 8, 10]  # 3 is skipped
        assert batch_result["error_count"] == 1
        errors = batch_result["errors"]
        assert len(errors) == 1
        assert errors[0][0] == 2  # Index of failed item

    def test_batch_on_error_skip_continues(self) -> None:
        """On error skip mode continues processing without collecting errors.

        Note: skip_count not tracked in current implementation.
        Skipped items don't appear in results or errors.
        """

        def operation(x: int) -> int:
            if x % 2 == 0:
                msg = "Even numbers fail"
                raise ValueError(msg)
            return x

        result = FlextUtilities.batch([1, 2, 3, 4, 5], operation, on_error="skip")
        FlextTestsMatchers.assert_success(result)
        batch_result = result.unwrap()
        assert batch_result["results"] == [1, 3, 5]
        # Skip mode doesn't collect errors, so error_count is 0
        assert batch_result["error_count"] == 0

    def test_batch_on_error_fail_stops(self) -> None:
        """On error fail mode stops immediately and returns failure.

        Business Rule: on_error='fail' returns FlextResult.fail() on first error.
        No partial results are returned - the entire batch fails.
        """

        def operation(x: int) -> int:
            if x == 3:
                msg = "Error on 3"
                raise ValueError(msg)
            return x * 2

        result = FlextUtilities.batch([1, 2, 3, 4, 5], operation, on_error="fail")
        # fail mode returns failure result, not success
        assert result.is_failure
        assert result.error is not None
        assert "Item 2 failed" in result.error  # Index 2 is value 3

    # =========================================================================
    # Progress tracking (NOT YET IMPLEMENTED)
    # =========================================================================

    def test_batch_progress_callback_called(self) -> None:
        """Progress callback is called during batch processing."""
        progress_calls: list[tuple[int, int]] = []

        def progress_callback(current: int, total: int) -> None:
            """Track progress callback invocations."""
            progress_calls.append((current, total))

        items = [1, 2, 3, 4, 5]
        result = FlextUtilities.batch(
            items,
            lambda x: x * 2,
            progress=progress_callback,
        )
        FlextTestsMatchers.assert_success(result)
        
        # Verify callback was called: start (0, 5) and end (5, 5)
        assert len(progress_calls) >= 2
        assert progress_calls[0] == (0, 5)  # Start
        assert progress_calls[-1] == (5, 5)  # End
        
        # Verify intermediate calls (one per item with default interval=1)
        assert (1, 5) in progress_calls
        assert (2, 5) in progress_calls
        assert (3, 5) in progress_calls
        assert (4, 5) in progress_calls

    def test_batch_progress_interval_respected(self) -> None:
        """Progress callback respects progress_interval parameter."""
        progress_calls: list[tuple[int, int]] = []

        def progress_callback(current: int, total: int) -> None:
            """Track progress callback invocations."""
            progress_calls.append((current, total))

        items = list(range(10))  # 0-9
        result = FlextUtilities.batch(
            items,
            lambda x: x * 2,
            progress=progress_callback,
            progress_interval=3,  # Call every 3 items
        )
        FlextTestsMatchers.assert_success(result)
        
        # Should be called at: 0, 3, 6, 9, 10 (start + intervals + end)
        assert progress_calls[0] == (0, 10)  # Start
        assert (3, 10) in progress_calls
        assert (6, 10) in progress_calls
        assert (9, 10) in progress_calls
        assert progress_calls[-1] == (10, 10)  # End
        
        # Should NOT be called at 1, 2, 4, 5, 7, 8 (not multiples of 3)
        assert (1, 10) not in progress_calls
        assert (2, 10) not in progress_calls
        assert (4, 10) not in progress_calls

    # =========================================================================
    # Validation
    # =========================================================================

    def test_batch_pre_validate_filters_items(self) -> None:
        """Pre-validation filters items before processing."""
        processed_items: list[int] = []

        def operation(x: int) -> int:
            """Track which items were processed."""
            processed_items.append(x)
            return x * 2

        def pre_validator(x: int) -> bool:
            """Only process even numbers."""
            return x % 2 == 0

        items = [1, 2, 3, 4, 5, 6]
        result = FlextUtilities.batch(
            items,
            operation,
            pre_validate=pre_validator,
        )
        FlextTestsMatchers.assert_success(result)
        batch_result = result.unwrap()
        
        # Only even numbers should be processed
        assert processed_items == [2, 4, 6]
        assert batch_result["results"] == [4, 8, 12]
        assert batch_result["total"] == 6  # Total items (before filtering)
        assert batch_result["success_count"] == 3  # Processed items

    def test_batch_pre_validate_with_all_filtered(self) -> None:
        """Pre-validation handles case where all items are filtered."""
        def operation(x: int) -> int:
            """Operation that never runs."""
            return x * 2

        def pre_validator(_x: int) -> bool:
            """Filter out all items."""
            return False

        items = [1, 2, 3]
        result = FlextUtilities.batch(
            items,
            operation,
            pre_validate=pre_validator,
        )
        FlextTestsMatchers.assert_success(result)
        batch_result = result.unwrap()
        
        # No items processed, but total still reflects original count
        assert batch_result["results"] == []
        assert batch_result["total"] == 3
        assert batch_result["success_count"] == 0

    def test_batch_post_validate_filters_results(self) -> None:
        """Post-validation filters results after processing."""
        def operation(x: int) -> int:
            """Return processed value."""
            return x * 2

        def post_validator(result: int) -> bool:
            """Only keep results >= 6."""
            return result >= 6

        items = [1, 2, 3, 4, 5]
        result = FlextUtilities.batch(
            items,
            operation,
            post_validate=post_validator,
        )
        FlextTestsMatchers.assert_success(result)
        batch_result = result.unwrap()
        
        # Results 2, 4 filtered out (only 6, 8, 10 kept)
        assert batch_result["results"] == [6, 8, 10]
        assert batch_result["total"] == 5
        assert batch_result["success_count"] == 3  # After filtering

    def test_batch_post_validate_with_flextresult(self) -> None:
        """Post-validation works with FlextResult return types."""
        def operation(x: int) -> FlextResult[int]:
            """Return FlextResult."""
            return FlextResult[int].ok(x * 2)

        def post_validator(result: int) -> bool:
            """Only keep results >= 6."""
            return result >= 6

        items = [1, 2, 3, 4, 5]
        result = FlextUtilities.batch(
            items,
            operation,
            post_validate=post_validator,
        )
        FlextTestsMatchers.assert_success(result)
        batch_result = result.unwrap()
        
        # Results 2, 4 filtered out (only 6, 8, 10 kept)
        assert batch_result["results"] == [6, 8, 10]
        assert batch_result["success_count"] == 3

    # =========================================================================
    # BatchResult properties - derived from dict
    # =========================================================================

    def test_batch_result_all_succeeded_true(self) -> None:
        """all_succeeded is True when no errors (derived from error_count==0)."""
        result = FlextUtilities.batch([1, 2, 3], lambda x: x)
        FlextTestsMatchers.assert_success(result)
        batch_result = result.unwrap()
        # all_succeeded can be computed from error_count
        assert batch_result["error_count"] == 0

    def test_batch_result_all_succeeded_false(self) -> None:
        """all_succeeded is False when errors exist (derived from error_count>0)."""

        def operation(x: int) -> int:
            if x == 2:
                msg = "Fail"
                raise ValueError(msg)
            return x

        result = FlextUtilities.batch([1, 2, 3], operation, on_error="collect")
        FlextTestsMatchers.assert_success(result)
        batch_result = result.unwrap()
        # all_succeeded would be False when error_count > 0
        error_count = batch_result["error_count"]
        assert error_count > 0

    def test_batch_result_get_result_by_index(self) -> None:
        """Can get result by list index from results list."""
        result = FlextUtilities.batch([10, 20, 30], lambda x: x + 1)
        FlextTestsMatchers.assert_success(result)
        batch_result = result.unwrap()
        # Results are accessed by list indexing
        results = cast("list[int]", batch_result["results"])
        assert results[0] == 11
        assert results[1] == 21
        assert results[2] == 31

    def test_batch_result_get_error_by_index(self) -> None:
        """Can get error from errors list by tuple index."""

        def operation(x: int) -> int:
            if x == 20:
                msg = "Error on 20"
                raise ValueError(msg)
            return x

        result = FlextUtilities.batch([10, 20, 30], operation, on_error="collect")
        FlextTestsMatchers.assert_success(result)
        batch_result = result.unwrap()
        # Errors are tuples of (index, error_message)
        errors = batch_result["errors"]
        assert len(errors) == 1
        assert errors[0][0] == 1  # Index of failed item
        assert "20" in errors[0][1]  # Error message

    # =========================================================================
    # Edge cases
    # =========================================================================

    def test_batch_with_none_in_items(self) -> None:
        """Batch handles None values in items."""
        result = FlextUtilities.batch(
            [1, None, 3],
            lambda x: x or 0,
        )
        FlextTestsMatchers.assert_success(result)
        batch_result = result.unwrap()
        assert batch_result["results"] == [1, 0, 3]

    def test_batch_string_items(self) -> None:
        """Batch processes string items."""
        result = FlextUtilities.batch(
            ["hello", "world"],
            lambda s: s.upper(),
        )
        FlextTestsMatchers.assert_success(result)
        batch_result = result.unwrap()
        assert batch_result["results"] == ["HELLO", "WORLD"]

    def test_batch_dict_items(self) -> None:
        """Batch processes dict items."""
        items = [{"a": 1}, {"a": 2}, {"a": 3}]
        result = FlextUtilities.batch(
            items,
            lambda d: d["a"] * 2,
        )
        FlextTestsMatchers.assert_success(result)
        batch_result = result.unwrap()
        assert batch_result["results"] == [2, 4, 6]

    def test_batch_flatten_results(self) -> None:
        """Flatten parameter flattens nested lists in results."""
        def operation(x: int) -> list[int]:
            """Return list of values."""
            return [x, x * 2, x * 3]

        items = [1, 2]
        result = FlextUtilities.batch(
            items,
            operation,
            flatten=True,
        )
        FlextTestsMatchers.assert_success(result)
        batch_result = result.unwrap()
        
        # Results should be flattened: [1,2,3,2,4,6] instead of [[1,2,3],[2,4,6]]
        assert batch_result["results"] == [1, 2, 3, 2, 4, 6]
        assert batch_result["total"] == 2
        assert batch_result["success_count"] == 6  # Flattened count

    def test_batch_flatten_with_mixed_types(self) -> None:
        """Flatten handles mixed list and non-list results."""
        def operation(x: int) -> list[int] | int:
            """Return list for even, int for odd."""
            if x % 2 == 0:
                return [x, x * 2]
            return x * 3

        items = [1, 2, 3]
        result = FlextUtilities.batch(
            items,
            operation,
            flatten=True,
        )
        FlextTestsMatchers.assert_success(result)
        batch_result = result.unwrap()
        
        # Results: [3] (from 1), [2, 4] (from 2), [9] (from 3) -> [3, 2, 4, 9]
        assert batch_result["results"] == [3, 2, 4, 9]
        assert batch_result["success_count"] == 4

    def test_batch_flatten_with_tuples(self) -> None:
        """Flatten handles tuples as well as lists."""
        def operation(x: int) -> tuple[int, int]:
            """Return tuple."""
            return (x, x * 2)

        items = [1, 2]
        result = FlextUtilities.batch(
            items,
            operation,
            flatten=True,
        )
        FlextTestsMatchers.assert_success(result)
        batch_result = result.unwrap()
        
        # Tuples should be flattened too
        assert batch_result["results"] == [1, 2, 2, 4]
        assert batch_result["success_count"] == 4

    def test_batch_large_items_chunked(self) -> None:
        """Batch handles large item count.

        Note: size= parameter is reserved but not implemented for chunking.
        This test validates processing many items without chunking.
        """
        items = list(range(100))
        result = FlextUtilities.batch(items, lambda x: x + 1)
        FlextTestsMatchers.assert_success(result)
        batch_result = result.unwrap()
        assert batch_result["total"] == 100
        assert batch_result["success_count"] == 100
        assert batch_result["results"] == list(range(1, 101))
