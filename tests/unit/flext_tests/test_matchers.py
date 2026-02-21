"""Unit tests for flext_tests.matchers module.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import warnings

import pytest

from flext_core import FlextResult, t
from flext_tests import tm
from flext_tests.constants import c


class TestFlextTestsMatchers:
    """Test suite for FlextTestsMatchers class."""

    def test_assert_result_success_passes(self) -> None:
        """Test tm.ok() with successful result."""
        result = FlextResult[str].ok("success")

        # Should not raise
        value = tm.ok(result)
        assert value == "success"

    def test_assert_result_success_fails(self) -> None:
        """Test tm.ok() with failed result."""
        result = FlextResult[str].fail("error")

        with pytest.raises(AssertionError, match="Expected success but got failure"):
            tm.ok(result)

    def test_assert_result_success_custom_message(self) -> None:
        """Test tm.ok() with custom error message."""
        result = FlextResult[str].fail("error")

        with pytest.raises(AssertionError, match="Custom message"):
            tm.ok(result, msg="Custom message")

    def test_assert_result_failure_passes(self) -> None:
        """Test tm.fail() with failed result."""
        result = FlextResult[str].fail("error")

        # Should not raise
        error = tm.fail(result)
        assert error == "error"

    def test_assert_result_failure_fails(self) -> None:
        """Test tm.fail() with successful result."""
        result = FlextResult[str].ok("success")

        with pytest.raises(AssertionError, match="Expected failure but got success"):
            tm.fail(result)

    def test_assert_result_failure_with_expected_error(self) -> None:
        """Test tm.fail() with expected error substring."""
        result = FlextResult[str].fail("Database connection failed")

        # Should not raise
        error = tm.fail(result, contains="connection")
        assert "connection" in error

    def test_assert_result_failure_expected_error_not_found(self) -> None:
        """Test tm.fail() when expected error substring not found."""
        result = FlextResult[str].fail("Database error")

        with pytest.raises(
            AssertionError,
            match=r"Expected.*to contain 'connection'",
        ):
            tm.fail(result, contains="connection")

    def test_assert_dict_contains_passes(self) -> None:
        """Test tm.that() with contains parameter for dict."""
        data = {"key1": "value1", "key2": "value2"}
        expected = {"key1": "value1"}

        # Should not raise - use kv for key-value pairs
        tm.that(data, kv=expected)

    def test_assert_dict_contains_missing_key(self) -> None:
        """Test tm.that() with missing key."""
        data = {"key1": "value1"}
        expected = {"key2": "value2"}

        with pytest.raises(AssertionError, match="Key 'key2' not found in mapping"):
            tm.that(data, kv=expected)

    def test_assert_dict_contains_wrong_value(self) -> None:
        """Test tm.that() with wrong value."""
        data = {"key1": "value1"}
        expected = {"key1": "wrong_value"}

        with pytest.raises(
            AssertionError,
            match="expected 'wrong_value', got 'value1'",
        ):
            tm.that(data, kv=expected)

    def test_assert_list_contains_passes(self) -> None:
        """Test tm.that() with has parameter for list."""
        items = ["item1", "item2", "item3"]

        # Should not raise
        tm.that(items, has="item2")

    def test_assert_list_contains_missing_item(self) -> None:
        """Test tm.that() with item not in list."""
        items = ["item1", "item2"]

        with pytest.raises(AssertionError, match=r"Expected.*to contain 'item3'"):
            tm.that(items, has="item3")

    def test_assert_valid_email_passes(self) -> None:
        """Test tm.that() with email pattern match."""
        # Should not raise
        tm.that("test@example.com", match=c.Tests.Matcher.EMAIL_PATTERN)

    def test_assert_valid_email_fails(self) -> None:
        """Test tm.that() with invalid email."""
        with pytest.raises(AssertionError, match="Assertion failed"):
            tm.that("invalid-email", match=c.Tests.Matcher.EMAIL_PATTERN)

    def test_assert_valid_email_edge_cases(self) -> None:
        """Test tm.that() with various email edge cases."""
        valid_emails = [
            "user.name@domain.co.uk",
            "test+tag@example.com",
            "a@b.co",
        ]
        invalid_emails = [
            "invalid",
            "@example.com",
            "test@",
            "test.example.com",  # Missing @
        ]

        for email in valid_emails:
            # Should not raise
            tm.that(email, match=c.Tests.Matcher.EMAIL_PATTERN)

        for email in invalid_emails:
            with pytest.raises(AssertionError):
                tm.that(email, match=c.Tests.Matcher.EMAIL_PATTERN)

    def test_assert_config_valid_passes(self) -> None:
        """Test tm.that() with keys parameter for config validation."""
        config: dict[str, t.GeneralValueType] = {
            "service_type": "api",
            "environment": "test",
            "timeout": 30,
        }

        # Should not raise - validate keys and timeout
        tm.that(config, keys=["service_type", "environment", "timeout"])
        tm.that(config["timeout"], is_=int, gt=0)

    def test_assert_config_valid_missing_required_key(self) -> None:
        """Test tm.that() with missing required key."""
        config = {"service_type": "api"}  # Missing environment

        with pytest.raises(AssertionError, match="Missing required keys"):
            tm.that(config, keys=["service_type", "environment", "timeout"])

    def test_assert_config_valid_invalid_timeout(self) -> None:
        """Test tm.that() with invalid timeout type."""
        config = {
            "service_type": "api",
            "environment": "test",
            "timeout": "invalid",  # Should be positive int
        }

        with pytest.raises(AssertionError, match="Assertion failed"):
            tm.that(config["timeout"], is_=int, gt=0)

    def test_assert_config_valid_zero_timeout(self) -> None:
        """Test tm.that() with zero timeout."""
        config: dict[str, t.GeneralValueType] = {
            "service_type": "api",
            "environment": "test",
            "timeout": 0,  # Should be positive
        }

        with pytest.raises(AssertionError, match="Assertion failed"):
            tm.that(config["timeout"], is_=int, gt=0)

    # =========================================================================
    # COMPREHENSIVE TESTS FOR ENHANCED tm.ok()
    # =========================================================================

    def test_ok_with_eq_parameter(self) -> None:
        """Test tm.ok() with eq parameter."""
        result = FlextResult[int].ok(42)
        value = tm.ok(result, eq=42)
        assert value == 42

    def test_ok_with_eq_parameter_fails(self) -> None:
        """Test tm.ok() with eq parameter fails when value doesn't match."""
        result = FlextResult[int].ok(42)
        with pytest.raises(AssertionError):
            tm.ok(result, eq=43)

    def test_ok_with_ne_parameter(self) -> None:
        """Test tm.ok() with ne parameter."""
        result = FlextResult[int].ok(42)
        value = tm.ok(result, ne=43)
        assert value == 42

    def test_ok_with_is_parameter(self) -> None:
        """Test tm.ok() with is_ parameter."""
        result = FlextResult[str].ok("test")
        value = tm.ok(result, is_=str)
        assert value == "test"

    def test_ok_with_is_tuple_parameter(self) -> None:
        """Test tm.ok() with is_ tuple parameter."""
        result = FlextResult[str].ok("test")
        value = tm.ok(result, is_=(str, bytes))
        assert value == "test"

    def test_ok_with_has_parameter(self) -> None:
        """Test tm.ok() with has parameter."""
        result = FlextResult[list[str]].ok(["a", "b", "c"])
        value = tm.ok(result, has="a")
        assert value == ["a", "b", "c"]

    def test_ok_with_has_sequence_parameter(self) -> None:
        """Test tm.ok() with has sequence parameter."""
        result = FlextResult[list[str]].ok(["a", "b", "c"])
        value = tm.ok(result, has=["a", "b"])
        assert value == ["a", "b", "c"]

    def test_ok_with_lacks_parameter(self) -> None:
        """Test tm.ok() with lacks parameter."""
        result = FlextResult[list[str]].ok(["a", "b", "c"])
        value = tm.ok(result, lacks="d")
        assert value == ["a", "b", "c"]

    def test_ok_with_len_exact_parameter(self) -> None:
        """Test tm.ok() with len exact parameter."""
        result = FlextResult[list[str]].ok(["a", "b", "c"])
        value = tm.ok(result, len=3)
        assert value == ["a", "b", "c"]

    def test_ok_with_len_range_parameter(self) -> None:
        """Test tm.ok() with len range parameter."""
        result = FlextResult[list[str]].ok(["a", "b", "c"])
        value = tm.ok(result, len=(2, 4))
        assert value == ["a", "b", "c"]

    def test_ok_with_deep_parameter(self) -> None:
        """Test tm.ok() with deep parameter."""
        data: dict[str, t.GeneralValueType] = {"user": {"name": "John", "age": 30}}
        result = FlextResult[dict[str, t.GeneralValueType]].ok(data)
        value = tm.ok(result, deep={"user.name": "John"})
        assert value == data

    def test_ok_with_deep_predicate_parameter(self) -> None:
        """Test tm.ok() with deep predicate parameter."""
        data: dict[str, t.GeneralValueType] = {"user": {"email": "test@example.com"}}
        result = FlextResult[dict[str, t.GeneralValueType]].ok(data)
        value = tm.ok(result, deep={"user.email": lambda e: "@" in str(e)})
        assert value == data

    def test_ok_with_path_parameter(self) -> None:
        """Test tm.ok() with path parameter."""
        data: dict[str, t.GeneralValueType] = {"user": {"name": "John"}}
        result = FlextResult[dict[str, t.GeneralValueType]].ok(data)
        value = tm.ok(result, path="user.name", eq="John")
        # path extraction returns the extracted value, not the original
        assert value == "John"

    def test_ok_with_where_parameter(self) -> None:
        """Test tm.ok() with where parameter."""
        result = FlextResult[int].ok(42)
        value = tm.ok(result, where=lambda x: x > 0)
        assert value == 42

    def test_ok_with_where_parameter_fails(self) -> None:
        """Test tm.ok() with where parameter fails when predicate returns False."""
        result = FlextResult[int].ok(42)
        with pytest.raises(AssertionError):
            tm.ok(result, where=lambda x: x < 0)

    def test_ok_with_starts_parameter(self) -> None:
        """Test tm.ok() with starts parameter."""
        result = FlextResult[str].ok("Hello World")
        value = tm.ok(result, starts="Hello")
        assert value == "Hello World"

    def test_ok_with_ends_parameter(self) -> None:
        """Test tm.ok() with ends parameter."""
        result = FlextResult[str].ok("Hello World")
        value = tm.ok(result, ends="World")
        assert value == "Hello World"

    def test_ok_with_match_parameter(self) -> None:
        """Test tm.ok() with match parameter."""
        result = FlextResult[str].ok("test@example.com")
        value = tm.ok(result, match=r"^[\w.]+@[\w.]+$")
        assert value == "test@example.com"

    def test_ok_with_gt_parameter(self) -> None:
        """Test tm.ok() with gt parameter."""
        result = FlextResult[int].ok(42)
        value = tm.ok(result, gt=0)
        assert value == 42

    def test_ok_with_gte_parameter(self) -> None:
        """Test tm.ok() with gte parameter."""
        result = FlextResult[int].ok(42)
        value = tm.ok(result, gte=42)
        assert value == 42

    def test_ok_with_lt_parameter(self) -> None:
        """Test tm.ok() with lt parameter."""
        result = FlextResult[int].ok(42)
        value = tm.ok(result, lt=100)
        assert value == 42

    def test_ok_with_lte_parameter(self) -> None:
        """Test tm.ok() with lte parameter."""
        result = FlextResult[int].ok(42)
        value = tm.ok(result, lte=42)
        assert value == 42

    def test_ok_with_none_parameter(self) -> None:
        """Test tm.ok() with none parameter."""
        result = FlextResult[str | None].ok("test")
        value = tm.ok(result, none=False)
        assert value == "test"

    def test_ok_with_empty_parameter(self) -> None:
        """Test tm.ok() with empty parameter."""
        result = FlextResult[list[str]].ok(["a"])
        value = tm.ok(result, empty=False)
        assert value == ["a"]

    # =========================================================================
    # COMPREHENSIVE TESTS FOR ENHANCED tm.fail()
    # =========================================================================

    def test_fail_with_has_parameter(self) -> None:
        """Test tm.fail() with has parameter."""
        result = FlextResult[str].fail("Database connection failed")
        error = tm.fail(result, has="connection")
        assert error == "Database connection failed"

    def test_fail_with_has_sequence_parameter(self) -> None:
        """Test tm.fail() with has sequence parameter."""
        result = FlextResult[str].fail("Database connection failed")
        error = tm.fail(result, has=["Database", "connection"])
        assert error == "Database connection failed"

    def test_fail_with_lacks_parameter(self) -> None:
        """Test tm.fail() with lacks parameter."""
        result = FlextResult[str].fail("Database error")
        error = tm.fail(result, lacks="internal")
        assert error == "Database error"

    def test_fail_with_starts_parameter(self) -> None:
        """Test tm.fail() with starts parameter."""
        result = FlextResult[str].fail("Error: connection failed")
        error = tm.fail(result, starts="Error:")
        assert error == "Error: connection failed"

    def test_fail_with_ends_parameter(self) -> None:
        """Test tm.fail() with ends parameter."""
        result = FlextResult[str].fail("connection failed")
        error = tm.fail(result, ends="failed")
        assert error == "connection failed"

    def test_fail_with_match_parameter(self) -> None:
        """Test tm.fail() with match parameter."""
        result = FlextResult[str].fail("Error: 404")
        error = tm.fail(result, match=r"Error: \d+")
        assert error == "Error: 404"

    def test_fail_with_code_parameter(self) -> None:
        """Test tm.fail() with code parameter."""
        result = FlextResult[str].fail("error", error_code="VALIDATION")
        error = tm.fail(result, code="VALIDATION")
        assert error == "error"

    def test_fail_with_code_has_parameter(self) -> None:
        """Test tm.fail() with code_has parameter."""
        result = FlextResult[str].fail("error", error_code="VALIDATION_ERROR")
        error = tm.fail(result, code_has="VALIDATION")
        assert error == "error"

    def test_fail_with_data_parameter(self) -> None:
        """Test tm.fail() with data parameter."""
        result = FlextResult[str].fail("error", error_data={"field": "email"})
        error = tm.fail(result, data={"field": "email"})
        assert error == "error"

    # =========================================================================
    # COMPREHENSIVE TESTS FOR ENHANCED tm.that()
    # =========================================================================

    def test_that_with_eq_parameter(self) -> None:
        """Test tm.that() with eq parameter."""
        tm.that(42, eq=42)

    def test_that_with_ne_parameter(self) -> None:
        """Test tm.that() with ne parameter."""
        tm.that(42, ne=43)

    def test_that_with_is_parameter(self) -> None:
        """Test tm.that() with is_ parameter."""
        tm.that("test", is_=str)

    def test_that_with_is_tuple_parameter(self) -> None:
        """Test tm.that() with is_ tuple parameter."""
        tm.that("test", is_=(str, bytes))

    def test_that_with_not_parameter(self) -> None:
        """Test tm.that() with not_ parameter."""
        tm.that("test", not_=int)

    def test_that_with_none_parameter(self) -> None:
        """Test tm.that() with none parameter."""
        tm.that("test", none=False)
        tm.that(None, none=True)

    def test_that_with_empty_parameter(self) -> None:
        """Test tm.that() with empty parameter."""
        tm.that(["a"], empty=False)
        tm.that([], empty=True)

    def test_that_with_has_parameter(self) -> None:
        """Test tm.that() with has parameter."""
        tm.that(["a", "b", "c"], has="a")

    def test_that_with_has_sequence_parameter(self) -> None:
        """Test tm.that() with has sequence parameter."""
        tm.that(["a", "b", "c"], has=["a", "b"])

    def test_that_with_lacks_parameter(self) -> None:
        """Test tm.that() with lacks parameter."""
        tm.that(["a", "b", "c"], lacks="d")

    def test_that_with_first_parameter(self) -> None:
        """Test tm.that() with first parameter."""
        tm.that(["a", "b", "c"], first="a")

    def test_that_with_last_parameter(self) -> None:
        """Test tm.that() with last parameter."""
        tm.that(["a", "b", "c"], last="c")

    def test_that_with_all_type_parameter(self) -> None:
        """Test tm.that() with all_ type parameter."""
        tm.that(["a", "b", "c"], all_=str)

    def test_that_with_all_predicate_parameter(self) -> None:
        """Test tm.that() with all_ predicate parameter."""
        tm.that([1, 2, 3], all_=lambda x: x > 0)

    def test_that_with_any_type_parameter(self) -> None:
        """Test tm.that() with any_ type parameter."""
        tm.that(["a", 1, "c"], any_=int)

    def test_that_with_any_predicate_parameter(self) -> None:
        """Test tm.that() with any_ predicate parameter."""
        tm.that([1, 2, 3], any_=lambda x: x > 2)

    def test_that_with_sorted_parameter(self) -> None:
        """Test tm.that() with sorted parameter."""
        tm.that([1, 2, 3], sorted=True)

    def test_that_with_unique_parameter(self) -> None:
        """Test tm.that() with unique parameter."""
        tm.that([1, 2, 3], unique=True)

    def test_that_with_keys_parameter(self) -> None:
        """Test tm.that() with keys parameter."""
        tm.that({"a": 1, "b": 2}, keys=["a", "b"])

    def test_that_with_lacks_keys_parameter(self) -> None:
        """Test tm.that() with lacks_keys parameter."""
        tm.that({"a": 1}, lacks_keys=["b"])

    def test_that_with_values_parameter(self) -> None:
        """Test tm.that() with values parameter."""
        tm.that({"a": 1, "b": 2}, values=[1, 2])

    def test_that_with_kv_tuple_parameter(self) -> None:
        """Test tm.that() with kv tuple parameter."""
        tm.that({"a": 1}, kv=("a", 1))

    def test_that_with_kv_mapping_parameter(self) -> None:
        """Test tm.that() with kv mapping parameter."""
        tm.that({"a": 1, "b": 2}, kv={"a": 1, "b": 2})

    def test_that_with_attrs_parameter(self) -> None:
        """Test tm.that() with attrs parameter."""

        class TestClass:
            def __init__(self) -> None:
                self.attr1 = "value1"
                self.attr2 = "value2"

        obj = TestClass()
        tm.that(obj, attrs=["attr1", "attr2"])

    def test_that_with_methods_parameter(self) -> None:
        """Test tm.that() with methods parameter."""

        class TestClass:
            def method1(self) -> None:
                pass

            def method2(self) -> None:
                pass

        obj = TestClass()
        tm.that(obj, methods=["method1", "method2"])

    def test_that_with_attr_eq_tuple_parameter(self) -> None:
        """Test tm.that() with attr_eq tuple parameter."""

        class TestClass:
            def __init__(self) -> None:
                self.attr = "value"

        obj = TestClass()
        tm.that(obj, attr_eq=("attr", "value"))

    def test_that_with_attr_eq_mapping_parameter(self) -> None:
        """Test tm.that() with attr_eq mapping parameter."""

        class TestClass:
            def __init__(self) -> None:
                self.attr1 = "value1"
                self.attr2 = "value2"

        obj = TestClass()
        tm.that(obj, attr_eq={"attr1": "value1", "attr2": "value2"})

    def test_that_with_ok_parameter(self) -> None:
        """Test tm.that() with ok parameter for FlextResult."""
        result = FlextResult[str].ok("success")
        tm.that(result, ok=True)

    def test_that_with_error_parameter(self) -> None:
        """Test tm.that() with error parameter for FlextResult."""
        result = FlextResult[str].fail("error")
        tm.that(result, error="error")

    def test_that_with_deep_parameter(self) -> None:
        """Test tm.that() with deep parameter."""
        data = {"user": {"name": "John", "age": 30}}
        tm.that(data, deep={"user.name": "John"})

    def test_that_with_where_parameter(self) -> None:
        """Test tm.that() with where parameter."""
        tm.that(42, where=lambda x: x > 0)

    def test_that_with_all_alias_parameter(self) -> None:
        """Test tm.that() with all alias parameter (accepts both all_ and all)."""
        tm.that(["a", "b", "c"], all=str)  # Using 'all' instead of 'all_'

    def test_that_with_any_alias_parameter(self) -> None:
        """Test tm.that() with any alias parameter (accepts both any_ and any)."""
        tm.that(["a", 1, "c"], any=int)  # Using 'any' instead of 'any_'

    # =========================================================================
    # TESTS FOR tm.check() FLUENT API
    # =========================================================================

    def test_check_returns_chain(self) -> None:
        """Test tm.check() returns Chain object."""
        result = FlextResult[int].ok(42)
        chain = tm.check(result)
        assert chain is not None
        assert hasattr(chain, "result")

    # =========================================================================
    # TESTS FOR tm.scope() CONTEXT MANAGER
    # =========================================================================

    def test_scope_basic_usage(self) -> None:
        """Test tm.scope() basic usage."""
        with tm.scope() as scope:
            assert scope is not None
            assert hasattr(scope, "config")
            assert hasattr(scope, "container")
            assert hasattr(scope, "context")

    def test_scope_with_config(self) -> None:
        """Test tm.scope() with config parameter."""
        with tm.scope(config={"debug": True}) as scope:
            assert scope.config["debug"] is True

    def test_scope_with_container(self) -> None:
        """Test tm.scope() with container parameter."""
        mock_service = "test_service_value"
        with tm.scope(container={"service": mock_service}) as scope:
            assert scope.container["service"] == mock_service

    def test_scope_with_context(self) -> None:
        """Test tm.scope() with context parameter."""
        with tm.scope(context={"user_id": 123}) as scope:
            assert scope.context["user_id"] == 123


    def test_ok_invalid_parameter_type(self) -> None:
        """Test tm.ok() with invalid parameter type raises ValueError."""
        result = FlextResult[int].ok(42)
        with pytest.raises(ValueError, match="Parameter validation failed"):
            tm.ok(result, len="invalid")  # len should be int or tuple[int, int]

    def test_fail_invalid_parameter_type(self) -> None:
        """Test tm.fail() with invalid parameter type raises ValueError."""
        result = FlextResult[str].fail("error")
        with pytest.raises(ValueError, match="Parameter validation failed"):
            tm.fail(result, code=123)  # code should be str

    def test_that_invalid_parameter_type(self) -> None:
        """Test tm.that() with invalid parameter type raises ValueError."""
        with pytest.raises(ValueError, match="Parameter validation failed"):
            tm.that([1, 2, 3], len="invalid")  # len should be int or tuple[int, int]

    def test_scope_invalid_parameter_type(self) -> None:
        """Test tm.scope() with invalid parameter type raises ValueError."""
        with pytest.raises(ValueError, match="Parameter validation failed"):
            with tm.scope(env="invalid"):  # env should be Mapping[str, str]
                pass
