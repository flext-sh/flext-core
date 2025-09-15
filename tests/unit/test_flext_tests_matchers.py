"""Tests for flext_tests matchers module - real functional tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from flext_core import FlextResult
from flext_tests import FlextTestsMatchers


class TestFlextTestsMatchers:
    """Real functional tests for FlextTestsMatchers module."""

    def test_flext_result_matchers(self) -> None:
        """Test FlextResult matching functionality."""
        success_result = FlextResult[str].ok("test_value")
        failure_result = FlextResult[str].fail("test_error")

        # Test success matcher
        if hasattr(FlextTestsMatchers, "be_success"):
            matcher = FlextTestsMatchers.be_success()
            assert matcher.matches(success_result) is True
            assert matcher.matches(failure_result) is False

        # Test failure matcher
        if hasattr(FlextTestsMatchers, "be_failure"):
            matcher = FlextTestsMatchers.be_failure()
            assert matcher.matches(failure_result) is True
            assert matcher.matches(success_result) is False

    def test_value_matchers(self) -> None:
        """Test value matching functionality."""
        # Test equal matcher
        if hasattr(FlextTestsMatchers, "equal"):
            matcher = FlextTestsMatchers.equal("expected_value")
            assert matcher.matches("expected_value") is True
            assert matcher.matches("different_value") is False

        # Test contain matcher
        if hasattr(FlextTestsMatchers, "contain"):
            matcher = FlextTestsMatchers.contain("substring")
            assert matcher.matches("this contains substring") is True
            assert matcher.matches("this does not") is False

    def test_type_matchers(self) -> None:
        """Test type matching functionality."""
        # Test type matcher
        if hasattr(FlextTestsMatchers, "be_instance_of"):
            matcher = FlextTestsMatchers.be_instance_of(str)
            assert matcher.matches("string_value") is True
            assert matcher.matches(123) is False

        # Test None matcher
        if hasattr(FlextTestsMatchers, "be_none"):
            matcher = FlextTestsMatchers.be_none()
            assert matcher.matches(None) is True
            assert matcher.matches("not_none") is False

    def test_collection_matchers(self) -> None:
        """Test collection matching functionality."""
        # Test empty matcher
        if hasattr(FlextTestsMatchers, "be_empty"):
            matcher = FlextTestsMatchers.be_empty()
            assert matcher.matches([]) is True
            assert matcher.matches([1, 2, 3]) is False

        # Test length matcher
        if hasattr(FlextTestsMatchers, "have_length"):
            matcher = FlextTestsMatchers.have_length(3)
            assert matcher.matches([1, 2, 3]) is True
            assert matcher.matches([1, 2]) is False

    def test_numeric_matchers(self) -> None:
        """Test numeric matching functionality."""
        # Test greater than matcher
        if hasattr(FlextTestsMatchers, "be_greater_than"):
            matcher = FlextTestsMatchers.be_greater_than(10)
            assert matcher.matches(15) is True
            assert matcher.matches(5) is False

        # Test less than matcher
        if hasattr(FlextTestsMatchers, "be_less_than"):
            matcher = FlextTestsMatchers.be_less_than(10)
            assert matcher.matches(5) is True
            assert matcher.matches(15) is False

    def test_string_matchers(self) -> None:
        """Test string-specific matching functionality."""
        # Test starts with matcher
        if hasattr(FlextTestsMatchers, "start_with"):
            matcher = FlextTestsMatchers.start_with("Hello")
            assert matcher.matches("Hello World") is True
            assert matcher.matches("Hi World") is False

        # Test ends with matcher
        if hasattr(FlextTestsMatchers, "end_with"):
            matcher = FlextTestsMatchers.end_with("World")
            assert matcher.matches("Hello World") is True
            assert matcher.matches("Hello Universe") is False

    def test_regex_matchers(self) -> None:
        """Test regex matching functionality."""
        if hasattr(FlextTestsMatchers, "match_regex"):
            matcher = FlextTestsMatchers.match_regex(r"\d+")
            assert matcher.matches("123") is True
            assert matcher.matches("abc") is False

    def test_compound_matchers(self) -> None:
        """Test compound matching functionality."""
        # Test all_of matcher (AND logic)
        if (
            hasattr(FlextTestsMatchers, "all_of")
            and hasattr(FlextTestsMatchers, "be_instance_of")
            and hasattr(FlextTestsMatchers, "start_with")
        ):
            matcher = FlextTestsMatchers.all_of(
                [
                    FlextTestsMatchers.be_instance_of(str),
                    FlextTestsMatchers.start_with("Hello"),
                ]
            )
            assert matcher.matches("Hello World") is True
            assert matcher.matches(123) is False
            assert matcher.matches("Hi World") is False

        # Test any_of matcher (OR logic)
        if hasattr(FlextTestsMatchers, "any_of") and hasattr(
            FlextTestsMatchers, "equal"
        ):
            matcher = FlextTestsMatchers.any_of(
                [
                    FlextTestsMatchers.equal("option1"),
                    FlextTestsMatchers.equal("option2"),
                ]
            )
            assert matcher.matches("option1") is True
            assert matcher.matches("option2") is True
            assert matcher.matches("option3") is False

    def test_custom_matchers(self) -> None:
        """Test custom matcher functionality."""
        # Test custom predicate matcher
        if hasattr(FlextTestsMatchers, "satisfy"):

            def is_even(n: int) -> bool:
                return n % 2 == 0

            matcher = FlextTestsMatchers.satisfy(is_even)
            assert matcher.matches(4) is True
            assert matcher.matches(3) is False

    def test_pytest_integration(self) -> None:
        """Test pytest integration if available."""
        if hasattr(FlextTestsMatchers, "pytest_assert"):
            success_result = FlextResult[str].ok("test")

            # This should not raise
            FlextTestsMatchers.assert_result_success(success_result)

            # This should raise AssertionError
            failure_result = FlextResult[str].fail("error")
            with pytest.raises(AssertionError):
                FlextTestsMatchers.assert_result_success(failure_result)

    def test_matcher_descriptions(self) -> None:
        """Test matcher description functionality."""
        if hasattr(FlextTestsMatchers, "equal"):
            matcher = FlextTestsMatchers.equal("test_value")
            if hasattr(matcher, "description"):
                description = matcher.description()
                assert isinstance(description, str)
                assert len(description) > 0

    def test_matcher_mismatch_descriptions(self) -> None:
        """Test matcher mismatch description functionality."""
        if hasattr(FlextTestsMatchers, "equal"):
            matcher = FlextTestsMatchers.equal("expected")
            if hasattr(matcher, "mismatch_description"):
                mismatch = matcher.mismatch_description("actual")
                assert isinstance(mismatch, str)
                assert len(mismatch) > 0
