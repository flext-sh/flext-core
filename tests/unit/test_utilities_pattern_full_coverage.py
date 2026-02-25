"""Tests for FlextUtilitiesPattern.match() - pattern matching utility.

Module: flext_core._utilities.pattern
Coverage target: lines 76-85

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from flext_core import u


class TestPatternMatch:
    """Tests for u.Pattern.match()."""

    def test_match_first_matching_pattern(self) -> None:
        """Returns result of first matching predicate's handler."""
        result = u.Pattern.match(
            5,
            (lambda value: value > 10, lambda value: "large"),
            (lambda value: value > 0, lambda value: "positive"),
        )
        assert result == "positive"

    def test_match_returns_first_match_not_best_match(self) -> None:
        """When multiple patterns match, returns the FIRST match."""
        result = u.Pattern.match(
            15,
            (lambda value: value > 10, lambda value: "large"),
            (lambda value: value > 0, lambda value: "positive"),
        )
        assert result == "large"

    def test_match_uses_default_when_no_match(self) -> None:
        """When no pattern matches, default handler is called."""
        result = u.Pattern.match(
            -5,
            (lambda value: value > 10, lambda value: "large"),
            (lambda value: value > 0, lambda value: "positive"),
            default=lambda value: f"negative: {value}",
        )
        assert result == "negative: -5"

    def test_match_raises_when_no_match_and_no_default(self) -> None:
        """Raises ValueError when no pattern matches and no default."""
        with pytest.raises(ValueError, match="No pattern matched"):
            u.Pattern.match(
                -5,
                (lambda value: value > 10, lambda value: "large"),
                (lambda value: value > 0, lambda value: "positive"),
            )

    def test_match_with_no_patterns_and_default(self) -> None:
        """With zero patterns and a default, returns default result."""
        result = u.Pattern.match(
            42,
            default=lambda value: value * 2,
        )
        assert result == 84

    def test_match_with_no_patterns_and_no_default_raises(self) -> None:
        """With zero patterns and no default, raises ValueError."""
        with pytest.raises(ValueError, match="No pattern matched"):
            u.Pattern.match(42)

    def test_match_handler_receives_original_value(self) -> None:
        """Handler receives the original value passed to match."""
        result = u.Pattern.match(
            "hello",
            (lambda value: isinstance(value, str), lambda value: value.upper()),
        )
        assert result == "HELLO"
