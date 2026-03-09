"""Tests for FlextUtilitiesText - text cleaning, truncation, safe_string, format.

Module: flext_core._utilities.text
Coverage target: lines 33, 82-83, 109

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re

import pytest
from hypothesis import given, strategies as st

from flext_core import u

from .contracts.text_contract import TextUtilityContract


class TestTextLogger:
    """Tests for u.Text.logger property."""

    def test_logger_property_returns_logger(self) -> None:
        """Logger property returns a structlog logger instance."""
        logger = u.Text().logger
        assert logger is not None
        assert hasattr(logger, "info")


class TestSafeString(TextUtilityContract):
    """Tests for u.Text.safe_string()."""

    @pytest.mark.parametrize(
        ("value", "message"),
        TextUtilityContract.SAFE_STRING_INVALID_CASES,
    )
    def test_safe_string_invalid_values_raise(
        self, value: str | None, message: str
    ) -> None:
        """Invalid values raise ValueError with actionable message."""
        with pytest.raises(ValueError, match=message):
            u.Text.safe_string(value)

    @pytest.mark.parametrize(
        ("value", "expected"),
        TextUtilityContract.SAFE_STRING_VALID_CASES,
    )
    def test_safe_string_valid_values_are_stripped(
        self, value: str, expected: str
    ) -> None:
        """Valid strings are stripped and preserved."""
        self.assert_safe_string_valid(value, expected)


class TestFormatAppId(TextUtilityContract):
    """Tests for u.Text.format_app_id()."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            pytest.param(name, expected, id=f"format-{index}")
            for index, (name, expected) in enumerate(
                TextUtilityContract.FORMAT_APP_ID_CASES
            )
        ],
    )
    def test_format_app_id_examples(self, name: str, expected: str) -> None:
        """Known examples are normalized as expected."""
        self.assert_format_app_id(name, expected)

    @given(st.text())
    def test_format_app_id_keeps_length(self, name: str) -> None:
        """Only character substitution is performed, never insertion/removal."""
        assert len(u.Text.format_app_id(name)) == len(name)

    @given(st.text())
    def test_format_app_id_normalization_rules(self, name: str) -> None:
        """Result is lowercase and contains no spaces or underscores."""
        formatted = u.Text.format_app_id(name)
        assert formatted == name.lower().replace(" ", "-").replace("_", "-")
        assert " " not in formatted
        assert "_" not in formatted


class TestCleanText(TextUtilityContract):
    """Tests for u.Text.clean_text()."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        TextUtilityContract.CLEAN_TEXT_CASES,
    )
    def test_clean_text_examples(self, raw: str, expected: str) -> None:
        """Clean text should remove control chars and normalize whitespace."""
        self.assert_clean_text(raw, expected)

    @given(st.text())
    def test_clean_text_never_contains_repeated_whitespace(self, raw: str) -> None:
        """Property-based guarantee for whitespace normalization."""
        cleaned = u.Text.clean_text(raw)
        assert re.search(r"\s{2,}", cleaned) is None
        assert cleaned == cleaned.strip()
