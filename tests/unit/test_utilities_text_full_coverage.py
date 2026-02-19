"""Tests for FlextUtilitiesText - text cleaning, truncation, safe_string, format.

Module: flext_core._utilities.text
Coverage target: lines 33, 82-83, 109

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from flext_core import u


class TestTextLogger:
    """Tests for u.Text.logger property."""

    def test_logger_property_returns_logger(self) -> None:
        """logger property returns a structlog logger instance."""
        text_util = u.Text()
        logger = text_util.logger
        assert logger is not None
        # Structlog loggers have standard methods
        assert hasattr(logger, "info")


class TestSafeString:
    """Tests for u.Text.safe_string() - covers lines 82-83."""

    def test_safe_string_none_raises(self) -> None:
        """None input raises ValueError."""
        with pytest.raises(ValueError, match="Text cannot be None"):
            u.Text.safe_string(None)

    def test_safe_string_empty_raises(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="empty or whitespace-only"):
            u.Text.safe_string("")

    def test_safe_string_whitespace_only_raises(self) -> None:
        """Whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="empty or whitespace-only"):
            u.Text.safe_string("   ")

    def test_safe_string_valid_returns_stripped(self) -> None:
        """Valid string is returned stripped."""
        assert u.Text.safe_string("  hello  ") == "hello"


class TestFormatAppId:
    """Tests for u.Text.format_app_id() - covers line 109."""

    def test_format_app_id_lowercases(self) -> None:
        """Name is lowercased."""
        assert u.Text.format_app_id("MyApp") == "myapp"

    def test_format_app_id_replaces_spaces_with_hyphens(self) -> None:
        """Spaces become hyphens."""
        assert u.Text.format_app_id("My App") == "my-app"

    def test_format_app_id_replaces_underscores_with_hyphens(self) -> None:
        """Underscores become hyphens."""
        assert u.Text.format_app_id("my_app") == "my-app"

    def test_format_app_id_combined(self) -> None:
        """Combined spaces, underscores, and case."""
        assert u.Text.format_app_id("My Application_Name") == "my-application-name"
