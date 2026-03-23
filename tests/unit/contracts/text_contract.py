"""Shared OO contracts for text utility behavior.

Centralizes scenarios and assertions to reduce duplication across unit suites.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from tests import u


class TextUtilityContract:
    """Reusable contract mixin for text utility tests.

    MRO-friendly base class: concrete test classes inherit from this contract and can
    add suite-specific assertions without duplicating core behavior cases.
    """

    SAFE_STRING_VALID_CASES: ClassVar[Sequence[tuple[str, str]]] = [
        ("hello", "hello"),
        ("  hello  ", "hello"),
        ("\nhello\t", "hello"),
        ("  olá mundo  ", "olá mundo"),
    ]
    SAFE_STRING_INVALID_CASES: ClassVar[Sequence[tuple[str | None, str]]] = [
        (None, "Text cannot be None"),
        ("", "empty or whitespace-only"),
        ("   ", "empty or whitespace-only"),
        ("\n\t", "empty or whitespace-only"),
    ]
    FORMAT_APP_ID_CASES: ClassVar[Sequence[tuple[str, str]]] = [
        ("MyApp", "myapp"),
        ("My App", "my-app"),
        ("my_app", "my-app"),
        ("My Application_Name", "my-application-name"),
    ]
    CLEAN_TEXT_CASES: ClassVar[Sequence[tuple[str, str]]] = [
        ("  hello   world  ", "hello world"),
        ("hello\x00world", "helloworld"),
        ("\nline1\tline2\r\n", "line1line2"),
    ]

    @staticmethod
    def assert_safe_string_valid(raw: str, expected: str) -> None:
        """Assert safe string normalization for valid input."""
        assert u.safe_string(raw) == expected

    @staticmethod
    def assert_clean_text(raw: str, expected: str) -> None:
        """Assert clean_text normalization."""
        assert u.clean_text(raw) == expected

    @staticmethod
    def assert_format_app_id(raw: str, expected: str) -> None:
        """Assert app id formatting behavior."""
        assert u.format_app_id(raw) == expected
