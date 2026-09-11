"""Test data cases mixin for FLEXT tests - reusable parametrize data.

Provides shared test case collections for data-driven testing across all test suites.
Access via c.Tests.FORMAT_APP_ID_CASES, etc.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from flext_infra import t


class FlextTestsConstantsDataCases:
    """Reusable test data cases for parametrized testing."""

    # App ID normalization test cases: (raw, normalized)
    FORMAT_APP_ID_CASES: Final[t.StrPairTuple] = (
        ("FlextApp", "flextapp"),
        ("flext-app", "flextapp"),
        ("FLEXT_APP", "flextapp"),
        ("flext_app_v2", "flext_app_v2"),
        ("MyApp2025", "myapp2025"),
        ("app", "app"),
    )

    # Safe string valid test cases: (raw, expected)
    SAFE_STRING_VALID_CASES: Final[t.StrPairTuple] = (
        ("  valid_name  ", "valid_name"),
        ("identifier_123", "identifier_123"),
        ("_private", "_private"),
        ("snake_case_name", "snake_case_name"),
        ("a", "a"),
        ("A_B_C", "A_B_C"),
    )

    # Safe string invalid test cases: (raw, reason)
    SAFE_STRING_INVALID_CASES: Final[tuple[tuple[str | None, str], ...]] = (
        ("", "empty string"),
        ("   ", "whitespace only"),
        ("invalid-name", "hyphen not allowed"),
        ("invalid.name", "dot not allowed"),
        ("invalid name", "space not allowed"),
        ("123start", "numeric prefix"),
        ("invalid@symbol", "special character"),
        (None, "null value"),
    )


__all__ = ["FlextTestsConstantsDataCases"]
