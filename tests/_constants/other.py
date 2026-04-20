"""Constants mixin for other.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Sequence,
)
from typing import ClassVar


class TestsFlextCoreConstantsOther:
    SAFE_STRING_VALID_CASES: ClassVar[Sequence[tuple[str, str]]] = [
        ("hello", "hello"),
        ("  hello  ", "hello"),
        ("hello\t", "hello"),
        ("  olá mundo  ", "olá mundo"),
    ]
    SAFE_STRING_INVALID_CASES: ClassVar[Sequence[tuple[str | None, str]]] = [
        (None, "Text cannot be None"),
        ("", "empty or whitespace-only"),
        ("   ", "empty or whitespace-only"),
        ("\t", "empty or whitespace-only"),
    ]
    FORMAT_APP_ID_CASES: ClassVar[Sequence[tuple[str, str]]] = [
        ("MyApp", "myapp"),
        ("My App", "my-app"),
        ("my_app", "my-app"),
        ("My Application_Name", "my-application-name"),
    ]
