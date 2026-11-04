# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
tests/_protocols/_mixins/test_data_values
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flext_tests import FlextTestsProtocols


class TestsFlextProtocolsMixinsTestDataValues:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class Data(FlextTestsProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def field_name(self) -> str:
            """Default test field name."""
            ...

        @property
        def config_key(self) -> str:
            """Default test settings key."""
            ...

        @property
        def username(self) -> str:
            """Default test username."""
            ...

        @property
        def email(self) -> str:
            """Default test email."""
            ...

        @property
        def string_value(self) -> str:
            """Default test string value."""
            ...

        @property
        def input_data(self) -> str:
            """Default test input data."""
            ...

        @property
        def request_data(self) -> str:
            """Default test request data."""
            ...

        @property
        def result_data(self) -> str:
            """Default test result data."""
            ...

        @property
        def message(self) -> str:
            """Default test message."""
            ...


__all__: list[str] = [
    "TestsFlextProtocolsMixinsTestDataValues",
]
