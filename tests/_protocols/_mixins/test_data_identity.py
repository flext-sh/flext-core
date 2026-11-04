# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
tests/_protocols/_mixins/test_data_identity
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flext_tests import FlextTestsProtocols


class TestsFlextProtocolsMixinsTestDataIdentity:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class Names(FlextTestsProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def module_name(self) -> str:
            """Default test module name."""
            ...

        @property
        def handler_name(self) -> str:
            """Default test handler name."""
            ...

        @property
        def chain_name(self) -> str:
            """Default test chain name."""
            ...

        @property
        def command_type(self) -> str:
            """Default test command type."""
            ...

        @property
        def query_type(self) -> str:
            """Default test query type."""
            ...

        @property
        def logger_name(self) -> str:
            """Default test logger name."""
            ...

        @property
        def app_name(self) -> str:
            """Default test application name."""
            ...

        @property
        def validation_app(self) -> str:
            """Default validation test application name."""
            ...

        @property
        def source_service(self) -> str:
            """Default source service name."""
            ...


__all__: list[str] = [
    "TestsFlextProtocolsMixinsTestDataIdentity",
]
