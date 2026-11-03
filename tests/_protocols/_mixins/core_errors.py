# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
tests/_protocols/_mixins/core_errors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from flext_tests import FlextTestsProtocols

from tests import t


class TestsFlextProtocolsMixinsCoreErrors:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class FailureEnvelope(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        error: str
        error_code: str | None
        error_data: t.JsonMapping | None
        exception: BaseException | None
        success: Literal[False]
        failure: Literal[True]

    @runtime_checkable
    class AttrObject(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def name(self) -> str:
            """Attribute recursive container name."""
            ...

        @property
        def value(self) -> int:
            """Attribute recursive container value."""
            ...

    @runtime_checkable
    class SampleModel(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        name: str
        value: int

    @runtime_checkable
    class TargetModel(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        value: int


__all__: list[str] = [
    "TestsFlextProtocolsMixinsCoreErrors",
]
