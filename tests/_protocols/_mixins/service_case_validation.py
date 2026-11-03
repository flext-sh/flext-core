# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
tests/_protocols/_mixins/service_case_validation
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_tests import FlextTestsProtocols

from tests import t

if TYPE_CHECKING:
    from flext_tests import p


class TestsFlextProtocolsMixinsServiceCaseValidation:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ValidationScenario(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def name(self) -> str:
            """Unique scenario name."""
            ...

        @property
        def validator_type(self) -> str:
            """Validator category under test."""
            ...

        @property
        def input_value(self) -> t.JsonValue | None:
            """Input value passed to validator."""
            ...

        @property
        def input_params(self) -> t.JsonPayload | None:
            """Optional validator parameters for scenario execution."""
            ...

        @property
        def should_succeed(self) -> bool:
            """Whether scenario expects validation success."""
            ...

        @property
        def expected_value(self) -> t.JsonValue | None:
            """Expected normalized value when validation succeeds."""
            ...

        @property
        def expected_error_contains(self) -> str | None:
            """Expected error substring when validation fails."""
            ...

        @property
        def description(self) -> str | None:
            """Human-readable scenario description."""
            ...

    @runtime_checkable
    class Operation(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def success_count(self) -> int:
            """Successful operations."""
            ...

        @property
        def failure_count(self) -> int:
            """Failed operations."""
            ...

        @property
        def skipped_count(self) -> int:
            """Skipped operations."""
            ...

        @property
        def metadata(self) -> t.JsonMapping:
            """Additional operation metadata."""
            ...

    @runtime_checkable
    class Conversion(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def converted(self) -> t.JsonList:
            """Converted records."""
            ...

        @property
        def errors(self) -> Sequence[str]:
            """Conversion errors."""
            ...

        @property
        def warnings(self) -> Sequence[str]:
            """Conversion warnings."""
            ...

        @property
        def skipped(self) -> t.JsonList:
            """Skipped records."""
            ...

        @property
        def metadata(self) -> t.JsonMapping:
            """Additional conversion metadata."""
            ...

    @runtime_checkable
    class ParserScenario(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def name(self) -> str:
            """Unique parser scenario name."""
            ...

        @property
        def parser_method(self) -> str:
            """Parser method to execute."""
            ...

        @property
        def input_data(self) -> str:
            """Raw parser input data."""
            ...

        @property
        def expected_output(self) -> t.JsonValue | None:
            """Expected parsed output for successful scenarios."""
            ...

        @property
        def should_succeed(self) -> bool:
            """Whether parser scenario expects success."""
            ...

        @property
        def error_contains(self) -> str | None:
            """Expected parser error substring."""
            ...

        @property
        def description(self) -> str | None:
            """Human-readable scenario description."""
            ...

    @runtime_checkable
    class PublicParseCase(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def name(self) -> str:
            """Unique scenario name."""
            ...

        @property
        def input_value(self) -> t.JsonPayload | None:
            """Public value passed to u.parse()."""
            ...

        @property
        def target(self) -> type:
            """Public target type passed to u.parse()."""
            ...

        @property
        def options(self) -> p.BaseModel | None:
            """Optional ParseOptions instance."""
            ...

        @property
        def should_succeed(self) -> bool:
            """Whether parsing should succeed."""
            ...

        @property
        def expected_value(self) -> t.JsonPayload | None:
            """Expected parsed scalar or enum value."""
            ...

        @property
        def expected_data(self) -> t.JsonMapping | None:
            """Expected parsed model_dump payload."""
            ...

        @property
        def error_contains(self) -> str | None:
            """Expected failure error substring."""
            ...

        @property
        def description(self) -> str | None:
            """Human-readable scenario description."""
            ...


__all__: list[str] = [
    "TestsFlextProtocolsMixinsServiceCaseValidation",
]
