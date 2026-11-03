# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
tests/_protocols/_mixins/service_case_core
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_tests import FlextTestsProtocols

from tests import t

if TYPE_CHECKING:
    from flext_tests import p


class TestsFlextProtocolsMixinsServiceCaseCore:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class ServiceUserData(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def user_id(self) -> int:
            """User identifier."""
            ...

        @property
        def name(self) -> str:
            """User name."""
            ...

    @runtime_checkable
    class ServiceUserService(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def settings_type(self) -> p.SettingsType | None:
            """FlextSettings class used to initialize the service runtime."""
            ...

        @property
        def runtime_settings(self) -> p.Settings | None:
            """Pre-built settings instance used directly for the runtime."""
            ...

        @property
        def settings_overrides(self) -> t.JsonMapping | None:
            """Settings overrides applied at instantiation."""
            ...

        @property
        def initial_context(self) -> p.Context | None:
            """Initial context for the service scope."""
            ...

    @runtime_checkable
    class ServiceTestCase(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def service_type(self) -> str | None:
            """Service type for factory-driven tests."""
            ...

        @property
        def input_value(self) -> str | None:
            """Primary service input."""
            ...

        @property
        def user_id(self) -> str | None:
            """User identifier for documented tests."""
            ...

        @property
        def expected_success(self) -> bool:
            """Whether service call is expected to succeed."""
            ...

        @property
        def expected_error(self) -> str | None:
            """Expected error substring for failure cases."""
            ...

        @property
        def description(self) -> str:
            """Human-readable test case description."""
            ...

        @property
        def extra_param(self) -> int:
            """Auxiliary numeric parameter."""
            ...

    @runtime_checkable
    class RailwayTestCase(FlextTestsProtocols.Model, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def user_ids(self) -> Sequence[str]:
            """User identifiers used in pipeline."""
            ...

        @property
        def operations(self) -> Sequence[str]:
            """Pipeline operations to execute."""
            ...

        @property
        def expected_pipeline_length(self) -> int:
            """Expected number of pipeline stages."""
            ...

        @property
        def should_fail_at(self) -> int | None:
            """Optional pipeline step expected to fail."""
            ...

        @property
        def description(self) -> str:
            """Human-readable railway test case description."""
            ...


__all__: list[str] = [
    "TestsFlextProtocolsMixinsServiceCaseCore",
]
