# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Io package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes
    from tests.infra.unit.io.test_infra_json_io import (
        TestFlextInfraJsonService,
        TestFlextInfraJsonService as s,
    )
    from tests.infra.unit.io.test_infra_output_edge_cases import (
        TestInfraOutputEdgeCases,
        TestInfraOutputNoColor,
        TestMroFacadeMethods,
    )
    from tests.infra.unit.io.test_infra_output_formatting import (
        ANSI_RE,
        TestInfraOutputHeader,
        TestInfraOutputMessages,
        TestInfraOutputProgress,
        TestInfraOutputStatus,
        TestInfraOutputSummary,
    )
    from tests.infra.unit.io.test_infra_terminal_detection import (
        TestShouldUseColor,
        TestShouldUseUnicode,
    )

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ANSI_RE": ("tests.infra.unit.io.test_infra_output_formatting", "ANSI_RE"),
    "TestFlextInfraJsonService": (
        "tests.infra.unit.io.test_infra_json_io",
        "TestFlextInfraJsonService",
    ),
    "TestInfraOutputEdgeCases": (
        "tests.infra.unit.io.test_infra_output_edge_cases",
        "TestInfraOutputEdgeCases",
    ),
    "TestInfraOutputHeader": (
        "tests.infra.unit.io.test_infra_output_formatting",
        "TestInfraOutputHeader",
    ),
    "TestInfraOutputMessages": (
        "tests.infra.unit.io.test_infra_output_formatting",
        "TestInfraOutputMessages",
    ),
    "TestInfraOutputNoColor": (
        "tests.infra.unit.io.test_infra_output_edge_cases",
        "TestInfraOutputNoColor",
    ),
    "TestInfraOutputProgress": (
        "tests.infra.unit.io.test_infra_output_formatting",
        "TestInfraOutputProgress",
    ),
    "TestInfraOutputStatus": (
        "tests.infra.unit.io.test_infra_output_formatting",
        "TestInfraOutputStatus",
    ),
    "TestInfraOutputSummary": (
        "tests.infra.unit.io.test_infra_output_formatting",
        "TestInfraOutputSummary",
    ),
    "TestMroFacadeMethods": (
        "tests.infra.unit.io.test_infra_output_edge_cases",
        "TestMroFacadeMethods",
    ),
    "TestShouldUseColor": (
        "tests.infra.unit.io.test_infra_terminal_detection",
        "TestShouldUseColor",
    ),
    "TestShouldUseUnicode": (
        "tests.infra.unit.io.test_infra_terminal_detection",
        "TestShouldUseUnicode",
    ),
    "s": ("tests.infra.unit.io.test_infra_json_io", "TestFlextInfraJsonService"),
}

__all__ = [
    "ANSI_RE",
    "TestFlextInfraJsonService",
    "TestInfraOutputEdgeCases",
    "TestInfraOutputHeader",
    "TestInfraOutputMessages",
    "TestInfraOutputNoColor",
    "TestInfraOutputProgress",
    "TestInfraOutputStatus",
    "TestInfraOutputSummary",
    "TestMroFacadeMethods",
    "TestShouldUseColor",
    "TestShouldUseUnicode",
    "s",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
