# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Io package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from tests.infra.unit.io.test_infra_json_io import (
        TestFlextInfraJsonService,
        TestFlextInfraJsonService as s,
    )
    from tests.infra.unit.io.test_infra_output import (
        ANSI_RE,
        TestInfraOutputEdgeCases,
        TestInfraOutputHeader,
        TestInfraOutputMessages,
        TestInfraOutputNoColor,
        TestInfraOutputProgress,
        TestInfraOutputStatus,
        TestInfraOutputSummary,
        TestMroFacadeMethods,
        TestShouldUseColor,
        TestShouldUseUnicode,
    )

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ANSI_RE": ("tests.infra.unit.io.test_infra_output", "ANSI_RE"),
    "TestFlextInfraJsonService": ("tests.infra.unit.io.test_infra_json_io", "TestFlextInfraJsonService"),
    "TestInfraOutputEdgeCases": ("tests.infra.unit.io.test_infra_output", "TestInfraOutputEdgeCases"),
    "TestInfraOutputHeader": ("tests.infra.unit.io.test_infra_output", "TestInfraOutputHeader"),
    "TestInfraOutputMessages": ("tests.infra.unit.io.test_infra_output", "TestInfraOutputMessages"),
    "TestInfraOutputNoColor": ("tests.infra.unit.io.test_infra_output", "TestInfraOutputNoColor"),
    "TestInfraOutputProgress": ("tests.infra.unit.io.test_infra_output", "TestInfraOutputProgress"),
    "TestInfraOutputStatus": ("tests.infra.unit.io.test_infra_output", "TestInfraOutputStatus"),
    "TestInfraOutputSummary": ("tests.infra.unit.io.test_infra_output", "TestInfraOutputSummary"),
    "TestMroFacadeMethods": ("tests.infra.unit.io.test_infra_output", "TestMroFacadeMethods"),
    "TestShouldUseColor": ("tests.infra.unit.io.test_infra_output", "TestShouldUseColor"),
    "TestShouldUseUnicode": ("tests.infra.unit.io.test_infra_output", "TestShouldUseUnicode"),
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


def __getattr__(name: str) -> Any:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
