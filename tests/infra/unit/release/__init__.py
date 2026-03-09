# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Release package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from tests.infra.unit.release.main import (
        TestReleaseInit,
        TestReleaseMainFlow,
        TestReleaseMainParsing,
        TestReleaseMainTagResolution,
        TestReleaseMainVersionResolution,
        TestResolveVersionInteractive,
    )
    from tests.infra.unit.release.orchestrator import (
        TestFlextInfraReleaseOrchestrator,
        TestFlextInfraReleaseOrchestratorChangeCollection,
        TestFlextInfraReleaseOrchestratorDispatchPhase,
        TestFlextInfraReleaseOrchestratorPhaseBuild,
        TestFlextInfraReleaseOrchestratorPhaseVersion,
    )

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "TestFlextInfraReleaseOrchestrator": (
        "tests.infra.unit.release.orchestrator",
        "TestFlextInfraReleaseOrchestrator",
    ),
    "TestFlextInfraReleaseOrchestratorChangeCollection": (
        "tests.infra.unit.release.orchestrator",
        "TestFlextInfraReleaseOrchestratorChangeCollection",
    ),
    "TestFlextInfraReleaseOrchestratorDispatchPhase": (
        "tests.infra.unit.release.orchestrator",
        "TestFlextInfraReleaseOrchestratorDispatchPhase",
    ),
    "TestFlextInfraReleaseOrchestratorPhaseBuild": (
        "tests.infra.unit.release.orchestrator",
        "TestFlextInfraReleaseOrchestratorPhaseBuild",
    ),
    "TestFlextInfraReleaseOrchestratorPhaseVersion": (
        "tests.infra.unit.release.orchestrator",
        "TestFlextInfraReleaseOrchestratorPhaseVersion",
    ),
    "TestReleaseInit": ("tests.infra.unit.release.main", "TestReleaseInit"),
    "TestReleaseMainFlow": ("tests.infra.unit.release.main", "TestReleaseMainFlow"),
    "TestReleaseMainParsing": (
        "tests.infra.unit.release.main",
        "TestReleaseMainParsing",
    ),
    "TestReleaseMainTagResolution": (
        "tests.infra.unit.release.main",
        "TestReleaseMainTagResolution",
    ),
    "TestReleaseMainVersionResolution": (
        "tests.infra.unit.release.main",
        "TestReleaseMainVersionResolution",
    ),
    "TestResolveVersionInteractive": (
        "tests.infra.unit.release.main",
        "TestResolveVersionInteractive",
    ),
}

__all__ = [
    "TestFlextInfraReleaseOrchestrator",
    "TestFlextInfraReleaseOrchestratorChangeCollection",
    "TestFlextInfraReleaseOrchestratorDispatchPhase",
    "TestFlextInfraReleaseOrchestratorPhaseBuild",
    "TestFlextInfraReleaseOrchestratorPhaseVersion",
    "TestReleaseInit",
    "TestReleaseMainFlow",
    "TestReleaseMainParsing",
    "TestReleaseMainTagResolution",
    "TestReleaseMainVersionResolution",
    "TestResolveVersionInteractive",
]


def __getattr__(name: str) -> Any:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
