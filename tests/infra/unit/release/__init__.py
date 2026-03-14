# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Release package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes
    from tests.infra.unit.release.flow import TestReleaseMainFlow
    from tests.infra.unit.release.main import TestReleaseMainParsing
    from tests.infra.unit.release.orchestrator import TestReleaseOrchestratorExecute
    from tests.infra.unit.release.orchestrator_git import (
        TestCollectChanges,
        TestCreateBranches,
        TestCreateTag,
        TestPreviousTag,
        TestPushRelease,
    )
    from tests.infra.unit.release.orchestrator_helpers import (
        TestBuildTargets,
        TestBumpNextDev,
        TestDispatchPhase,
        TestGenerateNotes,
        TestRunMake,
        TestUpdateChangelog,
        TestVersionFiles,
    )
    from tests.infra.unit.release.orchestrator_phases import (
        TestPhaseBuild,
        TestPhaseValidate,
        TestPhaseVersion,
    )
    from tests.infra.unit.release.orchestrator_publish import (
        TestPhasePublish,
        workspace_root,
    )
    from tests.infra.unit.release.release_init import TestReleaseInit
    from tests.infra.unit.release.version_resolution import (
        TestReleaseMainTagResolution,
        TestReleaseMainVersionResolution,
        TestResolveVersionInteractive,
    )

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "TestBuildTargets": (
        "tests.infra.unit.release.orchestrator_helpers",
        "TestBuildTargets",
    ),
    "TestBumpNextDev": (
        "tests.infra.unit.release.orchestrator_helpers",
        "TestBumpNextDev",
    ),
    "TestCollectChanges": (
        "tests.infra.unit.release.orchestrator_git",
        "TestCollectChanges",
    ),
    "TestCreateBranches": (
        "tests.infra.unit.release.orchestrator_git",
        "TestCreateBranches",
    ),
    "TestCreateTag": ("tests.infra.unit.release.orchestrator_git", "TestCreateTag"),
    "TestDispatchPhase": (
        "tests.infra.unit.release.orchestrator_helpers",
        "TestDispatchPhase",
    ),
    "TestGenerateNotes": (
        "tests.infra.unit.release.orchestrator_helpers",
        "TestGenerateNotes",
    ),
    "TestPhaseBuild": (
        "tests.infra.unit.release.orchestrator_phases",
        "TestPhaseBuild",
    ),
    "TestPhasePublish": (
        "tests.infra.unit.release.orchestrator_publish",
        "TestPhasePublish",
    ),
    "TestPhaseValidate": (
        "tests.infra.unit.release.orchestrator_phases",
        "TestPhaseValidate",
    ),
    "TestPhaseVersion": (
        "tests.infra.unit.release.orchestrator_phases",
        "TestPhaseVersion",
    ),
    "TestPreviousTag": ("tests.infra.unit.release.orchestrator_git", "TestPreviousTag"),
    "TestPushRelease": ("tests.infra.unit.release.orchestrator_git", "TestPushRelease"),
    "TestReleaseInit": ("tests.infra.unit.release.release_init", "TestReleaseInit"),
    "TestReleaseMainFlow": ("tests.infra.unit.release.flow", "TestReleaseMainFlow"),
    "TestReleaseMainParsing": (
        "tests.infra.unit.release.main",
        "TestReleaseMainParsing",
    ),
    "TestReleaseMainTagResolution": (
        "tests.infra.unit.release.version_resolution",
        "TestReleaseMainTagResolution",
    ),
    "TestReleaseMainVersionResolution": (
        "tests.infra.unit.release.version_resolution",
        "TestReleaseMainVersionResolution",
    ),
    "TestReleaseOrchestratorExecute": (
        "tests.infra.unit.release.orchestrator",
        "TestReleaseOrchestratorExecute",
    ),
    "TestResolveVersionInteractive": (
        "tests.infra.unit.release.version_resolution",
        "TestResolveVersionInteractive",
    ),
    "TestRunMake": ("tests.infra.unit.release.orchestrator_helpers", "TestRunMake"),
    "TestUpdateChangelog": (
        "tests.infra.unit.release.orchestrator_helpers",
        "TestUpdateChangelog",
    ),
    "TestVersionFiles": (
        "tests.infra.unit.release.orchestrator_helpers",
        "TestVersionFiles",
    ),
    "workspace_root": (
        "tests.infra.unit.release.orchestrator_publish",
        "workspace_root",
    ),
}

__all__ = [
    "TestBuildTargets",
    "TestBumpNextDev",
    "TestCollectChanges",
    "TestCreateBranches",
    "TestCreateTag",
    "TestDispatchPhase",
    "TestGenerateNotes",
    "TestPhaseBuild",
    "TestPhasePublish",
    "TestPhaseValidate",
    "TestPhaseVersion",
    "TestPreviousTag",
    "TestPushRelease",
    "TestReleaseInit",
    "TestReleaseMainFlow",
    "TestReleaseMainParsing",
    "TestReleaseMainTagResolution",
    "TestReleaseMainVersionResolution",
    "TestReleaseOrchestratorExecute",
    "TestResolveVersionInteractive",
    "TestRunMake",
    "TestUpdateChangelog",
    "TestVersionFiles",
    "workspace_root",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
