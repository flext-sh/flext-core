# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Github package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes
    from tests.infra.unit.github.linter import TestFlextInfraWorkflowLinter
    from tests.infra.unit.github.main import (
        TestRunLint,
        TestRunWorkflows,
        run_lint,
        run_pr,
        run_workflows,
    )
    from tests.infra.unit.github.main_dispatch import (
        TestRunPrWorkspace,
        run_pr_workspace,
    )
    from tests.infra.unit.github.main_integration import TestMain, main
    from tests.infra.unit.github.pr import (
        TestCreate,
        TestFlextInfraPrManager,
        TestStatus,
    )
    from tests.infra.unit.github.pr_cli import (
        TestMainFunction,
        TestParseArgs,
        TestSelectorFunction,
    )
    from tests.infra.unit.github.pr_init import TestGithubInit
    from tests.infra.unit.github.pr_operations import (
        TestChecks,
        TestClose,
        TestMerge,
        TestTriggerRelease,
        TestView,
    )
    from tests.infra.unit.github.pr_workspace import (
        TestCheckpoint,
        TestFlextInfraPrWorkspaceManager,
        TestRunPr,
    )
    from tests.infra.unit.github.pr_workspace_orchestrate import (
        TestOrchestrate,
        TestStaticMethods,
    )
    from tests.infra.unit.github.workflows import (
        TestFlextInfraWorkflowSyncer,
        TestRenderTemplate,
        TestSyncOperation,
        TestSyncProject,
    )
    from tests.infra.unit.github.workflows_workspace import (
        TestSyncWorkspace,
        TestWriteReport,
    )

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "TestCheckpoint": ("tests.infra.unit.github.pr_workspace", "TestCheckpoint"),
    "TestChecks": ("tests.infra.unit.github.pr_operations", "TestChecks"),
    "TestClose": ("tests.infra.unit.github.pr_operations", "TestClose"),
    "TestCreate": ("tests.infra.unit.github.pr", "TestCreate"),
    "TestFlextInfraPrManager": (
        "tests.infra.unit.github.pr",
        "TestFlextInfraPrManager",
    ),
    "TestFlextInfraPrWorkspaceManager": (
        "tests.infra.unit.github.pr_workspace",
        "TestFlextInfraPrWorkspaceManager",
    ),
    "TestFlextInfraWorkflowLinter": (
        "tests.infra.unit.github.linter",
        "TestFlextInfraWorkflowLinter",
    ),
    "TestFlextInfraWorkflowSyncer": (
        "tests.infra.unit.github.workflows",
        "TestFlextInfraWorkflowSyncer",
    ),
    "TestGithubInit": ("tests.infra.unit.github.pr_init", "TestGithubInit"),
    "TestMain": ("tests.infra.unit.github.main_integration", "TestMain"),
    "TestMainFunction": ("tests.infra.unit.github.pr_cli", "TestMainFunction"),
    "TestMerge": ("tests.infra.unit.github.pr_operations", "TestMerge"),
    "TestOrchestrate": (
        "tests.infra.unit.github.pr_workspace_orchestrate",
        "TestOrchestrate",
    ),
    "TestParseArgs": ("tests.infra.unit.github.pr_cli", "TestParseArgs"),
    "TestRenderTemplate": ("tests.infra.unit.github.workflows", "TestRenderTemplate"),
    "TestRunLint": ("tests.infra.unit.github.main", "TestRunLint"),
    "TestRunPr": ("tests.infra.unit.github.pr_workspace", "TestRunPr"),
    "TestRunPrWorkspace": (
        "tests.infra.unit.github.main_dispatch",
        "TestRunPrWorkspace",
    ),
    "TestRunWorkflows": ("tests.infra.unit.github.main", "TestRunWorkflows"),
    "TestSelectorFunction": ("tests.infra.unit.github.pr_cli", "TestSelectorFunction"),
    "TestStaticMethods": (
        "tests.infra.unit.github.pr_workspace_orchestrate",
        "TestStaticMethods",
    ),
    "TestStatus": ("tests.infra.unit.github.pr", "TestStatus"),
    "TestSyncOperation": ("tests.infra.unit.github.workflows", "TestSyncOperation"),
    "TestSyncProject": ("tests.infra.unit.github.workflows", "TestSyncProject"),
    "TestSyncWorkspace": (
        "tests.infra.unit.github.workflows_workspace",
        "TestSyncWorkspace",
    ),
    "TestTriggerRelease": (
        "tests.infra.unit.github.pr_operations",
        "TestTriggerRelease",
    ),
    "TestView": ("tests.infra.unit.github.pr_operations", "TestView"),
    "TestWriteReport": (
        "tests.infra.unit.github.workflows_workspace",
        "TestWriteReport",
    ),
    "main": ("tests.infra.unit.github.main_integration", "main"),
    "run_lint": ("tests.infra.unit.github.main", "run_lint"),
    "run_pr": ("tests.infra.unit.github.main", "run_pr"),
    "run_pr_workspace": ("tests.infra.unit.github.main_dispatch", "run_pr_workspace"),
    "run_workflows": ("tests.infra.unit.github.main", "run_workflows"),
}

__all__ = [
    "TestCheckpoint",
    "TestChecks",
    "TestClose",
    "TestCreate",
    "TestFlextInfraPrManager",
    "TestFlextInfraPrWorkspaceManager",
    "TestFlextInfraWorkflowLinter",
    "TestFlextInfraWorkflowSyncer",
    "TestGithubInit",
    "TestMain",
    "TestMainFunction",
    "TestMerge",
    "TestOrchestrate",
    "TestParseArgs",
    "TestRenderTemplate",
    "TestRunLint",
    "TestRunPr",
    "TestRunPrWorkspace",
    "TestRunWorkflows",
    "TestSelectorFunction",
    "TestStaticMethods",
    "TestStatus",
    "TestSyncOperation",
    "TestSyncProject",
    "TestSyncWorkspace",
    "TestTriggerRelease",
    "TestView",
    "TestWriteReport",
    "main",
    "run_lint",
    "run_pr",
    "run_pr_workspace",
    "run_workflows",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
