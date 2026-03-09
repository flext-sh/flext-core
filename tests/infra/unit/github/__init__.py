# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Github package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from tests.infra.unit.github.linter import TestFlextInfraWorkflowLinter
    from tests.infra.unit.github.main import (
        TestMain,
        TestRunLint,
        TestRunPrWorkspace,
        TestRunWorkflows,
        main,
        run_lint,
        run_pr,
        run_pr_workspace,
        run_workflows,
    )
    from tests.infra.unit.github.pr import (
        TestChecks,
        TestClose,
        TestCreate,
        TestFlextInfraPrManager,
        TestGithubInit,
        TestMainFunction,
        TestMerge,
        TestParseArgs,
        TestSelectorFunction,
        TestStatus,
        TestTriggerRelease,
        TestView,
    )
    from tests.infra.unit.github.pr_workspace import (
        TestCheckpoint,
        TestFlextInfraPrWorkspaceManager,
        TestOrchestrate,
        TestRunPr,
        TestStaticMethods,
    )
    from tests.infra.unit.github.workflows import (
        TestFlextInfraWorkflowSyncer,
        TestRenderTemplate,
        TestSyncOperation,
        TestSyncProject,
    )

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "TestCheckpoint": ("tests.infra.unit.github.pr_workspace", "TestCheckpoint"),
    "TestChecks": ("tests.infra.unit.github.pr", "TestChecks"),
    "TestClose": ("tests.infra.unit.github.pr", "TestClose"),
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
    "TestGithubInit": ("tests.infra.unit.github.pr", "TestGithubInit"),
    "TestMain": ("tests.infra.unit.github.main", "TestMain"),
    "TestMainFunction": ("tests.infra.unit.github.pr", "TestMainFunction"),
    "TestMerge": ("tests.infra.unit.github.pr", "TestMerge"),
    "TestOrchestrate": ("tests.infra.unit.github.pr_workspace", "TestOrchestrate"),
    "TestParseArgs": ("tests.infra.unit.github.pr", "TestParseArgs"),
    "TestRenderTemplate": ("tests.infra.unit.github.workflows", "TestRenderTemplate"),
    "TestRunLint": ("tests.infra.unit.github.main", "TestRunLint"),
    "TestRunPr": ("tests.infra.unit.github.pr_workspace", "TestRunPr"),
    "TestRunPrWorkspace": ("tests.infra.unit.github.main", "TestRunPrWorkspace"),
    "TestRunWorkflows": ("tests.infra.unit.github.main", "TestRunWorkflows"),
    "TestSelectorFunction": ("tests.infra.unit.github.pr", "TestSelectorFunction"),
    "TestStaticMethods": ("tests.infra.unit.github.pr_workspace", "TestStaticMethods"),
    "TestStatus": ("tests.infra.unit.github.pr", "TestStatus"),
    "TestSyncOperation": ("tests.infra.unit.github.workflows", "TestSyncOperation"),
    "TestSyncProject": ("tests.infra.unit.github.workflows", "TestSyncProject"),
    "TestTriggerRelease": ("tests.infra.unit.github.pr", "TestTriggerRelease"),
    "TestView": ("tests.infra.unit.github.pr", "TestView"),
    "main": ("tests.infra.unit.github.main", "main"),
    "run_lint": ("tests.infra.unit.github.main", "run_lint"),
    "run_pr": ("tests.infra.unit.github.main", "run_pr"),
    "run_pr_workspace": ("tests.infra.unit.github.main", "run_pr_workspace"),
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
    "TestTriggerRelease",
    "TestView",
    "main",
    "run_lint",
    "run_pr",
    "run_pr_workspace",
    "run_workflows",
]


def __getattr__(name: str) -> Any:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
