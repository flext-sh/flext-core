# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Check package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes
    from tests.infra.unit.check.cli import (
        test_resolve_gates_maps_type_alias,
        test_run_cli_run_returns_one_for_fail,
        test_run_cli_run_returns_two_for_error,
        test_run_cli_run_returns_zero_for_pass,
        test_run_cli_with_fail_fast_flag,
        test_run_cli_with_multiple_projects,
    )
    from tests.infra.unit.check.extended_cli_entry import (
        TestCheckMainEntryPoint,
        TestFixPyrelfyCLI,
        TestRunCLIExtended,
        TestWorkspaceCheckCLI,
    )
    from tests.infra.unit.check.extended_config_fixer import (
        TestConfigFixerEnsureProjectExcludes,
        TestConfigFixerExecute,
        TestConfigFixerFindPyprojectFiles,
        TestConfigFixerFixSearchPaths,
        TestConfigFixerProcessFile,
        TestConfigFixerRemoveIgnoreSubConfig,
        TestConfigFixerRun,
        TestConfigFixerToArray,
    )
    from tests.infra.unit.check.extended_config_fixer_errors import (
        TestConfigFixerPathResolution,
        TestConfigFixerRunMethods,
        TestConfigFixerRunWithVerbose,
        TestProcessFileReadError,
    )
    from tests.infra.unit.check.extended_error_reporting import (
        RunStub,
        TestErrorReporting,
        TestGoFmtEmptyLinesInOutput,
        TestMarkdownReportEmptyGates,
        TestMypyEmptyLinesInOutput,
        TestRuffFormatDuplicateFiles,
    )
    from tests.infra.unit.check.extended_gate_bandit_markdown import (
        TestWorkspaceCheckerRunBandit,
        TestWorkspaceCheckerRunMarkdown,
    )
    from tests.infra.unit.check.extended_gate_go_cmd import (
        TestWorkspaceCheckerCollectMarkdownFiles,
        TestWorkspaceCheckerRunCommand,
        TestWorkspaceCheckerRunGo,
    )
    from tests.infra.unit.check.extended_gate_mypy_pyright import (
        TestWorkspaceCheckerRunMypy,
        TestWorkspaceCheckerRunPyright,
    )
    from tests.infra.unit.check.extended_models import (
        TestCheckIssueFormatted,
        TestProjectResultProperties,
        TestWorkspaceCheckerErrorSummary,
    )
    from tests.infra.unit.check.extended_project_runners import TestJsonWriteFailure
    from tests.infra.unit.check.extended_projects import (
        TestCheckProjectRunners,
        TestLintAndFormatPublicMethods,
    )
    from tests.infra.unit.check.extended_reports import (
        TestMarkdownReportSkipsEmptyGates,
        TestMarkdownReportWithErrors,
        TestWorkspaceCheckerMarkdownReport,
        TestWorkspaceCheckerMarkdownReportEdgeCases,
        TestWorkspaceCheckerSARIFReport,
        TestWorkspaceCheckerSARIFReportEdgeCases,
    )
    from tests.infra.unit.check.extended_resolve_gates import (
        TestWorkspaceCheckerParseGateCSV,
        TestWorkspaceCheckerResolveGates,
    )
    from tests.infra.unit.check.extended_run_projects import (
        CheckProjectStub,
        TestRunProjectsBehavior,
        TestRunProjectsReports,
        TestRunProjectsValidation,
        TestRunSingleProject,
    )
    from tests.infra.unit.check.extended_runners import TestRunMypy, TestRunPyrefly
    from tests.infra.unit.check.extended_runners_extra import (
        TestRunBandit,
        TestRunMarkdown,
        TestRunPyright,
    )
    from tests.infra.unit.check.extended_runners_go import TestRunGo
    from tests.infra.unit.check.extended_runners_ruff import (
        RunCallable,
        TestCollectMarkdownFiles,
        TestRunCommand,
        TestRunRuffFormat,
        TestRunRuffLint,
    )
    from tests.infra.unit.check.extended_workspace_init import (
        TestWorkspaceCheckerBuildGateResult,
        TestWorkspaceCheckerBuildGateResult as r,
        TestWorkspaceCheckerDirsWithPy,
        TestWorkspaceCheckerExecute,
        TestWorkspaceCheckerExistingCheckDirs,
        TestWorkspaceCheckerInitialization,
        TestWorkspaceCheckerInitOSError,
        TestWorkspaceCheckerResolveWorkspaceRootFallback,
    )
    from tests.infra.unit.check.fix_pyrefly_config import (
        test_fix_pyrefly_config_main_executes_real_cli_help,
    )
    from tests.infra.unit.check.init import TestFlextInfraCheck
    from tests.infra.unit.check.main import test_check_main_executes_real_cli
    from tests.infra.unit.check.pyrefly import TestFlextInfraConfigFixer
    from tests.infra.unit.check.workspace import TestFlextInfraWorkspaceChecker
    from tests.infra.unit.check.workspace_check import (
        test_workspace_check_main_returns_error_without_projects,
    )

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "CheckProjectStub": (
        "tests.infra.unit.check.extended_run_projects",
        "CheckProjectStub",
    ),
    "RunCallable": ("tests.infra.unit.check.extended_runners_ruff", "RunCallable"),
    "RunStub": ("tests.infra.unit.check.extended_error_reporting", "RunStub"),
    "TestCheckIssueFormatted": (
        "tests.infra.unit.check.extended_models",
        "TestCheckIssueFormatted",
    ),
    "TestCheckMainEntryPoint": (
        "tests.infra.unit.check.extended_cli_entry",
        "TestCheckMainEntryPoint",
    ),
    "TestCheckProjectRunners": (
        "tests.infra.unit.check.extended_projects",
        "TestCheckProjectRunners",
    ),
    "TestCollectMarkdownFiles": (
        "tests.infra.unit.check.extended_runners_ruff",
        "TestCollectMarkdownFiles",
    ),
    "TestConfigFixerEnsureProjectExcludes": (
        "tests.infra.unit.check.extended_config_fixer",
        "TestConfigFixerEnsureProjectExcludes",
    ),
    "TestConfigFixerExecute": (
        "tests.infra.unit.check.extended_config_fixer",
        "TestConfigFixerExecute",
    ),
    "TestConfigFixerFindPyprojectFiles": (
        "tests.infra.unit.check.extended_config_fixer",
        "TestConfigFixerFindPyprojectFiles",
    ),
    "TestConfigFixerFixSearchPaths": (
        "tests.infra.unit.check.extended_config_fixer",
        "TestConfigFixerFixSearchPaths",
    ),
    "TestConfigFixerPathResolution": (
        "tests.infra.unit.check.extended_config_fixer_errors",
        "TestConfigFixerPathResolution",
    ),
    "TestConfigFixerProcessFile": (
        "tests.infra.unit.check.extended_config_fixer",
        "TestConfigFixerProcessFile",
    ),
    "TestConfigFixerRemoveIgnoreSubConfig": (
        "tests.infra.unit.check.extended_config_fixer",
        "TestConfigFixerRemoveIgnoreSubConfig",
    ),
    "TestConfigFixerRun": (
        "tests.infra.unit.check.extended_config_fixer",
        "TestConfigFixerRun",
    ),
    "TestConfigFixerRunMethods": (
        "tests.infra.unit.check.extended_config_fixer_errors",
        "TestConfigFixerRunMethods",
    ),
    "TestConfigFixerRunWithVerbose": (
        "tests.infra.unit.check.extended_config_fixer_errors",
        "TestConfigFixerRunWithVerbose",
    ),
    "TestConfigFixerToArray": (
        "tests.infra.unit.check.extended_config_fixer",
        "TestConfigFixerToArray",
    ),
    "TestErrorReporting": (
        "tests.infra.unit.check.extended_error_reporting",
        "TestErrorReporting",
    ),
    "TestFixPyrelfyCLI": (
        "tests.infra.unit.check.extended_cli_entry",
        "TestFixPyrelfyCLI",
    ),
    "TestFlextInfraCheck": ("tests.infra.unit.check.init", "TestFlextInfraCheck"),
    "TestFlextInfraConfigFixer": (
        "tests.infra.unit.check.pyrefly",
        "TestFlextInfraConfigFixer",
    ),
    "TestFlextInfraWorkspaceChecker": (
        "tests.infra.unit.check.workspace",
        "TestFlextInfraWorkspaceChecker",
    ),
    "TestGoFmtEmptyLinesInOutput": (
        "tests.infra.unit.check.extended_error_reporting",
        "TestGoFmtEmptyLinesInOutput",
    ),
    "TestJsonWriteFailure": (
        "tests.infra.unit.check.extended_project_runners",
        "TestJsonWriteFailure",
    ),
    "TestLintAndFormatPublicMethods": (
        "tests.infra.unit.check.extended_projects",
        "TestLintAndFormatPublicMethods",
    ),
    "TestMarkdownReportEmptyGates": (
        "tests.infra.unit.check.extended_error_reporting",
        "TestMarkdownReportEmptyGates",
    ),
    "TestMarkdownReportSkipsEmptyGates": (
        "tests.infra.unit.check.extended_reports",
        "TestMarkdownReportSkipsEmptyGates",
    ),
    "TestMarkdownReportWithErrors": (
        "tests.infra.unit.check.extended_reports",
        "TestMarkdownReportWithErrors",
    ),
    "TestMypyEmptyLinesInOutput": (
        "tests.infra.unit.check.extended_error_reporting",
        "TestMypyEmptyLinesInOutput",
    ),
    "TestProcessFileReadError": (
        "tests.infra.unit.check.extended_config_fixer_errors",
        "TestProcessFileReadError",
    ),
    "TestProjectResultProperties": (
        "tests.infra.unit.check.extended_models",
        "TestProjectResultProperties",
    ),
    "TestRuffFormatDuplicateFiles": (
        "tests.infra.unit.check.extended_error_reporting",
        "TestRuffFormatDuplicateFiles",
    ),
    "TestRunBandit": ("tests.infra.unit.check.extended_runners_extra", "TestRunBandit"),
    "TestRunCLIExtended": (
        "tests.infra.unit.check.extended_cli_entry",
        "TestRunCLIExtended",
    ),
    "TestRunCommand": (
        "tests.infra.unit.check.extended_runners_ruff",
        "TestRunCommand",
    ),
    "TestRunGo": ("tests.infra.unit.check.extended_runners_go", "TestRunGo"),
    "TestRunMarkdown": (
        "tests.infra.unit.check.extended_runners_extra",
        "TestRunMarkdown",
    ),
    "TestRunMypy": ("tests.infra.unit.check.extended_runners", "TestRunMypy"),
    "TestRunProjectsBehavior": (
        "tests.infra.unit.check.extended_run_projects",
        "TestRunProjectsBehavior",
    ),
    "TestRunProjectsReports": (
        "tests.infra.unit.check.extended_run_projects",
        "TestRunProjectsReports",
    ),
    "TestRunProjectsValidation": (
        "tests.infra.unit.check.extended_run_projects",
        "TestRunProjectsValidation",
    ),
    "TestRunPyrefly": ("tests.infra.unit.check.extended_runners", "TestRunPyrefly"),
    "TestRunPyright": (
        "tests.infra.unit.check.extended_runners_extra",
        "TestRunPyright",
    ),
    "TestRunRuffFormat": (
        "tests.infra.unit.check.extended_runners_ruff",
        "TestRunRuffFormat",
    ),
    "TestRunRuffLint": (
        "tests.infra.unit.check.extended_runners_ruff",
        "TestRunRuffLint",
    ),
    "TestRunSingleProject": (
        "tests.infra.unit.check.extended_run_projects",
        "TestRunSingleProject",
    ),
    "TestWorkspaceCheckCLI": (
        "tests.infra.unit.check.extended_cli_entry",
        "TestWorkspaceCheckCLI",
    ),
    "TestWorkspaceCheckerBuildGateResult": (
        "tests.infra.unit.check.extended_workspace_init",
        "TestWorkspaceCheckerBuildGateResult",
    ),
    "TestWorkspaceCheckerCollectMarkdownFiles": (
        "tests.infra.unit.check.extended_gate_go_cmd",
        "TestWorkspaceCheckerCollectMarkdownFiles",
    ),
    "TestWorkspaceCheckerDirsWithPy": (
        "tests.infra.unit.check.extended_workspace_init",
        "TestWorkspaceCheckerDirsWithPy",
    ),
    "TestWorkspaceCheckerErrorSummary": (
        "tests.infra.unit.check.extended_models",
        "TestWorkspaceCheckerErrorSummary",
    ),
    "TestWorkspaceCheckerExecute": (
        "tests.infra.unit.check.extended_workspace_init",
        "TestWorkspaceCheckerExecute",
    ),
    "TestWorkspaceCheckerExistingCheckDirs": (
        "tests.infra.unit.check.extended_workspace_init",
        "TestWorkspaceCheckerExistingCheckDirs",
    ),
    "TestWorkspaceCheckerInitOSError": (
        "tests.infra.unit.check.extended_workspace_init",
        "TestWorkspaceCheckerInitOSError",
    ),
    "TestWorkspaceCheckerInitialization": (
        "tests.infra.unit.check.extended_workspace_init",
        "TestWorkspaceCheckerInitialization",
    ),
    "TestWorkspaceCheckerMarkdownReport": (
        "tests.infra.unit.check.extended_reports",
        "TestWorkspaceCheckerMarkdownReport",
    ),
    "TestWorkspaceCheckerMarkdownReportEdgeCases": (
        "tests.infra.unit.check.extended_reports",
        "TestWorkspaceCheckerMarkdownReportEdgeCases",
    ),
    "TestWorkspaceCheckerParseGateCSV": (
        "tests.infra.unit.check.extended_resolve_gates",
        "TestWorkspaceCheckerParseGateCSV",
    ),
    "TestWorkspaceCheckerResolveGates": (
        "tests.infra.unit.check.extended_resolve_gates",
        "TestWorkspaceCheckerResolveGates",
    ),
    "TestWorkspaceCheckerResolveWorkspaceRootFallback": (
        "tests.infra.unit.check.extended_workspace_init",
        "TestWorkspaceCheckerResolveWorkspaceRootFallback",
    ),
    "TestWorkspaceCheckerRunBandit": (
        "tests.infra.unit.check.extended_gate_bandit_markdown",
        "TestWorkspaceCheckerRunBandit",
    ),
    "TestWorkspaceCheckerRunCommand": (
        "tests.infra.unit.check.extended_gate_go_cmd",
        "TestWorkspaceCheckerRunCommand",
    ),
    "TestWorkspaceCheckerRunGo": (
        "tests.infra.unit.check.extended_gate_go_cmd",
        "TestWorkspaceCheckerRunGo",
    ),
    "TestWorkspaceCheckerRunMarkdown": (
        "tests.infra.unit.check.extended_gate_bandit_markdown",
        "TestWorkspaceCheckerRunMarkdown",
    ),
    "TestWorkspaceCheckerRunMypy": (
        "tests.infra.unit.check.extended_gate_mypy_pyright",
        "TestWorkspaceCheckerRunMypy",
    ),
    "TestWorkspaceCheckerRunPyright": (
        "tests.infra.unit.check.extended_gate_mypy_pyright",
        "TestWorkspaceCheckerRunPyright",
    ),
    "TestWorkspaceCheckerSARIFReport": (
        "tests.infra.unit.check.extended_reports",
        "TestWorkspaceCheckerSARIFReport",
    ),
    "TestWorkspaceCheckerSARIFReportEdgeCases": (
        "tests.infra.unit.check.extended_reports",
        "TestWorkspaceCheckerSARIFReportEdgeCases",
    ),
    "r": (
        "tests.infra.unit.check.extended_workspace_init",
        "TestWorkspaceCheckerBuildGateResult",
    ),
    "test_check_main_executes_real_cli": (
        "tests.infra.unit.check.main",
        "test_check_main_executes_real_cli",
    ),
    "test_fix_pyrefly_config_main_executes_real_cli_help": (
        "tests.infra.unit.check.fix_pyrefly_config",
        "test_fix_pyrefly_config_main_executes_real_cli_help",
    ),
    "test_resolve_gates_maps_type_alias": (
        "tests.infra.unit.check.cli",
        "test_resolve_gates_maps_type_alias",
    ),
    "test_run_cli_run_returns_one_for_fail": (
        "tests.infra.unit.check.cli",
        "test_run_cli_run_returns_one_for_fail",
    ),
    "test_run_cli_run_returns_two_for_error": (
        "tests.infra.unit.check.cli",
        "test_run_cli_run_returns_two_for_error",
    ),
    "test_run_cli_run_returns_zero_for_pass": (
        "tests.infra.unit.check.cli",
        "test_run_cli_run_returns_zero_for_pass",
    ),
    "test_run_cli_with_fail_fast_flag": (
        "tests.infra.unit.check.cli",
        "test_run_cli_with_fail_fast_flag",
    ),
    "test_run_cli_with_multiple_projects": (
        "tests.infra.unit.check.cli",
        "test_run_cli_with_multiple_projects",
    ),
    "test_workspace_check_main_returns_error_without_projects": (
        "tests.infra.unit.check.workspace_check",
        "test_workspace_check_main_returns_error_without_projects",
    ),
}

__all__ = [
    "CheckProjectStub",
    "RunCallable",
    "RunStub",
    "TestCheckIssueFormatted",
    "TestCheckMainEntryPoint",
    "TestCheckProjectRunners",
    "TestCollectMarkdownFiles",
    "TestConfigFixerEnsureProjectExcludes",
    "TestConfigFixerExecute",
    "TestConfigFixerFindPyprojectFiles",
    "TestConfigFixerFixSearchPaths",
    "TestConfigFixerPathResolution",
    "TestConfigFixerProcessFile",
    "TestConfigFixerRemoveIgnoreSubConfig",
    "TestConfigFixerRun",
    "TestConfigFixerRunMethods",
    "TestConfigFixerRunWithVerbose",
    "TestConfigFixerToArray",
    "TestErrorReporting",
    "TestFixPyrelfyCLI",
    "TestFlextInfraCheck",
    "TestFlextInfraConfigFixer",
    "TestFlextInfraWorkspaceChecker",
    "TestGoFmtEmptyLinesInOutput",
    "TestJsonWriteFailure",
    "TestLintAndFormatPublicMethods",
    "TestMarkdownReportEmptyGates",
    "TestMarkdownReportSkipsEmptyGates",
    "TestMarkdownReportWithErrors",
    "TestMypyEmptyLinesInOutput",
    "TestProcessFileReadError",
    "TestProjectResultProperties",
    "TestRuffFormatDuplicateFiles",
    "TestRunBandit",
    "TestRunCLIExtended",
    "TestRunCommand",
    "TestRunGo",
    "TestRunMarkdown",
    "TestRunMypy",
    "TestRunProjectsBehavior",
    "TestRunProjectsReports",
    "TestRunProjectsValidation",
    "TestRunPyrefly",
    "TestRunPyright",
    "TestRunRuffFormat",
    "TestRunRuffLint",
    "TestRunSingleProject",
    "TestWorkspaceCheckCLI",
    "TestWorkspaceCheckerBuildGateResult",
    "TestWorkspaceCheckerCollectMarkdownFiles",
    "TestWorkspaceCheckerDirsWithPy",
    "TestWorkspaceCheckerErrorSummary",
    "TestWorkspaceCheckerExecute",
    "TestWorkspaceCheckerExistingCheckDirs",
    "TestWorkspaceCheckerInitOSError",
    "TestWorkspaceCheckerInitialization",
    "TestWorkspaceCheckerMarkdownReport",
    "TestWorkspaceCheckerMarkdownReportEdgeCases",
    "TestWorkspaceCheckerParseGateCSV",
    "TestWorkspaceCheckerResolveGates",
    "TestWorkspaceCheckerResolveWorkspaceRootFallback",
    "TestWorkspaceCheckerRunBandit",
    "TestWorkspaceCheckerRunCommand",
    "TestWorkspaceCheckerRunGo",
    "TestWorkspaceCheckerRunMarkdown",
    "TestWorkspaceCheckerRunMypy",
    "TestWorkspaceCheckerRunPyright",
    "TestWorkspaceCheckerSARIFReport",
    "TestWorkspaceCheckerSARIFReportEdgeCases",
    "r",
    "test_check_main_executes_real_cli",
    "test_fix_pyrefly_config_main_executes_real_cli_help",
    "test_resolve_gates_maps_type_alias",
    "test_run_cli_run_returns_one_for_fail",
    "test_run_cli_run_returns_two_for_error",
    "test_run_cli_run_returns_zero_for_pass",
    "test_run_cli_with_fail_fast_flag",
    "test_run_cli_with_multiple_projects",
    "test_workspace_check_main_returns_error_without_projects",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
