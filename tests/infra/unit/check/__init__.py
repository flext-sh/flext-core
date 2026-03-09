# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Check package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from tests.infra.unit.check.cli import (
        test_resolve_gates_maps_type_alias,
        test_run_cli_run_returns_one_for_fail,
        test_run_cli_run_returns_two_for_error,
        test_run_cli_run_returns_zero_for_pass,
        test_run_cli_with_fail_fast_flag,
        test_run_cli_with_multiple_projects,
    )
    from tests.infra.unit.check.extended import (
        TestCheckIssueFormatted,
        TestCheckMainEntryPoint,
        TestConfigFixerApplyFixesEmptyProject,
        TestConfigFixerEnsureProjectExcludes,
        TestConfigFixerExecute,
        TestConfigFixerFindPyprojectFiles,
        TestConfigFixerFixSearchPaths,
        TestConfigFixerPathRelativeToError,
        TestConfigFixerPathResolution,
        TestConfigFixerProcessFile,
        TestConfigFixerProcessFileErrors,
        TestConfigFixerRemoveIgnoreSubConfig,
        TestConfigFixerRun,
        TestConfigFixerRunMethods,
        TestConfigFixerRunWithVerbose,
        TestConfigFixerToArray,
        TestFixPyrelfyCLI,
        TestGoFormatEmptyLines,
        TestGoFormatEmptyLineSkipping,
        TestGoFormatParsing,
        TestJsonWriteFailure,
        TestLintAndFormatPublicMethods,
        TestMarkdownLinting,
        TestMarkdownReportSkipsEmptyGates,
        TestMarkdownReportWithErrors,
        TestMypyEmptyLines,
        TestMypyEmptyLineSkipping,
        TestMypyJSONParsing,
        TestProcessFileReadError,
        TestProjectResultProperties,
        TestRuffFormatDeduplication,
        TestRuffFormatDuplicates,
        TestRuffFormatDuplicateSkipping,
        TestRuffFormatEmptyLines,
        TestRunCLI,
        TestWorkspaceCheckCLI,
        TestWorkspaceCheckerBuildGateResult,
        TestWorkspaceCheckerBuildGateResult as r,
        TestWorkspaceCheckerCheckProjectMethods,
        TestWorkspaceCheckerCollectMarkdownFiles,
        TestWorkspaceCheckerDirsWithPy,
        TestWorkspaceCheckerErrorReporting,
        TestWorkspaceCheckerErrorReportingMultipleProjects,
        TestWorkspaceCheckerErrorSummary,
        TestWorkspaceCheckerExecute,
        TestWorkspaceCheckerExistingCheckDirs,
        TestWorkspaceCheckerGoFmtEmptyLines,
        TestWorkspaceCheckerGoFmtEmptyLinesInOutput,
        TestWorkspaceCheckerInitialization,
        TestWorkspaceCheckerInitOSError,
        TestWorkspaceCheckerMarkdownReport,
        TestWorkspaceCheckerMarkdownReportEdgeCases,
        TestWorkspaceCheckerMarkdownReportEmptyGates,
        TestWorkspaceCheckerMypyEmptyLines,
        TestWorkspaceCheckerMypyEmptyLinesInOutput,
        TestWorkspaceCheckerParseGateCSV,
        TestWorkspaceCheckerResolveGates,
        TestWorkspaceCheckerResolveWorkspaceRootFallback,
        TestWorkspaceCheckerRuffFormatDuplicateFiles,
        TestWorkspaceCheckerRuffFormatDuplicates,
        TestWorkspaceCheckerRun,
        TestWorkspaceCheckerRunBandit,
        TestWorkspaceCheckerRunCommand,
        TestWorkspaceCheckerRunGo,
        TestWorkspaceCheckerRunMarkdown,
        TestWorkspaceCheckerRunMypy,
        TestWorkspaceCheckerRunProjects,
        TestWorkspaceCheckerRunPyrefly,
        TestWorkspaceCheckerRunPyright,
        TestWorkspaceCheckerRunRuffFormat,
        TestWorkspaceCheckerRunRuffLint,
        TestWorkspaceCheckerSARIFReport,
        TestWorkspaceCheckerSARIFReportEdgeCases,
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

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "TestCheckIssueFormatted": (
        "tests.infra.unit.check.extended",
        "TestCheckIssueFormatted",
    ),
    "TestCheckMainEntryPoint": (
        "tests.infra.unit.check.extended",
        "TestCheckMainEntryPoint",
    ),
    "TestConfigFixerApplyFixesEmptyProject": (
        "tests.infra.unit.check.extended",
        "TestConfigFixerApplyFixesEmptyProject",
    ),
    "TestConfigFixerEnsureProjectExcludes": (
        "tests.infra.unit.check.extended",
        "TestConfigFixerEnsureProjectExcludes",
    ),
    "TestConfigFixerExecute": (
        "tests.infra.unit.check.extended",
        "TestConfigFixerExecute",
    ),
    "TestConfigFixerFindPyprojectFiles": (
        "tests.infra.unit.check.extended",
        "TestConfigFixerFindPyprojectFiles",
    ),
    "TestConfigFixerFixSearchPaths": (
        "tests.infra.unit.check.extended",
        "TestConfigFixerFixSearchPaths",
    ),
    "TestConfigFixerPathRelativeToError": (
        "tests.infra.unit.check.extended",
        "TestConfigFixerPathRelativeToError",
    ),
    "TestConfigFixerPathResolution": (
        "tests.infra.unit.check.extended",
        "TestConfigFixerPathResolution",
    ),
    "TestConfigFixerProcessFile": (
        "tests.infra.unit.check.extended",
        "TestConfigFixerProcessFile",
    ),
    "TestConfigFixerProcessFileErrors": (
        "tests.infra.unit.check.extended",
        "TestConfigFixerProcessFileErrors",
    ),
    "TestConfigFixerRemoveIgnoreSubConfig": (
        "tests.infra.unit.check.extended",
        "TestConfigFixerRemoveIgnoreSubConfig",
    ),
    "TestConfigFixerRun": ("tests.infra.unit.check.extended", "TestConfigFixerRun"),
    "TestConfigFixerRunMethods": (
        "tests.infra.unit.check.extended",
        "TestConfigFixerRunMethods",
    ),
    "TestConfigFixerRunWithVerbose": (
        "tests.infra.unit.check.extended",
        "TestConfigFixerRunWithVerbose",
    ),
    "TestConfigFixerToArray": (
        "tests.infra.unit.check.extended",
        "TestConfigFixerToArray",
    ),
    "TestFixPyrelfyCLI": ("tests.infra.unit.check.extended", "TestFixPyrelfyCLI"),
    "TestFlextInfraCheck": ("tests.infra.unit.check.init", "TestFlextInfraCheck"),
    "TestFlextInfraConfigFixer": (
        "tests.infra.unit.check.pyrefly",
        "TestFlextInfraConfigFixer",
    ),
    "TestFlextInfraWorkspaceChecker": (
        "tests.infra.unit.check.workspace",
        "TestFlextInfraWorkspaceChecker",
    ),
    "TestGoFormatEmptyLineSkipping": (
        "tests.infra.unit.check.extended",
        "TestGoFormatEmptyLineSkipping",
    ),
    "TestGoFormatEmptyLines": (
        "tests.infra.unit.check.extended",
        "TestGoFormatEmptyLines",
    ),
    "TestGoFormatParsing": ("tests.infra.unit.check.extended", "TestGoFormatParsing"),
    "TestJsonWriteFailure": ("tests.infra.unit.check.extended", "TestJsonWriteFailure"),
    "TestLintAndFormatPublicMethods": (
        "tests.infra.unit.check.extended",
        "TestLintAndFormatPublicMethods",
    ),
    "TestMarkdownLinting": ("tests.infra.unit.check.extended", "TestMarkdownLinting"),
    "TestMarkdownReportSkipsEmptyGates": (
        "tests.infra.unit.check.extended",
        "TestMarkdownReportSkipsEmptyGates",
    ),
    "TestMarkdownReportWithErrors": (
        "tests.infra.unit.check.extended",
        "TestMarkdownReportWithErrors",
    ),
    "TestMypyEmptyLineSkipping": (
        "tests.infra.unit.check.extended",
        "TestMypyEmptyLineSkipping",
    ),
    "TestMypyEmptyLines": ("tests.infra.unit.check.extended", "TestMypyEmptyLines"),
    "TestMypyJSONParsing": ("tests.infra.unit.check.extended", "TestMypyJSONParsing"),
    "TestProcessFileReadError": (
        "tests.infra.unit.check.extended",
        "TestProcessFileReadError",
    ),
    "TestProjectResultProperties": (
        "tests.infra.unit.check.extended",
        "TestProjectResultProperties",
    ),
    "TestRuffFormatDeduplication": (
        "tests.infra.unit.check.extended",
        "TestRuffFormatDeduplication",
    ),
    "TestRuffFormatDuplicateSkipping": (
        "tests.infra.unit.check.extended",
        "TestRuffFormatDuplicateSkipping",
    ),
    "TestRuffFormatDuplicates": (
        "tests.infra.unit.check.extended",
        "TestRuffFormatDuplicates",
    ),
    "TestRuffFormatEmptyLines": (
        "tests.infra.unit.check.extended",
        "TestRuffFormatEmptyLines",
    ),
    "TestRunCLI": ("tests.infra.unit.check.extended", "TestRunCLI"),
    "TestWorkspaceCheckCLI": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckCLI",
    ),
    "TestWorkspaceCheckerBuildGateResult": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerBuildGateResult",
    ),
    "TestWorkspaceCheckerCheckProjectMethods": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerCheckProjectMethods",
    ),
    "TestWorkspaceCheckerCollectMarkdownFiles": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerCollectMarkdownFiles",
    ),
    "TestWorkspaceCheckerDirsWithPy": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerDirsWithPy",
    ),
    "TestWorkspaceCheckerErrorReporting": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerErrorReporting",
    ),
    "TestWorkspaceCheckerErrorReportingMultipleProjects": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerErrorReportingMultipleProjects",
    ),
    "TestWorkspaceCheckerErrorSummary": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerErrorSummary",
    ),
    "TestWorkspaceCheckerExecute": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerExecute",
    ),
    "TestWorkspaceCheckerExistingCheckDirs": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerExistingCheckDirs",
    ),
    "TestWorkspaceCheckerGoFmtEmptyLines": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerGoFmtEmptyLines",
    ),
    "TestWorkspaceCheckerGoFmtEmptyLinesInOutput": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerGoFmtEmptyLinesInOutput",
    ),
    "TestWorkspaceCheckerInitOSError": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerInitOSError",
    ),
    "TestWorkspaceCheckerInitialization": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerInitialization",
    ),
    "TestWorkspaceCheckerMarkdownReport": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerMarkdownReport",
    ),
    "TestWorkspaceCheckerMarkdownReportEdgeCases": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerMarkdownReportEdgeCases",
    ),
    "TestWorkspaceCheckerMarkdownReportEmptyGates": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerMarkdownReportEmptyGates",
    ),
    "TestWorkspaceCheckerMypyEmptyLines": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerMypyEmptyLines",
    ),
    "TestWorkspaceCheckerMypyEmptyLinesInOutput": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerMypyEmptyLinesInOutput",
    ),
    "TestWorkspaceCheckerParseGateCSV": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerParseGateCSV",
    ),
    "TestWorkspaceCheckerResolveGates": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerResolveGates",
    ),
    "TestWorkspaceCheckerResolveWorkspaceRootFallback": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerResolveWorkspaceRootFallback",
    ),
    "TestWorkspaceCheckerRuffFormatDuplicateFiles": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerRuffFormatDuplicateFiles",
    ),
    "TestWorkspaceCheckerRuffFormatDuplicates": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerRuffFormatDuplicates",
    ),
    "TestWorkspaceCheckerRun": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerRun",
    ),
    "TestWorkspaceCheckerRunBandit": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerRunBandit",
    ),
    "TestWorkspaceCheckerRunCommand": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerRunCommand",
    ),
    "TestWorkspaceCheckerRunGo": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerRunGo",
    ),
    "TestWorkspaceCheckerRunMarkdown": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerRunMarkdown",
    ),
    "TestWorkspaceCheckerRunMypy": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerRunMypy",
    ),
    "TestWorkspaceCheckerRunProjects": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerRunProjects",
    ),
    "TestWorkspaceCheckerRunPyrefly": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerRunPyrefly",
    ),
    "TestWorkspaceCheckerRunPyright": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerRunPyright",
    ),
    "TestWorkspaceCheckerRunRuffFormat": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerRunRuffFormat",
    ),
    "TestWorkspaceCheckerRunRuffLint": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerRunRuffLint",
    ),
    "TestWorkspaceCheckerSARIFReport": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerSARIFReport",
    ),
    "TestWorkspaceCheckerSARIFReportEdgeCases": (
        "tests.infra.unit.check.extended",
        "TestWorkspaceCheckerSARIFReportEdgeCases",
    ),
    "r": ("tests.infra.unit.check.extended", "TestWorkspaceCheckerBuildGateResult"),
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
    "TestCheckIssueFormatted",
    "TestCheckMainEntryPoint",
    "TestConfigFixerApplyFixesEmptyProject",
    "TestConfigFixerEnsureProjectExcludes",
    "TestConfigFixerExecute",
    "TestConfigFixerFindPyprojectFiles",
    "TestConfigFixerFixSearchPaths",
    "TestConfigFixerPathRelativeToError",
    "TestConfigFixerPathResolution",
    "TestConfigFixerProcessFile",
    "TestConfigFixerProcessFileErrors",
    "TestConfigFixerRemoveIgnoreSubConfig",
    "TestConfigFixerRun",
    "TestConfigFixerRunMethods",
    "TestConfigFixerRunWithVerbose",
    "TestConfigFixerToArray",
    "TestFixPyrelfyCLI",
    "TestFlextInfraCheck",
    "TestFlextInfraConfigFixer",
    "TestFlextInfraWorkspaceChecker",
    "TestGoFormatEmptyLineSkipping",
    "TestGoFormatEmptyLines",
    "TestGoFormatParsing",
    "TestJsonWriteFailure",
    "TestLintAndFormatPublicMethods",
    "TestMarkdownLinting",
    "TestMarkdownReportSkipsEmptyGates",
    "TestMarkdownReportWithErrors",
    "TestMypyEmptyLineSkipping",
    "TestMypyEmptyLines",
    "TestMypyJSONParsing",
    "TestProcessFileReadError",
    "TestProjectResultProperties",
    "TestRuffFormatDeduplication",
    "TestRuffFormatDuplicateSkipping",
    "TestRuffFormatDuplicates",
    "TestRuffFormatEmptyLines",
    "TestRunCLI",
    "TestWorkspaceCheckCLI",
    "TestWorkspaceCheckerBuildGateResult",
    "TestWorkspaceCheckerCheckProjectMethods",
    "TestWorkspaceCheckerCollectMarkdownFiles",
    "TestWorkspaceCheckerDirsWithPy",
    "TestWorkspaceCheckerErrorReporting",
    "TestWorkspaceCheckerErrorReportingMultipleProjects",
    "TestWorkspaceCheckerErrorSummary",
    "TestWorkspaceCheckerExecute",
    "TestWorkspaceCheckerExistingCheckDirs",
    "TestWorkspaceCheckerGoFmtEmptyLines",
    "TestWorkspaceCheckerGoFmtEmptyLinesInOutput",
    "TestWorkspaceCheckerInitOSError",
    "TestWorkspaceCheckerInitialization",
    "TestWorkspaceCheckerMarkdownReport",
    "TestWorkspaceCheckerMarkdownReportEdgeCases",
    "TestWorkspaceCheckerMarkdownReportEmptyGates",
    "TestWorkspaceCheckerMypyEmptyLines",
    "TestWorkspaceCheckerMypyEmptyLinesInOutput",
    "TestWorkspaceCheckerParseGateCSV",
    "TestWorkspaceCheckerResolveGates",
    "TestWorkspaceCheckerResolveWorkspaceRootFallback",
    "TestWorkspaceCheckerRuffFormatDuplicateFiles",
    "TestWorkspaceCheckerRuffFormatDuplicates",
    "TestWorkspaceCheckerRun",
    "TestWorkspaceCheckerRunBandit",
    "TestWorkspaceCheckerRunCommand",
    "TestWorkspaceCheckerRunGo",
    "TestWorkspaceCheckerRunMarkdown",
    "TestWorkspaceCheckerRunMypy",
    "TestWorkspaceCheckerRunProjects",
    "TestWorkspaceCheckerRunPyrefly",
    "TestWorkspaceCheckerRunPyright",
    "TestWorkspaceCheckerRunRuffFormat",
    "TestWorkspaceCheckerRunRuffLint",
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


def __getattr__(name: str) -> Any:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
