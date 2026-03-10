# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Codegen package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from tests.infra.unit.codegen.autofix import (
        fixer,
        test_files_modified_tracks_affected_files,
        test_flexcore_excluded_from_run,
        test_in_context_typevar_not_flagged,
        test_project_without_src_returns_empty,
        test_standalone_final_detected_as_fixable,
        test_standalone_typealias_detected_as_fixable,
        test_standalone_typevar_detected_as_fixable,
        test_syntax_error_files_skipped,
    )
    from tests.infra.unit.codegen.census import (
        TestCensusReportModel,
        TestCensusViolationModel,
        TestExcludedProjects,
        TestFixabilityClassification,
        TestParseViolationInvalid,
        TestParseViolationValid,
        TestViolationPattern,
        census,
    )
    from tests.infra.unit.codegen.constants_quality_gate import (
        test_handle_constants_quality_gate_json_exits_with_int,
        test_handle_constants_quality_gate_text_exits_with_int,
        test_main_constants_quality_gate_dispatch,
        test_main_constants_quality_gate_parses_before_report,
        test_quality_gate_real_workspace_run,
        test_quality_gate_success_verdict_helper,
    )
    from tests.infra.unit.codegen.init import (
        test_codegen_dir_returns_all_exports,
        test_codegen_getattr_raises_attribute_error,
        test_codegen_lazy_imports_work,
    )
    from tests.infra.unit.codegen.lazy_init import (
        TestBuildSiblingExportIndex,
        TestExtractExports,
        TestExtractInlineConstants,
        TestExtractInlineConstants as c,
        TestExtractVersionExports,
        TestFlextInfraCodegenLazyInit,
        TestGenerateFile,
        TestGenerateTypeChecking,
        TestInferPackage,
        TestMergeChildExports,
        TestProcessDirectory,
        TestReadExistingDocstring,
        TestResolveAliases,
        TestRunRuffFix,
        TestScanAstPublicDefs,
        TestShouldBubbleUp,
        test_codegen_init_getattr_raises_attribute_error,
    )
    from tests.infra.unit.codegen.lazy_init_tests import (
        TestAllDirectoriesScanned,
        TestCheckOnlyMode,
        TestEdgeCases,
        TestExcludedDirectories,
    )
    from tests.infra.unit.codegen.main import (
        TestHandleLazyInit,
        TestMainCommandDispatch,
        TestMainEntryPoint,
    )
    from tests.infra.unit.codegen.pipeline import test_codegen_pipeline_end_to_end
    from tests.infra.unit.codegen.scaffolder import (
        TestGeneratedClassNamingConvention,
        TestGeneratedFilesAreValidPython,
        TestScaffoldProjectCreatesSrcModules,
        TestScaffoldProjectCreatesTestsModules,
        TestScaffoldProjectIdempotency,
        TestScaffoldProjectNoop,
    )

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "TestAllDirectoriesScanned": (
        "tests.infra.unit.codegen.lazy_init_tests",
        "TestAllDirectoriesScanned",
    ),
    "TestBuildSiblingExportIndex": (
        "tests.infra.unit.codegen.lazy_init",
        "TestBuildSiblingExportIndex",
    ),
    "TestCensusReportModel": (
        "tests.infra.unit.codegen.census",
        "TestCensusReportModel",
    ),
    "TestCensusViolationModel": (
        "tests.infra.unit.codegen.census",
        "TestCensusViolationModel",
    ),
    "TestCheckOnlyMode": (
        "tests.infra.unit.codegen.lazy_init_tests",
        "TestCheckOnlyMode",
    ),
    "TestEdgeCases": ("tests.infra.unit.codegen.lazy_init_tests", "TestEdgeCases"),
    "TestExcludedDirectories": (
        "tests.infra.unit.codegen.lazy_init_tests",
        "TestExcludedDirectories",
    ),
    "TestExcludedProjects": ("tests.infra.unit.codegen.census", "TestExcludedProjects"),
    "TestExtractExports": ("tests.infra.unit.codegen.lazy_init", "TestExtractExports"),
    "TestExtractInlineConstants": (
        "tests.infra.unit.codegen.lazy_init",
        "TestExtractInlineConstants",
    ),
    "TestExtractVersionExports": (
        "tests.infra.unit.codegen.lazy_init",
        "TestExtractVersionExports",
    ),
    "TestFixabilityClassification": (
        "tests.infra.unit.codegen.census",
        "TestFixabilityClassification",
    ),
    "TestFlextInfraCodegenLazyInit": (
        "tests.infra.unit.codegen.lazy_init",
        "TestFlextInfraCodegenLazyInit",
    ),
    "TestGenerateFile": ("tests.infra.unit.codegen.lazy_init", "TestGenerateFile"),
    "TestGenerateTypeChecking": (
        "tests.infra.unit.codegen.lazy_init",
        "TestGenerateTypeChecking",
    ),
    "TestGeneratedClassNamingConvention": (
        "tests.infra.unit.codegen.scaffolder",
        "TestGeneratedClassNamingConvention",
    ),
    "TestGeneratedFilesAreValidPython": (
        "tests.infra.unit.codegen.scaffolder",
        "TestGeneratedFilesAreValidPython",
    ),
    "TestHandleLazyInit": ("tests.infra.unit.codegen.main", "TestHandleLazyInit"),
    "TestInferPackage": ("tests.infra.unit.codegen.lazy_init", "TestInferPackage"),
    "TestMainCommandDispatch": (
        "tests.infra.unit.codegen.main",
        "TestMainCommandDispatch",
    ),
    "TestMainEntryPoint": ("tests.infra.unit.codegen.main", "TestMainEntryPoint"),
    "TestMergeChildExports": (
        "tests.infra.unit.codegen.lazy_init",
        "TestMergeChildExports",
    ),
    "TestParseViolationInvalid": (
        "tests.infra.unit.codegen.census",
        "TestParseViolationInvalid",
    ),
    "TestParseViolationValid": (
        "tests.infra.unit.codegen.census",
        "TestParseViolationValid",
    ),
    "TestProcessDirectory": (
        "tests.infra.unit.codegen.lazy_init",
        "TestProcessDirectory",
    ),
    "TestReadExistingDocstring": (
        "tests.infra.unit.codegen.lazy_init",
        "TestReadExistingDocstring",
    ),
    "TestResolveAliases": ("tests.infra.unit.codegen.lazy_init", "TestResolveAliases"),
    "TestRunRuffFix": ("tests.infra.unit.codegen.lazy_init", "TestRunRuffFix"),
    "TestScaffoldProjectCreatesSrcModules": (
        "tests.infra.unit.codegen.scaffolder",
        "TestScaffoldProjectCreatesSrcModules",
    ),
    "TestScaffoldProjectCreatesTestsModules": (
        "tests.infra.unit.codegen.scaffolder",
        "TestScaffoldProjectCreatesTestsModules",
    ),
    "TestScaffoldProjectIdempotency": (
        "tests.infra.unit.codegen.scaffolder",
        "TestScaffoldProjectIdempotency",
    ),
    "TestScaffoldProjectNoop": (
        "tests.infra.unit.codegen.scaffolder",
        "TestScaffoldProjectNoop",
    ),
    "TestScanAstPublicDefs": (
        "tests.infra.unit.codegen.lazy_init",
        "TestScanAstPublicDefs",
    ),
    "TestShouldBubbleUp": ("tests.infra.unit.codegen.lazy_init", "TestShouldBubbleUp"),
    "TestViolationPattern": ("tests.infra.unit.codegen.census", "TestViolationPattern"),
    "c": ("tests.infra.unit.codegen.lazy_init", "TestExtractInlineConstants"),
    "census": ("tests.infra.unit.codegen.census", "census"),
    "fixer": ("tests.infra.unit.codegen.autofix", "fixer"),
    "test_codegen_dir_returns_all_exports": (
        "tests.infra.unit.codegen.init",
        "test_codegen_dir_returns_all_exports",
    ),
    "test_codegen_getattr_raises_attribute_error": (
        "tests.infra.unit.codegen.init",
        "test_codegen_getattr_raises_attribute_error",
    ),
    "test_codegen_init_getattr_raises_attribute_error": (
        "tests.infra.unit.codegen.lazy_init",
        "test_codegen_init_getattr_raises_attribute_error",
    ),
    "test_codegen_lazy_imports_work": (
        "tests.infra.unit.codegen.init",
        "test_codegen_lazy_imports_work",
    ),
    "test_codegen_pipeline_end_to_end": (
        "tests.infra.unit.codegen.pipeline",
        "test_codegen_pipeline_end_to_end",
    ),
    "test_files_modified_tracks_affected_files": (
        "tests.infra.unit.codegen.autofix",
        "test_files_modified_tracks_affected_files",
    ),
    "test_flexcore_excluded_from_run": (
        "tests.infra.unit.codegen.autofix",
        "test_flexcore_excluded_from_run",
    ),
    "test_handle_constants_quality_gate_json_exits_with_int": (
        "tests.infra.unit.codegen.constants_quality_gate",
        "test_handle_constants_quality_gate_json_exits_with_int",
    ),
    "test_handle_constants_quality_gate_text_exits_with_int": (
        "tests.infra.unit.codegen.constants_quality_gate",
        "test_handle_constants_quality_gate_text_exits_with_int",
    ),
    "test_in_context_typevar_not_flagged": (
        "tests.infra.unit.codegen.autofix",
        "test_in_context_typevar_not_flagged",
    ),
    "test_main_constants_quality_gate_dispatch": (
        "tests.infra.unit.codegen.constants_quality_gate",
        "test_main_constants_quality_gate_dispatch",
    ),
    "test_main_constants_quality_gate_parses_before_report": (
        "tests.infra.unit.codegen.constants_quality_gate",
        "test_main_constants_quality_gate_parses_before_report",
    ),
    "test_project_without_src_returns_empty": (
        "tests.infra.unit.codegen.autofix",
        "test_project_without_src_returns_empty",
    ),
    "test_quality_gate_real_workspace_run": (
        "tests.infra.unit.codegen.constants_quality_gate",
        "test_quality_gate_real_workspace_run",
    ),
    "test_quality_gate_success_verdict_helper": (
        "tests.infra.unit.codegen.constants_quality_gate",
        "test_quality_gate_success_verdict_helper",
    ),
    "test_standalone_final_detected_as_fixable": (
        "tests.infra.unit.codegen.autofix",
        "test_standalone_final_detected_as_fixable",
    ),
    "test_standalone_typealias_detected_as_fixable": (
        "tests.infra.unit.codegen.autofix",
        "test_standalone_typealias_detected_as_fixable",
    ),
    "test_standalone_typevar_detected_as_fixable": (
        "tests.infra.unit.codegen.autofix",
        "test_standalone_typevar_detected_as_fixable",
    ),
    "test_syntax_error_files_skipped": (
        "tests.infra.unit.codegen.autofix",
        "test_syntax_error_files_skipped",
    ),
}

__all__ = [
    "TestAllDirectoriesScanned",
    "TestBuildSiblingExportIndex",
    "TestCensusReportModel",
    "TestCensusViolationModel",
    "TestCheckOnlyMode",
    "TestEdgeCases",
    "TestExcludedDirectories",
    "TestExcludedProjects",
    "TestExtractExports",
    "TestExtractInlineConstants",
    "TestExtractVersionExports",
    "TestFixabilityClassification",
    "TestFlextInfraCodegenLazyInit",
    "TestGenerateFile",
    "TestGenerateTypeChecking",
    "TestGeneratedClassNamingConvention",
    "TestGeneratedFilesAreValidPython",
    "TestHandleLazyInit",
    "TestInferPackage",
    "TestMainCommandDispatch",
    "TestMainEntryPoint",
    "TestMergeChildExports",
    "TestParseViolationInvalid",
    "TestParseViolationValid",
    "TestProcessDirectory",
    "TestReadExistingDocstring",
    "TestResolveAliases",
    "TestRunRuffFix",
    "TestScaffoldProjectCreatesSrcModules",
    "TestScaffoldProjectCreatesTestsModules",
    "TestScaffoldProjectIdempotency",
    "TestScaffoldProjectNoop",
    "TestScanAstPublicDefs",
    "TestShouldBubbleUp",
    "TestViolationPattern",
    "c",
    "census",
    "fixer",
    "test_codegen_dir_returns_all_exports",
    "test_codegen_getattr_raises_attribute_error",
    "test_codegen_init_getattr_raises_attribute_error",
    "test_codegen_lazy_imports_work",
    "test_codegen_pipeline_end_to_end",
    "test_files_modified_tracks_affected_files",
    "test_flexcore_excluded_from_run",
    "test_handle_constants_quality_gate_json_exits_with_int",
    "test_handle_constants_quality_gate_text_exits_with_int",
    "test_in_context_typevar_not_flagged",
    "test_main_constants_quality_gate_dispatch",
    "test_main_constants_quality_gate_parses_before_report",
    "test_project_without_src_returns_empty",
    "test_quality_gate_real_workspace_run",
    "test_quality_gate_success_verdict_helper",
    "test_standalone_final_detected_as_fixable",
    "test_standalone_typealias_detected_as_fixable",
    "test_standalone_typevar_detected_as_fixable",
    "test_syntax_error_files_skipped",
]


def __getattr__(name: str) -> Any:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
