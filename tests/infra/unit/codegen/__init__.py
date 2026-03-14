# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Codegen package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes
    from tests.infra.unit.codegen.autofix import (
        fixer,
        test_in_context_typevar_not_flagged,
        test_standalone_final_detected_as_fixable,
        test_standalone_typealias_detected_as_fixable,
        test_standalone_typevar_detected_as_fixable,
        test_syntax_error_files_skipped,
    )
    from tests.infra.unit.codegen.autofix_workspace import (
        test_files_modified_tracks_affected_files,
        test_flexcore_excluded_from_run,
        test_project_without_src_returns_empty,
    )
    from tests.infra.unit.codegen.census import (
        TestFixabilityClassification,
        TestParseViolationInvalid,
        TestParseViolationValid,
        census,
    )
    from tests.infra.unit.codegen.census_models import (
        TestCensusReportModel,
        TestCensusViolationModel,
        TestExcludedProjects,
        TestViolationPattern,
    )
    from tests.infra.unit.codegen.constants_quality_gate import (
        TestConstantsQualityGateCLIDispatch,
        TestConstantsQualityGateVerdict,
    )
    from tests.infra.unit.codegen.init import (
        test_codegen_dir_returns_all_exports,
        test_codegen_getattr_raises_attribute_error,
        test_codegen_lazy_imports_work,
    )
    from tests.infra.unit.codegen.lazy_init_generation import (
        TestGenerateFile,
        TestGenerateTypeChecking,
        TestResolveAliases,
        TestRunRuffFix,
        test_codegen_init_getattr_raises_attribute_error,
    )
    from tests.infra.unit.codegen.lazy_init_helpers import (
        TestBuildSiblingExportIndex,
        TestExtractExports,
        TestInferPackage,
        TestReadExistingDocstring,
    )
    from tests.infra.unit.codegen.lazy_init_process import TestProcessDirectory
    from tests.infra.unit.codegen.lazy_init_service import TestFlextInfraCodegenLazyInit
    from tests.infra.unit.codegen.lazy_init_tests import (
        TestAllDirectoriesScanned,
        TestCheckOnlyMode,
        TestEdgeCases,
        TestExcludedDirectories,
    )
    from tests.infra.unit.codegen.lazy_init_transforms import (
        TestExtractInlineConstants,
        TestExtractInlineConstants as c,
        TestExtractVersionExports,
        TestMergeChildExports,
        TestScanAstPublicDefs,
        TestShouldBubbleUp,
    )
    from tests.infra.unit.codegen.main import (
        TestHandleLazyInit,
        TestMainCommandDispatch,
        TestMainEntryPoint,
    )
    from tests.infra.unit.codegen.pipeline import test_codegen_pipeline_end_to_end
    from tests.infra.unit.codegen.scaffolder import (
        TestScaffoldProjectCreatesSrcModules,
        TestScaffoldProjectCreatesTestsModules,
        TestScaffoldProjectIdempotency,
        TestScaffoldProjectNoop,
    )
    from tests.infra.unit.codegen.scaffolder_naming import (
        TestGeneratedClassNamingConvention,
        TestGeneratedFilesAreValidPython,
    )

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "TestAllDirectoriesScanned": (
        "tests.infra.unit.codegen.lazy_init_tests",
        "TestAllDirectoriesScanned",
    ),
    "TestBuildSiblingExportIndex": (
        "tests.infra.unit.codegen.lazy_init_helpers",
        "TestBuildSiblingExportIndex",
    ),
    "TestCensusReportModel": (
        "tests.infra.unit.codegen.census_models",
        "TestCensusReportModel",
    ),
    "TestCensusViolationModel": (
        "tests.infra.unit.codegen.census_models",
        "TestCensusViolationModel",
    ),
    "TestCheckOnlyMode": (
        "tests.infra.unit.codegen.lazy_init_tests",
        "TestCheckOnlyMode",
    ),
    "TestConstantsQualityGateCLIDispatch": (
        "tests.infra.unit.codegen.constants_quality_gate",
        "TestConstantsQualityGateCLIDispatch",
    ),
    "TestConstantsQualityGateVerdict": (
        "tests.infra.unit.codegen.constants_quality_gate",
        "TestConstantsQualityGateVerdict",
    ),
    "TestEdgeCases": ("tests.infra.unit.codegen.lazy_init_tests", "TestEdgeCases"),
    "TestExcludedDirectories": (
        "tests.infra.unit.codegen.lazy_init_tests",
        "TestExcludedDirectories",
    ),
    "TestExcludedProjects": (
        "tests.infra.unit.codegen.census_models",
        "TestExcludedProjects",
    ),
    "TestExtractExports": (
        "tests.infra.unit.codegen.lazy_init_helpers",
        "TestExtractExports",
    ),
    "TestExtractInlineConstants": (
        "tests.infra.unit.codegen.lazy_init_transforms",
        "TestExtractInlineConstants",
    ),
    "TestExtractVersionExports": (
        "tests.infra.unit.codegen.lazy_init_transforms",
        "TestExtractVersionExports",
    ),
    "TestFixabilityClassification": (
        "tests.infra.unit.codegen.census",
        "TestFixabilityClassification",
    ),
    "TestFlextInfraCodegenLazyInit": (
        "tests.infra.unit.codegen.lazy_init_service",
        "TestFlextInfraCodegenLazyInit",
    ),
    "TestGenerateFile": (
        "tests.infra.unit.codegen.lazy_init_generation",
        "TestGenerateFile",
    ),
    "TestGenerateTypeChecking": (
        "tests.infra.unit.codegen.lazy_init_generation",
        "TestGenerateTypeChecking",
    ),
    "TestGeneratedClassNamingConvention": (
        "tests.infra.unit.codegen.scaffolder_naming",
        "TestGeneratedClassNamingConvention",
    ),
    "TestGeneratedFilesAreValidPython": (
        "tests.infra.unit.codegen.scaffolder_naming",
        "TestGeneratedFilesAreValidPython",
    ),
    "TestHandleLazyInit": ("tests.infra.unit.codegen.main", "TestHandleLazyInit"),
    "TestInferPackage": (
        "tests.infra.unit.codegen.lazy_init_helpers",
        "TestInferPackage",
    ),
    "TestMainCommandDispatch": (
        "tests.infra.unit.codegen.main",
        "TestMainCommandDispatch",
    ),
    "TestMainEntryPoint": ("tests.infra.unit.codegen.main", "TestMainEntryPoint"),
    "TestMergeChildExports": (
        "tests.infra.unit.codegen.lazy_init_transforms",
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
        "tests.infra.unit.codegen.lazy_init_process",
        "TestProcessDirectory",
    ),
    "TestReadExistingDocstring": (
        "tests.infra.unit.codegen.lazy_init_helpers",
        "TestReadExistingDocstring",
    ),
    "TestResolveAliases": (
        "tests.infra.unit.codegen.lazy_init_generation",
        "TestResolveAliases",
    ),
    "TestRunRuffFix": (
        "tests.infra.unit.codegen.lazy_init_generation",
        "TestRunRuffFix",
    ),
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
        "tests.infra.unit.codegen.lazy_init_transforms",
        "TestScanAstPublicDefs",
    ),
    "TestShouldBubbleUp": (
        "tests.infra.unit.codegen.lazy_init_transforms",
        "TestShouldBubbleUp",
    ),
    "TestViolationPattern": (
        "tests.infra.unit.codegen.census_models",
        "TestViolationPattern",
    ),
    "c": (
        "tests.infra.unit.codegen.lazy_init_transforms",
        "TestExtractInlineConstants",
    ),
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
        "tests.infra.unit.codegen.lazy_init_generation",
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
        "tests.infra.unit.codegen.autofix_workspace",
        "test_files_modified_tracks_affected_files",
    ),
    "test_flexcore_excluded_from_run": (
        "tests.infra.unit.codegen.autofix_workspace",
        "test_flexcore_excluded_from_run",
    ),
    "test_in_context_typevar_not_flagged": (
        "tests.infra.unit.codegen.autofix",
        "test_in_context_typevar_not_flagged",
    ),
    "test_project_without_src_returns_empty": (
        "tests.infra.unit.codegen.autofix_workspace",
        "test_project_without_src_returns_empty",
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
    "TestConstantsQualityGateCLIDispatch",
    "TestConstantsQualityGateVerdict",
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
    "test_in_context_typevar_not_flagged",
    "test_project_without_src_returns_empty",
    "test_standalone_final_detected_as_fixable",
    "test_standalone_typealias_detected_as_fixable",
    "test_standalone_typevar_detected_as_fixable",
    "test_syntax_error_files_skipped",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
