# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Infra package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes
    from tests.infra import unit
    from tests.infra.constants import FlextInfraTestConstants, c
    from tests.infra.fixtures import (
        real_docs_project,
        real_makefile_project,
        real_python_package,
        real_toml_project,
        real_workspace,
    )
    from tests.infra.fixtures_git import real_git_repo
    from tests.infra.git_service import RealGitService
    from tests.infra.helpers import FlextInfraTestHelpers, h
    from tests.infra.models import FlextInfraTestModels, m
    from tests.infra.protocols import FlextInfraTestProtocols, p
    from tests.infra.runner_service import RealSubprocessRunner
    from tests.infra.scenarios import (
        DependencyScenario,
        DependencyScenarios,
        GitScenario,
        GitScenarios,
        SubprocessScenario,
        SubprocessScenarios,
        WorkspaceScenario,
        WorkspaceScenarios,
    )
    from tests.infra.typings import FlextInfraTestTypes, t
    from tests.infra.unit import (
        basemk,
        check,
        codegen,
        container,
        core,
        deps,
        discovery,
        docs,
        github,
        io,
        refactor,
        release,
    )
    from tests.infra.unit._utilities.test_discovery_consolidated import (
        TestDiscoveryDiscoverProjects,
        TestDiscoveryFindAllPyprojectFiles,
        TestDiscoveryIterPythonFiles,
        TestDiscoveryProjectRoots,
    )
    from tests.infra.unit._utilities.test_formatting import TestFormattingRunRuffFix
    from tests.infra.unit._utilities.test_iteration import (
        TestIterWorkspacePythonModules,
    )
    from tests.infra.unit._utilities.test_parsing import (
        TestParsingModuleAst,
        TestParsingModuleCst,
    )
    from tests.infra.unit._utilities.test_safety import (
        TestSafetyCheckpoint,
        TestSafetyRollback,
        TestSafetyWorkspaceValidation,
    )
    from tests.infra.unit._utilities.test_scanning import (
        MockScanner,
        TestScanFileBatch,
        TestScanModels,
    )
    from tests.infra.unit.basemk.engine import (
        test_basemk_cli_generate_to_file,
        test_basemk_cli_generate_to_stdout,
        test_basemk_engine_execute_calls_render_all,
        test_basemk_engine_render_all_handles_template_error,
        test_basemk_engine_render_all_returns_string,
        test_basemk_engine_render_all_with_valid_config,
        test_generator_fails_for_invalid_make_syntax,
        test_generator_renders_with_config_override,
        test_generator_write_saves_output_file,
        test_render_all_generates_large_makefile,
        test_render_all_has_no_scripts_path_references,
    )
    from tests.infra.unit.basemk.generator import (
        test_generator_execute_returns_generated_content,
        test_generator_generate_propagates_render_failure,
        test_generator_generate_with_basemk_config_object,
        test_generator_generate_with_dict_config,
        test_generator_generate_with_invalid_dict_config,
        test_generator_generate_with_none_config_uses_default,
        test_generator_initializes_with_custom_engine,
        test_generator_initializes_with_default_engine,
        test_generator_write_creates_parent_directories,
        test_generator_write_fails_without_output_or_stream,
        test_generator_write_to_file,
        test_generator_write_to_stream,
    )
    from tests.infra.unit.basemk.generator_edge_cases import (
        test_generator_normalize_config_with_basemk_config,
        test_generator_normalize_config_with_dict,
        test_generator_normalize_config_with_invalid_dict,
        test_generator_normalize_config_with_none,
        test_generator_validate_generated_output_handles_oserror,
        test_generator_write_handles_file_permission_error,
        test_generator_write_to_stream_handles_oserror,
    )
    from tests.infra.unit.basemk.init import TestFlextInfraBaseMk
    from tests.infra.unit.basemk.main import (
        test_basemk_build_config_with_none,
        test_basemk_build_config_with_project_name,
        test_basemk_main_calls_sys_exit,
        test_basemk_main_ensures_structlog_configured,
        test_basemk_main_output_to_stdout,
        test_basemk_main_with_generate_command,
        test_basemk_main_with_generation_failure,
        test_basemk_main_with_invalid_command,
        test_basemk_main_with_no_command,
        test_basemk_main_with_none_argv,
        test_basemk_main_with_output_file,
        test_basemk_main_with_project_name,
        test_basemk_main_with_write_failure,
    )
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
    from tests.infra.unit.container.test_infra_container import (
        TestInfraContainerFunctions,
        TestInfraMroPattern,
        TestInfraServiceRetrieval,
    )
    from tests.infra.unit.core.basemk_validator import (
        TestBaseMkValidatorCore,
        TestBaseMkValidatorEdgeCases,
        TestBaseMkValidatorSha256,
        v,
    )
    from tests.infra.unit.core.init import TestCoreModuleInit
    from tests.infra.unit.core.inventory import (
        TestInventoryServiceCore,
        TestInventoryServiceReports,
        TestInventoryServiceScripts,
    )
    from tests.infra.unit.core.main import (
        TestMainBaseMkValidate,
        TestMainCliRouting,
        TestMainInventory,
        TestMainScan,
    )
    from tests.infra.unit.core.pytest_diag import (
        TestPytestDiagExtractorCore,
        TestPytestDiagLogParsing,
        TestPytestDiagParseXml,
    )
    from tests.infra.unit.core.scanner import (
        TestScannerCore,
        TestScannerHelpers,
        TestScannerMultiFile,
    )
    from tests.infra.unit.core.skill_validator import (
        TestNormalizeStringList,
        TestSafeLoadYaml,
        TestSkillValidatorAstGrepCount,
        TestSkillValidatorCore,
        TestSkillValidatorRenderTemplate,
    )
    from tests.infra.unit.core.stub_chain import (
        TestStubChainAnalyze,
        TestStubChainCore,
        TestStubChainDiscoverProjects,
        TestStubChainIsInternal,
        TestStubChainStubExists,
        TestStubChainValidate,
    )
    from tests.infra.unit.deps.test_detection_classify import (
        TestBuildProjectReport,
        TestClassifyIssues,
    )
    from tests.infra.unit.deps.test_detection_deptry import TestRunDeptry
    from tests.infra.unit.deps.test_detection_models import (
        TestFlextInfraDependencyDetectionModels,
        TestFlextInfraDependencyDetectionService,
        TestFlextInfraDependencyDetectionService as s,
        TestToInfraValue,
    )
    from tests.infra.unit.deps.test_detection_pip_check import TestRunPipCheck
    from tests.infra.unit.deps.test_detection_typings import (
        TestLoadDependencyLimits,
        TestRunMypyStubHints,
    )
    from tests.infra.unit.deps.test_detection_typings_flow import (
        TestModuleAndTypingsFlow,
    )
    from tests.infra.unit.deps.test_detection_uncovered import (
        TestDetectionUncoveredLines,
    )
    from tests.infra.unit.deps.test_detection_wrappers import (
        TestModuleLevelWrappers,
        test_discover_projects_wrapper,
        test_get_current_typings_from_pyproject_wrapper,
        test_get_required_typings_wrapper,
        test_run_deptry_wrapper,
        test_run_mypy_stub_hints_wrapper,
        test_run_pip_check_wrapper,
    )
    from tests.infra.unit.deps.test_detector_detect import (
        TestFlextInfraRuntimeDevDependencyDetectorRunDetect,
    )
    from tests.infra.unit.deps.test_detector_detect_failures import (
        TestDetectorRunFailures,
    )
    from tests.infra.unit.deps.test_detector_init import (
        TestFlextInfraRuntimeDevDependencyDetectorInit,
    )
    from tests.infra.unit.deps.test_detector_main import (
        TestFlextInfraRuntimeDevDependencyDetectorRunTypings,
        TestMainFunction,
    )
    from tests.infra.unit.deps.test_detector_models import (
        TestFlextInfraDependencyDetectorModels,
    )
    from tests.infra.unit.deps.test_detector_report import (
        TestFlextInfraRuntimeDevDependencyDetectorRunReport,
    )
    from tests.infra.unit.deps.test_detector_report_flags import TestDetectorReportFlags
    from tests.infra.unit.deps.test_extra_paths_manager import (
        TestConstants,
        TestFlextInfraExtraPathsManager,
        TestGetDepPaths,
        TestSyncOne,
    )
    from tests.infra.unit.deps.test_extra_paths_pep621 import (
        TestPathDepPathsPep621,
        TestPathDepPathsPoetry,
        test_helpers_alias_exposed,
    )
    from tests.infra.unit.deps.test_extra_paths_sync import (
        pyright_content,
        test_main_success_modes,
        test_main_sync_failure,
        test_sync_extra_paths_missing_root_pyproject,
        test_sync_extra_paths_success_modes,
        test_sync_extra_paths_sync_failure,
        test_sync_one_edge_cases,
    )
    from tests.infra.unit.deps.test_init import TestFlextInfraDeps
    from tests.infra.unit.deps.test_internal_sync_discovery import (
        TestCollectInternalDeps,
        TestParseGitmodules,
        TestParseRepoMap,
    )
    from tests.infra.unit.deps.test_internal_sync_discovery_edge import (
        TestCollectInternalDepsEdgeCases,
    )
    from tests.infra.unit.deps.test_internal_sync_resolve import (
        TestInferOwnerFromOrigin,
        TestResolveRef,
        TestSynthesizedRepoMap,
    )
    from tests.infra.unit.deps.test_internal_sync_sync import TestSync
    from tests.infra.unit.deps.test_internal_sync_sync_edge import (
        TestSyncMethodEdgeCases,
    )
    from tests.infra.unit.deps.test_internal_sync_sync_edge_more import (
        TestSyncMethodEdgeCasesMore,
    )
    from tests.infra.unit.deps.test_internal_sync_update import (
        TestEnsureCheckout,
        TestEnsureSymlink,
        TestEnsureSymlinkEdgeCases,
    )
    from tests.infra.unit.deps.test_internal_sync_update_checkout_edge import (
        TestEnsureCheckoutEdgeCases,
    )
    from tests.infra.unit.deps.test_internal_sync_validation import (
        TestFlextInfraInternalDependencySyncService,
        TestIsInternalPathDep,
        TestIsRelativeTo,
        TestOwnerFromRemoteUrl,
        TestValidateGitRefEdgeCases,
    )
    from tests.infra.unit.deps.test_internal_sync_workspace import (
        TestIsWorkspaceMode,
        TestWorkspaceRootFromEnv,
        TestWorkspaceRootFromParents,
    )
    from tests.infra.unit.deps.test_main import (
        TestMainHelpAndErrors,
        TestMainReturnValues,
        TestSubcommandMapping,
    )
    from tests.infra.unit.deps.test_main_dispatch import (
        TestMainDelegation,
        TestMainExceptionHandling,
        TestMainModuleImport,
        TestMainSubcommandDispatch,
        TestMainSysArgvModification,
        test_string_zero_return_value,
    )
    from tests.infra.unit.deps.test_modernizer_comments import (
        TestInjectCommentsPhase,
        test_inject_comments_phase_apply_banner,
        test_inject_comments_phase_apply_broken_group_section,
        test_inject_comments_phase_apply_markers,
        test_inject_comments_phase_apply_with_optional_dependencies_dev,
    )
    from tests.infra.unit.deps.test_modernizer_consolidate import (
        TestConsolidateGroupsPhase,
        test_consolidate_groups_phase_apply_removes_old_groups,
        test_consolidate_groups_phase_apply_with_empty_poetry_group,
    )
    from tests.infra.unit.deps.test_modernizer_helpers import (
        array,
        as_string_list,
        canonical_dev_dependencies,
        dedupe_specs,
        dep_name,
        doc,
        ensure_table,
        project_dev_groups,
        test_array,
        test_as_string_list,
        test_as_string_list_toml_item,
        test_canonical_dev_dependencies,
        test_dedupe_specs,
        test_dep_name,
        test_ensure_table,
        test_project_dev_groups,
        test_project_dev_groups_missing_sections,
        test_unwrap_item,
        test_unwrap_item_toml_item,
        unwrap_item,
    )
    from tests.infra.unit.deps.test_modernizer_main import (
        TestFlextInfraPyprojectModernizer,
        TestModernizerRunAndMain,
    )
    from tests.infra.unit.deps.test_modernizer_main_extra import (
        TestModernizerEdgeCases,
        TestModernizerUncoveredLines,
        test_flext_infra_pyproject_modernizer_find_pyproject_files,
        test_flext_infra_pyproject_modernizer_process_file_invalid_toml,
    )
    from tests.infra.unit.deps.test_modernizer_pyrefly import (
        TestEnsurePyreflyConfigPhase,
        test_ensure_pyrefly_config_phase_apply_errors,
        test_ensure_pyrefly_config_phase_apply_ignore_errors,
        test_ensure_pyrefly_config_phase_apply_python_version,
        test_ensure_pyrefly_config_phase_apply_search_path,
    )
    from tests.infra.unit.deps.test_modernizer_pyright import (
        TestEnsurePyrightConfigPhase,
    )
    from tests.infra.unit.deps.test_modernizer_pytest import (
        TestEnsurePytestConfigPhase,
        test_ensure_pytest_config_phase_apply_markers,
        test_ensure_pytest_config_phase_apply_minversion,
        test_ensure_pytest_config_phase_apply_python_classes,
    )
    from tests.infra.unit.deps.test_modernizer_workspace import (
        TestParser,
        TestReadDoc,
        test_workspace_root_doc_construction,
    )
    from tests.infra.unit.deps.test_path_sync_helpers import (
        extract_dep_name,
        test_extract_dep_name,
        test_extract_requirement_name,
        test_helpers_alias_is_reachable_helpers,
        test_target_path,
    )
    from tests.infra.unit.deps.test_path_sync_init import (
        TestDetectMode,
        TestFlextInfraDependencyPathSync,
        TestPathSyncEdgeCases,
        test_detect_mode_with_nonexistent_path,
        test_detect_mode_with_path_object,
    )
    from tests.infra.unit.deps.test_path_sync_main import (
        TestMain,
        test_helpers_alias_is_reachable_main,
    )
    from tests.infra.unit.deps.test_path_sync_main_edges import TestMainEdgeCases
    from tests.infra.unit.deps.test_path_sync_main_more import (
        test_main_discovery_failure,
        test_main_no_changes_needed,
        test_main_project_invalid_toml,
        test_main_project_no_name,
        test_main_project_non_string_name,
        test_main_with_changes_and_dry_run,
        test_main_with_changes_no_dry_run,
        test_workspace_root_fallback,
    )
    from tests.infra.unit.deps.test_path_sync_main_project_obj import (
        test_helpers_alias_is_reachable_project_obj,
        test_main_project_obj_not_dict_first_loop,
        test_main_project_obj_not_dict_second_loop,
    )
    from tests.infra.unit.deps.test_path_sync_rewrite_deps import (
        TestRewriteDepPaths,
        rewrite_dep_paths,
        test_rewrite_dep_paths_dry_run,
        test_rewrite_dep_paths_read_failure,
        test_rewrite_dep_paths_with_internal_names,
        test_rewrite_dep_paths_with_no_deps,
    )
    from tests.infra.unit.deps.test_path_sync_rewrite_pep621 import (
        TestRewritePep621,
        test_rewrite_pep621_invalid_path_dep_regex,
        test_rewrite_pep621_no_project_table,
        test_rewrite_pep621_non_string_item,
    )
    from tests.infra.unit.deps.test_path_sync_rewrite_poetry import (
        TestRewritePoetry,
        test_rewrite_poetry_no_poetry_table,
        test_rewrite_poetry_no_tool_table,
        test_rewrite_poetry_with_non_dict_value,
    )
    from tests.infra.unit.discovery.test_infra_discovery import (
        TestFlextInfraDiscoveryService,
    )
    from tests.infra.unit.discovery.test_infra_discovery_edge_cases import (
        TestFlextInfraDiscoveryServiceUncoveredLines,
    )
    from tests.infra.unit.docs.auditor import (
        TestAuditorCore,
        TestAuditorNormalize,
        auditor,
        is_external,
        normalize_link,
        should_skip_target,
    )
    from tests.infra.unit.docs.auditor_budgets import TestLoadAuditBudgets
    from tests.infra.unit.docs.auditor_cli import (
        TestAuditorMainCli,
        TestAuditorScopeFailure,
    )
    from tests.infra.unit.docs.auditor_links import (
        TestAuditorBrokenLinks,
        TestAuditorToMarkdown,
    )
    from tests.infra.unit.docs.auditor_scope import (
        TestAuditorForbiddenTerms,
        TestAuditorScope,
    )
    from tests.infra.unit.docs.builder import TestBuilderCore, builder
    from tests.infra.unit.docs.builder_scope import TestBuilderScope
    from tests.infra.unit.docs.fixer import TestFixerCore
    from tests.infra.unit.docs.fixer_internals import (
        TestFixerMaybeFixLink,
        TestFixerProcessFile,
        TestFixerScope,
        TestFixerToc,
    )
    from tests.infra.unit.docs.generator import TestGeneratorCore
    from tests.infra.unit.docs.generator_internals import (
        TestGeneratorHelpers,
        TestGeneratorScope,
        gen,
    )
    from tests.infra.unit.docs.init import TestFlextInfraDocs
    from tests.infra.unit.docs.main import TestRunAudit, TestRunFix
    from tests.infra.unit.docs.main_commands import (
        TestRunBuild,
        TestRunGenerate,
        TestRunValidate,
    )
    from tests.infra.unit.docs.main_entry import TestMainRouting, TestMainWithFlags
    from tests.infra.unit.docs.shared import TestBuildScopes, TestFlextInfraDocScope
    from tests.infra.unit.docs.shared_iter import (
        TestIterMarkdownFiles,
        TestSelectedProjectNames,
    )
    from tests.infra.unit.docs.shared_write import TestWriteJson, TestWriteMarkdown
    from tests.infra.unit.docs.validator import TestValidateCore, TestValidateReport
    from tests.infra.unit.docs.validator_internals import (
        TestAdrHelpers,
        TestMaybeWriteTodo,
        TestValidateScope,
        validator,
    )
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
    from tests.infra.unit.github.pr import (
        TestCreate,
        TestFlextInfraPrManager,
        TestStatus,
    )
    from tests.infra.unit.github.pr_cli import TestParseArgs, TestSelectorFunction
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
    from tests.infra.unit.io.test_infra_json_io import TestFlextInfraJsonService
    from tests.infra.unit.io.test_infra_output_edge_cases import (
        TestInfraOutputEdgeCases,
        TestInfraOutputNoColor,
        TestMroFacadeMethods,
    )
    from tests.infra.unit.io.test_infra_output_formatting import (
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
    from tests.infra.unit.refactor.test_infra_refactor_analysis import (
        test_build_impact_map_extracts_rename_entries,
        test_build_impact_map_extracts_signature_entries,
        test_main_analyze_violations_is_read_only,
        test_main_analyze_violations_writes_json_report,
        test_violation_analysis_counts_massive_patterns,
        test_violation_analyzer_skips_non_utf8_files,
    )
    from tests.infra.unit.refactor.test_infra_refactor_class_and_propagation import (
        test_class_reconstructor_reorders_each_contiguous_method_block,
        test_class_reconstructor_reorders_methods_by_config,
        test_class_reconstructor_skips_interleaved_non_method_members,
        test_mro_checker_keeps_external_attribute_base,
        test_mro_redundancy_checker_removes_nested_attribute_inheritance,
        test_signature_propagation_removes_and_adds_keywords,
        test_signature_propagation_renames_call_keyword,
        test_symbol_propagation_keeps_alias_reference_when_asname_used,
        test_symbol_propagation_renames_import_and_local_references,
        test_symbol_propagation_updates_mro_base_references,
    )
    from tests.infra.unit.refactor.test_infra_refactor_engine import (
        test_engine_always_enables_class_nesting_file_rule,
        test_refactor_files_skips_non_python_inputs,
        test_refactor_project_scans_tests_and_scripts_dirs,
        test_rule_dispatch_fails_on_invalid_pattern_rule_config,
        test_rule_dispatch_fails_on_unknown_rule_mapping,
        test_rule_dispatch_keeps_legacy_id_fallback_mapping,
        test_rule_dispatch_prefers_fix_action_metadata,
    )
    from tests.infra.unit.refactor.test_infra_refactor_import_modernizer import (
        test_import_modernizer_adds_c_when_existing_c_is_aliased,
        test_import_modernizer_does_not_rewrite_function_parameter_shadow,
        test_import_modernizer_does_not_rewrite_rebound_local_name_usage,
        test_import_modernizer_partial_import_keeps_unmapped_symbols,
        test_import_modernizer_partial_import_with_asname_keeps_unmapped_alias,
        test_import_modernizer_skips_rewrite_when_runtime_alias_shadowed_in_function,
        test_import_modernizer_skips_when_runtime_alias_name_is_blocked,
        test_import_modernizer_updates_aliased_symbol_usage,
        test_lazy_import_rule_hoists_import_to_module_level,
        test_lazy_import_rule_uses_fix_action_for_hoist,
    )
    from tests.infra.unit.refactor.test_infra_refactor_legacy_and_annotations import (
        test_ensure_future_annotations_after_docstring,
        test_ensure_future_annotations_moves_existing_import_to_top,
        test_legacy_import_bypass_collapses_to_primary_import,
        test_legacy_rule_uses_fix_action_remove_for_aliases,
        test_legacy_wrapper_forwarding_keywords_is_inlined_as_alias,
        test_legacy_wrapper_forwarding_varargs_is_inlined_as_alias,
        test_legacy_wrapper_function_is_inlined_as_alias,
        test_legacy_wrapper_non_passthrough_is_not_inlined,
    )
    from tests.infra.unit.refactor.test_infra_refactor_pattern_corrections import (
        test_pattern_rule_converts_dict_annotations_to_mapping,
        test_pattern_rule_keeps_dict_param_when_copy_used,
        test_pattern_rule_keeps_dict_param_when_subscript_mutated,
        test_pattern_rule_keeps_type_cast_when_not_nested_object_cast,
        test_pattern_rule_optionally_converts_return_annotations_to_mapping,
        test_pattern_rule_removes_configured_redundant_casts,
        test_pattern_rule_removes_nested_type_object_cast_chain,
        test_pattern_rule_skips_overload_signatures,
    )
    from tests.infra.unit.refactor.test_infra_refactor_safety import (
        EngineSafetyStub,
        test_refactor_project_integrates_safety_manager,
    )
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
    from tests.infra.unit.test_infra_constants_core import (
        TestFlextInfraConstantsExcludedNamespace,
        TestFlextInfraConstantsFilesNamespace,
        TestFlextInfraConstantsGatesNamespace,
        TestFlextInfraConstantsPathsNamespace,
        TestFlextInfraConstantsStatusNamespace,
    )
    from tests.infra.unit.test_infra_constants_extra import (
        TestFlextInfraConstantsAlias,
        TestFlextInfraConstantsCheckNamespace,
        TestFlextInfraConstantsConsistency,
        TestFlextInfraConstantsEncodingNamespace,
        TestFlextInfraConstantsGithubNamespace,
        TestFlextInfraConstantsImmutability,
    )
    from tests.infra.unit.test_infra_git import (
        TestFlextInfraGitService,
        TestGitPush,
        TestGitTagOperations,
        TestRemovedCompatibilityMethods,
        git_repo,
    )
    from tests.infra.unit.test_infra_init_lazy_core import TestFlextInfraInitLazyLoading
    from tests.infra.unit.test_infra_init_lazy_submodules import (
        TestFlextInfraSubmoduleInitLazyLoading,
    )
    from tests.infra.unit.test_infra_main import (
        test_main_all_groups_defined,
        test_main_group_modules_are_valid,
        test_main_help_flag_returns_zero,
        test_main_returns_error_when_no_args,
        test_main_unknown_group_returns_error,
    )
    from tests.infra.unit.test_infra_maintenance_init import TestFlextInfraMaintenance
    from tests.infra.unit.test_infra_maintenance_main import (
        TestMaintenanceMainEnforcer,
        TestMaintenanceMainSuccess,
    )
    from tests.infra.unit.test_infra_maintenance_python_version import (
        TestDiscoverProjects,
        TestEnforcerExecute,
        TestEnsurePythonVersionFile,
        TestReadRequiredMinor,
        TestWorkspaceRoot,
    )
    from tests.infra.unit.test_infra_paths import TestFlextInfraPathResolver
    from tests.infra.unit.test_infra_patterns_core import (
        TestFlextInfraPatternsMarkdown,
        TestFlextInfraPatternsTooling,
    )
    from tests.infra.unit.test_infra_patterns_extra import (
        TestFlextInfraPatternsEdgeCases,
        TestFlextInfraPatternsPatternTypes,
    )
    from tests.infra.unit.test_infra_protocols import TestFlextInfraProtocolsImport
    from tests.infra.unit.test_infra_reporting_core import (
        TestFlextInfraReportingServiceCore,
    )
    from tests.infra.unit.test_infra_reporting_extra import (
        TestFlextInfraReportingServiceExtra,
    )
    from tests.infra.unit.test_infra_selection import TestFlextInfraUtilitiesSelection
    from tests.infra.unit.test_infra_subprocess_core import (
        runner,
        test_capture_cases,
        test_run_cases,
        test_run_raw_cases,
    )
    from tests.infra.unit.test_infra_subprocess_extra import (
        TestFlextInfraCommandRunnerExtra,
    )
    from tests.infra.unit.test_infra_templates import (
        engine,
        test_engine_constants_shared,
        test_multiple_instances_independent,
        test_render_failure,
        test_render_success,
        test_template_constants,
    )
    from tests.infra.unit.test_infra_toml_io import (
        TestFlextInfraTomlDocument,
        TestFlextInfraTomlHelpers,
        TestFlextInfraTomlRead,
    )
    from tests.infra.unit.test_infra_typings import TestFlextInfraTypesImport
    from tests.infra.unit.test_infra_utilities import TestFlextInfraUtilitiesImport
    from tests.infra.unit.test_infra_version_core import TestFlextInfraVersionClass
    from tests.infra.unit.test_infra_version_extra import (
        TestFlextInfraVersionModuleLevel,
        TestFlextInfraVersionPackageInfo,
    )
    from tests.infra.unit.test_infra_versioning import (
        service,
        test_bump_version_invalid,
        test_bump_version_result_type,
        test_bump_version_valid,
        test_current_workspace_version,
        test_parse_semver_invalid,
        test_parse_semver_result_type,
        test_parse_semver_valid,
        test_release_tag_from_branch_invalid,
        test_release_tag_from_branch_result_type,
        test_release_tag_from_branch_valid,
        test_replace_project_version,
    )
    from tests.infra.unit.test_infra_workspace_cli import (
        test_workspace_cli_migrate_command,
        test_workspace_cli_migrate_output_contains_summary,
    )
    from tests.infra.unit.test_infra_workspace_detector import (
        TestDetectorBasicDetection,
        TestDetectorGitRunScenarios,
        TestDetectorRepoNameExtraction,
        detector,
    )
    from tests.infra.unit.test_infra_workspace_init import TestFlextInfraWorkspace
    from tests.infra.unit.test_infra_workspace_main import (
        TestMainCli,
        TestRunDetect,
        TestRunMigrate,
        TestRunOrchestrate,
        TestRunSync,
    )
    from tests.infra.unit.test_infra_workspace_migrator import (
        test_migrator_apply_updates_project_files,
        test_migrator_discovery_failure,
        test_migrator_dry_run_reports_changes_without_writes,
        test_migrator_execute_returns_failure,
        test_migrator_handles_missing_pyproject_gracefully,
        test_migrator_no_changes_needed,
        test_migrator_preserves_custom_makefile_content,
        test_migrator_workspace_root_not_exists,
        test_migrator_workspace_root_project_detection,
    )
    from tests.infra.unit.test_infra_workspace_migrator_deps import (
        test_migrate_makefile_not_found_non_dry_run,
        test_migrate_pyproject_flext_core_non_dry_run,
        test_migrator_has_flext_core_dependency_in_poetry,
        test_migrator_has_flext_core_dependency_poetry_deps_not_table,
        test_migrator_has_flext_core_dependency_poetry_table_missing,
        test_workspace_migrator_error_handling_on_invalid_workspace,
        test_workspace_migrator_makefile_not_found_dry_run,
        test_workspace_migrator_makefile_read_error,
        test_workspace_migrator_pyproject_write_error,
    )
    from tests.infra.unit.test_infra_workspace_migrator_dryrun import (
        test_migrator_flext_core_dry_run,
        test_migrator_flext_core_project_skipped,
        test_migrator_gitignore_already_normalized_dry_run,
        test_migrator_makefile_not_found_dry_run,
        test_migrator_makefile_read_failure,
        test_migrator_pyproject_not_found_dry_run,
    )
    from tests.infra.unit.test_infra_workspace_migrator_errors import (
        TestMigratorReadFailures,
        TestMigratorWriteFailures,
    )
    from tests.infra.unit.test_infra_workspace_migrator_internal import (
        TestMigratorEdgeCases,
        TestMigratorInternalMakefile,
        TestMigratorInternalPyproject,
    )
    from tests.infra.unit.test_infra_workspace_migrator_pyproject import (
        TestMigratorDryRun,
        TestMigratorFlextCore,
        TestMigratorPoetryDeps,
    )
    from tests.infra.unit.test_infra_workspace_orchestrator import (
        TestOrchestratorBasic,
        TestOrchestratorFailures,
        orchestrator,
    )
    from tests.infra.unit.test_infra_workspace_sync import (
        SetupFn,
        svc,
        test_atomic_write_fail,
        test_atomic_write_ok,
        test_cli_result_by_project_root,
        test_gitignore_entry_scenarios,
        test_gitignore_sync_failure,
        test_gitignore_write_failure,
        test_sync_basemk_scenarios,
        test_sync_error_scenarios,
        test_sync_root_validation,
        test_sync_success_scenarios,
    )
    from tests.infra.utilities import FlextInfraTestUtilities, u
    from tests.infra.workspace_factory import WorkspaceFactory
    from tests.infra.workspace_scenarios import (
        BrokenScenario,
        EmptyScenario,
        FullScenario,
        MinimalScenario,
    )

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BrokenScenario": ("tests.infra.workspace_scenarios", "BrokenScenario"),
    "CheckProjectStub": (
        "tests.infra.unit.check.extended_run_projects",
        "CheckProjectStub",
    ),
    "DependencyScenario": ("tests.infra.scenarios", "DependencyScenario"),
    "DependencyScenarios": ("tests.infra.scenarios", "DependencyScenarios"),
    "EmptyScenario": ("tests.infra.workspace_scenarios", "EmptyScenario"),
    "EngineSafetyStub": (
        "tests.infra.unit.refactor.test_infra_refactor_safety",
        "EngineSafetyStub",
    ),
    "FlextInfraTestConstants": ("tests.infra.constants", "FlextInfraTestConstants"),
    "FlextInfraTestHelpers": ("tests.infra.helpers", "FlextInfraTestHelpers"),
    "FlextInfraTestModels": ("tests.infra.models", "FlextInfraTestModels"),
    "FlextInfraTestProtocols": ("tests.infra.protocols", "FlextInfraTestProtocols"),
    "FlextInfraTestTypes": ("tests.infra.typings", "FlextInfraTestTypes"),
    "FlextInfraTestUtilities": ("tests.infra.utilities", "FlextInfraTestUtilities"),
    "FullScenario": ("tests.infra.workspace_scenarios", "FullScenario"),
    "GitScenario": ("tests.infra.scenarios", "GitScenario"),
    "GitScenarios": ("tests.infra.scenarios", "GitScenarios"),
    "MinimalScenario": ("tests.infra.workspace_scenarios", "MinimalScenario"),
    "MockScanner": ("tests.infra.unit._utilities.test_scanning", "MockScanner"),
    "RealGitService": ("tests.infra.git_service", "RealGitService"),
    "RealSubprocessRunner": ("tests.infra.runner_service", "RealSubprocessRunner"),
    "RunCallable": ("tests.infra.unit.check.extended_runners_ruff", "RunCallable"),
    "RunStub": ("tests.infra.unit.check.extended_error_reporting", "RunStub"),
    "SetupFn": ("tests.infra.unit.test_infra_workspace_sync", "SetupFn"),
    "SubprocessScenario": ("tests.infra.scenarios", "SubprocessScenario"),
    "SubprocessScenarios": ("tests.infra.scenarios", "SubprocessScenarios"),
    "TestAdrHelpers": ("tests.infra.unit.docs.validator_internals", "TestAdrHelpers"),
    "TestAllDirectoriesScanned": (
        "tests.infra.unit.codegen.lazy_init_tests",
        "TestAllDirectoriesScanned",
    ),
    "TestAuditorBrokenLinks": (
        "tests.infra.unit.docs.auditor_links",
        "TestAuditorBrokenLinks",
    ),
    "TestAuditorCore": ("tests.infra.unit.docs.auditor", "TestAuditorCore"),
    "TestAuditorForbiddenTerms": (
        "tests.infra.unit.docs.auditor_scope",
        "TestAuditorForbiddenTerms",
    ),
    "TestAuditorMainCli": ("tests.infra.unit.docs.auditor_cli", "TestAuditorMainCli"),
    "TestAuditorNormalize": ("tests.infra.unit.docs.auditor", "TestAuditorNormalize"),
    "TestAuditorScope": ("tests.infra.unit.docs.auditor_scope", "TestAuditorScope"),
    "TestAuditorScopeFailure": (
        "tests.infra.unit.docs.auditor_cli",
        "TestAuditorScopeFailure",
    ),
    "TestAuditorToMarkdown": (
        "tests.infra.unit.docs.auditor_links",
        "TestAuditorToMarkdown",
    ),
    "TestBaseMkValidatorCore": (
        "tests.infra.unit.core.basemk_validator",
        "TestBaseMkValidatorCore",
    ),
    "TestBaseMkValidatorEdgeCases": (
        "tests.infra.unit.core.basemk_validator",
        "TestBaseMkValidatorEdgeCases",
    ),
    "TestBaseMkValidatorSha256": (
        "tests.infra.unit.core.basemk_validator",
        "TestBaseMkValidatorSha256",
    ),
    "TestBuildProjectReport": (
        "tests.infra.unit.deps.test_detection_classify",
        "TestBuildProjectReport",
    ),
    "TestBuildScopes": ("tests.infra.unit.docs.shared", "TestBuildScopes"),
    "TestBuildSiblingExportIndex": (
        "tests.infra.unit.codegen.lazy_init_helpers",
        "TestBuildSiblingExportIndex",
    ),
    "TestBuildTargets": (
        "tests.infra.unit.release.orchestrator_helpers",
        "TestBuildTargets",
    ),
    "TestBuilderCore": ("tests.infra.unit.docs.builder", "TestBuilderCore"),
    "TestBuilderScope": ("tests.infra.unit.docs.builder_scope", "TestBuilderScope"),
    "TestBumpNextDev": (
        "tests.infra.unit.release.orchestrator_helpers",
        "TestBumpNextDev",
    ),
    "TestCensusReportModel": (
        "tests.infra.unit.codegen.census_models",
        "TestCensusReportModel",
    ),
    "TestCensusViolationModel": (
        "tests.infra.unit.codegen.census_models",
        "TestCensusViolationModel",
    ),
    "TestCheckIssueFormatted": (
        "tests.infra.unit.check.extended_models",
        "TestCheckIssueFormatted",
    ),
    "TestCheckMainEntryPoint": (
        "tests.infra.unit.check.extended_cli_entry",
        "TestCheckMainEntryPoint",
    ),
    "TestCheckOnlyMode": (
        "tests.infra.unit.codegen.lazy_init_tests",
        "TestCheckOnlyMode",
    ),
    "TestCheckProjectRunners": (
        "tests.infra.unit.check.extended_projects",
        "TestCheckProjectRunners",
    ),
    "TestCheckpoint": ("tests.infra.unit.github.pr_workspace", "TestCheckpoint"),
    "TestChecks": ("tests.infra.unit.github.pr_operations", "TestChecks"),
    "TestClassifyIssues": (
        "tests.infra.unit.deps.test_detection_classify",
        "TestClassifyIssues",
    ),
    "TestClose": ("tests.infra.unit.github.pr_operations", "TestClose"),
    "TestCollectChanges": (
        "tests.infra.unit.release.orchestrator_git",
        "TestCollectChanges",
    ),
    "TestCollectInternalDeps": (
        "tests.infra.unit.deps.test_internal_sync_discovery",
        "TestCollectInternalDeps",
    ),
    "TestCollectInternalDepsEdgeCases": (
        "tests.infra.unit.deps.test_internal_sync_discovery_edge",
        "TestCollectInternalDepsEdgeCases",
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
    "TestConsolidateGroupsPhase": (
        "tests.infra.unit.deps.test_modernizer_consolidate",
        "TestConsolidateGroupsPhase",
    ),
    "TestConstants": (
        "tests.infra.unit.deps.test_extra_paths_manager",
        "TestConstants",
    ),
    "TestConstantsQualityGateCLIDispatch": (
        "tests.infra.unit.codegen.constants_quality_gate",
        "TestConstantsQualityGateCLIDispatch",
    ),
    "TestConstantsQualityGateVerdict": (
        "tests.infra.unit.codegen.constants_quality_gate",
        "TestConstantsQualityGateVerdict",
    ),
    "TestCoreModuleInit": ("tests.infra.unit.core.init", "TestCoreModuleInit"),
    "TestCreate": ("tests.infra.unit.github.pr", "TestCreate"),
    "TestCreateBranches": (
        "tests.infra.unit.release.orchestrator_git",
        "TestCreateBranches",
    ),
    "TestCreateTag": ("tests.infra.unit.release.orchestrator_git", "TestCreateTag"),
    "TestDetectMode": ("tests.infra.unit.deps.test_path_sync_init", "TestDetectMode"),
    "TestDetectionUncoveredLines": (
        "tests.infra.unit.deps.test_detection_uncovered",
        "TestDetectionUncoveredLines",
    ),
    "TestDetectorBasicDetection": (
        "tests.infra.unit.test_infra_workspace_detector",
        "TestDetectorBasicDetection",
    ),
    "TestDetectorGitRunScenarios": (
        "tests.infra.unit.test_infra_workspace_detector",
        "TestDetectorGitRunScenarios",
    ),
    "TestDetectorRepoNameExtraction": (
        "tests.infra.unit.test_infra_workspace_detector",
        "TestDetectorRepoNameExtraction",
    ),
    "TestDetectorReportFlags": (
        "tests.infra.unit.deps.test_detector_report_flags",
        "TestDetectorReportFlags",
    ),
    "TestDetectorRunFailures": (
        "tests.infra.unit.deps.test_detector_detect_failures",
        "TestDetectorRunFailures",
    ),
    "TestDiscoverProjects": (
        "tests.infra.unit.test_infra_maintenance_python_version",
        "TestDiscoverProjects",
    ),
    "TestDiscoveryDiscoverProjects": (
        "tests.infra.unit._utilities.test_discovery_consolidated",
        "TestDiscoveryDiscoverProjects",
    ),
    "TestDiscoveryFindAllPyprojectFiles": (
        "tests.infra.unit._utilities.test_discovery_consolidated",
        "TestDiscoveryFindAllPyprojectFiles",
    ),
    "TestDiscoveryIterPythonFiles": (
        "tests.infra.unit._utilities.test_discovery_consolidated",
        "TestDiscoveryIterPythonFiles",
    ),
    "TestDiscoveryProjectRoots": (
        "tests.infra.unit._utilities.test_discovery_consolidated",
        "TestDiscoveryProjectRoots",
    ),
    "TestDispatchPhase": (
        "tests.infra.unit.release.orchestrator_helpers",
        "TestDispatchPhase",
    ),
    "TestEdgeCases": ("tests.infra.unit.codegen.lazy_init_tests", "TestEdgeCases"),
    "TestEnforcerExecute": (
        "tests.infra.unit.test_infra_maintenance_python_version",
        "TestEnforcerExecute",
    ),
    "TestEnsureCheckout": (
        "tests.infra.unit.deps.test_internal_sync_update",
        "TestEnsureCheckout",
    ),
    "TestEnsureCheckoutEdgeCases": (
        "tests.infra.unit.deps.test_internal_sync_update_checkout_edge",
        "TestEnsureCheckoutEdgeCases",
    ),
    "TestEnsurePyreflyConfigPhase": (
        "tests.infra.unit.deps.test_modernizer_pyrefly",
        "TestEnsurePyreflyConfigPhase",
    ),
    "TestEnsurePyrightConfigPhase": (
        "tests.infra.unit.deps.test_modernizer_pyright",
        "TestEnsurePyrightConfigPhase",
    ),
    "TestEnsurePytestConfigPhase": (
        "tests.infra.unit.deps.test_modernizer_pytest",
        "TestEnsurePytestConfigPhase",
    ),
    "TestEnsurePythonVersionFile": (
        "tests.infra.unit.test_infra_maintenance_python_version",
        "TestEnsurePythonVersionFile",
    ),
    "TestEnsureSymlink": (
        "tests.infra.unit.deps.test_internal_sync_update",
        "TestEnsureSymlink",
    ),
    "TestEnsureSymlinkEdgeCases": (
        "tests.infra.unit.deps.test_internal_sync_update",
        "TestEnsureSymlinkEdgeCases",
    ),
    "TestErrorReporting": (
        "tests.infra.unit.check.extended_error_reporting",
        "TestErrorReporting",
    ),
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
    "TestFixPyrelfyCLI": (
        "tests.infra.unit.check.extended_cli_entry",
        "TestFixPyrelfyCLI",
    ),
    "TestFixabilityClassification": (
        "tests.infra.unit.codegen.census",
        "TestFixabilityClassification",
    ),
    "TestFixerCore": ("tests.infra.unit.docs.fixer", "TestFixerCore"),
    "TestFixerMaybeFixLink": (
        "tests.infra.unit.docs.fixer_internals",
        "TestFixerMaybeFixLink",
    ),
    "TestFixerProcessFile": (
        "tests.infra.unit.docs.fixer_internals",
        "TestFixerProcessFile",
    ),
    "TestFixerScope": ("tests.infra.unit.docs.fixer_internals", "TestFixerScope"),
    "TestFixerToc": ("tests.infra.unit.docs.fixer_internals", "TestFixerToc"),
    "TestFlextInfraBaseMk": ("tests.infra.unit.basemk.init", "TestFlextInfraBaseMk"),
    "TestFlextInfraCheck": ("tests.infra.unit.check.init", "TestFlextInfraCheck"),
    "TestFlextInfraCodegenLazyInit": (
        "tests.infra.unit.codegen.lazy_init_service",
        "TestFlextInfraCodegenLazyInit",
    ),
    "TestFlextInfraCommandRunnerExtra": (
        "tests.infra.unit.test_infra_subprocess_extra",
        "TestFlextInfraCommandRunnerExtra",
    ),
    "TestFlextInfraConfigFixer": (
        "tests.infra.unit.check.pyrefly",
        "TestFlextInfraConfigFixer",
    ),
    "TestFlextInfraConstantsAlias": (
        "tests.infra.unit.test_infra_constants_extra",
        "TestFlextInfraConstantsAlias",
    ),
    "TestFlextInfraConstantsCheckNamespace": (
        "tests.infra.unit.test_infra_constants_extra",
        "TestFlextInfraConstantsCheckNamespace",
    ),
    "TestFlextInfraConstantsConsistency": (
        "tests.infra.unit.test_infra_constants_extra",
        "TestFlextInfraConstantsConsistency",
    ),
    "TestFlextInfraConstantsEncodingNamespace": (
        "tests.infra.unit.test_infra_constants_extra",
        "TestFlextInfraConstantsEncodingNamespace",
    ),
    "TestFlextInfraConstantsExcludedNamespace": (
        "tests.infra.unit.test_infra_constants_core",
        "TestFlextInfraConstantsExcludedNamespace",
    ),
    "TestFlextInfraConstantsFilesNamespace": (
        "tests.infra.unit.test_infra_constants_core",
        "TestFlextInfraConstantsFilesNamespace",
    ),
    "TestFlextInfraConstantsGatesNamespace": (
        "tests.infra.unit.test_infra_constants_core",
        "TestFlextInfraConstantsGatesNamespace",
    ),
    "TestFlextInfraConstantsGithubNamespace": (
        "tests.infra.unit.test_infra_constants_extra",
        "TestFlextInfraConstantsGithubNamespace",
    ),
    "TestFlextInfraConstantsImmutability": (
        "tests.infra.unit.test_infra_constants_extra",
        "TestFlextInfraConstantsImmutability",
    ),
    "TestFlextInfraConstantsPathsNamespace": (
        "tests.infra.unit.test_infra_constants_core",
        "TestFlextInfraConstantsPathsNamespace",
    ),
    "TestFlextInfraConstantsStatusNamespace": (
        "tests.infra.unit.test_infra_constants_core",
        "TestFlextInfraConstantsStatusNamespace",
    ),
    "TestFlextInfraDependencyDetectionModels": (
        "tests.infra.unit.deps.test_detection_models",
        "TestFlextInfraDependencyDetectionModels",
    ),
    "TestFlextInfraDependencyDetectionService": (
        "tests.infra.unit.deps.test_detection_models",
        "TestFlextInfraDependencyDetectionService",
    ),
    "TestFlextInfraDependencyDetectorModels": (
        "tests.infra.unit.deps.test_detector_models",
        "TestFlextInfraDependencyDetectorModels",
    ),
    "TestFlextInfraDependencyPathSync": (
        "tests.infra.unit.deps.test_path_sync_init",
        "TestFlextInfraDependencyPathSync",
    ),
    "TestFlextInfraDeps": ("tests.infra.unit.deps.test_init", "TestFlextInfraDeps"),
    "TestFlextInfraDiscoveryService": (
        "tests.infra.unit.discovery.test_infra_discovery",
        "TestFlextInfraDiscoveryService",
    ),
    "TestFlextInfraDiscoveryServiceUncoveredLines": (
        "tests.infra.unit.discovery.test_infra_discovery_edge_cases",
        "TestFlextInfraDiscoveryServiceUncoveredLines",
    ),
    "TestFlextInfraDocScope": (
        "tests.infra.unit.docs.shared",
        "TestFlextInfraDocScope",
    ),
    "TestFlextInfraDocs": ("tests.infra.unit.docs.init", "TestFlextInfraDocs"),
    "TestFlextInfraExtraPathsManager": (
        "tests.infra.unit.deps.test_extra_paths_manager",
        "TestFlextInfraExtraPathsManager",
    ),
    "TestFlextInfraGitService": (
        "tests.infra.unit.test_infra_git",
        "TestFlextInfraGitService",
    ),
    "TestFlextInfraInitLazyLoading": (
        "tests.infra.unit.test_infra_init_lazy_core",
        "TestFlextInfraInitLazyLoading",
    ),
    "TestFlextInfraInternalDependencySyncService": (
        "tests.infra.unit.deps.test_internal_sync_validation",
        "TestFlextInfraInternalDependencySyncService",
    ),
    "TestFlextInfraJsonService": (
        "tests.infra.unit.io.test_infra_json_io",
        "TestFlextInfraJsonService",
    ),
    "TestFlextInfraMaintenance": (
        "tests.infra.unit.test_infra_maintenance_init",
        "TestFlextInfraMaintenance",
    ),
    "TestFlextInfraPathResolver": (
        "tests.infra.unit.test_infra_paths",
        "TestFlextInfraPathResolver",
    ),
    "TestFlextInfraPatternsEdgeCases": (
        "tests.infra.unit.test_infra_patterns_extra",
        "TestFlextInfraPatternsEdgeCases",
    ),
    "TestFlextInfraPatternsMarkdown": (
        "tests.infra.unit.test_infra_patterns_core",
        "TestFlextInfraPatternsMarkdown",
    ),
    "TestFlextInfraPatternsPatternTypes": (
        "tests.infra.unit.test_infra_patterns_extra",
        "TestFlextInfraPatternsPatternTypes",
    ),
    "TestFlextInfraPatternsTooling": (
        "tests.infra.unit.test_infra_patterns_core",
        "TestFlextInfraPatternsTooling",
    ),
    "TestFlextInfraPrManager": (
        "tests.infra.unit.github.pr",
        "TestFlextInfraPrManager",
    ),
    "TestFlextInfraPrWorkspaceManager": (
        "tests.infra.unit.github.pr_workspace",
        "TestFlextInfraPrWorkspaceManager",
    ),
    "TestFlextInfraProtocolsImport": (
        "tests.infra.unit.test_infra_protocols",
        "TestFlextInfraProtocolsImport",
    ),
    "TestFlextInfraPyprojectModernizer": (
        "tests.infra.unit.deps.test_modernizer_main",
        "TestFlextInfraPyprojectModernizer",
    ),
    "TestFlextInfraReportingServiceCore": (
        "tests.infra.unit.test_infra_reporting_core",
        "TestFlextInfraReportingServiceCore",
    ),
    "TestFlextInfraReportingServiceExtra": (
        "tests.infra.unit.test_infra_reporting_extra",
        "TestFlextInfraReportingServiceExtra",
    ),
    "TestFlextInfraRuntimeDevDependencyDetectorInit": (
        "tests.infra.unit.deps.test_detector_init",
        "TestFlextInfraRuntimeDevDependencyDetectorInit",
    ),
    "TestFlextInfraRuntimeDevDependencyDetectorRunDetect": (
        "tests.infra.unit.deps.test_detector_detect",
        "TestFlextInfraRuntimeDevDependencyDetectorRunDetect",
    ),
    "TestFlextInfraRuntimeDevDependencyDetectorRunReport": (
        "tests.infra.unit.deps.test_detector_report",
        "TestFlextInfraRuntimeDevDependencyDetectorRunReport",
    ),
    "TestFlextInfraRuntimeDevDependencyDetectorRunTypings": (
        "tests.infra.unit.deps.test_detector_main",
        "TestFlextInfraRuntimeDevDependencyDetectorRunTypings",
    ),
    "TestFlextInfraSubmoduleInitLazyLoading": (
        "tests.infra.unit.test_infra_init_lazy_submodules",
        "TestFlextInfraSubmoduleInitLazyLoading",
    ),
    "TestFlextInfraTomlDocument": (
        "tests.infra.unit.test_infra_toml_io",
        "TestFlextInfraTomlDocument",
    ),
    "TestFlextInfraTomlHelpers": (
        "tests.infra.unit.test_infra_toml_io",
        "TestFlextInfraTomlHelpers",
    ),
    "TestFlextInfraTomlRead": (
        "tests.infra.unit.test_infra_toml_io",
        "TestFlextInfraTomlRead",
    ),
    "TestFlextInfraTypesImport": (
        "tests.infra.unit.test_infra_typings",
        "TestFlextInfraTypesImport",
    ),
    "TestFlextInfraUtilitiesImport": (
        "tests.infra.unit.test_infra_utilities",
        "TestFlextInfraUtilitiesImport",
    ),
    "TestFlextInfraUtilitiesSelection": (
        "tests.infra.unit.test_infra_selection",
        "TestFlextInfraUtilitiesSelection",
    ),
    "TestFlextInfraVersionClass": (
        "tests.infra.unit.test_infra_version_core",
        "TestFlextInfraVersionClass",
    ),
    "TestFlextInfraVersionModuleLevel": (
        "tests.infra.unit.test_infra_version_extra",
        "TestFlextInfraVersionModuleLevel",
    ),
    "TestFlextInfraVersionPackageInfo": (
        "tests.infra.unit.test_infra_version_extra",
        "TestFlextInfraVersionPackageInfo",
    ),
    "TestFlextInfraWorkflowLinter": (
        "tests.infra.unit.github.linter",
        "TestFlextInfraWorkflowLinter",
    ),
    "TestFlextInfraWorkflowSyncer": (
        "tests.infra.unit.github.workflows",
        "TestFlextInfraWorkflowSyncer",
    ),
    "TestFlextInfraWorkspace": (
        "tests.infra.unit.test_infra_workspace_init",
        "TestFlextInfraWorkspace",
    ),
    "TestFlextInfraWorkspaceChecker": (
        "tests.infra.unit.check.workspace",
        "TestFlextInfraWorkspaceChecker",
    ),
    "TestFormattingRunRuffFix": (
        "tests.infra.unit._utilities.test_formatting",
        "TestFormattingRunRuffFix",
    ),
    "TestGenerateFile": (
        "tests.infra.unit.codegen.lazy_init_generation",
        "TestGenerateFile",
    ),
    "TestGenerateNotes": (
        "tests.infra.unit.release.orchestrator_helpers",
        "TestGenerateNotes",
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
    "TestGeneratorCore": ("tests.infra.unit.docs.generator", "TestGeneratorCore"),
    "TestGeneratorHelpers": (
        "tests.infra.unit.docs.generator_internals",
        "TestGeneratorHelpers",
    ),
    "TestGeneratorScope": (
        "tests.infra.unit.docs.generator_internals",
        "TestGeneratorScope",
    ),
    "TestGetDepPaths": (
        "tests.infra.unit.deps.test_extra_paths_manager",
        "TestGetDepPaths",
    ),
    "TestGitPush": ("tests.infra.unit.test_infra_git", "TestGitPush"),
    "TestGitTagOperations": ("tests.infra.unit.test_infra_git", "TestGitTagOperations"),
    "TestGithubInit": ("tests.infra.unit.github.pr_init", "TestGithubInit"),
    "TestGoFmtEmptyLinesInOutput": (
        "tests.infra.unit.check.extended_error_reporting",
        "TestGoFmtEmptyLinesInOutput",
    ),
    "TestHandleLazyInit": ("tests.infra.unit.codegen.main", "TestHandleLazyInit"),
    "TestInferOwnerFromOrigin": (
        "tests.infra.unit.deps.test_internal_sync_resolve",
        "TestInferOwnerFromOrigin",
    ),
    "TestInferPackage": (
        "tests.infra.unit.codegen.lazy_init_helpers",
        "TestInferPackage",
    ),
    "TestInfraContainerFunctions": (
        "tests.infra.unit.container.test_infra_container",
        "TestInfraContainerFunctions",
    ),
    "TestInfraMroPattern": (
        "tests.infra.unit.container.test_infra_container",
        "TestInfraMroPattern",
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
    "TestInfraServiceRetrieval": (
        "tests.infra.unit.container.test_infra_container",
        "TestInfraServiceRetrieval",
    ),
    "TestInjectCommentsPhase": (
        "tests.infra.unit.deps.test_modernizer_comments",
        "TestInjectCommentsPhase",
    ),
    "TestInventoryServiceCore": (
        "tests.infra.unit.core.inventory",
        "TestInventoryServiceCore",
    ),
    "TestInventoryServiceReports": (
        "tests.infra.unit.core.inventory",
        "TestInventoryServiceReports",
    ),
    "TestInventoryServiceScripts": (
        "tests.infra.unit.core.inventory",
        "TestInventoryServiceScripts",
    ),
    "TestIsInternalPathDep": (
        "tests.infra.unit.deps.test_internal_sync_validation",
        "TestIsInternalPathDep",
    ),
    "TestIsRelativeTo": (
        "tests.infra.unit.deps.test_internal_sync_validation",
        "TestIsRelativeTo",
    ),
    "TestIsWorkspaceMode": (
        "tests.infra.unit.deps.test_internal_sync_workspace",
        "TestIsWorkspaceMode",
    ),
    "TestIterMarkdownFiles": (
        "tests.infra.unit.docs.shared_iter",
        "TestIterMarkdownFiles",
    ),
    "TestIterWorkspacePythonModules": (
        "tests.infra.unit._utilities.test_iteration",
        "TestIterWorkspacePythonModules",
    ),
    "TestJsonWriteFailure": (
        "tests.infra.unit.check.extended_project_runners",
        "TestJsonWriteFailure",
    ),
    "TestLintAndFormatPublicMethods": (
        "tests.infra.unit.check.extended_projects",
        "TestLintAndFormatPublicMethods",
    ),
    "TestLoadAuditBudgets": (
        "tests.infra.unit.docs.auditor_budgets",
        "TestLoadAuditBudgets",
    ),
    "TestLoadDependencyLimits": (
        "tests.infra.unit.deps.test_detection_typings",
        "TestLoadDependencyLimits",
    ),
    "TestMain": ("tests.infra.unit.deps.test_path_sync_main", "TestMain"),
    "TestMainBaseMkValidate": ("tests.infra.unit.core.main", "TestMainBaseMkValidate"),
    "TestMainCli": ("tests.infra.unit.test_infra_workspace_main", "TestMainCli"),
    "TestMainCliRouting": ("tests.infra.unit.core.main", "TestMainCliRouting"),
    "TestMainCommandDispatch": (
        "tests.infra.unit.codegen.main",
        "TestMainCommandDispatch",
    ),
    "TestMainDelegation": (
        "tests.infra.unit.deps.test_main_dispatch",
        "TestMainDelegation",
    ),
    "TestMainEdgeCases": (
        "tests.infra.unit.deps.test_path_sync_main_edges",
        "TestMainEdgeCases",
    ),
    "TestMainEntryPoint": ("tests.infra.unit.codegen.main", "TestMainEntryPoint"),
    "TestMainExceptionHandling": (
        "tests.infra.unit.deps.test_main_dispatch",
        "TestMainExceptionHandling",
    ),
    "TestMainFunction": (
        "tests.infra.unit.deps.test_detector_main",
        "TestMainFunction",
    ),
    "TestMainHelpAndErrors": (
        "tests.infra.unit.deps.test_main",
        "TestMainHelpAndErrors",
    ),
    "TestMainInventory": ("tests.infra.unit.core.main", "TestMainInventory"),
    "TestMainModuleImport": (
        "tests.infra.unit.deps.test_main_dispatch",
        "TestMainModuleImport",
    ),
    "TestMainReturnValues": ("tests.infra.unit.deps.test_main", "TestMainReturnValues"),
    "TestMainRouting": ("tests.infra.unit.docs.main_entry", "TestMainRouting"),
    "TestMainScan": ("tests.infra.unit.core.main", "TestMainScan"),
    "TestMainSubcommandDispatch": (
        "tests.infra.unit.deps.test_main_dispatch",
        "TestMainSubcommandDispatch",
    ),
    "TestMainSysArgvModification": (
        "tests.infra.unit.deps.test_main_dispatch",
        "TestMainSysArgvModification",
    ),
    "TestMainWithFlags": ("tests.infra.unit.docs.main_entry", "TestMainWithFlags"),
    "TestMaintenanceMainEnforcer": (
        "tests.infra.unit.test_infra_maintenance_main",
        "TestMaintenanceMainEnforcer",
    ),
    "TestMaintenanceMainSuccess": (
        "tests.infra.unit.test_infra_maintenance_main",
        "TestMaintenanceMainSuccess",
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
    "TestMaybeWriteTodo": (
        "tests.infra.unit.docs.validator_internals",
        "TestMaybeWriteTodo",
    ),
    "TestMerge": ("tests.infra.unit.github.pr_operations", "TestMerge"),
    "TestMergeChildExports": (
        "tests.infra.unit.codegen.lazy_init_transforms",
        "TestMergeChildExports",
    ),
    "TestMigratorDryRun": (
        "tests.infra.unit.test_infra_workspace_migrator_pyproject",
        "TestMigratorDryRun",
    ),
    "TestMigratorEdgeCases": (
        "tests.infra.unit.test_infra_workspace_migrator_internal",
        "TestMigratorEdgeCases",
    ),
    "TestMigratorFlextCore": (
        "tests.infra.unit.test_infra_workspace_migrator_pyproject",
        "TestMigratorFlextCore",
    ),
    "TestMigratorInternalMakefile": (
        "tests.infra.unit.test_infra_workspace_migrator_internal",
        "TestMigratorInternalMakefile",
    ),
    "TestMigratorInternalPyproject": (
        "tests.infra.unit.test_infra_workspace_migrator_internal",
        "TestMigratorInternalPyproject",
    ),
    "TestMigratorPoetryDeps": (
        "tests.infra.unit.test_infra_workspace_migrator_pyproject",
        "TestMigratorPoetryDeps",
    ),
    "TestMigratorReadFailures": (
        "tests.infra.unit.test_infra_workspace_migrator_errors",
        "TestMigratorReadFailures",
    ),
    "TestMigratorWriteFailures": (
        "tests.infra.unit.test_infra_workspace_migrator_errors",
        "TestMigratorWriteFailures",
    ),
    "TestModernizerEdgeCases": (
        "tests.infra.unit.deps.test_modernizer_main_extra",
        "TestModernizerEdgeCases",
    ),
    "TestModernizerRunAndMain": (
        "tests.infra.unit.deps.test_modernizer_main",
        "TestModernizerRunAndMain",
    ),
    "TestModernizerUncoveredLines": (
        "tests.infra.unit.deps.test_modernizer_main_extra",
        "TestModernizerUncoveredLines",
    ),
    "TestModuleAndTypingsFlow": (
        "tests.infra.unit.deps.test_detection_typings_flow",
        "TestModuleAndTypingsFlow",
    ),
    "TestModuleLevelWrappers": (
        "tests.infra.unit.deps.test_detection_wrappers",
        "TestModuleLevelWrappers",
    ),
    "TestMroFacadeMethods": (
        "tests.infra.unit.io.test_infra_output_edge_cases",
        "TestMroFacadeMethods",
    ),
    "TestMypyEmptyLinesInOutput": (
        "tests.infra.unit.check.extended_error_reporting",
        "TestMypyEmptyLinesInOutput",
    ),
    "TestNormalizeStringList": (
        "tests.infra.unit.core.skill_validator",
        "TestNormalizeStringList",
    ),
    "TestOrchestrate": (
        "tests.infra.unit.github.pr_workspace_orchestrate",
        "TestOrchestrate",
    ),
    "TestOrchestratorBasic": (
        "tests.infra.unit.test_infra_workspace_orchestrator",
        "TestOrchestratorBasic",
    ),
    "TestOrchestratorFailures": (
        "tests.infra.unit.test_infra_workspace_orchestrator",
        "TestOrchestratorFailures",
    ),
    "TestOwnerFromRemoteUrl": (
        "tests.infra.unit.deps.test_internal_sync_validation",
        "TestOwnerFromRemoteUrl",
    ),
    "TestParseArgs": ("tests.infra.unit.github.pr_cli", "TestParseArgs"),
    "TestParseGitmodules": (
        "tests.infra.unit.deps.test_internal_sync_discovery",
        "TestParseGitmodules",
    ),
    "TestParseRepoMap": (
        "tests.infra.unit.deps.test_internal_sync_discovery",
        "TestParseRepoMap",
    ),
    "TestParseViolationInvalid": (
        "tests.infra.unit.codegen.census",
        "TestParseViolationInvalid",
    ),
    "TestParseViolationValid": (
        "tests.infra.unit.codegen.census",
        "TestParseViolationValid",
    ),
    "TestParser": ("tests.infra.unit.deps.test_modernizer_workspace", "TestParser"),
    "TestParsingModuleAst": (
        "tests.infra.unit._utilities.test_parsing",
        "TestParsingModuleAst",
    ),
    "TestParsingModuleCst": (
        "tests.infra.unit._utilities.test_parsing",
        "TestParsingModuleCst",
    ),
    "TestPathDepPathsPep621": (
        "tests.infra.unit.deps.test_extra_paths_pep621",
        "TestPathDepPathsPep621",
    ),
    "TestPathDepPathsPoetry": (
        "tests.infra.unit.deps.test_extra_paths_pep621",
        "TestPathDepPathsPoetry",
    ),
    "TestPathSyncEdgeCases": (
        "tests.infra.unit.deps.test_path_sync_init",
        "TestPathSyncEdgeCases",
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
    "TestProcessDirectory": (
        "tests.infra.unit.codegen.lazy_init_process",
        "TestProcessDirectory",
    ),
    "TestProcessFileReadError": (
        "tests.infra.unit.check.extended_config_fixer_errors",
        "TestProcessFileReadError",
    ),
    "TestProjectResultProperties": (
        "tests.infra.unit.check.extended_models",
        "TestProjectResultProperties",
    ),
    "TestPushRelease": ("tests.infra.unit.release.orchestrator_git", "TestPushRelease"),
    "TestPytestDiagExtractorCore": (
        "tests.infra.unit.core.pytest_diag",
        "TestPytestDiagExtractorCore",
    ),
    "TestPytestDiagLogParsing": (
        "tests.infra.unit.core.pytest_diag",
        "TestPytestDiagLogParsing",
    ),
    "TestPytestDiagParseXml": (
        "tests.infra.unit.core.pytest_diag",
        "TestPytestDiagParseXml",
    ),
    "TestReadDoc": ("tests.infra.unit.deps.test_modernizer_workspace", "TestReadDoc"),
    "TestReadExistingDocstring": (
        "tests.infra.unit.codegen.lazy_init_helpers",
        "TestReadExistingDocstring",
    ),
    "TestReadRequiredMinor": (
        "tests.infra.unit.test_infra_maintenance_python_version",
        "TestReadRequiredMinor",
    ),
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
    "TestRemovedCompatibilityMethods": (
        "tests.infra.unit.test_infra_git",
        "TestRemovedCompatibilityMethods",
    ),
    "TestRenderTemplate": ("tests.infra.unit.github.workflows", "TestRenderTemplate"),
    "TestResolveAliases": (
        "tests.infra.unit.codegen.lazy_init_generation",
        "TestResolveAliases",
    ),
    "TestResolveRef": (
        "tests.infra.unit.deps.test_internal_sync_resolve",
        "TestResolveRef",
    ),
    "TestResolveVersionInteractive": (
        "tests.infra.unit.release.version_resolution",
        "TestResolveVersionInteractive",
    ),
    "TestRewriteDepPaths": (
        "tests.infra.unit.deps.test_path_sync_rewrite_deps",
        "TestRewriteDepPaths",
    ),
    "TestRewritePep621": (
        "tests.infra.unit.deps.test_path_sync_rewrite_pep621",
        "TestRewritePep621",
    ),
    "TestRewritePoetry": (
        "tests.infra.unit.deps.test_path_sync_rewrite_poetry",
        "TestRewritePoetry",
    ),
    "TestRuffFormatDuplicateFiles": (
        "tests.infra.unit.check.extended_error_reporting",
        "TestRuffFormatDuplicateFiles",
    ),
    "TestRunAudit": ("tests.infra.unit.docs.main", "TestRunAudit"),
    "TestRunBandit": ("tests.infra.unit.check.extended_runners_extra", "TestRunBandit"),
    "TestRunBuild": ("tests.infra.unit.docs.main_commands", "TestRunBuild"),
    "TestRunCLIExtended": (
        "tests.infra.unit.check.extended_cli_entry",
        "TestRunCLIExtended",
    ),
    "TestRunCommand": (
        "tests.infra.unit.check.extended_runners_ruff",
        "TestRunCommand",
    ),
    "TestRunDeptry": ("tests.infra.unit.deps.test_detection_deptry", "TestRunDeptry"),
    "TestRunDetect": ("tests.infra.unit.test_infra_workspace_main", "TestRunDetect"),
    "TestRunFix": ("tests.infra.unit.docs.main", "TestRunFix"),
    "TestRunGenerate": ("tests.infra.unit.docs.main_commands", "TestRunGenerate"),
    "TestRunGo": ("tests.infra.unit.check.extended_runners_go", "TestRunGo"),
    "TestRunLint": ("tests.infra.unit.github.main", "TestRunLint"),
    "TestRunMake": ("tests.infra.unit.release.orchestrator_helpers", "TestRunMake"),
    "TestRunMarkdown": (
        "tests.infra.unit.check.extended_runners_extra",
        "TestRunMarkdown",
    ),
    "TestRunMigrate": ("tests.infra.unit.test_infra_workspace_main", "TestRunMigrate"),
    "TestRunMypy": ("tests.infra.unit.check.extended_runners", "TestRunMypy"),
    "TestRunMypyStubHints": (
        "tests.infra.unit.deps.test_detection_typings",
        "TestRunMypyStubHints",
    ),
    "TestRunOrchestrate": (
        "tests.infra.unit.test_infra_workspace_main",
        "TestRunOrchestrate",
    ),
    "TestRunPipCheck": (
        "tests.infra.unit.deps.test_detection_pip_check",
        "TestRunPipCheck",
    ),
    "TestRunPr": ("tests.infra.unit.github.pr_workspace", "TestRunPr"),
    "TestRunPrWorkspace": (
        "tests.infra.unit.github.main_dispatch",
        "TestRunPrWorkspace",
    ),
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
    "TestRunRuffFix": (
        "tests.infra.unit.codegen.lazy_init_generation",
        "TestRunRuffFix",
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
    "TestRunSync": ("tests.infra.unit.test_infra_workspace_main", "TestRunSync"),
    "TestRunValidate": ("tests.infra.unit.docs.main_commands", "TestRunValidate"),
    "TestRunWorkflows": ("tests.infra.unit.github.main", "TestRunWorkflows"),
    "TestSafeLoadYaml": ("tests.infra.unit.core.skill_validator", "TestSafeLoadYaml"),
    "TestSafetyCheckpoint": (
        "tests.infra.unit._utilities.test_safety",
        "TestSafetyCheckpoint",
    ),
    "TestSafetyRollback": (
        "tests.infra.unit._utilities.test_safety",
        "TestSafetyRollback",
    ),
    "TestSafetyWorkspaceValidation": (
        "tests.infra.unit._utilities.test_safety",
        "TestSafetyWorkspaceValidation",
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
    "TestScanFileBatch": (
        "tests.infra.unit._utilities.test_scanning",
        "TestScanFileBatch",
    ),
    "TestScanModels": ("tests.infra.unit._utilities.test_scanning", "TestScanModels"),
    "TestScannerCore": ("tests.infra.unit.core.scanner", "TestScannerCore"),
    "TestScannerHelpers": ("tests.infra.unit.core.scanner", "TestScannerHelpers"),
    "TestScannerMultiFile": ("tests.infra.unit.core.scanner", "TestScannerMultiFile"),
    "TestSelectedProjectNames": (
        "tests.infra.unit.docs.shared_iter",
        "TestSelectedProjectNames",
    ),
    "TestSelectorFunction": ("tests.infra.unit.github.pr_cli", "TestSelectorFunction"),
    "TestShouldBubbleUp": (
        "tests.infra.unit.codegen.lazy_init_transforms",
        "TestShouldBubbleUp",
    ),
    "TestShouldUseColor": (
        "tests.infra.unit.io.test_infra_terminal_detection",
        "TestShouldUseColor",
    ),
    "TestShouldUseUnicode": (
        "tests.infra.unit.io.test_infra_terminal_detection",
        "TestShouldUseUnicode",
    ),
    "TestSkillValidatorAstGrepCount": (
        "tests.infra.unit.core.skill_validator",
        "TestSkillValidatorAstGrepCount",
    ),
    "TestSkillValidatorCore": (
        "tests.infra.unit.core.skill_validator",
        "TestSkillValidatorCore",
    ),
    "TestSkillValidatorRenderTemplate": (
        "tests.infra.unit.core.skill_validator",
        "TestSkillValidatorRenderTemplate",
    ),
    "TestStaticMethods": (
        "tests.infra.unit.github.pr_workspace_orchestrate",
        "TestStaticMethods",
    ),
    "TestStatus": ("tests.infra.unit.github.pr", "TestStatus"),
    "TestStubChainAnalyze": (
        "tests.infra.unit.core.stub_chain",
        "TestStubChainAnalyze",
    ),
    "TestStubChainCore": ("tests.infra.unit.core.stub_chain", "TestStubChainCore"),
    "TestStubChainDiscoverProjects": (
        "tests.infra.unit.core.stub_chain",
        "TestStubChainDiscoverProjects",
    ),
    "TestStubChainIsInternal": (
        "tests.infra.unit.core.stub_chain",
        "TestStubChainIsInternal",
    ),
    "TestStubChainStubExists": (
        "tests.infra.unit.core.stub_chain",
        "TestStubChainStubExists",
    ),
    "TestStubChainValidate": (
        "tests.infra.unit.core.stub_chain",
        "TestStubChainValidate",
    ),
    "TestSubcommandMapping": (
        "tests.infra.unit.deps.test_main",
        "TestSubcommandMapping",
    ),
    "TestSync": ("tests.infra.unit.deps.test_internal_sync_sync", "TestSync"),
    "TestSyncMethodEdgeCases": (
        "tests.infra.unit.deps.test_internal_sync_sync_edge",
        "TestSyncMethodEdgeCases",
    ),
    "TestSyncMethodEdgeCasesMore": (
        "tests.infra.unit.deps.test_internal_sync_sync_edge_more",
        "TestSyncMethodEdgeCasesMore",
    ),
    "TestSyncOne": ("tests.infra.unit.deps.test_extra_paths_manager", "TestSyncOne"),
    "TestSyncOperation": ("tests.infra.unit.github.workflows", "TestSyncOperation"),
    "TestSyncProject": ("tests.infra.unit.github.workflows", "TestSyncProject"),
    "TestSyncWorkspace": (
        "tests.infra.unit.github.workflows_workspace",
        "TestSyncWorkspace",
    ),
    "TestSynthesizedRepoMap": (
        "tests.infra.unit.deps.test_internal_sync_resolve",
        "TestSynthesizedRepoMap",
    ),
    "TestToInfraValue": (
        "tests.infra.unit.deps.test_detection_models",
        "TestToInfraValue",
    ),
    "TestTriggerRelease": (
        "tests.infra.unit.github.pr_operations",
        "TestTriggerRelease",
    ),
    "TestUpdateChangelog": (
        "tests.infra.unit.release.orchestrator_helpers",
        "TestUpdateChangelog",
    ),
    "TestValidateCore": ("tests.infra.unit.docs.validator", "TestValidateCore"),
    "TestValidateGitRefEdgeCases": (
        "tests.infra.unit.deps.test_internal_sync_validation",
        "TestValidateGitRefEdgeCases",
    ),
    "TestValidateReport": ("tests.infra.unit.docs.validator", "TestValidateReport"),
    "TestValidateScope": (
        "tests.infra.unit.docs.validator_internals",
        "TestValidateScope",
    ),
    "TestVersionFiles": (
        "tests.infra.unit.release.orchestrator_helpers",
        "TestVersionFiles",
    ),
    "TestView": ("tests.infra.unit.github.pr_operations", "TestView"),
    "TestViolationPattern": (
        "tests.infra.unit.codegen.census_models",
        "TestViolationPattern",
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
    "TestWorkspaceRoot": (
        "tests.infra.unit.test_infra_maintenance_python_version",
        "TestWorkspaceRoot",
    ),
    "TestWorkspaceRootFromEnv": (
        "tests.infra.unit.deps.test_internal_sync_workspace",
        "TestWorkspaceRootFromEnv",
    ),
    "TestWorkspaceRootFromParents": (
        "tests.infra.unit.deps.test_internal_sync_workspace",
        "TestWorkspaceRootFromParents",
    ),
    "TestWriteJson": ("tests.infra.unit.docs.shared_write", "TestWriteJson"),
    "TestWriteMarkdown": ("tests.infra.unit.docs.shared_write", "TestWriteMarkdown"),
    "TestWriteReport": (
        "tests.infra.unit.github.workflows_workspace",
        "TestWriteReport",
    ),
    "WorkspaceFactory": ("tests.infra.workspace_factory", "WorkspaceFactory"),
    "WorkspaceScenario": ("tests.infra.scenarios", "WorkspaceScenario"),
    "WorkspaceScenarios": ("tests.infra.scenarios", "WorkspaceScenarios"),
    "array": ("tests.infra.unit.deps.test_modernizer_helpers", "array"),
    "as_string_list": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "as_string_list",
    ),
    "auditor": ("tests.infra.unit.docs.auditor", "auditor"),
    "basemk": ("tests.infra.unit.basemk", ""),
    "builder": ("tests.infra.unit.docs.builder", "builder"),
    "c": ("tests.infra.constants", "c"),
    "canonical_dev_dependencies": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "canonical_dev_dependencies",
    ),
    "census": ("tests.infra.unit.codegen.census", "census"),
    "check": ("tests.infra.unit.check", ""),
    "codegen": ("tests.infra.unit.codegen", ""),
    "container": ("tests.infra.unit.container", ""),
    "core": ("tests.infra.unit.core", ""),
    "dedupe_specs": ("tests.infra.unit.deps.test_modernizer_helpers", "dedupe_specs"),
    "dep_name": ("tests.infra.unit.deps.test_modernizer_helpers", "dep_name"),
    "deps": ("tests.infra.unit.deps", ""),
    "detector": ("tests.infra.unit.test_infra_workspace_detector", "detector"),
    "discovery": ("tests.infra.unit.discovery", ""),
    "doc": ("tests.infra.unit.deps.test_modernizer_helpers", "doc"),
    "docs": ("tests.infra.unit.docs", ""),
    "engine": ("tests.infra.unit.test_infra_templates", "engine"),
    "ensure_table": ("tests.infra.unit.deps.test_modernizer_helpers", "ensure_table"),
    "extract_dep_name": (
        "tests.infra.unit.deps.test_path_sync_helpers",
        "extract_dep_name",
    ),
    "fixer": ("tests.infra.unit.codegen.autofix", "fixer"),
    "gen": ("tests.infra.unit.docs.generator_internals", "gen"),
    "git_repo": ("tests.infra.unit.test_infra_git", "git_repo"),
    "github": ("tests.infra.unit.github", ""),
    "h": ("tests.infra.helpers", "h"),
    "io": ("tests.infra.unit.io", ""),
    "is_external": ("tests.infra.unit.docs.auditor", "is_external"),
    "m": ("tests.infra.models", "m"),
    "normalize_link": ("tests.infra.unit.docs.auditor", "normalize_link"),
    "orchestrator": (
        "tests.infra.unit.test_infra_workspace_orchestrator",
        "orchestrator",
    ),
    "p": ("tests.infra.protocols", "p"),
    "project_dev_groups": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "project_dev_groups",
    ),
    "pyright_content": (
        "tests.infra.unit.deps.test_extra_paths_sync",
        "pyright_content",
    ),
    "r": (
        "tests.infra.unit.check.extended_workspace_init",
        "TestWorkspaceCheckerBuildGateResult",
    ),
    "real_docs_project": ("tests.infra.fixtures", "real_docs_project"),
    "real_git_repo": ("tests.infra.fixtures_git", "real_git_repo"),
    "real_makefile_project": ("tests.infra.fixtures", "real_makefile_project"),
    "real_python_package": ("tests.infra.fixtures", "real_python_package"),
    "real_toml_project": ("tests.infra.fixtures", "real_toml_project"),
    "real_workspace": ("tests.infra.fixtures", "real_workspace"),
    "refactor": ("tests.infra.unit.refactor", ""),
    "release": ("tests.infra.unit.release", ""),
    "rewrite_dep_paths": (
        "tests.infra.unit.deps.test_path_sync_rewrite_deps",
        "rewrite_dep_paths",
    ),
    "run_lint": ("tests.infra.unit.github.main", "run_lint"),
    "run_pr": ("tests.infra.unit.github.main", "run_pr"),
    "run_pr_workspace": ("tests.infra.unit.github.main_dispatch", "run_pr_workspace"),
    "run_workflows": ("tests.infra.unit.github.main", "run_workflows"),
    "runner": ("tests.infra.unit.test_infra_subprocess_core", "runner"),
    "s": (
        "tests.infra.unit.deps.test_detection_models",
        "TestFlextInfraDependencyDetectionService",
    ),
    "service": ("tests.infra.unit.test_infra_versioning", "service"),
    "should_skip_target": ("tests.infra.unit.docs.auditor", "should_skip_target"),
    "svc": ("tests.infra.unit.test_infra_workspace_sync", "svc"),
    "t": ("tests.infra.typings", "t"),
    "test_array": ("tests.infra.unit.deps.test_modernizer_helpers", "test_array"),
    "test_as_string_list": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "test_as_string_list",
    ),
    "test_as_string_list_toml_item": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "test_as_string_list_toml_item",
    ),
    "test_atomic_write_fail": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_atomic_write_fail",
    ),
    "test_atomic_write_ok": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_atomic_write_ok",
    ),
    "test_basemk_build_config_with_none": (
        "tests.infra.unit.basemk.main",
        "test_basemk_build_config_with_none",
    ),
    "test_basemk_build_config_with_project_name": (
        "tests.infra.unit.basemk.main",
        "test_basemk_build_config_with_project_name",
    ),
    "test_basemk_cli_generate_to_file": (
        "tests.infra.unit.basemk.engine",
        "test_basemk_cli_generate_to_file",
    ),
    "test_basemk_cli_generate_to_stdout": (
        "tests.infra.unit.basemk.engine",
        "test_basemk_cli_generate_to_stdout",
    ),
    "test_basemk_engine_execute_calls_render_all": (
        "tests.infra.unit.basemk.engine",
        "test_basemk_engine_execute_calls_render_all",
    ),
    "test_basemk_engine_render_all_handles_template_error": (
        "tests.infra.unit.basemk.engine",
        "test_basemk_engine_render_all_handles_template_error",
    ),
    "test_basemk_engine_render_all_returns_string": (
        "tests.infra.unit.basemk.engine",
        "test_basemk_engine_render_all_returns_string",
    ),
    "test_basemk_engine_render_all_with_valid_config": (
        "tests.infra.unit.basemk.engine",
        "test_basemk_engine_render_all_with_valid_config",
    ),
    "test_basemk_main_calls_sys_exit": (
        "tests.infra.unit.basemk.main",
        "test_basemk_main_calls_sys_exit",
    ),
    "test_basemk_main_ensures_structlog_configured": (
        "tests.infra.unit.basemk.main",
        "test_basemk_main_ensures_structlog_configured",
    ),
    "test_basemk_main_output_to_stdout": (
        "tests.infra.unit.basemk.main",
        "test_basemk_main_output_to_stdout",
    ),
    "test_basemk_main_with_generate_command": (
        "tests.infra.unit.basemk.main",
        "test_basemk_main_with_generate_command",
    ),
    "test_basemk_main_with_generation_failure": (
        "tests.infra.unit.basemk.main",
        "test_basemk_main_with_generation_failure",
    ),
    "test_basemk_main_with_invalid_command": (
        "tests.infra.unit.basemk.main",
        "test_basemk_main_with_invalid_command",
    ),
    "test_basemk_main_with_no_command": (
        "tests.infra.unit.basemk.main",
        "test_basemk_main_with_no_command",
    ),
    "test_basemk_main_with_none_argv": (
        "tests.infra.unit.basemk.main",
        "test_basemk_main_with_none_argv",
    ),
    "test_basemk_main_with_output_file": (
        "tests.infra.unit.basemk.main",
        "test_basemk_main_with_output_file",
    ),
    "test_basemk_main_with_project_name": (
        "tests.infra.unit.basemk.main",
        "test_basemk_main_with_project_name",
    ),
    "test_basemk_main_with_write_failure": (
        "tests.infra.unit.basemk.main",
        "test_basemk_main_with_write_failure",
    ),
    "test_build_impact_map_extracts_rename_entries": (
        "tests.infra.unit.refactor.test_infra_refactor_analysis",
        "test_build_impact_map_extracts_rename_entries",
    ),
    "test_build_impact_map_extracts_signature_entries": (
        "tests.infra.unit.refactor.test_infra_refactor_analysis",
        "test_build_impact_map_extracts_signature_entries",
    ),
    "test_bump_version_invalid": (
        "tests.infra.unit.test_infra_versioning",
        "test_bump_version_invalid",
    ),
    "test_bump_version_result_type": (
        "tests.infra.unit.test_infra_versioning",
        "test_bump_version_result_type",
    ),
    "test_bump_version_valid": (
        "tests.infra.unit.test_infra_versioning",
        "test_bump_version_valid",
    ),
    "test_canonical_dev_dependencies": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "test_canonical_dev_dependencies",
    ),
    "test_capture_cases": (
        "tests.infra.unit.test_infra_subprocess_core",
        "test_capture_cases",
    ),
    "test_check_main_executes_real_cli": (
        "tests.infra.unit.check.main",
        "test_check_main_executes_real_cli",
    ),
    "test_class_reconstructor_reorders_each_contiguous_method_block": (
        "tests.infra.unit.refactor.test_infra_refactor_class_and_propagation",
        "test_class_reconstructor_reorders_each_contiguous_method_block",
    ),
    "test_class_reconstructor_reorders_methods_by_config": (
        "tests.infra.unit.refactor.test_infra_refactor_class_and_propagation",
        "test_class_reconstructor_reorders_methods_by_config",
    ),
    "test_class_reconstructor_skips_interleaved_non_method_members": (
        "tests.infra.unit.refactor.test_infra_refactor_class_and_propagation",
        "test_class_reconstructor_skips_interleaved_non_method_members",
    ),
    "test_cli_result_by_project_root": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_cli_result_by_project_root",
    ),
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
    "test_consolidate_groups_phase_apply_removes_old_groups": (
        "tests.infra.unit.deps.test_modernizer_consolidate",
        "test_consolidate_groups_phase_apply_removes_old_groups",
    ),
    "test_consolidate_groups_phase_apply_with_empty_poetry_group": (
        "tests.infra.unit.deps.test_modernizer_consolidate",
        "test_consolidate_groups_phase_apply_with_empty_poetry_group",
    ),
    "test_current_workspace_version": (
        "tests.infra.unit.test_infra_versioning",
        "test_current_workspace_version",
    ),
    "test_dedupe_specs": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "test_dedupe_specs",
    ),
    "test_dep_name": ("tests.infra.unit.deps.test_modernizer_helpers", "test_dep_name"),
    "test_detect_mode_with_nonexistent_path": (
        "tests.infra.unit.deps.test_path_sync_init",
        "test_detect_mode_with_nonexistent_path",
    ),
    "test_detect_mode_with_path_object": (
        "tests.infra.unit.deps.test_path_sync_init",
        "test_detect_mode_with_path_object",
    ),
    "test_discover_projects_wrapper": (
        "tests.infra.unit.deps.test_detection_wrappers",
        "test_discover_projects_wrapper",
    ),
    "test_engine_always_enables_class_nesting_file_rule": (
        "tests.infra.unit.refactor.test_infra_refactor_engine",
        "test_engine_always_enables_class_nesting_file_rule",
    ),
    "test_engine_constants_shared": (
        "tests.infra.unit.test_infra_templates",
        "test_engine_constants_shared",
    ),
    "test_ensure_future_annotations_after_docstring": (
        "tests.infra.unit.refactor.test_infra_refactor_legacy_and_annotations",
        "test_ensure_future_annotations_after_docstring",
    ),
    "test_ensure_future_annotations_moves_existing_import_to_top": (
        "tests.infra.unit.refactor.test_infra_refactor_legacy_and_annotations",
        "test_ensure_future_annotations_moves_existing_import_to_top",
    ),
    "test_ensure_pyrefly_config_phase_apply_errors": (
        "tests.infra.unit.deps.test_modernizer_pyrefly",
        "test_ensure_pyrefly_config_phase_apply_errors",
    ),
    "test_ensure_pyrefly_config_phase_apply_ignore_errors": (
        "tests.infra.unit.deps.test_modernizer_pyrefly",
        "test_ensure_pyrefly_config_phase_apply_ignore_errors",
    ),
    "test_ensure_pyrefly_config_phase_apply_python_version": (
        "tests.infra.unit.deps.test_modernizer_pyrefly",
        "test_ensure_pyrefly_config_phase_apply_python_version",
    ),
    "test_ensure_pyrefly_config_phase_apply_search_path": (
        "tests.infra.unit.deps.test_modernizer_pyrefly",
        "test_ensure_pyrefly_config_phase_apply_search_path",
    ),
    "test_ensure_pytest_config_phase_apply_markers": (
        "tests.infra.unit.deps.test_modernizer_pytest",
        "test_ensure_pytest_config_phase_apply_markers",
    ),
    "test_ensure_pytest_config_phase_apply_minversion": (
        "tests.infra.unit.deps.test_modernizer_pytest",
        "test_ensure_pytest_config_phase_apply_minversion",
    ),
    "test_ensure_pytest_config_phase_apply_python_classes": (
        "tests.infra.unit.deps.test_modernizer_pytest",
        "test_ensure_pytest_config_phase_apply_python_classes",
    ),
    "test_ensure_table": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "test_ensure_table",
    ),
    "test_extract_dep_name": (
        "tests.infra.unit.deps.test_path_sync_helpers",
        "test_extract_dep_name",
    ),
    "test_extract_requirement_name": (
        "tests.infra.unit.deps.test_path_sync_helpers",
        "test_extract_requirement_name",
    ),
    "test_files_modified_tracks_affected_files": (
        "tests.infra.unit.codegen.autofix_workspace",
        "test_files_modified_tracks_affected_files",
    ),
    "test_fix_pyrefly_config_main_executes_real_cli_help": (
        "tests.infra.unit.check.fix_pyrefly_config",
        "test_fix_pyrefly_config_main_executes_real_cli_help",
    ),
    "test_flexcore_excluded_from_run": (
        "tests.infra.unit.codegen.autofix_workspace",
        "test_flexcore_excluded_from_run",
    ),
    "test_flext_infra_pyproject_modernizer_find_pyproject_files": (
        "tests.infra.unit.deps.test_modernizer_main_extra",
        "test_flext_infra_pyproject_modernizer_find_pyproject_files",
    ),
    "test_flext_infra_pyproject_modernizer_process_file_invalid_toml": (
        "tests.infra.unit.deps.test_modernizer_main_extra",
        "test_flext_infra_pyproject_modernizer_process_file_invalid_toml",
    ),
    "test_generator_execute_returns_generated_content": (
        "tests.infra.unit.basemk.generator",
        "test_generator_execute_returns_generated_content",
    ),
    "test_generator_fails_for_invalid_make_syntax": (
        "tests.infra.unit.basemk.engine",
        "test_generator_fails_for_invalid_make_syntax",
    ),
    "test_generator_generate_propagates_render_failure": (
        "tests.infra.unit.basemk.generator",
        "test_generator_generate_propagates_render_failure",
    ),
    "test_generator_generate_with_basemk_config_object": (
        "tests.infra.unit.basemk.generator",
        "test_generator_generate_with_basemk_config_object",
    ),
    "test_generator_generate_with_dict_config": (
        "tests.infra.unit.basemk.generator",
        "test_generator_generate_with_dict_config",
    ),
    "test_generator_generate_with_invalid_dict_config": (
        "tests.infra.unit.basemk.generator",
        "test_generator_generate_with_invalid_dict_config",
    ),
    "test_generator_generate_with_none_config_uses_default": (
        "tests.infra.unit.basemk.generator",
        "test_generator_generate_with_none_config_uses_default",
    ),
    "test_generator_initializes_with_custom_engine": (
        "tests.infra.unit.basemk.generator",
        "test_generator_initializes_with_custom_engine",
    ),
    "test_generator_initializes_with_default_engine": (
        "tests.infra.unit.basemk.generator",
        "test_generator_initializes_with_default_engine",
    ),
    "test_generator_normalize_config_with_basemk_config": (
        "tests.infra.unit.basemk.generator_edge_cases",
        "test_generator_normalize_config_with_basemk_config",
    ),
    "test_generator_normalize_config_with_dict": (
        "tests.infra.unit.basemk.generator_edge_cases",
        "test_generator_normalize_config_with_dict",
    ),
    "test_generator_normalize_config_with_invalid_dict": (
        "tests.infra.unit.basemk.generator_edge_cases",
        "test_generator_normalize_config_with_invalid_dict",
    ),
    "test_generator_normalize_config_with_none": (
        "tests.infra.unit.basemk.generator_edge_cases",
        "test_generator_normalize_config_with_none",
    ),
    "test_generator_renders_with_config_override": (
        "tests.infra.unit.basemk.engine",
        "test_generator_renders_with_config_override",
    ),
    "test_generator_validate_generated_output_handles_oserror": (
        "tests.infra.unit.basemk.generator_edge_cases",
        "test_generator_validate_generated_output_handles_oserror",
    ),
    "test_generator_write_creates_parent_directories": (
        "tests.infra.unit.basemk.generator",
        "test_generator_write_creates_parent_directories",
    ),
    "test_generator_write_fails_without_output_or_stream": (
        "tests.infra.unit.basemk.generator",
        "test_generator_write_fails_without_output_or_stream",
    ),
    "test_generator_write_handles_file_permission_error": (
        "tests.infra.unit.basemk.generator_edge_cases",
        "test_generator_write_handles_file_permission_error",
    ),
    "test_generator_write_saves_output_file": (
        "tests.infra.unit.basemk.engine",
        "test_generator_write_saves_output_file",
    ),
    "test_generator_write_to_file": (
        "tests.infra.unit.basemk.generator",
        "test_generator_write_to_file",
    ),
    "test_generator_write_to_stream": (
        "tests.infra.unit.basemk.generator",
        "test_generator_write_to_stream",
    ),
    "test_generator_write_to_stream_handles_oserror": (
        "tests.infra.unit.basemk.generator_edge_cases",
        "test_generator_write_to_stream_handles_oserror",
    ),
    "test_get_current_typings_from_pyproject_wrapper": (
        "tests.infra.unit.deps.test_detection_wrappers",
        "test_get_current_typings_from_pyproject_wrapper",
    ),
    "test_get_required_typings_wrapper": (
        "tests.infra.unit.deps.test_detection_wrappers",
        "test_get_required_typings_wrapper",
    ),
    "test_gitignore_entry_scenarios": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_gitignore_entry_scenarios",
    ),
    "test_gitignore_sync_failure": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_gitignore_sync_failure",
    ),
    "test_gitignore_write_failure": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_gitignore_write_failure",
    ),
    "test_helpers_alias_exposed": (
        "tests.infra.unit.deps.test_extra_paths_pep621",
        "test_helpers_alias_exposed",
    ),
    "test_helpers_alias_is_reachable_helpers": (
        "tests.infra.unit.deps.test_path_sync_helpers",
        "test_helpers_alias_is_reachable_helpers",
    ),
    "test_helpers_alias_is_reachable_main": (
        "tests.infra.unit.deps.test_path_sync_main",
        "test_helpers_alias_is_reachable_main",
    ),
    "test_helpers_alias_is_reachable_project_obj": (
        "tests.infra.unit.deps.test_path_sync_main_project_obj",
        "test_helpers_alias_is_reachable_project_obj",
    ),
    "test_import_modernizer_adds_c_when_existing_c_is_aliased": (
        "tests.infra.unit.refactor.test_infra_refactor_import_modernizer",
        "test_import_modernizer_adds_c_when_existing_c_is_aliased",
    ),
    "test_import_modernizer_does_not_rewrite_function_parameter_shadow": (
        "tests.infra.unit.refactor.test_infra_refactor_import_modernizer",
        "test_import_modernizer_does_not_rewrite_function_parameter_shadow",
    ),
    "test_import_modernizer_does_not_rewrite_rebound_local_name_usage": (
        "tests.infra.unit.refactor.test_infra_refactor_import_modernizer",
        "test_import_modernizer_does_not_rewrite_rebound_local_name_usage",
    ),
    "test_import_modernizer_partial_import_keeps_unmapped_symbols": (
        "tests.infra.unit.refactor.test_infra_refactor_import_modernizer",
        "test_import_modernizer_partial_import_keeps_unmapped_symbols",
    ),
    "test_import_modernizer_partial_import_with_asname_keeps_unmapped_alias": (
        "tests.infra.unit.refactor.test_infra_refactor_import_modernizer",
        "test_import_modernizer_partial_import_with_asname_keeps_unmapped_alias",
    ),
    "test_import_modernizer_skips_rewrite_when_runtime_alias_shadowed_in_function": (
        "tests.infra.unit.refactor.test_infra_refactor_import_modernizer",
        "test_import_modernizer_skips_rewrite_when_runtime_alias_shadowed_in_function",
    ),
    "test_import_modernizer_skips_when_runtime_alias_name_is_blocked": (
        "tests.infra.unit.refactor.test_infra_refactor_import_modernizer",
        "test_import_modernizer_skips_when_runtime_alias_name_is_blocked",
    ),
    "test_import_modernizer_updates_aliased_symbol_usage": (
        "tests.infra.unit.refactor.test_infra_refactor_import_modernizer",
        "test_import_modernizer_updates_aliased_symbol_usage",
    ),
    "test_in_context_typevar_not_flagged": (
        "tests.infra.unit.codegen.autofix",
        "test_in_context_typevar_not_flagged",
    ),
    "test_inject_comments_phase_apply_banner": (
        "tests.infra.unit.deps.test_modernizer_comments",
        "test_inject_comments_phase_apply_banner",
    ),
    "test_inject_comments_phase_apply_broken_group_section": (
        "tests.infra.unit.deps.test_modernizer_comments",
        "test_inject_comments_phase_apply_broken_group_section",
    ),
    "test_inject_comments_phase_apply_markers": (
        "tests.infra.unit.deps.test_modernizer_comments",
        "test_inject_comments_phase_apply_markers",
    ),
    "test_inject_comments_phase_apply_with_optional_dependencies_dev": (
        "tests.infra.unit.deps.test_modernizer_comments",
        "test_inject_comments_phase_apply_with_optional_dependencies_dev",
    ),
    "test_lazy_import_rule_hoists_import_to_module_level": (
        "tests.infra.unit.refactor.test_infra_refactor_import_modernizer",
        "test_lazy_import_rule_hoists_import_to_module_level",
    ),
    "test_lazy_import_rule_uses_fix_action_for_hoist": (
        "tests.infra.unit.refactor.test_infra_refactor_import_modernizer",
        "test_lazy_import_rule_uses_fix_action_for_hoist",
    ),
    "test_legacy_import_bypass_collapses_to_primary_import": (
        "tests.infra.unit.refactor.test_infra_refactor_legacy_and_annotations",
        "test_legacy_import_bypass_collapses_to_primary_import",
    ),
    "test_legacy_rule_uses_fix_action_remove_for_aliases": (
        "tests.infra.unit.refactor.test_infra_refactor_legacy_and_annotations",
        "test_legacy_rule_uses_fix_action_remove_for_aliases",
    ),
    "test_legacy_wrapper_forwarding_keywords_is_inlined_as_alias": (
        "tests.infra.unit.refactor.test_infra_refactor_legacy_and_annotations",
        "test_legacy_wrapper_forwarding_keywords_is_inlined_as_alias",
    ),
    "test_legacy_wrapper_forwarding_varargs_is_inlined_as_alias": (
        "tests.infra.unit.refactor.test_infra_refactor_legacy_and_annotations",
        "test_legacy_wrapper_forwarding_varargs_is_inlined_as_alias",
    ),
    "test_legacy_wrapper_function_is_inlined_as_alias": (
        "tests.infra.unit.refactor.test_infra_refactor_legacy_and_annotations",
        "test_legacy_wrapper_function_is_inlined_as_alias",
    ),
    "test_legacy_wrapper_non_passthrough_is_not_inlined": (
        "tests.infra.unit.refactor.test_infra_refactor_legacy_and_annotations",
        "test_legacy_wrapper_non_passthrough_is_not_inlined",
    ),
    "test_main_all_groups_defined": (
        "tests.infra.unit.test_infra_main",
        "test_main_all_groups_defined",
    ),
    "test_main_analyze_violations_is_read_only": (
        "tests.infra.unit.refactor.test_infra_refactor_analysis",
        "test_main_analyze_violations_is_read_only",
    ),
    "test_main_analyze_violations_writes_json_report": (
        "tests.infra.unit.refactor.test_infra_refactor_analysis",
        "test_main_analyze_violations_writes_json_report",
    ),
    "test_main_discovery_failure": (
        "tests.infra.unit.deps.test_path_sync_main_more",
        "test_main_discovery_failure",
    ),
    "test_main_group_modules_are_valid": (
        "tests.infra.unit.test_infra_main",
        "test_main_group_modules_are_valid",
    ),
    "test_main_help_flag_returns_zero": (
        "tests.infra.unit.test_infra_main",
        "test_main_help_flag_returns_zero",
    ),
    "test_main_no_changes_needed": (
        "tests.infra.unit.deps.test_path_sync_main_more",
        "test_main_no_changes_needed",
    ),
    "test_main_project_invalid_toml": (
        "tests.infra.unit.deps.test_path_sync_main_more",
        "test_main_project_invalid_toml",
    ),
    "test_main_project_no_name": (
        "tests.infra.unit.deps.test_path_sync_main_more",
        "test_main_project_no_name",
    ),
    "test_main_project_non_string_name": (
        "tests.infra.unit.deps.test_path_sync_main_more",
        "test_main_project_non_string_name",
    ),
    "test_main_project_obj_not_dict_first_loop": (
        "tests.infra.unit.deps.test_path_sync_main_project_obj",
        "test_main_project_obj_not_dict_first_loop",
    ),
    "test_main_project_obj_not_dict_second_loop": (
        "tests.infra.unit.deps.test_path_sync_main_project_obj",
        "test_main_project_obj_not_dict_second_loop",
    ),
    "test_main_returns_error_when_no_args": (
        "tests.infra.unit.test_infra_main",
        "test_main_returns_error_when_no_args",
    ),
    "test_main_success_modes": (
        "tests.infra.unit.deps.test_extra_paths_sync",
        "test_main_success_modes",
    ),
    "test_main_sync_failure": (
        "tests.infra.unit.deps.test_extra_paths_sync",
        "test_main_sync_failure",
    ),
    "test_main_unknown_group_returns_error": (
        "tests.infra.unit.test_infra_main",
        "test_main_unknown_group_returns_error",
    ),
    "test_main_with_changes_and_dry_run": (
        "tests.infra.unit.deps.test_path_sync_main_more",
        "test_main_with_changes_and_dry_run",
    ),
    "test_main_with_changes_no_dry_run": (
        "tests.infra.unit.deps.test_path_sync_main_more",
        "test_main_with_changes_no_dry_run",
    ),
    "test_migrate_makefile_not_found_non_dry_run": (
        "tests.infra.unit.test_infra_workspace_migrator_deps",
        "test_migrate_makefile_not_found_non_dry_run",
    ),
    "test_migrate_pyproject_flext_core_non_dry_run": (
        "tests.infra.unit.test_infra_workspace_migrator_deps",
        "test_migrate_pyproject_flext_core_non_dry_run",
    ),
    "test_migrator_apply_updates_project_files": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_apply_updates_project_files",
    ),
    "test_migrator_discovery_failure": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_discovery_failure",
    ),
    "test_migrator_dry_run_reports_changes_without_writes": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_dry_run_reports_changes_without_writes",
    ),
    "test_migrator_execute_returns_failure": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_execute_returns_failure",
    ),
    "test_migrator_flext_core_dry_run": (
        "tests.infra.unit.test_infra_workspace_migrator_dryrun",
        "test_migrator_flext_core_dry_run",
    ),
    "test_migrator_flext_core_project_skipped": (
        "tests.infra.unit.test_infra_workspace_migrator_dryrun",
        "test_migrator_flext_core_project_skipped",
    ),
    "test_migrator_gitignore_already_normalized_dry_run": (
        "tests.infra.unit.test_infra_workspace_migrator_dryrun",
        "test_migrator_gitignore_already_normalized_dry_run",
    ),
    "test_migrator_handles_missing_pyproject_gracefully": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_handles_missing_pyproject_gracefully",
    ),
    "test_migrator_has_flext_core_dependency_in_poetry": (
        "tests.infra.unit.test_infra_workspace_migrator_deps",
        "test_migrator_has_flext_core_dependency_in_poetry",
    ),
    "test_migrator_has_flext_core_dependency_poetry_deps_not_table": (
        "tests.infra.unit.test_infra_workspace_migrator_deps",
        "test_migrator_has_flext_core_dependency_poetry_deps_not_table",
    ),
    "test_migrator_has_flext_core_dependency_poetry_table_missing": (
        "tests.infra.unit.test_infra_workspace_migrator_deps",
        "test_migrator_has_flext_core_dependency_poetry_table_missing",
    ),
    "test_migrator_makefile_not_found_dry_run": (
        "tests.infra.unit.test_infra_workspace_migrator_dryrun",
        "test_migrator_makefile_not_found_dry_run",
    ),
    "test_migrator_makefile_read_failure": (
        "tests.infra.unit.test_infra_workspace_migrator_dryrun",
        "test_migrator_makefile_read_failure",
    ),
    "test_migrator_no_changes_needed": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_no_changes_needed",
    ),
    "test_migrator_preserves_custom_makefile_content": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_preserves_custom_makefile_content",
    ),
    "test_migrator_pyproject_not_found_dry_run": (
        "tests.infra.unit.test_infra_workspace_migrator_dryrun",
        "test_migrator_pyproject_not_found_dry_run",
    ),
    "test_migrator_workspace_root_not_exists": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_workspace_root_not_exists",
    ),
    "test_migrator_workspace_root_project_detection": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_workspace_root_project_detection",
    ),
    "test_mro_checker_keeps_external_attribute_base": (
        "tests.infra.unit.refactor.test_infra_refactor_class_and_propagation",
        "test_mro_checker_keeps_external_attribute_base",
    ),
    "test_mro_redundancy_checker_removes_nested_attribute_inheritance": (
        "tests.infra.unit.refactor.test_infra_refactor_class_and_propagation",
        "test_mro_redundancy_checker_removes_nested_attribute_inheritance",
    ),
    "test_multiple_instances_independent": (
        "tests.infra.unit.test_infra_templates",
        "test_multiple_instances_independent",
    ),
    "test_parse_semver_invalid": (
        "tests.infra.unit.test_infra_versioning",
        "test_parse_semver_invalid",
    ),
    "test_parse_semver_result_type": (
        "tests.infra.unit.test_infra_versioning",
        "test_parse_semver_result_type",
    ),
    "test_parse_semver_valid": (
        "tests.infra.unit.test_infra_versioning",
        "test_parse_semver_valid",
    ),
    "test_pattern_rule_converts_dict_annotations_to_mapping": (
        "tests.infra.unit.refactor.test_infra_refactor_pattern_corrections",
        "test_pattern_rule_converts_dict_annotations_to_mapping",
    ),
    "test_pattern_rule_keeps_dict_param_when_copy_used": (
        "tests.infra.unit.refactor.test_infra_refactor_pattern_corrections",
        "test_pattern_rule_keeps_dict_param_when_copy_used",
    ),
    "test_pattern_rule_keeps_dict_param_when_subscript_mutated": (
        "tests.infra.unit.refactor.test_infra_refactor_pattern_corrections",
        "test_pattern_rule_keeps_dict_param_when_subscript_mutated",
    ),
    "test_pattern_rule_keeps_type_cast_when_not_nested_object_cast": (
        "tests.infra.unit.refactor.test_infra_refactor_pattern_corrections",
        "test_pattern_rule_keeps_type_cast_when_not_nested_object_cast",
    ),
    "test_pattern_rule_optionally_converts_return_annotations_to_mapping": (
        "tests.infra.unit.refactor.test_infra_refactor_pattern_corrections",
        "test_pattern_rule_optionally_converts_return_annotations_to_mapping",
    ),
    "test_pattern_rule_removes_configured_redundant_casts": (
        "tests.infra.unit.refactor.test_infra_refactor_pattern_corrections",
        "test_pattern_rule_removes_configured_redundant_casts",
    ),
    "test_pattern_rule_removes_nested_type_object_cast_chain": (
        "tests.infra.unit.refactor.test_infra_refactor_pattern_corrections",
        "test_pattern_rule_removes_nested_type_object_cast_chain",
    ),
    "test_pattern_rule_skips_overload_signatures": (
        "tests.infra.unit.refactor.test_infra_refactor_pattern_corrections",
        "test_pattern_rule_skips_overload_signatures",
    ),
    "test_project_dev_groups": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "test_project_dev_groups",
    ),
    "test_project_dev_groups_missing_sections": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "test_project_dev_groups_missing_sections",
    ),
    "test_project_without_src_returns_empty": (
        "tests.infra.unit.codegen.autofix_workspace",
        "test_project_without_src_returns_empty",
    ),
    "test_refactor_files_skips_non_python_inputs": (
        "tests.infra.unit.refactor.test_infra_refactor_engine",
        "test_refactor_files_skips_non_python_inputs",
    ),
    "test_refactor_project_integrates_safety_manager": (
        "tests.infra.unit.refactor.test_infra_refactor_safety",
        "test_refactor_project_integrates_safety_manager",
    ),
    "test_refactor_project_scans_tests_and_scripts_dirs": (
        "tests.infra.unit.refactor.test_infra_refactor_engine",
        "test_refactor_project_scans_tests_and_scripts_dirs",
    ),
    "test_release_tag_from_branch_invalid": (
        "tests.infra.unit.test_infra_versioning",
        "test_release_tag_from_branch_invalid",
    ),
    "test_release_tag_from_branch_result_type": (
        "tests.infra.unit.test_infra_versioning",
        "test_release_tag_from_branch_result_type",
    ),
    "test_release_tag_from_branch_valid": (
        "tests.infra.unit.test_infra_versioning",
        "test_release_tag_from_branch_valid",
    ),
    "test_render_all_generates_large_makefile": (
        "tests.infra.unit.basemk.engine",
        "test_render_all_generates_large_makefile",
    ),
    "test_render_all_has_no_scripts_path_references": (
        "tests.infra.unit.basemk.engine",
        "test_render_all_has_no_scripts_path_references",
    ),
    "test_render_failure": (
        "tests.infra.unit.test_infra_templates",
        "test_render_failure",
    ),
    "test_render_success": (
        "tests.infra.unit.test_infra_templates",
        "test_render_success",
    ),
    "test_replace_project_version": (
        "tests.infra.unit.test_infra_versioning",
        "test_replace_project_version",
    ),
    "test_resolve_gates_maps_type_alias": (
        "tests.infra.unit.check.cli",
        "test_resolve_gates_maps_type_alias",
    ),
    "test_rewrite_dep_paths_dry_run": (
        "tests.infra.unit.deps.test_path_sync_rewrite_deps",
        "test_rewrite_dep_paths_dry_run",
    ),
    "test_rewrite_dep_paths_read_failure": (
        "tests.infra.unit.deps.test_path_sync_rewrite_deps",
        "test_rewrite_dep_paths_read_failure",
    ),
    "test_rewrite_dep_paths_with_internal_names": (
        "tests.infra.unit.deps.test_path_sync_rewrite_deps",
        "test_rewrite_dep_paths_with_internal_names",
    ),
    "test_rewrite_dep_paths_with_no_deps": (
        "tests.infra.unit.deps.test_path_sync_rewrite_deps",
        "test_rewrite_dep_paths_with_no_deps",
    ),
    "test_rewrite_pep621_invalid_path_dep_regex": (
        "tests.infra.unit.deps.test_path_sync_rewrite_pep621",
        "test_rewrite_pep621_invalid_path_dep_regex",
    ),
    "test_rewrite_pep621_no_project_table": (
        "tests.infra.unit.deps.test_path_sync_rewrite_pep621",
        "test_rewrite_pep621_no_project_table",
    ),
    "test_rewrite_pep621_non_string_item": (
        "tests.infra.unit.deps.test_path_sync_rewrite_pep621",
        "test_rewrite_pep621_non_string_item",
    ),
    "test_rewrite_poetry_no_poetry_table": (
        "tests.infra.unit.deps.test_path_sync_rewrite_poetry",
        "test_rewrite_poetry_no_poetry_table",
    ),
    "test_rewrite_poetry_no_tool_table": (
        "tests.infra.unit.deps.test_path_sync_rewrite_poetry",
        "test_rewrite_poetry_no_tool_table",
    ),
    "test_rewrite_poetry_with_non_dict_value": (
        "tests.infra.unit.deps.test_path_sync_rewrite_poetry",
        "test_rewrite_poetry_with_non_dict_value",
    ),
    "test_rule_dispatch_fails_on_invalid_pattern_rule_config": (
        "tests.infra.unit.refactor.test_infra_refactor_engine",
        "test_rule_dispatch_fails_on_invalid_pattern_rule_config",
    ),
    "test_rule_dispatch_fails_on_unknown_rule_mapping": (
        "tests.infra.unit.refactor.test_infra_refactor_engine",
        "test_rule_dispatch_fails_on_unknown_rule_mapping",
    ),
    "test_rule_dispatch_keeps_legacy_id_fallback_mapping": (
        "tests.infra.unit.refactor.test_infra_refactor_engine",
        "test_rule_dispatch_keeps_legacy_id_fallback_mapping",
    ),
    "test_rule_dispatch_prefers_fix_action_metadata": (
        "tests.infra.unit.refactor.test_infra_refactor_engine",
        "test_rule_dispatch_prefers_fix_action_metadata",
    ),
    "test_run_cases": ("tests.infra.unit.test_infra_subprocess_core", "test_run_cases"),
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
    "test_run_deptry_wrapper": (
        "tests.infra.unit.deps.test_detection_wrappers",
        "test_run_deptry_wrapper",
    ),
    "test_run_mypy_stub_hints_wrapper": (
        "tests.infra.unit.deps.test_detection_wrappers",
        "test_run_mypy_stub_hints_wrapper",
    ),
    "test_run_pip_check_wrapper": (
        "tests.infra.unit.deps.test_detection_wrappers",
        "test_run_pip_check_wrapper",
    ),
    "test_run_raw_cases": (
        "tests.infra.unit.test_infra_subprocess_core",
        "test_run_raw_cases",
    ),
    "test_signature_propagation_removes_and_adds_keywords": (
        "tests.infra.unit.refactor.test_infra_refactor_class_and_propagation",
        "test_signature_propagation_removes_and_adds_keywords",
    ),
    "test_signature_propagation_renames_call_keyword": (
        "tests.infra.unit.refactor.test_infra_refactor_class_and_propagation",
        "test_signature_propagation_renames_call_keyword",
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
    "test_string_zero_return_value": (
        "tests.infra.unit.deps.test_main_dispatch",
        "test_string_zero_return_value",
    ),
    "test_symbol_propagation_keeps_alias_reference_when_asname_used": (
        "tests.infra.unit.refactor.test_infra_refactor_class_and_propagation",
        "test_symbol_propagation_keeps_alias_reference_when_asname_used",
    ),
    "test_symbol_propagation_renames_import_and_local_references": (
        "tests.infra.unit.refactor.test_infra_refactor_class_and_propagation",
        "test_symbol_propagation_renames_import_and_local_references",
    ),
    "test_symbol_propagation_updates_mro_base_references": (
        "tests.infra.unit.refactor.test_infra_refactor_class_and_propagation",
        "test_symbol_propagation_updates_mro_base_references",
    ),
    "test_sync_basemk_scenarios": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_basemk_scenarios",
    ),
    "test_sync_error_scenarios": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_error_scenarios",
    ),
    "test_sync_extra_paths_missing_root_pyproject": (
        "tests.infra.unit.deps.test_extra_paths_sync",
        "test_sync_extra_paths_missing_root_pyproject",
    ),
    "test_sync_extra_paths_success_modes": (
        "tests.infra.unit.deps.test_extra_paths_sync",
        "test_sync_extra_paths_success_modes",
    ),
    "test_sync_extra_paths_sync_failure": (
        "tests.infra.unit.deps.test_extra_paths_sync",
        "test_sync_extra_paths_sync_failure",
    ),
    "test_sync_one_edge_cases": (
        "tests.infra.unit.deps.test_extra_paths_sync",
        "test_sync_one_edge_cases",
    ),
    "test_sync_root_validation": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_root_validation",
    ),
    "test_sync_success_scenarios": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_success_scenarios",
    ),
    "test_syntax_error_files_skipped": (
        "tests.infra.unit.codegen.autofix",
        "test_syntax_error_files_skipped",
    ),
    "test_target_path": (
        "tests.infra.unit.deps.test_path_sync_helpers",
        "test_target_path",
    ),
    "test_template_constants": (
        "tests.infra.unit.test_infra_templates",
        "test_template_constants",
    ),
    "test_unwrap_item": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "test_unwrap_item",
    ),
    "test_unwrap_item_toml_item": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "test_unwrap_item_toml_item",
    ),
    "test_violation_analysis_counts_massive_patterns": (
        "tests.infra.unit.refactor.test_infra_refactor_analysis",
        "test_violation_analysis_counts_massive_patterns",
    ),
    "test_violation_analyzer_skips_non_utf8_files": (
        "tests.infra.unit.refactor.test_infra_refactor_analysis",
        "test_violation_analyzer_skips_non_utf8_files",
    ),
    "test_workspace_check_main_returns_error_without_projects": (
        "tests.infra.unit.check.workspace_check",
        "test_workspace_check_main_returns_error_without_projects",
    ),
    "test_workspace_cli_migrate_command": (
        "tests.infra.unit.test_infra_workspace_cli",
        "test_workspace_cli_migrate_command",
    ),
    "test_workspace_cli_migrate_output_contains_summary": (
        "tests.infra.unit.test_infra_workspace_cli",
        "test_workspace_cli_migrate_output_contains_summary",
    ),
    "test_workspace_migrator_error_handling_on_invalid_workspace": (
        "tests.infra.unit.test_infra_workspace_migrator_deps",
        "test_workspace_migrator_error_handling_on_invalid_workspace",
    ),
    "test_workspace_migrator_makefile_not_found_dry_run": (
        "tests.infra.unit.test_infra_workspace_migrator_deps",
        "test_workspace_migrator_makefile_not_found_dry_run",
    ),
    "test_workspace_migrator_makefile_read_error": (
        "tests.infra.unit.test_infra_workspace_migrator_deps",
        "test_workspace_migrator_makefile_read_error",
    ),
    "test_workspace_migrator_pyproject_write_error": (
        "tests.infra.unit.test_infra_workspace_migrator_deps",
        "test_workspace_migrator_pyproject_write_error",
    ),
    "test_workspace_root_doc_construction": (
        "tests.infra.unit.deps.test_modernizer_workspace",
        "test_workspace_root_doc_construction",
    ),
    "test_workspace_root_fallback": (
        "tests.infra.unit.deps.test_path_sync_main_more",
        "test_workspace_root_fallback",
    ),
    "u": ("tests.infra.utilities", "u"),
    "unit": ("tests.infra.unit", ""),
    "unwrap_item": ("tests.infra.unit.deps.test_modernizer_helpers", "unwrap_item"),
    "v": ("tests.infra.unit.core.basemk_validator", "v"),
    "validator": ("tests.infra.unit.docs.validator_internals", "validator"),
    "workspace_root": (
        "tests.infra.unit.release.orchestrator_publish",
        "workspace_root",
    ),
}

__all__ = [
    "BrokenScenario",
    "CheckProjectStub",
    "DependencyScenario",
    "DependencyScenarios",
    "EmptyScenario",
    "EngineSafetyStub",
    "FlextInfraTestConstants",
    "FlextInfraTestHelpers",
    "FlextInfraTestModels",
    "FlextInfraTestProtocols",
    "FlextInfraTestTypes",
    "FlextInfraTestUtilities",
    "FullScenario",
    "GitScenario",
    "GitScenarios",
    "MinimalScenario",
    "MockScanner",
    "RealGitService",
    "RealSubprocessRunner",
    "RunCallable",
    "RunStub",
    "SetupFn",
    "SubprocessScenario",
    "SubprocessScenarios",
    "TestAdrHelpers",
    "TestAllDirectoriesScanned",
    "TestAuditorBrokenLinks",
    "TestAuditorCore",
    "TestAuditorForbiddenTerms",
    "TestAuditorMainCli",
    "TestAuditorNormalize",
    "TestAuditorScope",
    "TestAuditorScopeFailure",
    "TestAuditorToMarkdown",
    "TestBaseMkValidatorCore",
    "TestBaseMkValidatorEdgeCases",
    "TestBaseMkValidatorSha256",
    "TestBuildProjectReport",
    "TestBuildScopes",
    "TestBuildSiblingExportIndex",
    "TestBuildTargets",
    "TestBuilderCore",
    "TestBuilderScope",
    "TestBumpNextDev",
    "TestCensusReportModel",
    "TestCensusViolationModel",
    "TestCheckIssueFormatted",
    "TestCheckMainEntryPoint",
    "TestCheckOnlyMode",
    "TestCheckProjectRunners",
    "TestCheckpoint",
    "TestChecks",
    "TestClassifyIssues",
    "TestClose",
    "TestCollectChanges",
    "TestCollectInternalDeps",
    "TestCollectInternalDepsEdgeCases",
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
    "TestConsolidateGroupsPhase",
    "TestConstants",
    "TestConstantsQualityGateCLIDispatch",
    "TestConstantsQualityGateVerdict",
    "TestCoreModuleInit",
    "TestCreate",
    "TestCreateBranches",
    "TestCreateTag",
    "TestDetectMode",
    "TestDetectionUncoveredLines",
    "TestDetectorBasicDetection",
    "TestDetectorGitRunScenarios",
    "TestDetectorRepoNameExtraction",
    "TestDetectorReportFlags",
    "TestDetectorRunFailures",
    "TestDiscoverProjects",
    "TestDiscoveryDiscoverProjects",
    "TestDiscoveryFindAllPyprojectFiles",
    "TestDiscoveryIterPythonFiles",
    "TestDiscoveryProjectRoots",
    "TestDispatchPhase",
    "TestEdgeCases",
    "TestEnforcerExecute",
    "TestEnsureCheckout",
    "TestEnsureCheckoutEdgeCases",
    "TestEnsurePyreflyConfigPhase",
    "TestEnsurePyrightConfigPhase",
    "TestEnsurePytestConfigPhase",
    "TestEnsurePythonVersionFile",
    "TestEnsureSymlink",
    "TestEnsureSymlinkEdgeCases",
    "TestErrorReporting",
    "TestExcludedDirectories",
    "TestExcludedProjects",
    "TestExtractExports",
    "TestExtractInlineConstants",
    "TestExtractVersionExports",
    "TestFixPyrelfyCLI",
    "TestFixabilityClassification",
    "TestFixerCore",
    "TestFixerMaybeFixLink",
    "TestFixerProcessFile",
    "TestFixerScope",
    "TestFixerToc",
    "TestFlextInfraBaseMk",
    "TestFlextInfraCheck",
    "TestFlextInfraCodegenLazyInit",
    "TestFlextInfraCommandRunnerExtra",
    "TestFlextInfraConfigFixer",
    "TestFlextInfraConstantsAlias",
    "TestFlextInfraConstantsCheckNamespace",
    "TestFlextInfraConstantsConsistency",
    "TestFlextInfraConstantsEncodingNamespace",
    "TestFlextInfraConstantsExcludedNamespace",
    "TestFlextInfraConstantsFilesNamespace",
    "TestFlextInfraConstantsGatesNamespace",
    "TestFlextInfraConstantsGithubNamespace",
    "TestFlextInfraConstantsImmutability",
    "TestFlextInfraConstantsPathsNamespace",
    "TestFlextInfraConstantsStatusNamespace",
    "TestFlextInfraDependencyDetectionModels",
    "TestFlextInfraDependencyDetectionService",
    "TestFlextInfraDependencyDetectorModels",
    "TestFlextInfraDependencyPathSync",
    "TestFlextInfraDeps",
    "TestFlextInfraDiscoveryService",
    "TestFlextInfraDiscoveryServiceUncoveredLines",
    "TestFlextInfraDocScope",
    "TestFlextInfraDocs",
    "TestFlextInfraExtraPathsManager",
    "TestFlextInfraGitService",
    "TestFlextInfraInitLazyLoading",
    "TestFlextInfraInternalDependencySyncService",
    "TestFlextInfraJsonService",
    "TestFlextInfraMaintenance",
    "TestFlextInfraPathResolver",
    "TestFlextInfraPatternsEdgeCases",
    "TestFlextInfraPatternsMarkdown",
    "TestFlextInfraPatternsPatternTypes",
    "TestFlextInfraPatternsTooling",
    "TestFlextInfraPrManager",
    "TestFlextInfraPrWorkspaceManager",
    "TestFlextInfraProtocolsImport",
    "TestFlextInfraPyprojectModernizer",
    "TestFlextInfraReportingServiceCore",
    "TestFlextInfraReportingServiceExtra",
    "TestFlextInfraRuntimeDevDependencyDetectorInit",
    "TestFlextInfraRuntimeDevDependencyDetectorRunDetect",
    "TestFlextInfraRuntimeDevDependencyDetectorRunReport",
    "TestFlextInfraRuntimeDevDependencyDetectorRunTypings",
    "TestFlextInfraSubmoduleInitLazyLoading",
    "TestFlextInfraTomlDocument",
    "TestFlextInfraTomlHelpers",
    "TestFlextInfraTomlRead",
    "TestFlextInfraTypesImport",
    "TestFlextInfraUtilitiesImport",
    "TestFlextInfraUtilitiesSelection",
    "TestFlextInfraVersionClass",
    "TestFlextInfraVersionModuleLevel",
    "TestFlextInfraVersionPackageInfo",
    "TestFlextInfraWorkflowLinter",
    "TestFlextInfraWorkflowSyncer",
    "TestFlextInfraWorkspace",
    "TestFlextInfraWorkspaceChecker",
    "TestFormattingRunRuffFix",
    "TestGenerateFile",
    "TestGenerateNotes",
    "TestGenerateTypeChecking",
    "TestGeneratedClassNamingConvention",
    "TestGeneratedFilesAreValidPython",
    "TestGeneratorCore",
    "TestGeneratorHelpers",
    "TestGeneratorScope",
    "TestGetDepPaths",
    "TestGitPush",
    "TestGitTagOperations",
    "TestGithubInit",
    "TestGoFmtEmptyLinesInOutput",
    "TestHandleLazyInit",
    "TestInferOwnerFromOrigin",
    "TestInferPackage",
    "TestInfraContainerFunctions",
    "TestInfraMroPattern",
    "TestInfraOutputEdgeCases",
    "TestInfraOutputHeader",
    "TestInfraOutputMessages",
    "TestInfraOutputNoColor",
    "TestInfraOutputProgress",
    "TestInfraOutputStatus",
    "TestInfraOutputSummary",
    "TestInfraServiceRetrieval",
    "TestInjectCommentsPhase",
    "TestInventoryServiceCore",
    "TestInventoryServiceReports",
    "TestInventoryServiceScripts",
    "TestIsInternalPathDep",
    "TestIsRelativeTo",
    "TestIsWorkspaceMode",
    "TestIterMarkdownFiles",
    "TestIterWorkspacePythonModules",
    "TestJsonWriteFailure",
    "TestLintAndFormatPublicMethods",
    "TestLoadAuditBudgets",
    "TestLoadDependencyLimits",
    "TestMain",
    "TestMainBaseMkValidate",
    "TestMainCli",
    "TestMainCliRouting",
    "TestMainCommandDispatch",
    "TestMainDelegation",
    "TestMainEdgeCases",
    "TestMainEntryPoint",
    "TestMainExceptionHandling",
    "TestMainFunction",
    "TestMainHelpAndErrors",
    "TestMainInventory",
    "TestMainModuleImport",
    "TestMainReturnValues",
    "TestMainRouting",
    "TestMainScan",
    "TestMainSubcommandDispatch",
    "TestMainSysArgvModification",
    "TestMainWithFlags",
    "TestMaintenanceMainEnforcer",
    "TestMaintenanceMainSuccess",
    "TestMarkdownReportEmptyGates",
    "TestMarkdownReportSkipsEmptyGates",
    "TestMarkdownReportWithErrors",
    "TestMaybeWriteTodo",
    "TestMerge",
    "TestMergeChildExports",
    "TestMigratorDryRun",
    "TestMigratorEdgeCases",
    "TestMigratorFlextCore",
    "TestMigratorInternalMakefile",
    "TestMigratorInternalPyproject",
    "TestMigratorPoetryDeps",
    "TestMigratorReadFailures",
    "TestMigratorWriteFailures",
    "TestModernizerEdgeCases",
    "TestModernizerRunAndMain",
    "TestModernizerUncoveredLines",
    "TestModuleAndTypingsFlow",
    "TestModuleLevelWrappers",
    "TestMroFacadeMethods",
    "TestMypyEmptyLinesInOutput",
    "TestNormalizeStringList",
    "TestOrchestrate",
    "TestOrchestratorBasic",
    "TestOrchestratorFailures",
    "TestOwnerFromRemoteUrl",
    "TestParseArgs",
    "TestParseGitmodules",
    "TestParseRepoMap",
    "TestParseViolationInvalid",
    "TestParseViolationValid",
    "TestParser",
    "TestParsingModuleAst",
    "TestParsingModuleCst",
    "TestPathDepPathsPep621",
    "TestPathDepPathsPoetry",
    "TestPathSyncEdgeCases",
    "TestPhaseBuild",
    "TestPhasePublish",
    "TestPhaseValidate",
    "TestPhaseVersion",
    "TestPreviousTag",
    "TestProcessDirectory",
    "TestProcessFileReadError",
    "TestProjectResultProperties",
    "TestPushRelease",
    "TestPytestDiagExtractorCore",
    "TestPytestDiagLogParsing",
    "TestPytestDiagParseXml",
    "TestReadDoc",
    "TestReadExistingDocstring",
    "TestReadRequiredMinor",
    "TestReleaseInit",
    "TestReleaseMainFlow",
    "TestReleaseMainParsing",
    "TestReleaseMainTagResolution",
    "TestReleaseMainVersionResolution",
    "TestReleaseOrchestratorExecute",
    "TestRemovedCompatibilityMethods",
    "TestRenderTemplate",
    "TestResolveAliases",
    "TestResolveRef",
    "TestResolveVersionInteractive",
    "TestRewriteDepPaths",
    "TestRewritePep621",
    "TestRewritePoetry",
    "TestRuffFormatDuplicateFiles",
    "TestRunAudit",
    "TestRunBandit",
    "TestRunBuild",
    "TestRunCLIExtended",
    "TestRunCommand",
    "TestRunDeptry",
    "TestRunDetect",
    "TestRunFix",
    "TestRunGenerate",
    "TestRunGo",
    "TestRunLint",
    "TestRunMake",
    "TestRunMarkdown",
    "TestRunMigrate",
    "TestRunMypy",
    "TestRunMypyStubHints",
    "TestRunOrchestrate",
    "TestRunPipCheck",
    "TestRunPr",
    "TestRunPrWorkspace",
    "TestRunProjectsBehavior",
    "TestRunProjectsReports",
    "TestRunProjectsValidation",
    "TestRunPyrefly",
    "TestRunPyright",
    "TestRunRuffFix",
    "TestRunRuffFormat",
    "TestRunRuffLint",
    "TestRunSingleProject",
    "TestRunSync",
    "TestRunValidate",
    "TestRunWorkflows",
    "TestSafeLoadYaml",
    "TestSafetyCheckpoint",
    "TestSafetyRollback",
    "TestSafetyWorkspaceValidation",
    "TestScaffoldProjectCreatesSrcModules",
    "TestScaffoldProjectCreatesTestsModules",
    "TestScaffoldProjectIdempotency",
    "TestScaffoldProjectNoop",
    "TestScanAstPublicDefs",
    "TestScanFileBatch",
    "TestScanModels",
    "TestScannerCore",
    "TestScannerHelpers",
    "TestScannerMultiFile",
    "TestSelectedProjectNames",
    "TestSelectorFunction",
    "TestShouldBubbleUp",
    "TestShouldUseColor",
    "TestShouldUseUnicode",
    "TestSkillValidatorAstGrepCount",
    "TestSkillValidatorCore",
    "TestSkillValidatorRenderTemplate",
    "TestStaticMethods",
    "TestStatus",
    "TestStubChainAnalyze",
    "TestStubChainCore",
    "TestStubChainDiscoverProjects",
    "TestStubChainIsInternal",
    "TestStubChainStubExists",
    "TestStubChainValidate",
    "TestSubcommandMapping",
    "TestSync",
    "TestSyncMethodEdgeCases",
    "TestSyncMethodEdgeCasesMore",
    "TestSyncOne",
    "TestSyncOperation",
    "TestSyncProject",
    "TestSyncWorkspace",
    "TestSynthesizedRepoMap",
    "TestToInfraValue",
    "TestTriggerRelease",
    "TestUpdateChangelog",
    "TestValidateCore",
    "TestValidateGitRefEdgeCases",
    "TestValidateReport",
    "TestValidateScope",
    "TestVersionFiles",
    "TestView",
    "TestViolationPattern",
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
    "TestWorkspaceRoot",
    "TestWorkspaceRootFromEnv",
    "TestWorkspaceRootFromParents",
    "TestWriteJson",
    "TestWriteMarkdown",
    "TestWriteReport",
    "WorkspaceFactory",
    "WorkspaceScenario",
    "WorkspaceScenarios",
    "array",
    "as_string_list",
    "auditor",
    "basemk",
    "builder",
    "c",
    "canonical_dev_dependencies",
    "census",
    "check",
    "codegen",
    "container",
    "core",
    "dedupe_specs",
    "dep_name",
    "deps",
    "detector",
    "discovery",
    "doc",
    "docs",
    "engine",
    "ensure_table",
    "extract_dep_name",
    "fixer",
    "gen",
    "git_repo",
    "github",
    "h",
    "io",
    "is_external",
    "m",
    "normalize_link",
    "orchestrator",
    "p",
    "project_dev_groups",
    "pyright_content",
    "r",
    "real_docs_project",
    "real_git_repo",
    "real_makefile_project",
    "real_python_package",
    "real_toml_project",
    "real_workspace",
    "refactor",
    "release",
    "rewrite_dep_paths",
    "run_lint",
    "run_pr",
    "run_pr_workspace",
    "run_workflows",
    "runner",
    "s",
    "service",
    "should_skip_target",
    "svc",
    "t",
    "test_array",
    "test_as_string_list",
    "test_as_string_list_toml_item",
    "test_atomic_write_fail",
    "test_atomic_write_ok",
    "test_basemk_build_config_with_none",
    "test_basemk_build_config_with_project_name",
    "test_basemk_cli_generate_to_file",
    "test_basemk_cli_generate_to_stdout",
    "test_basemk_engine_execute_calls_render_all",
    "test_basemk_engine_render_all_handles_template_error",
    "test_basemk_engine_render_all_returns_string",
    "test_basemk_engine_render_all_with_valid_config",
    "test_basemk_main_calls_sys_exit",
    "test_basemk_main_ensures_structlog_configured",
    "test_basemk_main_output_to_stdout",
    "test_basemk_main_with_generate_command",
    "test_basemk_main_with_generation_failure",
    "test_basemk_main_with_invalid_command",
    "test_basemk_main_with_no_command",
    "test_basemk_main_with_none_argv",
    "test_basemk_main_with_output_file",
    "test_basemk_main_with_project_name",
    "test_basemk_main_with_write_failure",
    "test_build_impact_map_extracts_rename_entries",
    "test_build_impact_map_extracts_signature_entries",
    "test_bump_version_invalid",
    "test_bump_version_result_type",
    "test_bump_version_valid",
    "test_canonical_dev_dependencies",
    "test_capture_cases",
    "test_check_main_executes_real_cli",
    "test_class_reconstructor_reorders_each_contiguous_method_block",
    "test_class_reconstructor_reorders_methods_by_config",
    "test_class_reconstructor_skips_interleaved_non_method_members",
    "test_cli_result_by_project_root",
    "test_codegen_dir_returns_all_exports",
    "test_codegen_getattr_raises_attribute_error",
    "test_codegen_init_getattr_raises_attribute_error",
    "test_codegen_lazy_imports_work",
    "test_codegen_pipeline_end_to_end",
    "test_consolidate_groups_phase_apply_removes_old_groups",
    "test_consolidate_groups_phase_apply_with_empty_poetry_group",
    "test_current_workspace_version",
    "test_dedupe_specs",
    "test_dep_name",
    "test_detect_mode_with_nonexistent_path",
    "test_detect_mode_with_path_object",
    "test_discover_projects_wrapper",
    "test_engine_always_enables_class_nesting_file_rule",
    "test_engine_constants_shared",
    "test_ensure_future_annotations_after_docstring",
    "test_ensure_future_annotations_moves_existing_import_to_top",
    "test_ensure_pyrefly_config_phase_apply_errors",
    "test_ensure_pyrefly_config_phase_apply_ignore_errors",
    "test_ensure_pyrefly_config_phase_apply_python_version",
    "test_ensure_pyrefly_config_phase_apply_search_path",
    "test_ensure_pytest_config_phase_apply_markers",
    "test_ensure_pytest_config_phase_apply_minversion",
    "test_ensure_pytest_config_phase_apply_python_classes",
    "test_ensure_table",
    "test_extract_dep_name",
    "test_extract_requirement_name",
    "test_files_modified_tracks_affected_files",
    "test_fix_pyrefly_config_main_executes_real_cli_help",
    "test_flexcore_excluded_from_run",
    "test_flext_infra_pyproject_modernizer_find_pyproject_files",
    "test_flext_infra_pyproject_modernizer_process_file_invalid_toml",
    "test_generator_execute_returns_generated_content",
    "test_generator_fails_for_invalid_make_syntax",
    "test_generator_generate_propagates_render_failure",
    "test_generator_generate_with_basemk_config_object",
    "test_generator_generate_with_dict_config",
    "test_generator_generate_with_invalid_dict_config",
    "test_generator_generate_with_none_config_uses_default",
    "test_generator_initializes_with_custom_engine",
    "test_generator_initializes_with_default_engine",
    "test_generator_normalize_config_with_basemk_config",
    "test_generator_normalize_config_with_dict",
    "test_generator_normalize_config_with_invalid_dict",
    "test_generator_normalize_config_with_none",
    "test_generator_renders_with_config_override",
    "test_generator_validate_generated_output_handles_oserror",
    "test_generator_write_creates_parent_directories",
    "test_generator_write_fails_without_output_or_stream",
    "test_generator_write_handles_file_permission_error",
    "test_generator_write_saves_output_file",
    "test_generator_write_to_file",
    "test_generator_write_to_stream",
    "test_generator_write_to_stream_handles_oserror",
    "test_get_current_typings_from_pyproject_wrapper",
    "test_get_required_typings_wrapper",
    "test_gitignore_entry_scenarios",
    "test_gitignore_sync_failure",
    "test_gitignore_write_failure",
    "test_helpers_alias_exposed",
    "test_helpers_alias_is_reachable_helpers",
    "test_helpers_alias_is_reachable_main",
    "test_helpers_alias_is_reachable_project_obj",
    "test_import_modernizer_adds_c_when_existing_c_is_aliased",
    "test_import_modernizer_does_not_rewrite_function_parameter_shadow",
    "test_import_modernizer_does_not_rewrite_rebound_local_name_usage",
    "test_import_modernizer_partial_import_keeps_unmapped_symbols",
    "test_import_modernizer_partial_import_with_asname_keeps_unmapped_alias",
    "test_import_modernizer_skips_rewrite_when_runtime_alias_shadowed_in_function",
    "test_import_modernizer_skips_when_runtime_alias_name_is_blocked",
    "test_import_modernizer_updates_aliased_symbol_usage",
    "test_in_context_typevar_not_flagged",
    "test_inject_comments_phase_apply_banner",
    "test_inject_comments_phase_apply_broken_group_section",
    "test_inject_comments_phase_apply_markers",
    "test_inject_comments_phase_apply_with_optional_dependencies_dev",
    "test_lazy_import_rule_hoists_import_to_module_level",
    "test_lazy_import_rule_uses_fix_action_for_hoist",
    "test_legacy_import_bypass_collapses_to_primary_import",
    "test_legacy_rule_uses_fix_action_remove_for_aliases",
    "test_legacy_wrapper_forwarding_keywords_is_inlined_as_alias",
    "test_legacy_wrapper_forwarding_varargs_is_inlined_as_alias",
    "test_legacy_wrapper_function_is_inlined_as_alias",
    "test_legacy_wrapper_non_passthrough_is_not_inlined",
    "test_main_all_groups_defined",
    "test_main_analyze_violations_is_read_only",
    "test_main_analyze_violations_writes_json_report",
    "test_main_discovery_failure",
    "test_main_group_modules_are_valid",
    "test_main_help_flag_returns_zero",
    "test_main_no_changes_needed",
    "test_main_project_invalid_toml",
    "test_main_project_no_name",
    "test_main_project_non_string_name",
    "test_main_project_obj_not_dict_first_loop",
    "test_main_project_obj_not_dict_second_loop",
    "test_main_returns_error_when_no_args",
    "test_main_success_modes",
    "test_main_sync_failure",
    "test_main_unknown_group_returns_error",
    "test_main_with_changes_and_dry_run",
    "test_main_with_changes_no_dry_run",
    "test_migrate_makefile_not_found_non_dry_run",
    "test_migrate_pyproject_flext_core_non_dry_run",
    "test_migrator_apply_updates_project_files",
    "test_migrator_discovery_failure",
    "test_migrator_dry_run_reports_changes_without_writes",
    "test_migrator_execute_returns_failure",
    "test_migrator_flext_core_dry_run",
    "test_migrator_flext_core_project_skipped",
    "test_migrator_gitignore_already_normalized_dry_run",
    "test_migrator_handles_missing_pyproject_gracefully",
    "test_migrator_has_flext_core_dependency_in_poetry",
    "test_migrator_has_flext_core_dependency_poetry_deps_not_table",
    "test_migrator_has_flext_core_dependency_poetry_table_missing",
    "test_migrator_makefile_not_found_dry_run",
    "test_migrator_makefile_read_failure",
    "test_migrator_no_changes_needed",
    "test_migrator_preserves_custom_makefile_content",
    "test_migrator_pyproject_not_found_dry_run",
    "test_migrator_workspace_root_not_exists",
    "test_migrator_workspace_root_project_detection",
    "test_mro_checker_keeps_external_attribute_base",
    "test_mro_redundancy_checker_removes_nested_attribute_inheritance",
    "test_multiple_instances_independent",
    "test_parse_semver_invalid",
    "test_parse_semver_result_type",
    "test_parse_semver_valid",
    "test_pattern_rule_converts_dict_annotations_to_mapping",
    "test_pattern_rule_keeps_dict_param_when_copy_used",
    "test_pattern_rule_keeps_dict_param_when_subscript_mutated",
    "test_pattern_rule_keeps_type_cast_when_not_nested_object_cast",
    "test_pattern_rule_optionally_converts_return_annotations_to_mapping",
    "test_pattern_rule_removes_configured_redundant_casts",
    "test_pattern_rule_removes_nested_type_object_cast_chain",
    "test_pattern_rule_skips_overload_signatures",
    "test_project_dev_groups",
    "test_project_dev_groups_missing_sections",
    "test_project_without_src_returns_empty",
    "test_refactor_files_skips_non_python_inputs",
    "test_refactor_project_integrates_safety_manager",
    "test_refactor_project_scans_tests_and_scripts_dirs",
    "test_release_tag_from_branch_invalid",
    "test_release_tag_from_branch_result_type",
    "test_release_tag_from_branch_valid",
    "test_render_all_generates_large_makefile",
    "test_render_all_has_no_scripts_path_references",
    "test_render_failure",
    "test_render_success",
    "test_replace_project_version",
    "test_resolve_gates_maps_type_alias",
    "test_rewrite_dep_paths_dry_run",
    "test_rewrite_dep_paths_read_failure",
    "test_rewrite_dep_paths_with_internal_names",
    "test_rewrite_dep_paths_with_no_deps",
    "test_rewrite_pep621_invalid_path_dep_regex",
    "test_rewrite_pep621_no_project_table",
    "test_rewrite_pep621_non_string_item",
    "test_rewrite_poetry_no_poetry_table",
    "test_rewrite_poetry_no_tool_table",
    "test_rewrite_poetry_with_non_dict_value",
    "test_rule_dispatch_fails_on_invalid_pattern_rule_config",
    "test_rule_dispatch_fails_on_unknown_rule_mapping",
    "test_rule_dispatch_keeps_legacy_id_fallback_mapping",
    "test_rule_dispatch_prefers_fix_action_metadata",
    "test_run_cases",
    "test_run_cli_run_returns_one_for_fail",
    "test_run_cli_run_returns_two_for_error",
    "test_run_cli_run_returns_zero_for_pass",
    "test_run_cli_with_fail_fast_flag",
    "test_run_cli_with_multiple_projects",
    "test_run_deptry_wrapper",
    "test_run_mypy_stub_hints_wrapper",
    "test_run_pip_check_wrapper",
    "test_run_raw_cases",
    "test_signature_propagation_removes_and_adds_keywords",
    "test_signature_propagation_renames_call_keyword",
    "test_standalone_final_detected_as_fixable",
    "test_standalone_typealias_detected_as_fixable",
    "test_standalone_typevar_detected_as_fixable",
    "test_string_zero_return_value",
    "test_symbol_propagation_keeps_alias_reference_when_asname_used",
    "test_symbol_propagation_renames_import_and_local_references",
    "test_symbol_propagation_updates_mro_base_references",
    "test_sync_basemk_scenarios",
    "test_sync_error_scenarios",
    "test_sync_extra_paths_missing_root_pyproject",
    "test_sync_extra_paths_success_modes",
    "test_sync_extra_paths_sync_failure",
    "test_sync_one_edge_cases",
    "test_sync_root_validation",
    "test_sync_success_scenarios",
    "test_syntax_error_files_skipped",
    "test_target_path",
    "test_template_constants",
    "test_unwrap_item",
    "test_unwrap_item_toml_item",
    "test_violation_analysis_counts_massive_patterns",
    "test_violation_analyzer_skips_non_utf8_files",
    "test_workspace_check_main_returns_error_without_projects",
    "test_workspace_cli_migrate_command",
    "test_workspace_cli_migrate_output_contains_summary",
    "test_workspace_migrator_error_handling_on_invalid_workspace",
    "test_workspace_migrator_makefile_not_found_dry_run",
    "test_workspace_migrator_makefile_read_error",
    "test_workspace_migrator_pyproject_write_error",
    "test_workspace_root_doc_construction",
    "test_workspace_root_fallback",
    "u",
    "unit",
    "unwrap_item",
    "v",
    "validator",
    "workspace_root",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
