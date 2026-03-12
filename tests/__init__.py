# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""flext-core comprehensive test suite.

This package contains all tests for flext-core components.
Uses flext_tests directly for all generic test infrastructure.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from tests.base import TestsFlextServiceBase
    from tests.benchmark.test_container_memory import (
        TestContainerMemory,
        get_memory_usage,
    )
    from tests.benchmark.test_container_performance import (
        PerformanceBenchmark,
        TestContainerPerformance,
    )
    from tests.benchmark.test_refactor_nesting_performance import (
        TestPerformanceBenchmarks,
    )
    from tests.conftest import (
        FlextConsolidationContext,
        FlextScenarioRunner,
        FlextTestAutomationFramework,
        FunctionalExternalService,
        assert_rejects,
        assert_validates,
        automation_framework,
        clean_container,
        consolidation_context,
        empty_strings,
        flext_result_failure,
        flext_result_success,
        invalid_hostnames,
        invalid_port_numbers,
        invalid_uris,
        mock_external_service,
        out_of_range,
        parser_scenarios,
        rAssertionHelper,
        real_entity,
        real_value_object,
        reliability_scenarios,
        reset_global_container,
        result_assertion_helper,
        sample_data,
        scenario_runner,
        temp_dir,
        temp_directory,
        temp_file,
        test_context,
        test_framework,
        valid_hostnames,
        valid_port_numbers,
        valid_ranges,
        valid_strings,
        valid_uris,
        validation_scenarios,
        whitespace_strings,
    )
    from tests.constants import TestsFlextConstants, c
    from tests.helpers.factories import (
        FailingService,
        FailingServiceAuto,
        FailingServiceAutoFactory,
        FailingServiceFactory,
        GenericModelFactory,
        GetUserService as s,
        GetUserServiceAuto,
        GetUserServiceAutoFactory,
        GetUserServiceFactory,
        ServiceFactoryRegistry,
        ServiceTestCaseFactory,
        ServiceTestCases,
        TestDataGenerators,
        UserFactory,
        ValidatingService,
        ValidatingServiceAuto,
        ValidatingServiceAutoFactory,
        ValidatingServiceFactory,
        reset_all_factories,
    )
    from tests.helpers.scenarios import (
        ParserScenario,
        ParserScenarios,
        ReliabilityScenario,
        ReliabilityScenarios,
        ValidationScenario,
        ValidationScenarios,
    )
    from tests.infra.constants import FlextInfraTestConstants
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
    from tests.infra.models import FlextInfraTestModels
    from tests.infra.protocols import FlextInfraTestProtocols
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
        TestMainExceptionHandling,
        TestMainModuleImport,
        TestMainStructlogConfiguration,
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
        doc,
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
        test_rewrite_dep_paths_dry_run,
        test_rewrite_dep_paths_read_failure,
        test_rewrite_dep_paths_with_internal_names,
        test_rewrite_dep_paths_with_no_deps,
    )
    from tests.infra.unit.deps.test_path_sync_rewrite_pep621 import (
        TestRewritePep621,
        test_helpers_alias_is_reachable_pep621,
        test_rewrite_pep621_invalid_path_dep_regex,
        test_rewrite_pep621_no_project_table,
        test_rewrite_pep621_non_string_item,
    )
    from tests.infra.unit.deps.test_path_sync_rewrite_poetry import (
        TestRewritePoetry,
        test_helpers_alias_is_reachable_poetry,
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
    from tests.infra.utilities import FlextInfraTestUtilities
    from tests.infra.workspace_factory import WorkspaceFactory
    from tests.infra.workspace_scenarios import (
        BrokenScenario,
        EmptyScenario,
        FullScenario,
        MinimalScenario,
    )
    from tests.integration.patterns.test_advanced_patterns import TestFunction
    from tests.integration.patterns.test_architectural_patterns import (
        TestEnterprisePatterns,
        TestEventDrivenPatterns,
    )
    from tests.integration.patterns.test_patterns_commands import (
        CreateUserCommand,
        CreateUserCommandHandler,
        FailingCommand,
        FailingCommandHandler,
        FlextCommandId,
        FlextCommandType,
        TestFlextCommand,
        TestFlextCommandHandler,
        TestFlextCommandResults,
        UpdateUserCommand,
        UpdateUserCommandHandler,
    )
    from tests.integration.patterns.test_patterns_logging import (
        TestFlextContext,
        TestFlextLogger,
        TestFlextLoggerIntegration,
        TestFlextLoggerUsage,
        TestFlextLogLevel,
        make_result_logger,
    )
    from tests.integration.patterns.test_patterns_testing import (
        AssertionBuilder,
        FixtureBuilder,
        FlextTestBuilder,
        GivenWhenThenBuilder,
        MockScenario,
        ParameterizedTestBuilder,
        SuiteBuilder,
        TestAdvancedPatterns,
        TestComprehensiveIntegration,
        TestPerformanceAnalysis,
        TestPropertyBasedPatterns,
        TestRealWorldScenarios,
        arrange_act_assert,
        mark_test_pattern,
    )
    from tests.integration.test_config_integration import (
        ConfigTestCase,
        ConfigTestFactories,
        TestFlextSettingsSingletonIntegration,
        ThreadSafetyTest,
    )
    from tests.integration.test_infra_integration import (
        TestBaseMkGenerationFlow,
        TestContainerIntegration,
        TestCrossModuleIntegration,
        TestIntegrationWithRealCommandServices,
        TestOutputSingletonConsistency,
        TestPathResolverDiscoveryFlow,
        TestServicerChaining,
        TestWorkspaceDetectionOrchestrationFlow,
    )
    from tests.integration.test_integration import TestLibraryIntegration
    from tests.integration.test_migration_validation import (
        TestBackwardCompatibility,
        TestMigrationComplexity,
        TestMigrationScenario1,
        TestMigrationScenario2,
        TestMigrationScenario4,
        TestMigrationScenario5,
    )
    from tests.integration.test_refactor_nesting_file import (
        test_class_nesting_refactor_single_file_end_to_end,
    )
    from tests.integration.test_refactor_nesting_idempotency import TestIdempotency
    from tests.integration.test_refactor_nesting_project import TestProjectLevelRefactor
    from tests.integration.test_refactor_nesting_workspace import (
        TestWorkspaceLevelRefactor,
    )
    from tests.integration.test_refactor_policy_mro import (
        AlgarOudMigConstants,
        AlgarOudMigModels,
        AlgarOudMigProtocols,
        AlgarOudMigTypes,
        AlgarOudMigUtilities,
        FlextCliConstants,
        FlextCliModels,
        FlextCliProtocols,
        FlextCliTypes,
        FlextCliUtilities,
        FlextLdapConstants,
        FlextLdapModels,
        FlextLdapProtocols,
        FlextLdapTypes,
        FlextLdapUtilities,
        test_mro_resolver_accepts_expected_order,
        test_mro_resolver_rejects_wrong_order,
    )
    from tests.integration.test_service import (
        LifecycleService,
        NotificationService,
        ServiceConfig,
        TestFlextServiceIntegration,
        UserQueryService,
        UserServiceEntity,
        pytestmark,
    )
    from tests.integration.test_system import TestCompleteFlextSystemIntegration
    from tests.models import TestsFlextModels, m
    from tests.protocols import TestsFlextProtocols, p
    from tests.test_documented_patterns import (
        GetUserService,
        MultiOperationService,
        RailwayTestCase,
        SendEmailService,
        ServiceTestCase,
        TestAllPatternsIntegration,
        TestFactories,
        TestPattern1V1Explicit,
        TestPattern2V2Property,
        TestPattern3RailwayV1,
        TestPattern4RailwayV2Property,
        TestPattern5MonadicComposition,
        TestPattern6ErrorHandling,
        TestPattern7AutomaticInfrastructure,
        TestPattern8MultipleOperations,
        User,
        ValidationService,
    )
    from tests.test_service_result_property import TestServiceResultProperty
    from tests.test_utils import (
        AssertionHelpers,
        TestDataFactory,
        TestFixtureFactory,
        TestResult,
        TestResultCo,
        assertion_helpers,
        fixture_factory,
        test_data_factory,
    )
    from tests.typings import T, T_co, T_contra, TestsFlextTypes
    from tests.unit.conftest_infra import (
        infra_git,
        infra_git_repo,
        infra_io,
        infra_path,
        infra_patterns,
        infra_pr_manager,
        infra_pr_workspace_manager,
        infra_reporting,
        infra_safe_command_output,
        infra_selection,
        infra_subprocess,
        infra_templates,
        infra_test_workspace,
        infra_toml,
        infra_workflow_linter,
        infra_workflow_syncer,
    )
    from tests.unit.contracts.text_contract import TextUtilityContract
    from tests.unit.flext_tests.test_builders import TestFlextTestsBuilders
    from tests.unit.flext_tests.test_docker import (
        TestContainerInfo,
        TestContainerStatus,
        TestFlextTestsDocker,
        TestFlextTestsDockerWorkerId,
        TestFlextTestsDockerWorkspaceRoot,
    )
    from tests.unit.flext_tests.test_domains import TestFlextTestsDomains
    from tests.unit.flext_tests.test_factories import (
        TestConfig,
        TestFactoriesHelpers,
        TestFlextTestsFactoriesModernAPI,
        TestsFlextTestsFactoriesDict,
        TestsFlextTestsFactoriesGeneric,
        TestsFlextTestsFactoriesList,
        TestsFlextTestsFactoriesModel,
        TestsFlextTestsFactoriesRes,
        TestUser,
    )
    from tests.unit.flext_tests.test_files import (
        TestAssertExists,
        TestBatchOperations,
        TestCreateInStatic,
        TestFileInfo,
        TestFileInfoFromModels,
        TestFlextTestsFiles,
        TestFlextTestsFilesNewApi,
        TestInfoWithContentMeta,
        TestShortAlias,
    )
    from tests.unit.flext_tests.test_matchers import TestFlextTestsMatchers
    from tests.unit.flext_tests.test_utilities import (
        TestFlextTestsUtilitiesFactory,
        TestFlextTestsUtilitiesResult,
        TestFlextTestsUtilitiesResultCompat,
        TestFlextTestsUtilitiesTestContext,
    )
    from tests.unit.test_args_coverage_100 import TestFlextUtilitiesArgs
    from tests.unit.test_automated_container import TestAutomatedFlextContainer
    from tests.unit.test_automated_context import TestAutomatedFlextContext
    from tests.unit.test_automated_decorators import (
        TestAutomatedFlextDecorators,
        TestAutomatedFlextDecorators as d,
    )
    from tests.unit.test_automated_dispatcher import TestAutomatedFlextDispatcher
    from tests.unit.test_automated_exceptions import (
        TestAutomatedFlextExceptions,
        TestAutomatedFlextExceptions as e,
    )
    from tests.unit.test_automated_handlers import TestAutomatedFlextHandlers
    from tests.unit.test_automated_loggings import TestAutomatedFlextLoggings
    from tests.unit.test_automated_mixins import (
        TestAutomatedFlextMixins,
        TestAutomatedFlextMixins as x,
    )
    from tests.unit.test_automated_registry import TestAutomatedFlextRegistry
    from tests.unit.test_automated_result import TestAutomatedr
    from tests.unit.test_automated_runtime import TestAutomatedFlextRuntime
    from tests.unit.test_automated_service import TestAutomatedFlextService
    from tests.unit.test_automated_settings import TestAutomatedFlextSettings
    from tests.unit.test_automated_utilities import TestAutomatedFlextUtilities
    from tests.unit.test_collection_utilities_coverage_100 import (
        CoerceListValidatorScenario,
        CollectionScenarios,
        ParseMappingScenario,
        ParseSequenceScenario,
    )
    from tests.unit.test_collections_coverage_100 import (
        TestFlextModelsCollectionsCategories,
        TestFlextModelsCollectionsOptions,
        TestFlextModelsCollectionsResults,
        TestFlextModelsCollectionsSettings,
        TestFlextModelsCollectionsStatistics,
    )
    from tests.unit.test_config import TestFlextSettings
    from tests.unit.test_constants import TestFlextConstants
    from tests.unit.test_constants_full_coverage import (
        test_constants_auto_enum_and_bimapping_paths,
    )
    from tests.unit.test_container import TestFlextContainer
    from tests.unit.test_container_full_coverage import (
        test_additional_container_branches_cover_fluent_and_lookup_paths,
        test_additional_register_factory_and_unregister_paths,
        test_config_context_properties_and_defaults,
        test_configure_with_resource_register_and_factory_error_paths,
        test_container_remaining_branch_paths_in_sync_factory_and_getters,
        test_create_auto_register_factories_path,
        test_create_auto_register_factory_wrapper_callable_and_non_callable,
        test_create_scoped_instance_and_scoped_additional_branches,
        test_get_and_get_typed_resource_factory_paths,
        test_initialize_di_components_error_paths,
        test_initialize_di_components_second_type_error_branch,
        test_misc_unregistration_clear_and_reset,
        test_protocol_name_and_builder,
        test_provide_property_paths,
        test_register_existing_providers_full_paths_and_misc_methods,
        test_register_existing_providers_skips_and_register_core_fallback,
        test_scoped_config_context_branches,
        test_sync_config_namespace_paths,
        test_sync_config_registers_namespace_factories_and_fallbacks,
    )
    from tests.unit.test_context_coverage_100 import TestContext100Coverage
    from tests.unit.test_context_full_coverage import (
        test_clear_keys_values_items_and_validate_branches,
        test_container_and_service_domain_paths,
        test_create_merges_metadata_dict_branch,
        test_create_overloads_and_auto_correlation,
        test_export_paths_with_metadata_and_statistics,
        test_inactive_and_none_value_paths,
        test_narrow_contextvar_invalid_inputs,
        test_protocol_name_and_narrow_contextvar_exception_branch,
        test_set_set_all_get_validation_and_error_paths,
        test_update_statistics_remove_hook_and_clone_false_result,
    )
    from tests.unit.test_coverage_76_lines import (
        TestResultBasics,
        TestResultTransformations,
    )
    from tests.unit.test_coverage_context import (
        TestContextDataModel,
        TestCorrelationDomain,
        TestPerformanceDomain,
        TestServiceDomain,
        TestUtilitiesDomain,
    )
    from tests.unit.test_coverage_exceptions import (
        TestExceptionContext,
        TestExceptionEdgeCases,
        TestExceptionFactory,
        TestExceptionIntegration,
        TestExceptionMetrics,
        TestExceptionProperties,
        TestExceptionSerialization,
        TestFlextExceptionsHierarchy,
        TestHierarchicalExceptionSystem,
    )
    from tests.unit.test_coverage_loggings import (
        TestExceptionLogging,
        TestFactoryPatterns,
        TestGlobalContextManagement,
        TestInstanceCreation,
        TestLevelBasedContextManagement,
        TestLoggingIntegration,
        TestLoggingMethods,
        TestScopedContextManagement,
    )
    from tests.unit.test_coverage_models import (
        TestAggregateRoots,
        TestCommands,
        TestDomainEvents,
        TestEntities,
        TestMetadata,
        TestModelIntegration,
        TestModelSerialization,
        TestModelValidation,
        TestQueries,
        TestValues,
    )
    from tests.unit.test_decorators import TestFlextDecorators
    from tests.unit.test_decorators_discovery_full_coverage import (
        TestFactoryDecoratorsDiscoveryHasFactories,
        TestFactoryDecoratorsDiscoveryScanModule,
    )
    from tests.unit.test_decorators_full_coverage import (
        test_bind_operation_context_without_ensure_correlation_and_bind_failure,
        test_clear_operation_scope_and_handle_log_result_paths,
        test_combined_with_and_without_railway_uses_injection,
        test_deprecated_wrapper_emits_warning_and_returns_value,
        test_execute_retry_exponential_and_handle_exhaustion_raise_last_exception,
        test_execute_retry_loop_covers_default_linear_and_never_ran,
        test_handle_log_result_without_fallback_logger_and_non_dict_like_extra,
        test_handle_retry_exhaustion_falsey_exception_reaches_timeout_error,
        test_inject_sets_missing_dependency_from_container,
        test_log_operation_track_perf_exception_adds_duration,
        test_railway_and_retry_additional_paths,
        test_resolve_logger_prefers_logger_attribute,
        test_retry_unreachable_timeouterror_path,
        test_timeout_additional_success_and_reraise_timeout_paths,
        test_timeout_covers_exception_timeout_branch,
        test_timeout_raises_when_successful_call_exceeds_limit,
        test_timeout_reraises_original_exception_when_within_limit,
        test_track_performance_success_and_failure_paths,
        test_with_correlation_with_context_track_operation_and_factory,
    )
    from tests.unit.test_di_incremental import (
        TestContainerDIRealExecution,
        TestDependencyIntegrationRealExecution,
        TestDIBridgeRealExecution,
        TestRealWiringScenarios,
        TestServiceBootstrapWithDI,
    )
    from tests.unit.test_di_services_access import (
        TestConfigServiceViaDI,
        TestContextServiceViaDI,
        TestLoggerServiceViaDI,
        TestServicesIntegrationViaDI,
    )
    from tests.unit.test_dispatcher_di import TestDispatcherDI
    from tests.unit.test_dispatcher_full_coverage import (
        EventHandler,
        QueryHandler,
        SampleCommand,
        SampleEvent,
        SampleHandler,
        SampleQuery,
        UnregisteredCommand,
        dispatcher,
        test_callable_registration_with_attribute,
        test_dispatch_invalid_input_types,
        test_event_publishing_strict,
        test_exception_handling_in_dispatch,
        test_handler_attribute_discovery,
        test_invalid_registration_attempts,
        test_strict_registration_and_dispatch,
    )
    from tests.unit.test_dispatcher_minimal import (
        AutoCommand,
        AutoDiscoveryHandler,
        EchoHandler,
        EventSubscriber,
        ExplodingHandler,
        test_dispatch_after_handler_removal_fails,
        test_dispatch_auto_discovery_handler,
        test_dispatch_command_success,
        test_dispatch_handler_exception_returns_failure,
        test_dispatch_no_handler_fails,
        test_publish_event_to_subscriber,
        test_publish_no_subscribers_succeeds,
        test_register_handler_as_event_subscriber,
        test_register_handler_with_can_handle,
        test_register_handler_with_message_type,
        test_register_handler_without_route_fails,
    )
    from tests.unit.test_dispatcher_reliability import (
        test_circuit_breaker_transitions_and_metrics,
        test_rate_limiter_blocks_then_recovers,
        test_rate_limiter_jitter_application,
        test_retry_policy_behavior,
    )
    from tests.unit.test_dispatcher_reliability_full_coverage import (
        test_dispatcher_reliability_branch_paths,
    )
    from tests.unit.test_dispatcher_timeout_coverage_100 import (
        TestTimeoutEnforcerCleanup,
        TestTimeoutEnforcerEdgeCases,
        TestTimeoutEnforcerExecutorManagement,
        TestTimeoutEnforcerInitialization,
        TimeoutEnforcerScenarios,
    )
    from tests.unit.test_entity_coverage import TestEntityCoverageEdgeCases
    from tests.unit.test_enum_utilities_coverage_100 import (
        CoerceValidatorScenario,
        EnumScenarios,
        IsMemberScenario,
        IsSubsetScenario,
        ParseOrDefaultScenario,
        ParseScenario,
        TestuEnumCoerceByNameValidator,
        TestuEnumCoerceValidator,
        TestuEnumIsMember,
        TestuEnumIsSubset,
        TestuEnumMetadata,
        TestuEnumParse,
        TestuEnumParseOrDefault,
    )
    from tests.unit.test_exceptions import Teste
    from tests.unit.test_exceptions_full_coverage import (
        test_authentication_error_normalizes_extra_kwargs_into_context,
        test_base_error_normalize_metadata_merges_existing_metadata_model,
        test_exceptions_uncovered_metadata_paths,
        test_merge_metadata_context_paths,
        test_not_found_error_correlation_id_selection_and_extra_kwargs,
    )
    from tests.unit.test_final_75_percent_push import TestCoveragePush75Percent
    from tests.unit.test_handler_decorator_discovery import (
        TestHandlerDecoratorMetadata,
        TestHandlerDiscoveryClass,
        TestHandlerDiscoveryEdgeCases,
        TestHandlerDiscoveryIntegration,
        TestHandlerDiscoveryModule,
        TestHandlerDiscoveryServiceIntegration,
    )
    from tests.unit.test_handlers import TestFlextHandlers
    from tests.unit.test_handlers_full_coverage import (
        handlers_module,
        test_create_from_callable_branches,
        test_discovery_narrowed_function_paths,
        test_handler_type_literal_and_invalid,
        test_invalid_handler_mode_init_raises,
        test_run_pipeline_query_and_event_paths,
    )
    from tests.unit.test_loggings_error_paths_coverage import TestLoggingsErrorPaths
    from tests.unit.test_loggings_full_coverage import (
        test_loggings_bind_clear_level_error_paths,
        test_loggings_context_and_factory_paths,
        test_loggings_exception_and_adapter_paths,
        test_loggings_instance_and_message_format_paths,
        test_loggings_remaining_branch_paths,
        test_loggings_source_and_log_error_paths,
        test_loggings_uncovered_level_trace_path_and_exception_guards,
    )
    from tests.unit.test_loggings_strict_returns import (
        TestBackwardCompatDiscardReturnValue,
        TestCriticalReturnsResultBool,
        TestDebugReturnsResultBool,
        TestErrorReturnsResultBool,
        TestExceptionReturnsResultBool,
        TestInfoReturnsResultBool,
        TestLogReturnsResultBool,
        TestProtocolComplianceStructlogLogger,
        TestTraceReturnsResultBool,
        TestWarningReturnsResultBool,
    )
    from tests.unit.test_mixins import TestFlextMixinsNestedClasses
    from tests.unit.test_mixins_full_coverage import (
        test_mixins_container_registration_and_logger_paths,
        test_mixins_context_logging_and_cqrs_paths,
        test_mixins_context_stack_pop_initializes_missing_stack_attr,
        test_mixins_remaining_branch_paths,
        test_mixins_result_and_model_conversion_paths,
        test_mixins_runtime_bootstrap_and_track_paths,
        test_mixins_validation_and_protocol_paths,
    )
    from tests.unit.test_models import TestFlextModels
    from tests.unit.test_models_79_coverage import (
        TestFlextModelsAggregateRoot,
        TestFlextModelsCommand,
        TestFlextModelsDomainEvent,
        TestFlextModelsEdgeCases,
        TestFlextModelsEntity,
        TestFlextModelsIntegration,
        TestFlextModelsQuery,
        TestFlextModelsValue,
    )
    from tests.unit.test_models_base_full_coverage import (
        test_frozen_value_model_equality_and_hash,
        test_identifiable_unique_id_empty_rejected,
        test_metadata_attributes_accepts_basemodel_mapping,
        test_metadata_attributes_accepts_none,
        test_metadata_attributes_accepts_t_dict_and_mapping,
        test_metadata_attributes_rejects_basemodel_non_mapping_dump,
        test_metadata_attributes_rejects_non_mapping,
        test_timestampable_timestamp_conversion_and_json_serializer,
        test_timestamped_model_and_alias_and_canonical_symbols,
    )
    from tests.unit.test_models_collections_full_coverage import (
        test_categories_clear_and_symbols_are_available,
        test_config_hash_from_mapping_and_non_hashable,
        test_options_merge_conflict_paths_and_empty_merge_options,
        test_results_internal_conflict_paths_and_combine,
        test_rules_merge_combines_model_dump_values,
        test_statistics_from_dict_and_none_conflict_resolution,
    )
    from tests.unit.test_models_container import (
        ContainerModelsScenarios,
        TestFlextModelsContainer,
        TestFlextUtilitiesModelNormalizeToMetadata,
    )
    from tests.unit.test_models_container_full_coverage import (
        test_container_resource_registration_metadata_normalized,
    )
    from tests.unit.test_models_context_full_coverage import (
        test_context_data_metadata_normalizer_removed,
        test_context_data_normalize_and_json_checks,
        test_context_data_validate_dict_serializable_error_paths,
        test_context_data_validate_dict_serializable_none_and_mapping,
        test_context_data_validate_dict_serializable_real_dicts,
        test_context_export_serializable_and_validators,
        test_context_export_statistics_validator_and_computed_fields,
        test_context_export_validate_dict_serializable_mapping_and_models,
        test_context_export_validate_dict_serializable_valid,
        test_scope_data_validators_and_errors,
        test_statistics_and_custom_fields_validators,
        test_structlog_proxy_context_var_default_when_key_missing,
        test_structlog_proxy_context_var_get_set_reset_paths,
        test_to_general_value_dict_removed,
    )
    from tests.unit.test_models_cqrs_full_coverage import (
        test_command_pagination_limit,
        test_cqrs_query_resolve_deeper_and_int_pagination,
        test_flext_message_type_alias_adapter,
        test_handler_builder_fluent_methods,
        test_query_resolve_pagination_wrapper_and_fallback,
        test_query_validate_pagination_dict_and_default,
    )
    from tests.unit.test_models_entity_full_coverage import (
        test_entity_comparable_map_and_bulk_validation_paths,
    )
    from tests.unit.test_models_generic_full_coverage import (
        test_canonical_aliases_are_available,
        test_conversion_add_converted_and_error_metadata_append_paths,
        test_conversion_add_skipped_skip_reason_upsert_paths,
        test_conversion_add_warning_metadata_append_paths,
        test_conversion_start_and_complete_methods,
        test_operation_progress_start_operation_sets_runtime_fields,
    )
    from tests.unit.test_models_handler_full_coverage import (
        test_models_handler_branches,
        test_models_handler_uncovered_mode_and_reset_paths,
    )
    from tests.unit.test_models_service_full_coverage import (
        test_service_request_timeout_post_validator_messages,
        test_service_request_timeout_validator_branches,
    )
    from tests.unit.test_models_settings_full_coverage import (
        test_models_settings_branch_paths,
        test_models_settings_context_validator_and_non_standard_status_input,
    )
    from tests.unit.test_models_validation_full_coverage import (
        test_basic_imports_work,
        test_ensure_utc_datetime_adds_tzinfo_when_naive,
        test_ensure_utc_datetime_preserves_aware,
        test_ensure_utc_datetime_returns_none_on_none,
        test_facade_binding_is_correct,
        test_normalize_to_list_passes_list_through,
        test_normalize_to_list_wraps_int,
        test_normalize_to_list_wraps_scalar,
        test_strip_whitespace_preserves_clean,
        test_strip_whitespace_returns_empty_on_spaces,
        test_strip_whitespace_trims_leading_trailing,
        test_validate_config_dict_normalizes_dict,
        test_validate_tags_list_from_string,
        test_validate_tags_list_normalizes,
    )
    from tests.unit.test_namespace_validator import TestFlextInfraNamespaceValidator
    from tests.unit.test_pagination_coverage_100 import (
        ExtractPageParamsScenario,
        PaginationScenarios,
        PreparePaginationDataScenario,
        TestuPaginationBuildPaginationResponse,
        TestuPaginationExtractPageParams,
        TestuPaginationExtractPaginationConfig,
        TestuPaginationPreparePaginationData,
        TestuPaginationValidatePaginationParams,
        ValidatePaginationParamsScenario,
    )
    from tests.unit.test_phase2_coverage_final import TestPhase2FinalCoveragePush
    from tests.unit.test_protocols import TestFlextProtocols
    from tests.unit.test_protocols_full_coverage import (
        test_check_implements_protocol_false_non_runtime_protocol,
        test_implements_decorator_helper_methods_and_static_wrappers,
        test_implements_decorator_validation_error_message,
        test_protocol_base_name_methods_and_runtime_check_branch,
        test_protocol_meta_default_model_base_and_get_protocols_default,
        test_protocol_model_and_settings_methods,
    )
    from tests.unit.test_refactor_cli_models_workflow import (
        test_centralize_pydantic_cli_outputs_extended_metrics,
        test_namespace_enforce_cli_fails_on_manual_protocol_violation,
        test_ultrawork_models_cli_runs_dry_run_copy,
    )
    from tests.unit.test_refactor_migrate_to_class_mro import (
        test_discover_project_roots_without_nested_git_dirs,
        test_migrate_protocols_rewrites_references_with_p_alias,
        test_migrate_to_mro_inlines_alias_constant_into_constants_class,
        test_migrate_to_mro_moves_constant_and_rewrites_reference,
        test_migrate_to_mro_moves_manual_uppercase_assignment,
        test_migrate_to_mro_normalizes_facade_alias_to_c,
        test_migrate_to_mro_rejects_unknown_target,
        test_migrate_typings_rewrites_references_with_t_alias,
        test_mro_scanner_includes_constants_variants_in_all_scopes,
        test_refactor_utilities_iter_python_files_includes_examples_and_scripts,
    )
    from tests.unit.test_refactor_namespace_enforcer import (
        test_namespace_enforcer_apply_inserts_future_after_single_line_module_docstring,
        test_namespace_enforcer_apply_keeps_script_shebang_when_adding_future,
        test_namespace_enforcer_apply_moves_manual_protocol_to_protocols_file,
        test_namespace_enforcer_creates_missing_facades_and_rewrites_imports,
        test_namespace_enforcer_detects_cyclic_imports_in_tests_directory,
        test_namespace_enforcer_detects_internal_private_imports,
        test_namespace_enforcer_detects_manual_protocol_outside_canonical_files,
        test_namespace_enforcer_detects_manual_typings_and_compat_aliases,
        test_namespace_enforcer_detects_missing_runtime_alias_outside_src,
        test_namespace_enforcer_does_not_rewrite_indented_import_aliases,
        test_namespace_enforcer_does_not_rewrite_multiline_import_alias_blocks,
    )
    from tests.unit.test_refactor_policy_family_rules import (
        test_decorators_family_blocks_dispatcher_target,
        test_dispatcher_family_blocks_models_target,
        test_helper_consolidation_is_prechecked,
        test_models_family_blocks_utilities_target,
        test_runtime_family_blocks_non_runtime_target,
        test_utilities_family_allows_utilities_target,
    )
    from tests.unit.test_refactor_pydantic_centralizer import (
        test_centralizer_converts_typed_dict_factory_to_model,
        test_centralizer_does_not_touch_settings_module,
        test_centralizer_moves_dict_alias_in_typings_without_keyword_name,
        test_centralizer_moves_manual_type_aliases_to_models_file,
    )
    from tests.unit.test_registry import ConcreteTestHandler, TestFlextRegistry
    from tests.unit.test_registry_full_coverage import (
        test_create_auto_discover_and_mode_mapping,
        test_execute_and_register_handler_failure_paths,
        test_get_plugin_and_register_metadata_and_list_items_exception,
        test_summary_error_paths_and_bindings_failures,
        test_summary_properties_and_subclass_storage_reset,
    )
    from tests.unit.test_result import Testr
    from tests.unit.test_result_additional import (
        ExplodingGetattr,
        test_create_from_callable_and_repr,
        test_data_alias_matches_value,
        test_flow_through_short_circuits_on_failure,
        test_map_error_identity_and_transform,
        test_ok_raises_on_none,
        test_to_io_result_failure_path,
        test_with_resource_cleanup_runs,
    )
    from tests.unit.test_result_coverage_100 import TestrCoverage
    from tests.unit.test_result_exception_carrying import (
        TestAltPropagatesException,
        TestCreateFromCallableCarriesException,
        TestErrorOrPatternUnchanged,
        TestExceptionPropertyAccess,
        TestFailNoExceptionBackwardCompat,
        TestFailWithException,
        TestFlatMapPropagatesException,
        TestFromIOResultCarriesException,
        TestFromValidationCarriesException,
        TestLashPropagatesException,
        TestMapPropagatesException,
        TestMonadicOperationsUnchanged,
        TestOkNoneGuardStillRaises,
        TestSafeCarriesException,
        TestToIOChainsException,
        TestTraversePropagatesException,
    )
    from tests.unit.test_result_full_coverage import (
        test_from_validation_and_to_model_paths,
        test_init_fallback_and_lazy_result_property,
        test_lash_runtime_result_and_from_io_result_fallback,
        test_map_flat_map_and_then_paths,
        test_recover_tap_and_tap_error_paths,
        test_type_guards_and_protocol_name,
        test_validation_like_error_structure,
    )
    from tests.unit.test_runtime import TestFlextRuntime
    from tests.unit.test_runtime_coverage_100 import (
        TestRuntimeDictLike,
        TestRuntimeTypeChecking,
    )
    from tests.unit.test_runtime_full_coverage import (
        reset_runtime_state,
        runtime_cov_tests,
        runtime_tests,
        test_async_log_writer_paths,
        test_async_log_writer_shutdown_with_full_queue,
        test_config_bridge_and_trace_context_and_http_validation,
        test_configure_structlog_edge_paths,
        test_configure_structlog_print_logger_factory_fallback,
        test_dependency_integration_and_wiring_paths,
        test_dependency_registration_duplicate_guards,
        test_ensure_trace_context_dict_conversion_paths,
        test_get_logger_none_name_paths,
        test_model_helpers_remaining_paths,
        test_model_support_and_hash_compare_paths,
        test_normalization_edge_branches,
        test_reconfigure_and_reset_state_paths,
        test_reuse_existing_runtime_coverage_branches,
        test_reuse_existing_runtime_scenarios,
        test_runtime_create_instance_failure_branch,
        test_runtime_integration_tracking_paths,
        test_runtime_misc_remaining_paths,
        test_runtime_module_accessors_and_metadata,
        test_runtime_result_alias_compatibility,
        test_runtime_result_all_missed_branches,
        test_runtime_result_remaining_paths,
    )
    from tests.unit.test_service import TestsCore
    from tests.unit.test_service_additional import (
        RuntimeCloneService,
        test_get_service_info,
        test_is_valid_handles_validation_exception,
        test_result_property_raises_on_failure,
    )
    from tests.unit.test_service_bootstrap import (
        ConcreteTestService,
        TestServiceBootstrap,
    )
    from tests.unit.test_service_coverage_100 import (
        TestDomainResult,
        TestService,
        TestService100Coverage,
        TestServiceWithValidation,
    )
    from tests.unit.test_service_full_coverage import (
        test_service_create_initial_runtime_prefers_custom_config_type_and_context_property,
        test_service_create_runtime_container_overrides_branch,
        test_service_init_type_guards_and_properties,
    )
    from tests.unit.test_settings_full_coverage import (
        test_settings_materialize_and_context_overrides,
    )
    from tests.unit.test_transformer_class_nesting import (
        test_class_nesting_appends_to_existing_namespace_and_removes_pass,
        test_class_nesting_keeps_unmapped_top_level_classes,
        test_class_nesting_moves_top_level_class_into_new_namespace,
    )
    from tests.unit.test_transformer_helper_consolidation import (
        TestHelperConsolidationTransformer,
    )
    from tests.unit.test_transformer_nested_class_propagation import (
        NestedClassPropagationTransformer,
        test_nested_class_propagation_preserves_asname_and_rewrites_alias_usage,
        test_nested_class_propagation_updates_import_annotations_and_calls,
    )
    from tests.unit.test_typings import TestFlextTypings
    from tests.unit.test_typings_full_coverage import (
        TestConfigMapDictOps,
        TestDictMixinOperations,
        TestValidatorCallable,
        TestValidatorMapMixin,
    )
    from tests.unit.test_utilities import Testu
    from tests.unit.test_utilities_args_full_coverage import (
        UnknownHint,
        test_args_get_enum_params_annotated_unwrap_branch,
        test_args_get_enum_params_branches,
    )
    from tests.unit.test_utilities_cache_coverage_100 import (
        CacheScenarios,
        TestuCacheClearObjectCache,
        TestuCacheGenerateCacheKey,
        TestuCacheHasCacheAttributes,
        TestuCacheLogger,
        TestuCacheNormalizeComponent,
        TestuCacheSortDictKeys,
        TestuCacheSortKey,
    )
    from tests.unit.test_utilities_checker_full_coverage import (
        MissingType,
        test_checker_logger_and_safe_type_hints_fallback,
        test_extract_message_type_annotation_and_dict_subclass_paths,
        test_extract_message_type_from_handle_with_only_self,
        test_extract_message_type_from_parameter_branches,
        test_object_dict_and_type_error_fallback_paths,
    )
    from tests.unit.test_utilities_collection_coverage_100 import (
        TestuCollectionBatch,
        TestuCollectionChunk,
        TestuCollectionCoerceDictValidator,
        TestuCollectionCoerceListValidator,
        TestuCollectionCount,
        TestuCollectionFilter,
        TestuCollectionFind,
        TestuCollectionGroup,
        TestuCollectionMap,
        TestuCollectionMerge,
        TestuCollectionParseMapping,
        TestuCollectionParseSequence,
        TestuCollectionProcess,
    )
    from tests.unit.test_utilities_collection_full_coverage import (
        test_batch_fail_collect_flatten_and_progress,
        test_collection_batch_failure_error_capture_and_parse_sequence_outer_error,
        test_find_mapping_no_match_and_merge_error_paths,
        test_parse_mapping_outer_exception,
        test_process_outer_exception_and_coercion_branches,
    )
    from tests.unit.test_utilities_configuration_coverage_100 import (
        BadSingletonForTest,
        ConfigWithoutModelConfigForTest,
        DataclassConfigForTest,
        FailingOptionsForTest,
        OptionsModelForTest,
        SingletonWithoutGetGlobalForTest,
        SingletonWithoutModelDumpForTest,
        StrictOptionsForTest,
        TestConfigConstants,
        TestConfigModels,
        TestFlextUtilitiesConfiguration,
    )
    from tests.unit.test_utilities_configuration_full_coverage import (
        test_build_options_invalid_only_kwargs_returns_base,
        test_private_getters_exception_paths,
        test_register_singleton_register_factory_and_bulk_register_paths,
        test_resolve_env_file_and_log_level,
    )
    from tests.unit.test_utilities_context_full_coverage import (
        TestCloneContainer,
        TestCloneRuntime,
        TestCreateDatetimeProxy,
        TestCreateDictProxy,
        TestCreateStrProxy,
    )
    from tests.unit.test_utilities_conversion_full_coverage import (
        test_conversion_string_and_join_paths,
        test_to_flexible_value_and_safe_list_branches,
        test_to_flexible_value_fallback_none_branch_for_unsupported_type,
    )
    from tests.unit.test_utilities_coverage import TestUtilitiesCoverage
    from tests.unit.test_utilities_data_mapper import (
        TestMapperBuildFlagsDict,
        TestMapperCollectActiveKeys,
        TestMapperFilterDict,
        TestMapperInvertDict,
        TestMapperMapDictKeys,
        TestMapperTransformValues,
    )
    from tests.unit.test_utilities_deprecation_full_coverage import (
        test_deprecated_class_noop_init_branch,
    )
    from tests.unit.test_utilities_domain import (
        TestuDomain,
        create_compare_entities_cases,
        create_compare_value_objects_cases,
        create_hash_entity_cases,
        create_hash_value_object_cases,
        create_validate_entity_has_id_cases,
        create_validate_value_object_immutable_cases,
    )
    from tests.unit.test_utilities_domain_full_coverage import (
        TestDomainHashValue,
        TestDomainLogger,
        TestValidateValueImmutable,
        test_validate_value_object_immutable_exception_and_no_setattr_branch,
    )
    from tests.unit.test_utilities_enum_full_coverage import (
        Priority,
        Status,
        TextLike,
        test_auto_value_lowercases_input,
        test_bi_map_returns_forward_copy_and_inverse,
        test_create_discriminated_union_multiple_enums,
        test_create_enum_executes_factory_path,
        test_dispatch_coerce_mode_with_enum_string_and_other_object,
        test_dispatch_is_member_by_name_and_by_value,
        test_dispatch_is_name_mode,
        test_dispatch_parse_mode_with_enum_string_and_other_object,
        test_dispatch_unknown_mode_raises,
        test_get_enum_values_returns_immutable_sequence,
        test_members_uses_cache_on_second_call,
        test_names_uses_cache_on_second_call,
        test_private_coerce_with_enum_and_string,
        test_private_is_member_by_name,
        test_private_is_member_by_value,
        test_private_parse_success_and_failure,
        test_shortcuts_delegate_to_primary_methods,
    )
    from tests.unit.test_utilities_generators_full_coverage import (
        generators_module,
        test_enrich_and_ensure_trace_context_branches,
        test_ensure_dict_branches,
        test_generate_special_paths_and_dynamic_subclass,
        test_generators_additional_missed_paths,
        test_generators_mapping_non_dict_normalization_path,
        test_normalize_context_to_dict_error_paths,
    )
    from tests.unit.test_utilities_guards_full_coverage import (
        test_aliases_are_available,
        test_chk_exercises_missed_branches,
        test_configuration_mapping_and_dict_negative_branches,
        test_extract_mapping_or_none_branches,
        test_guard_in_has_empty_none_helpers,
        test_guard_instance_attribute_access_warnings,
        test_guards_bool_identity_branch_via_isinstance_fallback,
        test_guards_bool_shortcut_and_issubclass_typeerror,
        test_guards_handler_type_issubclass_typeerror_branch_direct,
        test_guards_issubclass_success_when_callable_is_patched,
        test_guards_issubclass_typeerror_when_class_not_treated_as_callable,
        test_is_flexible_value_covers_all_branches,
        test_is_general_value_type_negative_paths_and_callable,
        test_is_handler_type_branches,
        test_is_type_non_empty_unknown_and_tuple_and_fallback,
        test_is_type_protocol_fallback_branches,
        test_non_empty_and_normalize_branches,
        test_protocol_and_simple_guard_helpers,
    )
    from tests.unit.test_utilities_mapper_coverage_100 import (
        SimpleObj,
        TestuMapperAccessors,
        TestuMapperAdvanced,
        TestuMapperBuild,
        TestuMapperConversions,
        TestuMapperExtract,
        TestuMapperUtils,
    )
    from tests.unit.test_utilities_mapper_full_coverage import (
        AttrObject,
        BadBool,
        BadMapping,
        BadString,
        ExplodingLenList,
        mapper,
        test_accessor_take_pick_as_or_flat_and_agg_branches,
        test_at_take_and_as_branches,
        test_bad_string_and_bad_bool_raise_value_error,
        test_build_apply_transform_and_process_error_paths,
        test_construct_transform_and_deep_eq_branches,
        test_conversion_and_extract_success_branches,
        test_convert_default_fallback_matrix,
        test_convert_sequence_branch_returns_tuple,
        test_ensure_and_extract_array_index_helpers,
        test_extract_error_paths_and_prop_accessor,
        test_extract_field_value_and_ensure_variants,
        test_field_and_fields_multi_branches,
        test_filter_map_normalize_convert_helpers,
        test_general_value_helpers_and_logger,
        test_group_sort_unique_slice_chunk_branches,
        test_invert_and_json_conversion_branches,
        test_map_flags_collect_and_invert_branches,
        test_narrow_to_string_keyed_dict_and_mapping_paths,
        test_process_context_data_and_related_convenience,
        test_remaining_build_fields_construct_and_eq_paths,
        test_remaining_uncovered_branches,
        test_small_mapper_convenience_methods,
        test_transform_option_extract_and_step_helpers,
        test_type_guards_and_narrowing_failures,
    )
    from tests.unit.test_utilities_model_full_coverage import (
        test_merge_defaults_and_dump_paths,
        test_normalize_to_pydantic_dict_and_value_branches,
        test_update_exception_path,
        test_update_success_path_returns_ok_result,
    )
    from tests.unit.test_utilities_pagination_full_coverage import (
        test_pagination_response_string_fallbacks,
    )
    from tests.unit.test_utilities_parser_full_coverage import (
        test_parser_convert_and_norm_branches,
        test_parser_internal_helpers_additional_coverage,
        test_parser_parse_helpers_and_primitive_coercion_branches,
        test_parser_pipeline_and_pattern_branches,
        test_parser_remaining_branch_paths,
        test_parser_safe_length_and_parse_delimited_error_paths,
        test_parser_split_and_normalize_exception_paths,
        test_parser_success_and_edge_paths_cover_major_branches,
    )
    from tests.unit.test_utilities_reliability import TestFlextUtilitiesReliability
    from tests.unit.test_utilities_reliability_full_coverage import (
        test_utilities_reliability_branches,
        test_utilities_reliability_compose_returns_non_result_directly,
        test_utilities_reliability_uncovered_retry_compose_and_sequence_paths,
    )
    from tests.unit.test_utilities_string_parser import (
        StringParserTestFactory,
        TestuStringParser,
    )
    from tests.unit.test_utilities_text_full_coverage import (
        TestCleanText,
        TestFormatAppId,
        TestSafeString,
        TestTextLogger,
    )
    from tests.unit.test_utilities_type_checker_coverage_100 import (
        DictHandler,
        ExplicitTypeHandler,
        GenericHandler,
        IntHandler,
        NoHandleMethod,
        NonCallableHandle,
        ObjectHandler,
        StringHandler,
        TestuTypeChecker,
        TMessage,
    )
    from tests.unit.test_utilities_type_guards_coverage_100 import (
        TestuTypeGuardsIsDictNonEmpty,
        TestuTypeGuardsIsListNonEmpty,
        TestuTypeGuardsIsStringNonEmpty,
        TestuTypeGuardsNormalizeToMetadataValue,
        TypeGuardsScenarios,
    )
    from tests.unit.test_version import TestFlextVersion
    from tests.utilities import TestsFlextUtilities, u

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AlgarOudMigConstants": (
        "tests.integration.test_refactor_policy_mro",
        "AlgarOudMigConstants",
    ),
    "AlgarOudMigModels": (
        "tests.integration.test_refactor_policy_mro",
        "AlgarOudMigModels",
    ),
    "AlgarOudMigProtocols": (
        "tests.integration.test_refactor_policy_mro",
        "AlgarOudMigProtocols",
    ),
    "AlgarOudMigTypes": (
        "tests.integration.test_refactor_policy_mro",
        "AlgarOudMigTypes",
    ),
    "AlgarOudMigUtilities": (
        "tests.integration.test_refactor_policy_mro",
        "AlgarOudMigUtilities",
    ),
    "AssertionBuilder": (
        "tests.integration.patterns.test_patterns_testing",
        "AssertionBuilder",
    ),
    "AssertionHelpers": ("tests.test_utils", "AssertionHelpers"),
    "AttrObject": ("tests.unit.test_utilities_mapper_full_coverage", "AttrObject"),
    "AutoCommand": ("tests.unit.test_dispatcher_minimal", "AutoCommand"),
    "AutoDiscoveryHandler": (
        "tests.unit.test_dispatcher_minimal",
        "AutoDiscoveryHandler",
    ),
    "BadBool": ("tests.unit.test_utilities_mapper_full_coverage", "BadBool"),
    "BadMapping": ("tests.unit.test_utilities_mapper_full_coverage", "BadMapping"),
    "BadSingletonForTest": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "BadSingletonForTest",
    ),
    "BadString": ("tests.unit.test_utilities_mapper_full_coverage", "BadString"),
    "BrokenScenario": ("tests.infra.workspace_scenarios", "BrokenScenario"),
    "CacheScenarios": (
        "tests.unit.test_utilities_cache_coverage_100",
        "CacheScenarios",
    ),
    "CheckProjectStub": (
        "tests.infra.unit.check.extended_run_projects",
        "CheckProjectStub",
    ),
    "CoerceListValidatorScenario": (
        "tests.unit.test_collection_utilities_coverage_100",
        "CoerceListValidatorScenario",
    ),
    "CoerceValidatorScenario": (
        "tests.unit.test_enum_utilities_coverage_100",
        "CoerceValidatorScenario",
    ),
    "CollectionScenarios": (
        "tests.unit.test_collection_utilities_coverage_100",
        "CollectionScenarios",
    ),
    "ConcreteTestHandler": ("tests.unit.test_registry", "ConcreteTestHandler"),
    "ConcreteTestService": ("tests.unit.test_service_bootstrap", "ConcreteTestService"),
    "ConfigTestCase": ("tests.integration.test_config_integration", "ConfigTestCase"),
    "ConfigTestFactories": (
        "tests.integration.test_config_integration",
        "ConfigTestFactories",
    ),
    "ConfigWithoutModelConfigForTest": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "ConfigWithoutModelConfigForTest",
    ),
    "ContainerModelsScenarios": (
        "tests.unit.test_models_container",
        "ContainerModelsScenarios",
    ),
    "CreateUserCommand": (
        "tests.integration.patterns.test_patterns_commands",
        "CreateUserCommand",
    ),
    "CreateUserCommandHandler": (
        "tests.integration.patterns.test_patterns_commands",
        "CreateUserCommandHandler",
    ),
    "DataclassConfigForTest": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "DataclassConfigForTest",
    ),
    "DependencyScenario": ("tests.infra.scenarios", "DependencyScenario"),
    "DependencyScenarios": ("tests.infra.scenarios", "DependencyScenarios"),
    "DictHandler": (
        "tests.unit.test_utilities_type_checker_coverage_100",
        "DictHandler",
    ),
    "EchoHandler": ("tests.unit.test_dispatcher_minimal", "EchoHandler"),
    "EmptyScenario": ("tests.infra.workspace_scenarios", "EmptyScenario"),
    "EngineSafetyStub": (
        "tests.infra.unit.refactor.test_infra_refactor_safety",
        "EngineSafetyStub",
    ),
    "EnumScenarios": ("tests.unit.test_enum_utilities_coverage_100", "EnumScenarios"),
    "EventHandler": ("tests.unit.test_dispatcher_full_coverage", "EventHandler"),
    "EventSubscriber": ("tests.unit.test_dispatcher_minimal", "EventSubscriber"),
    "ExplicitTypeHandler": (
        "tests.unit.test_utilities_type_checker_coverage_100",
        "ExplicitTypeHandler",
    ),
    "ExplodingGetattr": ("tests.unit.test_result_additional", "ExplodingGetattr"),
    "ExplodingHandler": ("tests.unit.test_dispatcher_minimal", "ExplodingHandler"),
    "ExplodingLenList": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "ExplodingLenList",
    ),
    "ExtractPageParamsScenario": (
        "tests.unit.test_pagination_coverage_100",
        "ExtractPageParamsScenario",
    ),
    "FailingCommand": (
        "tests.integration.patterns.test_patterns_commands",
        "FailingCommand",
    ),
    "FailingCommandHandler": (
        "tests.integration.patterns.test_patterns_commands",
        "FailingCommandHandler",
    ),
    "FailingOptionsForTest": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "FailingOptionsForTest",
    ),
    "FailingService": ("tests.helpers.factories", "FailingService"),
    "FailingServiceAuto": ("tests.helpers.factories", "FailingServiceAuto"),
    "FailingServiceAutoFactory": (
        "tests.helpers.factories",
        "FailingServiceAutoFactory",
    ),
    "FailingServiceFactory": ("tests.helpers.factories", "FailingServiceFactory"),
    "FixtureBuilder": (
        "tests.integration.patterns.test_patterns_testing",
        "FixtureBuilder",
    ),
    "FlextCliConstants": (
        "tests.integration.test_refactor_policy_mro",
        "FlextCliConstants",
    ),
    "FlextCliModels": ("tests.integration.test_refactor_policy_mro", "FlextCliModels"),
    "FlextCliProtocols": (
        "tests.integration.test_refactor_policy_mro",
        "FlextCliProtocols",
    ),
    "FlextCliTypes": ("tests.integration.test_refactor_policy_mro", "FlextCliTypes"),
    "FlextCliUtilities": (
        "tests.integration.test_refactor_policy_mro",
        "FlextCliUtilities",
    ),
    "FlextCommandId": (
        "tests.integration.patterns.test_patterns_commands",
        "FlextCommandId",
    ),
    "FlextCommandType": (
        "tests.integration.patterns.test_patterns_commands",
        "FlextCommandType",
    ),
    "FlextConsolidationContext": ("tests.conftest", "FlextConsolidationContext"),
    "FlextInfraTestConstants": ("tests.infra.constants", "FlextInfraTestConstants"),
    "FlextInfraTestHelpers": ("tests.infra.helpers", "FlextInfraTestHelpers"),
    "FlextInfraTestModels": ("tests.infra.models", "FlextInfraTestModels"),
    "FlextInfraTestProtocols": ("tests.infra.protocols", "FlextInfraTestProtocols"),
    "FlextInfraTestTypes": ("tests.infra.typings", "FlextInfraTestTypes"),
    "FlextInfraTestUtilities": ("tests.infra.utilities", "FlextInfraTestUtilities"),
    "FlextLdapConstants": (
        "tests.integration.test_refactor_policy_mro",
        "FlextLdapConstants",
    ),
    "FlextLdapModels": (
        "tests.integration.test_refactor_policy_mro",
        "FlextLdapModels",
    ),
    "FlextLdapProtocols": (
        "tests.integration.test_refactor_policy_mro",
        "FlextLdapProtocols",
    ),
    "FlextLdapTypes": ("tests.integration.test_refactor_policy_mro", "FlextLdapTypes"),
    "FlextLdapUtilities": (
        "tests.integration.test_refactor_policy_mro",
        "FlextLdapUtilities",
    ),
    "rAssertionHelper": ("tests.conftest", "rAssertionHelper"),
    "FlextScenarioRunner": ("tests.conftest", "FlextScenarioRunner"),
    "FlextTestAutomationFramework": ("tests.conftest", "FlextTestAutomationFramework"),
    "FlextTestBuilder": (
        "tests.integration.patterns.test_patterns_testing",
        "FlextTestBuilder",
    ),
    "FullScenario": ("tests.infra.workspace_scenarios", "FullScenario"),
    "FunctionalExternalService": ("tests.conftest", "FunctionalExternalService"),
    "GenericHandler": (
        "tests.unit.test_utilities_type_checker_coverage_100",
        "GenericHandler",
    ),
    "GenericModelFactory": ("tests.helpers.factories", "GenericModelFactory"),
    "GetUserService": ("tests.test_documented_patterns", "GetUserService"),
    "GetUserServiceAuto": ("tests.helpers.factories", "GetUserServiceAuto"),
    "GetUserServiceAutoFactory": (
        "tests.helpers.factories",
        "GetUserServiceAutoFactory",
    ),
    "GetUserServiceFactory": ("tests.helpers.factories", "GetUserServiceFactory"),
    "GitScenario": ("tests.infra.scenarios", "GitScenario"),
    "GitScenarios": ("tests.infra.scenarios", "GitScenarios"),
    "GivenWhenThenBuilder": (
        "tests.integration.patterns.test_patterns_testing",
        "GivenWhenThenBuilder",
    ),
    "IntHandler": ("tests.unit.test_utilities_type_checker_coverage_100", "IntHandler"),
    "IsMemberScenario": (
        "tests.unit.test_enum_utilities_coverage_100",
        "IsMemberScenario",
    ),
    "IsSubsetScenario": (
        "tests.unit.test_enum_utilities_coverage_100",
        "IsSubsetScenario",
    ),
    "LifecycleService": ("tests.integration.test_service", "LifecycleService"),
    "MinimalScenario": ("tests.infra.workspace_scenarios", "MinimalScenario"),
    "MissingType": ("tests.unit.test_utilities_checker_full_coverage", "MissingType"),
    "MockScanner": ("tests.infra.unit._utilities.test_scanning", "MockScanner"),
    "MockScenario": (
        "tests.integration.patterns.test_patterns_testing",
        "MockScenario",
    ),
    "MultiOperationService": (
        "tests.test_documented_patterns",
        "MultiOperationService",
    ),
    "NestedClassPropagationTransformer": (
        "tests.unit.test_transformer_nested_class_propagation",
        "NestedClassPropagationTransformer",
    ),
    "NoHandleMethod": (
        "tests.unit.test_utilities_type_checker_coverage_100",
        "NoHandleMethod",
    ),
    "NonCallableHandle": (
        "tests.unit.test_utilities_type_checker_coverage_100",
        "NonCallableHandle",
    ),
    "NotificationService": ("tests.integration.test_service", "NotificationService"),
    "ObjectHandler": (
        "tests.unit.test_utilities_type_checker_coverage_100",
        "ObjectHandler",
    ),
    "OptionsModelForTest": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "OptionsModelForTest",
    ),
    "PaginationScenarios": (
        "tests.unit.test_pagination_coverage_100",
        "PaginationScenarios",
    ),
    "ParameterizedTestBuilder": (
        "tests.integration.patterns.test_patterns_testing",
        "ParameterizedTestBuilder",
    ),
    "ParseMappingScenario": (
        "tests.unit.test_collection_utilities_coverage_100",
        "ParseMappingScenario",
    ),
    "ParseOrDefaultScenario": (
        "tests.unit.test_enum_utilities_coverage_100",
        "ParseOrDefaultScenario",
    ),
    "ParseScenario": ("tests.unit.test_enum_utilities_coverage_100", "ParseScenario"),
    "ParseSequenceScenario": (
        "tests.unit.test_collection_utilities_coverage_100",
        "ParseSequenceScenario",
    ),
    "ParserScenario": ("tests.helpers.scenarios", "ParserScenario"),
    "ParserScenarios": ("tests.helpers.scenarios", "ParserScenarios"),
    "PerformanceBenchmark": (
        "tests.benchmark.test_container_performance",
        "PerformanceBenchmark",
    ),
    "PreparePaginationDataScenario": (
        "tests.unit.test_pagination_coverage_100",
        "PreparePaginationDataScenario",
    ),
    "Priority": ("tests.unit.test_utilities_enum_full_coverage", "Priority"),
    "QueryHandler": ("tests.unit.test_dispatcher_full_coverage", "QueryHandler"),
    "RailwayTestCase": ("tests.test_documented_patterns", "RailwayTestCase"),
    "RealGitService": ("tests.infra.git_service", "RealGitService"),
    "RealSubprocessRunner": ("tests.infra.runner_service", "RealSubprocessRunner"),
    "ReliabilityScenario": ("tests.helpers.scenarios", "ReliabilityScenario"),
    "ReliabilityScenarios": ("tests.helpers.scenarios", "ReliabilityScenarios"),
    "RunCallable": ("tests.infra.unit.check.extended_runners_ruff", "RunCallable"),
    "RunStub": ("tests.infra.unit.check.extended_error_reporting", "RunStub"),
    "RuntimeCloneService": (
        "tests.unit.test_service_additional",
        "RuntimeCloneService",
    ),
    "SampleCommand": ("tests.unit.test_dispatcher_full_coverage", "SampleCommand"),
    "SampleEvent": ("tests.unit.test_dispatcher_full_coverage", "SampleEvent"),
    "SampleHandler": ("tests.unit.test_dispatcher_full_coverage", "SampleHandler"),
    "SampleQuery": ("tests.unit.test_dispatcher_full_coverage", "SampleQuery"),
    "SendEmailService": ("tests.test_documented_patterns", "SendEmailService"),
    "ServiceConfig": ("tests.integration.test_service", "ServiceConfig"),
    "ServiceFactoryRegistry": ("tests.helpers.factories", "ServiceFactoryRegistry"),
    "ServiceTestCase": ("tests.test_documented_patterns", "ServiceTestCase"),
    "ServiceTestCaseFactory": ("tests.helpers.factories", "ServiceTestCaseFactory"),
    "ServiceTestCases": ("tests.helpers.factories", "ServiceTestCases"),
    "SetupFn": ("tests.infra.unit.test_infra_workspace_sync", "SetupFn"),
    "SimpleObj": ("tests.unit.test_utilities_mapper_coverage_100", "SimpleObj"),
    "SingletonWithoutGetGlobalForTest": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "SingletonWithoutGetGlobalForTest",
    ),
    "SingletonWithoutModelDumpForTest": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "SingletonWithoutModelDumpForTest",
    ),
    "Status": ("tests.unit.test_utilities_enum_full_coverage", "Status"),
    "StrictOptionsForTest": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "StrictOptionsForTest",
    ),
    "StringHandler": (
        "tests.unit.test_utilities_type_checker_coverage_100",
        "StringHandler",
    ),
    "StringParserTestFactory": (
        "tests.unit.test_utilities_string_parser",
        "StringParserTestFactory",
    ),
    "SubprocessScenario": ("tests.infra.scenarios", "SubprocessScenario"),
    "SubprocessScenarios": ("tests.infra.scenarios", "SubprocessScenarios"),
    "SuiteBuilder": (
        "tests.integration.patterns.test_patterns_testing",
        "SuiteBuilder",
    ),
    "T": ("tests.typings", "T"),
    "TMessage": ("tests.unit.test_utilities_type_checker_coverage_100", "TMessage"),
    "T_co": ("tests.typings", "T_co"),
    "T_contra": ("tests.typings", "T_contra"),
    "TestAdrHelpers": ("tests.infra.unit.docs.validator_internals", "TestAdrHelpers"),
    "TestAdvancedPatterns": (
        "tests.integration.patterns.test_patterns_testing",
        "TestAdvancedPatterns",
    ),
    "TestAggregateRoots": ("tests.unit.test_coverage_models", "TestAggregateRoots"),
    "TestAllDirectoriesScanned": (
        "tests.infra.unit.codegen.lazy_init_tests",
        "TestAllDirectoriesScanned",
    ),
    "TestAllPatternsIntegration": (
        "tests.test_documented_patterns",
        "TestAllPatternsIntegration",
    ),
    "TestAltPropagatesException": (
        "tests.unit.test_result_exception_carrying",
        "TestAltPropagatesException",
    ),
    "TestAssertExists": ("tests.unit.flext_tests.test_files", "TestAssertExists"),
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
    "TestAutomatedFlextContainer": (
        "tests.unit.test_automated_container",
        "TestAutomatedFlextContainer",
    ),
    "TestAutomatedFlextContext": (
        "tests.unit.test_automated_context",
        "TestAutomatedFlextContext",
    ),
    "TestAutomatedFlextDecorators": (
        "tests.unit.test_automated_decorators",
        "TestAutomatedFlextDecorators",
    ),
    "TestAutomatedFlextDispatcher": (
        "tests.unit.test_automated_dispatcher",
        "TestAutomatedFlextDispatcher",
    ),
    "TestAutomatedFlextExceptions": (
        "tests.unit.test_automated_exceptions",
        "TestAutomatedFlextExceptions",
    ),
    "TestAutomatedFlextHandlers": (
        "tests.unit.test_automated_handlers",
        "TestAutomatedFlextHandlers",
    ),
    "TestAutomatedFlextLoggings": (
        "tests.unit.test_automated_loggings",
        "TestAutomatedFlextLoggings",
    ),
    "TestAutomatedFlextMixins": (
        "tests.unit.test_automated_mixins",
        "TestAutomatedFlextMixins",
    ),
    "TestAutomatedFlextRegistry": (
        "tests.unit.test_automated_registry",
        "TestAutomatedFlextRegistry",
    ),
    "TestAutomatedr": (
        "tests.unit.test_automated_result",
        "TestAutomatedr",
    ),
    "TestAutomatedFlextRuntime": (
        "tests.unit.test_automated_runtime",
        "TestAutomatedFlextRuntime",
    ),
    "TestAutomatedFlextService": (
        "tests.unit.test_automated_service",
        "TestAutomatedFlextService",
    ),
    "TestAutomatedFlextSettings": (
        "tests.unit.test_automated_settings",
        "TestAutomatedFlextSettings",
    ),
    "TestAutomatedFlextUtilities": (
        "tests.unit.test_automated_utilities",
        "TestAutomatedFlextUtilities",
    ),
    "TestBackwardCompatDiscardReturnValue": (
        "tests.unit.test_loggings_strict_returns",
        "TestBackwardCompatDiscardReturnValue",
    ),
    "TestBackwardCompatibility": (
        "tests.integration.test_migration_validation",
        "TestBackwardCompatibility",
    ),
    "TestBaseMkGenerationFlow": (
        "tests.integration.test_infra_integration",
        "TestBaseMkGenerationFlow",
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
    "TestBatchOperations": ("tests.unit.flext_tests.test_files", "TestBatchOperations"),
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
    "TestCleanText": ("tests.unit.test_utilities_text_full_coverage", "TestCleanText"),
    "TestCloneContainer": (
        "tests.unit.test_utilities_context_full_coverage",
        "TestCloneContainer",
    ),
    "TestCloneRuntime": (
        "tests.unit.test_utilities_context_full_coverage",
        "TestCloneRuntime",
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
    "TestCommands": ("tests.unit.test_coverage_models", "TestCommands"),
    "TestCompleteFlextSystemIntegration": (
        "tests.integration.test_system",
        "TestCompleteFlextSystemIntegration",
    ),
    "TestComprehensiveIntegration": (
        "tests.integration.patterns.test_patterns_testing",
        "TestComprehensiveIntegration",
    ),
    "TestConfig": ("tests.unit.flext_tests.test_factories", "TestConfig"),
    "TestConfigConstants": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "TestConfigConstants",
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
    "TestConfigMapDictOps": (
        "tests.unit.test_typings_full_coverage",
        "TestConfigMapDictOps",
    ),
    "TestConfigModels": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "TestConfigModels",
    ),
    "TestConfigServiceViaDI": (
        "tests.unit.test_di_services_access",
        "TestConfigServiceViaDI",
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
    "TestContainerDIRealExecution": (
        "tests.unit.test_di_incremental",
        "TestContainerDIRealExecution",
    ),
    "TestContainerInfo": ("tests.unit.flext_tests.test_docker", "TestContainerInfo"),
    "TestContainerIntegration": (
        "tests.integration.test_infra_integration",
        "TestContainerIntegration",
    ),
    "TestContainerMemory": (
        "tests.benchmark.test_container_memory",
        "TestContainerMemory",
    ),
    "TestContainerPerformance": (
        "tests.benchmark.test_container_performance",
        "TestContainerPerformance",
    ),
    "TestContainerStatus": (
        "tests.unit.flext_tests.test_docker",
        "TestContainerStatus",
    ),
    "TestContext100Coverage": (
        "tests.unit.test_context_coverage_100",
        "TestContext100Coverage",
    ),
    "TestContextDataModel": (
        "tests.unit.test_coverage_context",
        "TestContextDataModel",
    ),
    "TestContextServiceViaDI": (
        "tests.unit.test_di_services_access",
        "TestContextServiceViaDI",
    ),
    "TestCoreModuleInit": ("tests.infra.unit.core.init", "TestCoreModuleInit"),
    "TestCorrelationDomain": (
        "tests.unit.test_coverage_context",
        "TestCorrelationDomain",
    ),
    "TestCoveragePush75Percent": (
        "tests.unit.test_final_75_percent_push",
        "TestCoveragePush75Percent",
    ),
    "TestCreate": ("tests.infra.unit.github.pr", "TestCreate"),
    "TestCreateBranches": (
        "tests.infra.unit.release.orchestrator_git",
        "TestCreateBranches",
    ),
    "TestCreateDatetimeProxy": (
        "tests.unit.test_utilities_context_full_coverage",
        "TestCreateDatetimeProxy",
    ),
    "TestCreateDictProxy": (
        "tests.unit.test_utilities_context_full_coverage",
        "TestCreateDictProxy",
    ),
    "TestCreateFromCallableCarriesException": (
        "tests.unit.test_result_exception_carrying",
        "TestCreateFromCallableCarriesException",
    ),
    "TestCreateInStatic": ("tests.unit.flext_tests.test_files", "TestCreateInStatic"),
    "TestCreateStrProxy": (
        "tests.unit.test_utilities_context_full_coverage",
        "TestCreateStrProxy",
    ),
    "TestCreateTag": ("tests.infra.unit.release.orchestrator_git", "TestCreateTag"),
    "TestCriticalReturnsResultBool": (
        "tests.unit.test_loggings_strict_returns",
        "TestCriticalReturnsResultBool",
    ),
    "TestCrossModuleIntegration": (
        "tests.integration.test_infra_integration",
        "TestCrossModuleIntegration",
    ),
    "TestDIBridgeRealExecution": (
        "tests.unit.test_di_incremental",
        "TestDIBridgeRealExecution",
    ),
    "TestDataFactory": ("tests.test_utils", "TestDataFactory"),
    "TestDataGenerators": ("tests.helpers.factories", "TestDataGenerators"),
    "TestDebugReturnsResultBool": (
        "tests.unit.test_loggings_strict_returns",
        "TestDebugReturnsResultBool",
    ),
    "TestDependencyIntegrationRealExecution": (
        "tests.unit.test_di_incremental",
        "TestDependencyIntegrationRealExecution",
    ),
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
    "TestDictMixinOperations": (
        "tests.unit.test_typings_full_coverage",
        "TestDictMixinOperations",
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
    "TestDispatcherDI": ("tests.unit.test_dispatcher_di", "TestDispatcherDI"),
    "TestDomainEvents": ("tests.unit.test_coverage_models", "TestDomainEvents"),
    "TestDomainHashValue": (
        "tests.unit.test_utilities_domain_full_coverage",
        "TestDomainHashValue",
    ),
    "TestDomainLogger": (
        "tests.unit.test_utilities_domain_full_coverage",
        "TestDomainLogger",
    ),
    "TestDomainResult": ("tests.unit.test_service_coverage_100", "TestDomainResult"),
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
    "TestEnterprisePatterns": (
        "tests.integration.patterns.test_architectural_patterns",
        "TestEnterprisePatterns",
    ),
    "TestEntities": ("tests.unit.test_coverage_models", "TestEntities"),
    "TestEntityCoverageEdgeCases": (
        "tests.unit.test_entity_coverage",
        "TestEntityCoverageEdgeCases",
    ),
    "TestErrorOrPatternUnchanged": (
        "tests.unit.test_result_exception_carrying",
        "TestErrorOrPatternUnchanged",
    ),
    "TestErrorReporting": (
        "tests.infra.unit.check.extended_error_reporting",
        "TestErrorReporting",
    ),
    "TestErrorReturnsResultBool": (
        "tests.unit.test_loggings_strict_returns",
        "TestErrorReturnsResultBool",
    ),
    "TestEventDrivenPatterns": (
        "tests.integration.patterns.test_architectural_patterns",
        "TestEventDrivenPatterns",
    ),
    "TestExceptionContext": (
        "tests.unit.test_coverage_exceptions",
        "TestExceptionContext",
    ),
    "TestExceptionEdgeCases": (
        "tests.unit.test_coverage_exceptions",
        "TestExceptionEdgeCases",
    ),
    "TestExceptionFactory": (
        "tests.unit.test_coverage_exceptions",
        "TestExceptionFactory",
    ),
    "TestExceptionIntegration": (
        "tests.unit.test_coverage_exceptions",
        "TestExceptionIntegration",
    ),
    "TestExceptionLogging": (
        "tests.unit.test_coverage_loggings",
        "TestExceptionLogging",
    ),
    "TestExceptionMetrics": (
        "tests.unit.test_coverage_exceptions",
        "TestExceptionMetrics",
    ),
    "TestExceptionProperties": (
        "tests.unit.test_coverage_exceptions",
        "TestExceptionProperties",
    ),
    "TestExceptionPropertyAccess": (
        "tests.unit.test_result_exception_carrying",
        "TestExceptionPropertyAccess",
    ),
    "TestExceptionReturnsResultBool": (
        "tests.unit.test_loggings_strict_returns",
        "TestExceptionReturnsResultBool",
    ),
    "TestExceptionSerialization": (
        "tests.unit.test_coverage_exceptions",
        "TestExceptionSerialization",
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
    "TestFactories": ("tests.test_documented_patterns", "TestFactories"),
    "TestFactoriesHelpers": (
        "tests.unit.flext_tests.test_factories",
        "TestFactoriesHelpers",
    ),
    "TestFactoryDecoratorsDiscoveryHasFactories": (
        "tests.unit.test_decorators_discovery_full_coverage",
        "TestFactoryDecoratorsDiscoveryHasFactories",
    ),
    "TestFactoryDecoratorsDiscoveryScanModule": (
        "tests.unit.test_decorators_discovery_full_coverage",
        "TestFactoryDecoratorsDiscoveryScanModule",
    ),
    "TestFactoryPatterns": ("tests.unit.test_coverage_loggings", "TestFactoryPatterns"),
    "TestFailNoExceptionBackwardCompat": (
        "tests.unit.test_result_exception_carrying",
        "TestFailNoExceptionBackwardCompat",
    ),
    "TestFailWithException": (
        "tests.unit.test_result_exception_carrying",
        "TestFailWithException",
    ),
    "TestFileInfo": ("tests.unit.flext_tests.test_files", "TestFileInfo"),
    "TestFileInfoFromModels": (
        "tests.unit.flext_tests.test_files",
        "TestFileInfoFromModels",
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
    "TestFixtureFactory": ("tests.test_utils", "TestFixtureFactory"),
    "TestFlatMapPropagatesException": (
        "tests.unit.test_result_exception_carrying",
        "TestFlatMapPropagatesException",
    ),
    "TestFlextCommand": (
        "tests.integration.patterns.test_patterns_commands",
        "TestFlextCommand",
    ),
    "TestFlextCommandHandler": (
        "tests.integration.patterns.test_patterns_commands",
        "TestFlextCommandHandler",
    ),
    "TestFlextCommandResults": (
        "tests.integration.patterns.test_patterns_commands",
        "TestFlextCommandResults",
    ),
    "TestFlextConstants": ("tests.unit.test_constants", "TestFlextConstants"),
    "TestFlextContainer": ("tests.unit.test_container", "TestFlextContainer"),
    "TestFlextContext": (
        "tests.integration.patterns.test_patterns_logging",
        "TestFlextContext",
    ),
    "TestFlextDecorators": ("tests.unit.test_decorators", "TestFlextDecorators"),
    "TestFlextExceptionsHierarchy": (
        "tests.unit.test_coverage_exceptions",
        "TestFlextExceptionsHierarchy",
    ),
    "TestFlextHandlers": ("tests.unit.test_handlers", "TestFlextHandlers"),
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
    "TestFlextInfraNamespaceValidator": (
        "tests.unit.test_namespace_validator",
        "TestFlextInfraNamespaceValidator",
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
    "TestFlextLogLevel": (
        "tests.integration.patterns.test_patterns_logging",
        "TestFlextLogLevel",
    ),
    "TestFlextLogger": (
        "tests.integration.patterns.test_patterns_logging",
        "TestFlextLogger",
    ),
    "TestFlextLoggerIntegration": (
        "tests.integration.patterns.test_patterns_logging",
        "TestFlextLoggerIntegration",
    ),
    "TestFlextLoggerUsage": (
        "tests.integration.patterns.test_patterns_logging",
        "TestFlextLoggerUsage",
    ),
    "TestFlextMixinsNestedClasses": (
        "tests.unit.test_mixins",
        "TestFlextMixinsNestedClasses",
    ),
    "TestFlextModels": ("tests.unit.test_models", "TestFlextModels"),
    "TestFlextModelsAggregateRoot": (
        "tests.unit.test_models_79_coverage",
        "TestFlextModelsAggregateRoot",
    ),
    "TestFlextModelsCollectionsCategories": (
        "tests.unit.test_collections_coverage_100",
        "TestFlextModelsCollectionsCategories",
    ),
    "TestFlextModelsCollectionsOptions": (
        "tests.unit.test_collections_coverage_100",
        "TestFlextModelsCollectionsOptions",
    ),
    "TestFlextModelsCollectionsResults": (
        "tests.unit.test_collections_coverage_100",
        "TestFlextModelsCollectionsResults",
    ),
    "TestFlextModelsCollectionsSettings": (
        "tests.unit.test_collections_coverage_100",
        "TestFlextModelsCollectionsSettings",
    ),
    "TestFlextModelsCollectionsStatistics": (
        "tests.unit.test_collections_coverage_100",
        "TestFlextModelsCollectionsStatistics",
    ),
    "TestFlextModelsCommand": (
        "tests.unit.test_models_79_coverage",
        "TestFlextModelsCommand",
    ),
    "TestFlextModelsContainer": (
        "tests.unit.test_models_container",
        "TestFlextModelsContainer",
    ),
    "TestFlextModelsDomainEvent": (
        "tests.unit.test_models_79_coverage",
        "TestFlextModelsDomainEvent",
    ),
    "TestFlextModelsEdgeCases": (
        "tests.unit.test_models_79_coverage",
        "TestFlextModelsEdgeCases",
    ),
    "TestFlextModelsEntity": (
        "tests.unit.test_models_79_coverage",
        "TestFlextModelsEntity",
    ),
    "TestFlextModelsIntegration": (
        "tests.unit.test_models_79_coverage",
        "TestFlextModelsIntegration",
    ),
    "TestFlextModelsQuery": (
        "tests.unit.test_models_79_coverage",
        "TestFlextModelsQuery",
    ),
    "TestFlextModelsValue": (
        "tests.unit.test_models_79_coverage",
        "TestFlextModelsValue",
    ),
    "TestFlextProtocols": ("tests.unit.test_protocols", "TestFlextProtocols"),
    "TestFlextRegistry": ("tests.unit.test_registry", "TestFlextRegistry"),
    "TestFlextRuntime": ("tests.unit.test_runtime", "TestFlextRuntime"),
    "TestFlextServiceIntegration": (
        "tests.integration.test_service",
        "TestFlextServiceIntegration",
    ),
    "TestFlextSettings": ("tests.unit.test_config", "TestFlextSettings"),
    "TestFlextSettingsSingletonIntegration": (
        "tests.integration.test_config_integration",
        "TestFlextSettingsSingletonIntegration",
    ),
    "TestFlextTestsBuilders": (
        "tests.unit.flext_tests.test_builders",
        "TestFlextTestsBuilders",
    ),
    "TestFlextTestsDocker": (
        "tests.unit.flext_tests.test_docker",
        "TestFlextTestsDocker",
    ),
    "TestFlextTestsDockerWorkerId": (
        "tests.unit.flext_tests.test_docker",
        "TestFlextTestsDockerWorkerId",
    ),
    "TestFlextTestsDockerWorkspaceRoot": (
        "tests.unit.flext_tests.test_docker",
        "TestFlextTestsDockerWorkspaceRoot",
    ),
    "TestFlextTestsDomains": (
        "tests.unit.flext_tests.test_domains",
        "TestFlextTestsDomains",
    ),
    "TestFlextTestsFactoriesModernAPI": (
        "tests.unit.flext_tests.test_factories",
        "TestFlextTestsFactoriesModernAPI",
    ),
    "TestFlextTestsFiles": ("tests.unit.flext_tests.test_files", "TestFlextTestsFiles"),
    "TestFlextTestsFilesNewApi": (
        "tests.unit.flext_tests.test_files",
        "TestFlextTestsFilesNewApi",
    ),
    "TestFlextTestsMatchers": (
        "tests.unit.flext_tests.test_matchers",
        "TestFlextTestsMatchers",
    ),
    "TestFlextTestsUtilitiesFactory": (
        "tests.unit.flext_tests.test_utilities",
        "TestFlextTestsUtilitiesFactory",
    ),
    "TestFlextTestsUtilitiesResult": (
        "tests.unit.flext_tests.test_utilities",
        "TestFlextTestsUtilitiesResult",
    ),
    "TestFlextTestsUtilitiesResultCompat": (
        "tests.unit.flext_tests.test_utilities",
        "TestFlextTestsUtilitiesResultCompat",
    ),
    "TestFlextTestsUtilitiesTestContext": (
        "tests.unit.flext_tests.test_utilities",
        "TestFlextTestsUtilitiesTestContext",
    ),
    "TestFlextTypings": ("tests.unit.test_typings", "TestFlextTypings"),
    "TestFlextUtilitiesArgs": (
        "tests.unit.test_args_coverage_100",
        "TestFlextUtilitiesArgs",
    ),
    "TestFlextUtilitiesConfiguration": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "TestFlextUtilitiesConfiguration",
    ),
    "TestFlextUtilitiesModelNormalizeToMetadata": (
        "tests.unit.test_models_container",
        "TestFlextUtilitiesModelNormalizeToMetadata",
    ),
    "TestFlextUtilitiesReliability": (
        "tests.unit.test_utilities_reliability",
        "TestFlextUtilitiesReliability",
    ),
    "TestFlextVersion": ("tests.unit.test_version", "TestFlextVersion"),
    "TestFormatAppId": (
        "tests.unit.test_utilities_text_full_coverage",
        "TestFormatAppId",
    ),
    "TestFormattingRunRuffFix": (
        "tests.infra.unit._utilities.test_formatting",
        "TestFormattingRunRuffFix",
    ),
    "TestFromIOResultCarriesException": (
        "tests.unit.test_result_exception_carrying",
        "TestFromIOResultCarriesException",
    ),
    "TestFromValidationCarriesException": (
        "tests.unit.test_result_exception_carrying",
        "TestFromValidationCarriesException",
    ),
    "TestFunction": (
        "tests.integration.patterns.test_advanced_patterns",
        "TestFunction",
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
    "TestGlobalContextManagement": (
        "tests.unit.test_coverage_loggings",
        "TestGlobalContextManagement",
    ),
    "TestGoFmtEmptyLinesInOutput": (
        "tests.infra.unit.check.extended_error_reporting",
        "TestGoFmtEmptyLinesInOutput",
    ),
    "TestHandleLazyInit": ("tests.infra.unit.codegen.main", "TestHandleLazyInit"),
    "TestHandlerDecoratorMetadata": (
        "tests.unit.test_handler_decorator_discovery",
        "TestHandlerDecoratorMetadata",
    ),
    "TestHandlerDiscoveryClass": (
        "tests.unit.test_handler_decorator_discovery",
        "TestHandlerDiscoveryClass",
    ),
    "TestHandlerDiscoveryEdgeCases": (
        "tests.unit.test_handler_decorator_discovery",
        "TestHandlerDiscoveryEdgeCases",
    ),
    "TestHandlerDiscoveryIntegration": (
        "tests.unit.test_handler_decorator_discovery",
        "TestHandlerDiscoveryIntegration",
    ),
    "TestHandlerDiscoveryModule": (
        "tests.unit.test_handler_decorator_discovery",
        "TestHandlerDiscoveryModule",
    ),
    "TestHandlerDiscoveryServiceIntegration": (
        "tests.unit.test_handler_decorator_discovery",
        "TestHandlerDiscoveryServiceIntegration",
    ),
    "TestHelperConsolidationTransformer": (
        "tests.unit.test_transformer_helper_consolidation",
        "TestHelperConsolidationTransformer",
    ),
    "TestHierarchicalExceptionSystem": (
        "tests.unit.test_coverage_exceptions",
        "TestHierarchicalExceptionSystem",
    ),
    "TestIdempotency": (
        "tests.integration.test_refactor_nesting_idempotency",
        "TestIdempotency",
    ),
    "TestInferOwnerFromOrigin": (
        "tests.infra.unit.deps.test_internal_sync_resolve",
        "TestInferOwnerFromOrigin",
    ),
    "TestInferPackage": (
        "tests.infra.unit.codegen.lazy_init_helpers",
        "TestInferPackage",
    ),
    "TestInfoReturnsResultBool": (
        "tests.unit.test_loggings_strict_returns",
        "TestInfoReturnsResultBool",
    ),
    "TestInfoWithContentMeta": (
        "tests.unit.flext_tests.test_files",
        "TestInfoWithContentMeta",
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
    "TestInstanceCreation": (
        "tests.unit.test_coverage_loggings",
        "TestInstanceCreation",
    ),
    "TestIntegrationWithRealCommandServices": (
        "tests.integration.test_infra_integration",
        "TestIntegrationWithRealCommandServices",
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
    "TestLashPropagatesException": (
        "tests.unit.test_result_exception_carrying",
        "TestLashPropagatesException",
    ),
    "TestLevelBasedContextManagement": (
        "tests.unit.test_coverage_loggings",
        "TestLevelBasedContextManagement",
    ),
    "TestLibraryIntegration": (
        "tests.integration.test_integration",
        "TestLibraryIntegration",
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
    "TestLogReturnsResultBool": (
        "tests.unit.test_loggings_strict_returns",
        "TestLogReturnsResultBool",
    ),
    "TestLoggerServiceViaDI": (
        "tests.unit.test_di_services_access",
        "TestLoggerServiceViaDI",
    ),
    "TestLoggingIntegration": (
        "tests.unit.test_coverage_loggings",
        "TestLoggingIntegration",
    ),
    "TestLoggingMethods": ("tests.unit.test_coverage_loggings", "TestLoggingMethods"),
    "TestLoggingsErrorPaths": (
        "tests.unit.test_loggings_error_paths_coverage",
        "TestLoggingsErrorPaths",
    ),
    "TestMain": ("tests.infra.unit.deps.test_path_sync_main", "TestMain"),
    "TestMainBaseMkValidate": ("tests.infra.unit.core.main", "TestMainBaseMkValidate"),
    "TestMainCli": ("tests.infra.unit.test_infra_workspace_main", "TestMainCli"),
    "TestMainCliRouting": ("tests.infra.unit.core.main", "TestMainCliRouting"),
    "TestMainCommandDispatch": (
        "tests.infra.unit.codegen.main",
        "TestMainCommandDispatch",
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
    "TestMainStructlogConfiguration": (
        "tests.infra.unit.deps.test_main_dispatch",
        "TestMainStructlogConfiguration",
    ),
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
    "TestMapPropagatesException": (
        "tests.unit.test_result_exception_carrying",
        "TestMapPropagatesException",
    ),
    "TestMapperBuildFlagsDict": (
        "tests.unit.test_utilities_data_mapper",
        "TestMapperBuildFlagsDict",
    ),
    "TestMapperCollectActiveKeys": (
        "tests.unit.test_utilities_data_mapper",
        "TestMapperCollectActiveKeys",
    ),
    "TestMapperFilterDict": (
        "tests.unit.test_utilities_data_mapper",
        "TestMapperFilterDict",
    ),
    "TestMapperInvertDict": (
        "tests.unit.test_utilities_data_mapper",
        "TestMapperInvertDict",
    ),
    "TestMapperMapDictKeys": (
        "tests.unit.test_utilities_data_mapper",
        "TestMapperMapDictKeys",
    ),
    "TestMapperTransformValues": (
        "tests.unit.test_utilities_data_mapper",
        "TestMapperTransformValues",
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
    "TestMetadata": ("tests.unit.test_coverage_models", "TestMetadata"),
    "TestMigrationComplexity": (
        "tests.integration.test_migration_validation",
        "TestMigrationComplexity",
    ),
    "TestMigrationScenario1": (
        "tests.integration.test_migration_validation",
        "TestMigrationScenario1",
    ),
    "TestMigrationScenario2": (
        "tests.integration.test_migration_validation",
        "TestMigrationScenario2",
    ),
    "TestMigrationScenario4": (
        "tests.integration.test_migration_validation",
        "TestMigrationScenario4",
    ),
    "TestMigrationScenario5": (
        "tests.integration.test_migration_validation",
        "TestMigrationScenario5",
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
    "TestModelIntegration": ("tests.unit.test_coverage_models", "TestModelIntegration"),
    "TestModelSerialization": (
        "tests.unit.test_coverage_models",
        "TestModelSerialization",
    ),
    "TestModelValidation": ("tests.unit.test_coverage_models", "TestModelValidation"),
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
    "TestMonadicOperationsUnchanged": (
        "tests.unit.test_result_exception_carrying",
        "TestMonadicOperationsUnchanged",
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
    "TestOkNoneGuardStillRaises": (
        "tests.unit.test_result_exception_carrying",
        "TestOkNoneGuardStillRaises",
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
    "TestOutputSingletonConsistency": (
        "tests.integration.test_infra_integration",
        "TestOutputSingletonConsistency",
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
    "TestPathResolverDiscoveryFlow": (
        "tests.integration.test_infra_integration",
        "TestPathResolverDiscoveryFlow",
    ),
    "TestPathSyncEdgeCases": (
        "tests.infra.unit.deps.test_path_sync_init",
        "TestPathSyncEdgeCases",
    ),
    "TestPattern1V1Explicit": (
        "tests.test_documented_patterns",
        "TestPattern1V1Explicit",
    ),
    "TestPattern2V2Property": (
        "tests.test_documented_patterns",
        "TestPattern2V2Property",
    ),
    "TestPattern3RailwayV1": (
        "tests.test_documented_patterns",
        "TestPattern3RailwayV1",
    ),
    "TestPattern4RailwayV2Property": (
        "tests.test_documented_patterns",
        "TestPattern4RailwayV2Property",
    ),
    "TestPattern5MonadicComposition": (
        "tests.test_documented_patterns",
        "TestPattern5MonadicComposition",
    ),
    "TestPattern6ErrorHandling": (
        "tests.test_documented_patterns",
        "TestPattern6ErrorHandling",
    ),
    "TestPattern7AutomaticInfrastructure": (
        "tests.test_documented_patterns",
        "TestPattern7AutomaticInfrastructure",
    ),
    "TestPattern8MultipleOperations": (
        "tests.test_documented_patterns",
        "TestPattern8MultipleOperations",
    ),
    "TestPerformanceAnalysis": (
        "tests.integration.patterns.test_patterns_testing",
        "TestPerformanceAnalysis",
    ),
    "TestPerformanceBenchmarks": (
        "tests.benchmark.test_refactor_nesting_performance",
        "TestPerformanceBenchmarks",
    ),
    "TestPerformanceDomain": (
        "tests.unit.test_coverage_context",
        "TestPerformanceDomain",
    ),
    "TestPhase2FinalCoveragePush": (
        "tests.unit.test_phase2_coverage_final",
        "TestPhase2FinalCoveragePush",
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
    "TestProjectLevelRefactor": (
        "tests.integration.test_refactor_nesting_project",
        "TestProjectLevelRefactor",
    ),
    "TestProjectResultProperties": (
        "tests.infra.unit.check.extended_models",
        "TestProjectResultProperties",
    ),
    "TestPropertyBasedPatterns": (
        "tests.integration.patterns.test_patterns_testing",
        "TestPropertyBasedPatterns",
    ),
    "TestProtocolComplianceStructlogLogger": (
        "tests.unit.test_loggings_strict_returns",
        "TestProtocolComplianceStructlogLogger",
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
    "TestQueries": ("tests.unit.test_coverage_models", "TestQueries"),
    "TestReadDoc": ("tests.infra.unit.deps.test_modernizer_workspace", "TestReadDoc"),
    "TestReadExistingDocstring": (
        "tests.infra.unit.codegen.lazy_init_helpers",
        "TestReadExistingDocstring",
    ),
    "TestReadRequiredMinor": (
        "tests.infra.unit.test_infra_maintenance_python_version",
        "TestReadRequiredMinor",
    ),
    "TestRealWiringScenarios": (
        "tests.unit.test_di_incremental",
        "TestRealWiringScenarios",
    ),
    "TestRealWorldScenarios": (
        "tests.integration.patterns.test_patterns_testing",
        "TestRealWorldScenarios",
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
    "TestResult": ("tests.test_utils", "TestResult"),
    "TestResultBasics": ("tests.unit.test_coverage_76_lines", "TestResultBasics"),
    "TestResultCo": ("tests.test_utils", "TestResultCo"),
    "TestResultTransformations": (
        "tests.unit.test_coverage_76_lines",
        "TestResultTransformations",
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
    "TestRuntimeDictLike": (
        "tests.unit.test_runtime_coverage_100",
        "TestRuntimeDictLike",
    ),
    "TestRuntimeTypeChecking": (
        "tests.unit.test_runtime_coverage_100",
        "TestRuntimeTypeChecking",
    ),
    "TestSafeCarriesException": (
        "tests.unit.test_result_exception_carrying",
        "TestSafeCarriesException",
    ),
    "TestSafeLoadYaml": ("tests.infra.unit.core.skill_validator", "TestSafeLoadYaml"),
    "TestSafeString": (
        "tests.unit.test_utilities_text_full_coverage",
        "TestSafeString",
    ),
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
    "TestScopedContextManagement": (
        "tests.unit.test_coverage_loggings",
        "TestScopedContextManagement",
    ),
    "TestSelectedProjectNames": (
        "tests.infra.unit.docs.shared_iter",
        "TestSelectedProjectNames",
    ),
    "TestSelectorFunction": ("tests.infra.unit.github.pr_cli", "TestSelectorFunction"),
    "TestService": ("tests.unit.test_service_coverage_100", "TestService"),
    "TestService100Coverage": (
        "tests.unit.test_service_coverage_100",
        "TestService100Coverage",
    ),
    "TestServiceBootstrap": (
        "tests.unit.test_service_bootstrap",
        "TestServiceBootstrap",
    ),
    "TestServiceBootstrapWithDI": (
        "tests.unit.test_di_incremental",
        "TestServiceBootstrapWithDI",
    ),
    "TestServiceDomain": ("tests.unit.test_coverage_context", "TestServiceDomain"),
    "TestServicerChaining": (
        "tests.integration.test_infra_integration",
        "TestServicerChaining",
    ),
    "TestServiceResultProperty": (
        "tests.test_service_result_property",
        "TestServiceResultProperty",
    ),
    "TestServiceWithValidation": (
        "tests.unit.test_service_coverage_100",
        "TestServiceWithValidation",
    ),
    "TestServicesIntegrationViaDI": (
        "tests.unit.test_di_services_access",
        "TestServicesIntegrationViaDI",
    ),
    "TestShortAlias": ("tests.unit.flext_tests.test_files", "TestShortAlias"),
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
    "TestTextLogger": (
        "tests.unit.test_utilities_text_full_coverage",
        "TestTextLogger",
    ),
    "TestTimeoutEnforcerCleanup": (
        "tests.unit.test_dispatcher_timeout_coverage_100",
        "TestTimeoutEnforcerCleanup",
    ),
    "TestTimeoutEnforcerEdgeCases": (
        "tests.unit.test_dispatcher_timeout_coverage_100",
        "TestTimeoutEnforcerEdgeCases",
    ),
    "TestTimeoutEnforcerExecutorManagement": (
        "tests.unit.test_dispatcher_timeout_coverage_100",
        "TestTimeoutEnforcerExecutorManagement",
    ),
    "TestTimeoutEnforcerInitialization": (
        "tests.unit.test_dispatcher_timeout_coverage_100",
        "TestTimeoutEnforcerInitialization",
    ),
    "TestToIOChainsException": (
        "tests.unit.test_result_exception_carrying",
        "TestToIOChainsException",
    ),
    "TestToInfraValue": (
        "tests.infra.unit.deps.test_detection_models",
        "TestToInfraValue",
    ),
    "TestTraceReturnsResultBool": (
        "tests.unit.test_loggings_strict_returns",
        "TestTraceReturnsResultBool",
    ),
    "TestTraversePropagatesException": (
        "tests.unit.test_result_exception_carrying",
        "TestTraversePropagatesException",
    ),
    "TestTriggerRelease": (
        "tests.infra.unit.github.pr_operations",
        "TestTriggerRelease",
    ),
    "TestUpdateChangelog": (
        "tests.infra.unit.release.orchestrator_helpers",
        "TestUpdateChangelog",
    ),
    "TestUser": ("tests.unit.flext_tests.test_factories", "TestUser"),
    "TestUtilitiesCoverage": (
        "tests.unit.test_utilities_coverage",
        "TestUtilitiesCoverage",
    ),
    "TestUtilitiesDomain": ("tests.unit.test_coverage_context", "TestUtilitiesDomain"),
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
    "TestValidateValueImmutable": (
        "tests.unit.test_utilities_domain_full_coverage",
        "TestValidateValueImmutable",
    ),
    "TestValidatorCallable": (
        "tests.unit.test_typings_full_coverage",
        "TestValidatorCallable",
    ),
    "TestValidatorMapMixin": (
        "tests.unit.test_typings_full_coverage",
        "TestValidatorMapMixin",
    ),
    "TestValues": ("tests.unit.test_coverage_models", "TestValues"),
    "TestVersionFiles": (
        "tests.infra.unit.release.orchestrator_helpers",
        "TestVersionFiles",
    ),
    "TestView": ("tests.infra.unit.github.pr_operations", "TestView"),
    "TestViolationPattern": (
        "tests.infra.unit.codegen.census_models",
        "TestViolationPattern",
    ),
    "TestWarningReturnsResultBool": (
        "tests.unit.test_loggings_strict_returns",
        "TestWarningReturnsResultBool",
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
    "TestWorkspaceDetectionOrchestrationFlow": (
        "tests.integration.test_infra_integration",
        "TestWorkspaceDetectionOrchestrationFlow",
    ),
    "TestWorkspaceLevelRefactor": (
        "tests.integration.test_refactor_nesting_workspace",
        "TestWorkspaceLevelRefactor",
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
    "Teste": ("tests.unit.test_exceptions", "Teste"),
    "Testr": ("tests.unit.test_result", "Testr"),
    "TestrCoverage": ("tests.unit.test_result_coverage_100", "TestrCoverage"),
    "TestsCore": ("tests.unit.test_service", "TestsCore"),
    "TestsFlextConstants": ("tests.constants", "TestsFlextConstants"),
    "TestsFlextModels": ("tests.models", "TestsFlextModels"),
    "TestsFlextProtocols": ("tests.protocols", "TestsFlextProtocols"),
    "TestsFlextServiceBase": ("tests.base", "TestsFlextServiceBase"),
    "TestsFlextTestsFactoriesDict": (
        "tests.unit.flext_tests.test_factories",
        "TestsFlextTestsFactoriesDict",
    ),
    "TestsFlextTestsFactoriesGeneric": (
        "tests.unit.flext_tests.test_factories",
        "TestsFlextTestsFactoriesGeneric",
    ),
    "TestsFlextTestsFactoriesList": (
        "tests.unit.flext_tests.test_factories",
        "TestsFlextTestsFactoriesList",
    ),
    "TestsFlextTestsFactoriesModel": (
        "tests.unit.flext_tests.test_factories",
        "TestsFlextTestsFactoriesModel",
    ),
    "TestsFlextTestsFactoriesRes": (
        "tests.unit.flext_tests.test_factories",
        "TestsFlextTestsFactoriesRes",
    ),
    "TestsFlextTypes": ("tests.typings", "TestsFlextTypes"),
    "TestsFlextUtilities": ("tests.utilities", "TestsFlextUtilities"),
    "Testu": ("tests.unit.test_utilities", "Testu"),
    "TestuCacheClearObjectCache": (
        "tests.unit.test_utilities_cache_coverage_100",
        "TestuCacheClearObjectCache",
    ),
    "TestuCacheGenerateCacheKey": (
        "tests.unit.test_utilities_cache_coverage_100",
        "TestuCacheGenerateCacheKey",
    ),
    "TestuCacheHasCacheAttributes": (
        "tests.unit.test_utilities_cache_coverage_100",
        "TestuCacheHasCacheAttributes",
    ),
    "TestuCacheLogger": (
        "tests.unit.test_utilities_cache_coverage_100",
        "TestuCacheLogger",
    ),
    "TestuCacheNormalizeComponent": (
        "tests.unit.test_utilities_cache_coverage_100",
        "TestuCacheNormalizeComponent",
    ),
    "TestuCacheSortDictKeys": (
        "tests.unit.test_utilities_cache_coverage_100",
        "TestuCacheSortDictKeys",
    ),
    "TestuCacheSortKey": (
        "tests.unit.test_utilities_cache_coverage_100",
        "TestuCacheSortKey",
    ),
    "TestuCollectionBatch": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionBatch",
    ),
    "TestuCollectionChunk": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionChunk",
    ),
    "TestuCollectionCoerceDictValidator": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionCoerceDictValidator",
    ),
    "TestuCollectionCoerceListValidator": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionCoerceListValidator",
    ),
    "TestuCollectionCount": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionCount",
    ),
    "TestuCollectionFilter": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionFilter",
    ),
    "TestuCollectionFind": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionFind",
    ),
    "TestuCollectionGroup": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionGroup",
    ),
    "TestuCollectionMap": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionMap",
    ),
    "TestuCollectionMerge": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionMerge",
    ),
    "TestuCollectionParseMapping": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionParseMapping",
    ),
    "TestuCollectionParseSequence": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionParseSequence",
    ),
    "TestuCollectionProcess": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionProcess",
    ),
    "TestuDomain": ("tests.unit.test_utilities_domain", "TestuDomain"),
    "TestuEnumCoerceByNameValidator": (
        "tests.unit.test_enum_utilities_coverage_100",
        "TestuEnumCoerceByNameValidator",
    ),
    "TestuEnumCoerceValidator": (
        "tests.unit.test_enum_utilities_coverage_100",
        "TestuEnumCoerceValidator",
    ),
    "TestuEnumIsMember": (
        "tests.unit.test_enum_utilities_coverage_100",
        "TestuEnumIsMember",
    ),
    "TestuEnumIsSubset": (
        "tests.unit.test_enum_utilities_coverage_100",
        "TestuEnumIsSubset",
    ),
    "TestuEnumMetadata": (
        "tests.unit.test_enum_utilities_coverage_100",
        "TestuEnumMetadata",
    ),
    "TestuEnumParse": ("tests.unit.test_enum_utilities_coverage_100", "TestuEnumParse"),
    "TestuEnumParseOrDefault": (
        "tests.unit.test_enum_utilities_coverage_100",
        "TestuEnumParseOrDefault",
    ),
    "TestuMapperAccessors": (
        "tests.unit.test_utilities_mapper_coverage_100",
        "TestuMapperAccessors",
    ),
    "TestuMapperAdvanced": (
        "tests.unit.test_utilities_mapper_coverage_100",
        "TestuMapperAdvanced",
    ),
    "TestuMapperBuild": (
        "tests.unit.test_utilities_mapper_coverage_100",
        "TestuMapperBuild",
    ),
    "TestuMapperConversions": (
        "tests.unit.test_utilities_mapper_coverage_100",
        "TestuMapperConversions",
    ),
    "TestuMapperExtract": (
        "tests.unit.test_utilities_mapper_coverage_100",
        "TestuMapperExtract",
    ),
    "TestuMapperUtils": (
        "tests.unit.test_utilities_mapper_coverage_100",
        "TestuMapperUtils",
    ),
    "TestuPaginationBuildPaginationResponse": (
        "tests.unit.test_pagination_coverage_100",
        "TestuPaginationBuildPaginationResponse",
    ),
    "TestuPaginationExtractPageParams": (
        "tests.unit.test_pagination_coverage_100",
        "TestuPaginationExtractPageParams",
    ),
    "TestuPaginationExtractPaginationConfig": (
        "tests.unit.test_pagination_coverage_100",
        "TestuPaginationExtractPaginationConfig",
    ),
    "TestuPaginationPreparePaginationData": (
        "tests.unit.test_pagination_coverage_100",
        "TestuPaginationPreparePaginationData",
    ),
    "TestuPaginationValidatePaginationParams": (
        "tests.unit.test_pagination_coverage_100",
        "TestuPaginationValidatePaginationParams",
    ),
    "TestuStringParser": (
        "tests.unit.test_utilities_string_parser",
        "TestuStringParser",
    ),
    "TestuTypeChecker": (
        "tests.unit.test_utilities_type_checker_coverage_100",
        "TestuTypeChecker",
    ),
    "TestuTypeGuardsIsDictNonEmpty": (
        "tests.unit.test_utilities_type_guards_coverage_100",
        "TestuTypeGuardsIsDictNonEmpty",
    ),
    "TestuTypeGuardsIsListNonEmpty": (
        "tests.unit.test_utilities_type_guards_coverage_100",
        "TestuTypeGuardsIsListNonEmpty",
    ),
    "TestuTypeGuardsIsStringNonEmpty": (
        "tests.unit.test_utilities_type_guards_coverage_100",
        "TestuTypeGuardsIsStringNonEmpty",
    ),
    "TestuTypeGuardsNormalizeToMetadataValue": (
        "tests.unit.test_utilities_type_guards_coverage_100",
        "TestuTypeGuardsNormalizeToMetadataValue",
    ),
    "TextLike": ("tests.unit.test_utilities_enum_full_coverage", "TextLike"),
    "TextUtilityContract": (
        "tests.unit.contracts.text_contract",
        "TextUtilityContract",
    ),
    "ThreadSafetyTest": (
        "tests.integration.test_config_integration",
        "ThreadSafetyTest",
    ),
    "TimeoutEnforcerScenarios": (
        "tests.unit.test_dispatcher_timeout_coverage_100",
        "TimeoutEnforcerScenarios",
    ),
    "TypeGuardsScenarios": (
        "tests.unit.test_utilities_type_guards_coverage_100",
        "TypeGuardsScenarios",
    ),
    "UnknownHint": ("tests.unit.test_utilities_args_full_coverage", "UnknownHint"),
    "UnregisteredCommand": (
        "tests.unit.test_dispatcher_full_coverage",
        "UnregisteredCommand",
    ),
    "UpdateUserCommand": (
        "tests.integration.patterns.test_patterns_commands",
        "UpdateUserCommand",
    ),
    "UpdateUserCommandHandler": (
        "tests.integration.patterns.test_patterns_commands",
        "UpdateUserCommandHandler",
    ),
    "User": ("tests.test_documented_patterns", "User"),
    "UserFactory": ("tests.helpers.factories", "UserFactory"),
    "UserQueryService": ("tests.integration.test_service", "UserQueryService"),
    "UserServiceEntity": ("tests.integration.test_service", "UserServiceEntity"),
    "ValidatePaginationParamsScenario": (
        "tests.unit.test_pagination_coverage_100",
        "ValidatePaginationParamsScenario",
    ),
    "ValidatingService": ("tests.helpers.factories", "ValidatingService"),
    "ValidatingServiceAuto": ("tests.helpers.factories", "ValidatingServiceAuto"),
    "ValidatingServiceAutoFactory": (
        "tests.helpers.factories",
        "ValidatingServiceAutoFactory",
    ),
    "ValidatingServiceFactory": ("tests.helpers.factories", "ValidatingServiceFactory"),
    "ValidationScenario": ("tests.helpers.scenarios", "ValidationScenario"),
    "ValidationScenarios": ("tests.helpers.scenarios", "ValidationScenarios"),
    "ValidationService": ("tests.test_documented_patterns", "ValidationService"),
    "WorkspaceFactory": ("tests.infra.workspace_factory", "WorkspaceFactory"),
    "WorkspaceScenario": ("tests.infra.scenarios", "WorkspaceScenario"),
    "WorkspaceScenarios": ("tests.infra.scenarios", "WorkspaceScenarios"),
    "arrange_act_assert": (
        "tests.integration.patterns.test_patterns_testing",
        "arrange_act_assert",
    ),
    "assert_rejects": ("tests.conftest", "assert_rejects"),
    "assert_validates": ("tests.conftest", "assert_validates"),
    "assertion_helpers": ("tests.test_utils", "assertion_helpers"),
    "auditor": ("tests.infra.unit.docs.auditor", "auditor"),
    "automation_framework": ("tests.conftest", "automation_framework"),
    "builder": ("tests.infra.unit.docs.builder", "builder"),
    "c": ("tests.constants", "c"),
    "census": ("tests.infra.unit.codegen.census", "census"),
    "clean_container": ("tests.conftest", "clean_container"),
    "consolidation_context": ("tests.conftest", "consolidation_context"),
    "create_compare_entities_cases": (
        "tests.unit.test_utilities_domain",
        "create_compare_entities_cases",
    ),
    "create_compare_value_objects_cases": (
        "tests.unit.test_utilities_domain",
        "create_compare_value_objects_cases",
    ),
    "create_hash_entity_cases": (
        "tests.unit.test_utilities_domain",
        "create_hash_entity_cases",
    ),
    "create_hash_value_object_cases": (
        "tests.unit.test_utilities_domain",
        "create_hash_value_object_cases",
    ),
    "create_validate_entity_has_id_cases": (
        "tests.unit.test_utilities_domain",
        "create_validate_entity_has_id_cases",
    ),
    "create_validate_value_object_immutable_cases": (
        "tests.unit.test_utilities_domain",
        "create_validate_value_object_immutable_cases",
    ),
    "d": ("tests.unit.test_automated_decorators", "TestAutomatedFlextDecorators"),
    "detector": ("tests.infra.unit.test_infra_workspace_detector", "detector"),
    "dispatcher": ("tests.unit.test_dispatcher_full_coverage", "dispatcher"),
    "doc": ("tests.infra.unit.deps.test_modernizer_helpers", "doc"),
    "e": ("tests.unit.test_automated_exceptions", "TestAutomatedFlextExceptions"),
    "empty_strings": ("tests.conftest", "empty_strings"),
    "engine": ("tests.infra.unit.test_infra_templates", "engine"),
    "fixer": ("tests.infra.unit.codegen.autofix", "fixer"),
    "fixture_factory": ("tests.test_utils", "fixture_factory"),
    "flext_result_failure": ("tests.conftest", "flext_result_failure"),
    "flext_result_success": ("tests.conftest", "flext_result_success"),
    "gen": ("tests.infra.unit.docs.generator_internals", "gen"),
    "generators_module": (
        "tests.unit.test_utilities_generators_full_coverage",
        "generators_module",
    ),
    "get_memory_usage": ("tests.benchmark.test_container_memory", "get_memory_usage"),
    "git_repo": ("tests.infra.unit.test_infra_git", "git_repo"),
    "h": ("tests.infra.helpers", "h"),
    "handlers_module": ("tests.unit.test_handlers_full_coverage", "handlers_module"),
    "infra_git": ("tests.unit.conftest_infra", "infra_git"),
    "infra_git_repo": ("tests.unit.conftest_infra", "infra_git_repo"),
    "infra_io": ("tests.unit.conftest_infra", "infra_io"),
    "infra_path": ("tests.unit.conftest_infra", "infra_path"),
    "infra_patterns": ("tests.unit.conftest_infra", "infra_patterns"),
    "infra_pr_manager": ("tests.unit.conftest_infra", "infra_pr_manager"),
    "infra_pr_workspace_manager": (
        "tests.unit.conftest_infra",
        "infra_pr_workspace_manager",
    ),
    "infra_reporting": ("tests.unit.conftest_infra", "infra_reporting"),
    "infra_safe_command_output": (
        "tests.unit.conftest_infra",
        "infra_safe_command_output",
    ),
    "infra_selection": ("tests.unit.conftest_infra", "infra_selection"),
    "infra_subprocess": ("tests.unit.conftest_infra", "infra_subprocess"),
    "infra_templates": ("tests.unit.conftest_infra", "infra_templates"),
    "infra_test_workspace": ("tests.unit.conftest_infra", "infra_test_workspace"),
    "infra_toml": ("tests.unit.conftest_infra", "infra_toml"),
    "infra_workflow_linter": ("tests.unit.conftest_infra", "infra_workflow_linter"),
    "infra_workflow_syncer": ("tests.unit.conftest_infra", "infra_workflow_syncer"),
    "invalid_hostnames": ("tests.conftest", "invalid_hostnames"),
    "invalid_port_numbers": ("tests.conftest", "invalid_port_numbers"),
    "invalid_uris": ("tests.conftest", "invalid_uris"),
    "is_external": ("tests.infra.unit.docs.auditor", "is_external"),
    "m": ("tests.models", "m"),
    "make_result_logger": (
        "tests.integration.patterns.test_patterns_logging",
        "make_result_logger",
    ),
    "mapper": ("tests.unit.test_utilities_mapper_full_coverage", "mapper"),
    "mark_test_pattern": (
        "tests.integration.patterns.test_patterns_testing",
        "mark_test_pattern",
    ),
    "mock_external_service": ("tests.conftest", "mock_external_service"),
    "normalize_link": ("tests.infra.unit.docs.auditor", "normalize_link"),
    "orchestrator": (
        "tests.infra.unit.test_infra_workspace_orchestrator",
        "orchestrator",
    ),
    "out_of_range": ("tests.conftest", "out_of_range"),
    "p": ("tests.protocols", "p"),
    "parser_scenarios": ("tests.conftest", "parser_scenarios"),
    "pyright_content": (
        "tests.infra.unit.deps.test_extra_paths_sync",
        "pyright_content",
    ),
    "pytestmark": ("tests.integration.test_service", "pytestmark"),
    "r": (
        "tests.infra.unit.check.extended_workspace_init",
        "TestWorkspaceCheckerBuildGateResult",
    ),
    "real_docs_project": ("tests.infra.fixtures", "real_docs_project"),
    "real_entity": ("tests.conftest", "real_entity"),
    "real_git_repo": ("tests.infra.fixtures_git", "real_git_repo"),
    "real_makefile_project": ("tests.infra.fixtures", "real_makefile_project"),
    "real_python_package": ("tests.infra.fixtures", "real_python_package"),
    "real_toml_project": ("tests.infra.fixtures", "real_toml_project"),
    "real_value_object": ("tests.conftest", "real_value_object"),
    "real_workspace": ("tests.infra.fixtures", "real_workspace"),
    "reliability_scenarios": ("tests.conftest", "reliability_scenarios"),
    "reset_all_factories": ("tests.helpers.factories", "reset_all_factories"),
    "reset_global_container": ("tests.conftest", "reset_global_container"),
    "reset_runtime_state": (
        "tests.unit.test_runtime_full_coverage",
        "reset_runtime_state",
    ),
    "result_assertion_helper": ("tests.conftest", "result_assertion_helper"),
    "run_lint": ("tests.infra.unit.github.main", "run_lint"),
    "run_pr": ("tests.infra.unit.github.main", "run_pr"),
    "run_pr_workspace": ("tests.infra.unit.github.main_dispatch", "run_pr_workspace"),
    "run_workflows": ("tests.infra.unit.github.main", "run_workflows"),
    "runner": ("tests.infra.unit.test_infra_subprocess_core", "runner"),
    "runtime_cov_tests": ("tests.unit.test_runtime_full_coverage", "runtime_cov_tests"),
    "runtime_tests": ("tests.unit.test_runtime_full_coverage", "runtime_tests"),
    "s": ("tests.helpers.factories", "GetUserService"),
    "sample_data": ("tests.conftest", "sample_data"),
    "scenario_runner": ("tests.conftest", "scenario_runner"),
    "service": ("tests.infra.unit.test_infra_versioning", "service"),
    "should_skip_target": ("tests.infra.unit.docs.auditor", "should_skip_target"),
    "svc": ("tests.infra.unit.test_infra_workspace_sync", "svc"),
    "t": ("tests.infra.typings", "t"),
    "temp_dir": ("tests.conftest", "temp_dir"),
    "temp_directory": ("tests.conftest", "temp_directory"),
    "temp_file": ("tests.conftest", "temp_file"),
    "test_accessor_take_pick_as_or_flat_and_agg_branches": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_accessor_take_pick_as_or_flat_and_agg_branches",
    ),
    "test_additional_container_branches_cover_fluent_and_lookup_paths": (
        "tests.unit.test_container_full_coverage",
        "test_additional_container_branches_cover_fluent_and_lookup_paths",
    ),
    "test_additional_register_factory_and_unregister_paths": (
        "tests.unit.test_container_full_coverage",
        "test_additional_register_factory_and_unregister_paths",
    ),
    "test_aliases_are_available": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_aliases_are_available",
    ),
    "test_args_get_enum_params_annotated_unwrap_branch": (
        "tests.unit.test_utilities_args_full_coverage",
        "test_args_get_enum_params_annotated_unwrap_branch",
    ),
    "test_args_get_enum_params_branches": (
        "tests.unit.test_utilities_args_full_coverage",
        "test_args_get_enum_params_branches",
    ),
    "test_array": ("tests.infra.unit.deps.test_modernizer_helpers", "test_array"),
    "test_as_string_list": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "test_as_string_list",
    ),
    "test_as_string_list_toml_item": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "test_as_string_list_toml_item",
    ),
    "test_async_log_writer_paths": (
        "tests.unit.test_runtime_full_coverage",
        "test_async_log_writer_paths",
    ),
    "test_async_log_writer_shutdown_with_full_queue": (
        "tests.unit.test_runtime_full_coverage",
        "test_async_log_writer_shutdown_with_full_queue",
    ),
    "test_at_take_and_as_branches": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_at_take_and_as_branches",
    ),
    "test_atomic_write_fail": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_atomic_write_fail",
    ),
    "test_atomic_write_ok": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_atomic_write_ok",
    ),
    "test_authentication_error_normalizes_extra_kwargs_into_context": (
        "tests.unit.test_exceptions_full_coverage",
        "test_authentication_error_normalizes_extra_kwargs_into_context",
    ),
    "test_auto_value_lowercases_input": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_auto_value_lowercases_input",
    ),
    "test_bad_string_and_bad_bool_raise_value_error": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_bad_string_and_bad_bool_raise_value_error",
    ),
    "test_base_error_normalize_metadata_merges_existing_metadata_model": (
        "tests.unit.test_exceptions_full_coverage",
        "test_base_error_normalize_metadata_merges_existing_metadata_model",
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
    "test_basic_imports_work": (
        "tests.unit.test_models_validation_full_coverage",
        "test_basic_imports_work",
    ),
    "test_batch_fail_collect_flatten_and_progress": (
        "tests.unit.test_utilities_collection_full_coverage",
        "test_batch_fail_collect_flatten_and_progress",
    ),
    "test_bi_map_returns_forward_copy_and_inverse": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_bi_map_returns_forward_copy_and_inverse",
    ),
    "test_bind_operation_context_without_ensure_correlation_and_bind_failure": (
        "tests.unit.test_decorators_full_coverage",
        "test_bind_operation_context_without_ensure_correlation_and_bind_failure",
    ),
    "test_build_apply_transform_and_process_error_paths": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_build_apply_transform_and_process_error_paths",
    ),
    "test_build_impact_map_extracts_rename_entries": (
        "tests.infra.unit.refactor.test_infra_refactor_analysis",
        "test_build_impact_map_extracts_rename_entries",
    ),
    "test_build_impact_map_extracts_signature_entries": (
        "tests.infra.unit.refactor.test_infra_refactor_analysis",
        "test_build_impact_map_extracts_signature_entries",
    ),
    "test_build_options_invalid_only_kwargs_returns_base": (
        "tests.unit.test_utilities_configuration_full_coverage",
        "test_build_options_invalid_only_kwargs_returns_base",
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
    "test_callable_registration_with_attribute": (
        "tests.unit.test_dispatcher_full_coverage",
        "test_callable_registration_with_attribute",
    ),
    "test_canonical_aliases_are_available": (
        "tests.unit.test_models_generic_full_coverage",
        "test_canonical_aliases_are_available",
    ),
    "test_canonical_dev_dependencies": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "test_canonical_dev_dependencies",
    ),
    "test_capture_cases": (
        "tests.infra.unit.test_infra_subprocess_core",
        "test_capture_cases",
    ),
    "test_categories_clear_and_symbols_are_available": (
        "tests.unit.test_models_collections_full_coverage",
        "test_categories_clear_and_symbols_are_available",
    ),
    "test_centralize_pydantic_cli_outputs_extended_metrics": (
        "tests.unit.test_refactor_cli_models_workflow",
        "test_centralize_pydantic_cli_outputs_extended_metrics",
    ),
    "test_centralizer_converts_typed_dict_factory_to_model": (
        "tests.unit.test_refactor_pydantic_centralizer",
        "test_centralizer_converts_typed_dict_factory_to_model",
    ),
    "test_centralizer_does_not_touch_settings_module": (
        "tests.unit.test_refactor_pydantic_centralizer",
        "test_centralizer_does_not_touch_settings_module",
    ),
    "test_centralizer_moves_dict_alias_in_typings_without_keyword_name": (
        "tests.unit.test_refactor_pydantic_centralizer",
        "test_centralizer_moves_dict_alias_in_typings_without_keyword_name",
    ),
    "test_centralizer_moves_manual_type_aliases_to_models_file": (
        "tests.unit.test_refactor_pydantic_centralizer",
        "test_centralizer_moves_manual_type_aliases_to_models_file",
    ),
    "test_check_implements_protocol_false_non_runtime_protocol": (
        "tests.unit.test_protocols_full_coverage",
        "test_check_implements_protocol_false_non_runtime_protocol",
    ),
    "test_check_main_executes_real_cli": (
        "tests.infra.unit.check.main",
        "test_check_main_executes_real_cli",
    ),
    "test_checker_logger_and_safe_type_hints_fallback": (
        "tests.unit.test_utilities_checker_full_coverage",
        "test_checker_logger_and_safe_type_hints_fallback",
    ),
    "test_chk_exercises_missed_branches": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_chk_exercises_missed_branches",
    ),
    "test_circuit_breaker_transitions_and_metrics": (
        "tests.unit.test_dispatcher_reliability",
        "test_circuit_breaker_transitions_and_metrics",
    ),
    "test_class_nesting_appends_to_existing_namespace_and_removes_pass": (
        "tests.unit.test_transformer_class_nesting",
        "test_class_nesting_appends_to_existing_namespace_and_removes_pass",
    ),
    "test_class_nesting_keeps_unmapped_top_level_classes": (
        "tests.unit.test_transformer_class_nesting",
        "test_class_nesting_keeps_unmapped_top_level_classes",
    ),
    "test_class_nesting_moves_top_level_class_into_new_namespace": (
        "tests.unit.test_transformer_class_nesting",
        "test_class_nesting_moves_top_level_class_into_new_namespace",
    ),
    "test_class_nesting_refactor_single_file_end_to_end": (
        "tests.integration.test_refactor_nesting_file",
        "test_class_nesting_refactor_single_file_end_to_end",
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
    "test_clear_keys_values_items_and_validate_branches": (
        "tests.unit.test_context_full_coverage",
        "test_clear_keys_values_items_and_validate_branches",
    ),
    "test_clear_operation_scope_and_handle_log_result_paths": (
        "tests.unit.test_decorators_full_coverage",
        "test_clear_operation_scope_and_handle_log_result_paths",
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
    "test_collection_batch_failure_error_capture_and_parse_sequence_outer_error": (
        "tests.unit.test_utilities_collection_full_coverage",
        "test_collection_batch_failure_error_capture_and_parse_sequence_outer_error",
    ),
    "test_combined_with_and_without_railway_uses_injection": (
        "tests.unit.test_decorators_full_coverage",
        "test_combined_with_and_without_railway_uses_injection",
    ),
    "test_command_pagination_limit": (
        "tests.unit.test_models_cqrs_full_coverage",
        "test_command_pagination_limit",
    ),
    "test_config_bridge_and_trace_context_and_http_validation": (
        "tests.unit.test_runtime_full_coverage",
        "test_config_bridge_and_trace_context_and_http_validation",
    ),
    "test_config_context_properties_and_defaults": (
        "tests.unit.test_container_full_coverage",
        "test_config_context_properties_and_defaults",
    ),
    "test_config_hash_from_mapping_and_non_hashable": (
        "tests.unit.test_models_collections_full_coverage",
        "test_config_hash_from_mapping_and_non_hashable",
    ),
    "test_configuration_mapping_and_dict_negative_branches": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_configuration_mapping_and_dict_negative_branches",
    ),
    "test_configure_structlog_edge_paths": (
        "tests.unit.test_runtime_full_coverage",
        "test_configure_structlog_edge_paths",
    ),
    "test_configure_structlog_print_logger_factory_fallback": (
        "tests.unit.test_runtime_full_coverage",
        "test_configure_structlog_print_logger_factory_fallback",
    ),
    "test_configure_with_resource_register_and_factory_error_paths": (
        "tests.unit.test_container_full_coverage",
        "test_configure_with_resource_register_and_factory_error_paths",
    ),
    "test_consolidate_groups_phase_apply_removes_old_groups": (
        "tests.infra.unit.deps.test_modernizer_consolidate",
        "test_consolidate_groups_phase_apply_removes_old_groups",
    ),
    "test_consolidate_groups_phase_apply_with_empty_poetry_group": (
        "tests.infra.unit.deps.test_modernizer_consolidate",
        "test_consolidate_groups_phase_apply_with_empty_poetry_group",
    ),
    "test_constants_auto_enum_and_bimapping_paths": (
        "tests.unit.test_constants_full_coverage",
        "test_constants_auto_enum_and_bimapping_paths",
    ),
    "test_construct_transform_and_deep_eq_branches": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_construct_transform_and_deep_eq_branches",
    ),
    "test_container_and_service_domain_paths": (
        "tests.unit.test_context_full_coverage",
        "test_container_and_service_domain_paths",
    ),
    "test_container_remaining_branch_paths_in_sync_factory_and_getters": (
        "tests.unit.test_container_full_coverage",
        "test_container_remaining_branch_paths_in_sync_factory_and_getters",
    ),
    "test_container_resource_registration_metadata_normalized": (
        "tests.unit.test_models_container_full_coverage",
        "test_container_resource_registration_metadata_normalized",
    ),
    "test_context": ("tests.conftest", "test_context"),
    "test_context_data_metadata_normalizer_removed": (
        "tests.unit.test_models_context_full_coverage",
        "test_context_data_metadata_normalizer_removed",
    ),
    "test_context_data_normalize_and_json_checks": (
        "tests.unit.test_models_context_full_coverage",
        "test_context_data_normalize_and_json_checks",
    ),
    "test_context_data_validate_dict_serializable_error_paths": (
        "tests.unit.test_models_context_full_coverage",
        "test_context_data_validate_dict_serializable_error_paths",
    ),
    "test_context_data_validate_dict_serializable_none_and_mapping": (
        "tests.unit.test_models_context_full_coverage",
        "test_context_data_validate_dict_serializable_none_and_mapping",
    ),
    "test_context_data_validate_dict_serializable_real_dicts": (
        "tests.unit.test_models_context_full_coverage",
        "test_context_data_validate_dict_serializable_real_dicts",
    ),
    "test_context_export_serializable_and_validators": (
        "tests.unit.test_models_context_full_coverage",
        "test_context_export_serializable_and_validators",
    ),
    "test_context_export_statistics_validator_and_computed_fields": (
        "tests.unit.test_models_context_full_coverage",
        "test_context_export_statistics_validator_and_computed_fields",
    ),
    "test_context_export_validate_dict_serializable_mapping_and_models": (
        "tests.unit.test_models_context_full_coverage",
        "test_context_export_validate_dict_serializable_mapping_and_models",
    ),
    "test_context_export_validate_dict_serializable_valid": (
        "tests.unit.test_models_context_full_coverage",
        "test_context_export_validate_dict_serializable_valid",
    ),
    "test_conversion_add_converted_and_error_metadata_append_paths": (
        "tests.unit.test_models_generic_full_coverage",
        "test_conversion_add_converted_and_error_metadata_append_paths",
    ),
    "test_conversion_add_skipped_skip_reason_upsert_paths": (
        "tests.unit.test_models_generic_full_coverage",
        "test_conversion_add_skipped_skip_reason_upsert_paths",
    ),
    "test_conversion_add_warning_metadata_append_paths": (
        "tests.unit.test_models_generic_full_coverage",
        "test_conversion_add_warning_metadata_append_paths",
    ),
    "test_conversion_and_extract_success_branches": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_conversion_and_extract_success_branches",
    ),
    "test_conversion_start_and_complete_methods": (
        "tests.unit.test_models_generic_full_coverage",
        "test_conversion_start_and_complete_methods",
    ),
    "test_conversion_string_and_join_paths": (
        "tests.unit.test_utilities_conversion_full_coverage",
        "test_conversion_string_and_join_paths",
    ),
    "test_convert_default_fallback_matrix": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_convert_default_fallback_matrix",
    ),
    "test_convert_sequence_branch_returns_tuple": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_convert_sequence_branch_returns_tuple",
    ),
    "test_cqrs_query_resolve_deeper_and_int_pagination": (
        "tests.unit.test_models_cqrs_full_coverage",
        "test_cqrs_query_resolve_deeper_and_int_pagination",
    ),
    "test_create_auto_discover_and_mode_mapping": (
        "tests.unit.test_registry_full_coverage",
        "test_create_auto_discover_and_mode_mapping",
    ),
    "test_create_auto_register_factories_path": (
        "tests.unit.test_container_full_coverage",
        "test_create_auto_register_factories_path",
    ),
    "test_create_auto_register_factory_wrapper_callable_and_non_callable": (
        "tests.unit.test_container_full_coverage",
        "test_create_auto_register_factory_wrapper_callable_and_non_callable",
    ),
    "test_create_discriminated_union_multiple_enums": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_create_discriminated_union_multiple_enums",
    ),
    "test_create_enum_executes_factory_path": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_create_enum_executes_factory_path",
    ),
    "test_create_from_callable_and_repr": (
        "tests.unit.test_result_additional",
        "test_create_from_callable_and_repr",
    ),
    "test_create_from_callable_branches": (
        "tests.unit.test_handlers_full_coverage",
        "test_create_from_callable_branches",
    ),
    "test_create_merges_metadata_dict_branch": (
        "tests.unit.test_context_full_coverage",
        "test_create_merges_metadata_dict_branch",
    ),
    "test_create_overloads_and_auto_correlation": (
        "tests.unit.test_context_full_coverage",
        "test_create_overloads_and_auto_correlation",
    ),
    "test_create_scoped_instance_and_scoped_additional_branches": (
        "tests.unit.test_container_full_coverage",
        "test_create_scoped_instance_and_scoped_additional_branches",
    ),
    "test_current_workspace_version": (
        "tests.infra.unit.test_infra_versioning",
        "test_current_workspace_version",
    ),
    "test_data_alias_matches_value": (
        "tests.unit.test_result_additional",
        "test_data_alias_matches_value",
    ),
    "test_data_factory": ("tests.test_utils", "test_data_factory"),
    "test_decorators_family_blocks_dispatcher_target": (
        "tests.unit.test_refactor_policy_family_rules",
        "test_decorators_family_blocks_dispatcher_target",
    ),
    "test_dedupe_specs": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "test_dedupe_specs",
    ),
    "test_dep_name": ("tests.infra.unit.deps.test_modernizer_helpers", "test_dep_name"),
    "test_dependency_integration_and_wiring_paths": (
        "tests.unit.test_runtime_full_coverage",
        "test_dependency_integration_and_wiring_paths",
    ),
    "test_dependency_registration_duplicate_guards": (
        "tests.unit.test_runtime_full_coverage",
        "test_dependency_registration_duplicate_guards",
    ),
    "test_deprecated_class_noop_init_branch": (
        "tests.unit.test_utilities_deprecation_full_coverage",
        "test_deprecated_class_noop_init_branch",
    ),
    "test_deprecated_wrapper_emits_warning_and_returns_value": (
        "tests.unit.test_decorators_full_coverage",
        "test_deprecated_wrapper_emits_warning_and_returns_value",
    ),
    "test_detect_mode_with_nonexistent_path": (
        "tests.infra.unit.deps.test_path_sync_init",
        "test_detect_mode_with_nonexistent_path",
    ),
    "test_detect_mode_with_path_object": (
        "tests.infra.unit.deps.test_path_sync_init",
        "test_detect_mode_with_path_object",
    ),
    "test_discover_project_roots_without_nested_git_dirs": (
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_discover_project_roots_without_nested_git_dirs",
    ),
    "test_discover_projects_wrapper": (
        "tests.infra.unit.deps.test_detection_wrappers",
        "test_discover_projects_wrapper",
    ),
    "test_discovery_narrowed_function_paths": (
        "tests.unit.test_handlers_full_coverage",
        "test_discovery_narrowed_function_paths",
    ),
    "test_dispatch_after_handler_removal_fails": (
        "tests.unit.test_dispatcher_minimal",
        "test_dispatch_after_handler_removal_fails",
    ),
    "test_dispatch_auto_discovery_handler": (
        "tests.unit.test_dispatcher_minimal",
        "test_dispatch_auto_discovery_handler",
    ),
    "test_dispatch_coerce_mode_with_enum_string_and_other_object": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_dispatch_coerce_mode_with_enum_string_and_other_object",
    ),
    "test_dispatch_command_success": (
        "tests.unit.test_dispatcher_minimal",
        "test_dispatch_command_success",
    ),
    "test_dispatch_handler_exception_returns_failure": (
        "tests.unit.test_dispatcher_minimal",
        "test_dispatch_handler_exception_returns_failure",
    ),
    "test_dispatch_invalid_input_types": (
        "tests.unit.test_dispatcher_full_coverage",
        "test_dispatch_invalid_input_types",
    ),
    "test_dispatch_is_member_by_name_and_by_value": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_dispatch_is_member_by_name_and_by_value",
    ),
    "test_dispatch_is_name_mode": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_dispatch_is_name_mode",
    ),
    "test_dispatch_no_handler_fails": (
        "tests.unit.test_dispatcher_minimal",
        "test_dispatch_no_handler_fails",
    ),
    "test_dispatch_parse_mode_with_enum_string_and_other_object": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_dispatch_parse_mode_with_enum_string_and_other_object",
    ),
    "test_dispatch_unknown_mode_raises": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_dispatch_unknown_mode_raises",
    ),
    "test_dispatcher_family_blocks_models_target": (
        "tests.unit.test_refactor_policy_family_rules",
        "test_dispatcher_family_blocks_models_target",
    ),
    "test_dispatcher_reliability_branch_paths": (
        "tests.unit.test_dispatcher_reliability_full_coverage",
        "test_dispatcher_reliability_branch_paths",
    ),
    "test_engine_always_enables_class_nesting_file_rule": (
        "tests.infra.unit.refactor.test_infra_refactor_engine",
        "test_engine_always_enables_class_nesting_file_rule",
    ),
    "test_engine_constants_shared": (
        "tests.infra.unit.test_infra_templates",
        "test_engine_constants_shared",
    ),
    "test_enrich_and_ensure_trace_context_branches": (
        "tests.unit.test_utilities_generators_full_coverage",
        "test_enrich_and_ensure_trace_context_branches",
    ),
    "test_ensure_and_extract_array_index_helpers": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_ensure_and_extract_array_index_helpers",
    ),
    "test_ensure_dict_branches": (
        "tests.unit.test_utilities_generators_full_coverage",
        "test_ensure_dict_branches",
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
    "test_ensure_trace_context_dict_conversion_paths": (
        "tests.unit.test_runtime_full_coverage",
        "test_ensure_trace_context_dict_conversion_paths",
    ),
    "test_ensure_utc_datetime_adds_tzinfo_when_naive": (
        "tests.unit.test_models_validation_full_coverage",
        "test_ensure_utc_datetime_adds_tzinfo_when_naive",
    ),
    "test_ensure_utc_datetime_preserves_aware": (
        "tests.unit.test_models_validation_full_coverage",
        "test_ensure_utc_datetime_preserves_aware",
    ),
    "test_ensure_utc_datetime_returns_none_on_none": (
        "tests.unit.test_models_validation_full_coverage",
        "test_ensure_utc_datetime_returns_none_on_none",
    ),
    "test_entity_comparable_map_and_bulk_validation_paths": (
        "tests.unit.test_models_entity_full_coverage",
        "test_entity_comparable_map_and_bulk_validation_paths",
    ),
    "test_event_publishing_strict": (
        "tests.unit.test_dispatcher_full_coverage",
        "test_event_publishing_strict",
    ),
    "test_exception_handling_in_dispatch": (
        "tests.unit.test_dispatcher_full_coverage",
        "test_exception_handling_in_dispatch",
    ),
    "test_exceptions_uncovered_metadata_paths": (
        "tests.unit.test_exceptions_full_coverage",
        "test_exceptions_uncovered_metadata_paths",
    ),
    "test_execute_and_register_handler_failure_paths": (
        "tests.unit.test_registry_full_coverage",
        "test_execute_and_register_handler_failure_paths",
    ),
    "test_execute_retry_exponential_and_handle_exhaustion_raise_last_exception": (
        "tests.unit.test_decorators_full_coverage",
        "test_execute_retry_exponential_and_handle_exhaustion_raise_last_exception",
    ),
    "test_execute_retry_loop_covers_default_linear_and_never_ran": (
        "tests.unit.test_decorators_full_coverage",
        "test_execute_retry_loop_covers_default_linear_and_never_ran",
    ),
    "test_export_paths_with_metadata_and_statistics": (
        "tests.unit.test_context_full_coverage",
        "test_export_paths_with_metadata_and_statistics",
    ),
    "test_extract_dep_name": (
        "tests.infra.unit.deps.test_path_sync_helpers",
        "test_extract_dep_name",
    ),
    "test_extract_error_paths_and_prop_accessor": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_extract_error_paths_and_prop_accessor",
    ),
    "test_extract_field_value_and_ensure_variants": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_extract_field_value_and_ensure_variants",
    ),
    "test_extract_mapping_or_none_branches": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_extract_mapping_or_none_branches",
    ),
    "test_extract_message_type_annotation_and_dict_subclass_paths": (
        "tests.unit.test_utilities_checker_full_coverage",
        "test_extract_message_type_annotation_and_dict_subclass_paths",
    ),
    "test_extract_message_type_from_handle_with_only_self": (
        "tests.unit.test_utilities_checker_full_coverage",
        "test_extract_message_type_from_handle_with_only_self",
    ),
    "test_extract_message_type_from_parameter_branches": (
        "tests.unit.test_utilities_checker_full_coverage",
        "test_extract_message_type_from_parameter_branches",
    ),
    "test_extract_requirement_name": (
        "tests.infra.unit.deps.test_path_sync_helpers",
        "test_extract_requirement_name",
    ),
    "test_facade_binding_is_correct": (
        "tests.unit.test_models_validation_full_coverage",
        "test_facade_binding_is_correct",
    ),
    "test_field_and_fields_multi_branches": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_field_and_fields_multi_branches",
    ),
    "test_files_modified_tracks_affected_files": (
        "tests.infra.unit.codegen.autofix_workspace",
        "test_files_modified_tracks_affected_files",
    ),
    "test_filter_map_normalize_convert_helpers": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_filter_map_normalize_convert_helpers",
    ),
    "test_find_mapping_no_match_and_merge_error_paths": (
        "tests.unit.test_utilities_collection_full_coverage",
        "test_find_mapping_no_match_and_merge_error_paths",
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
    "test_flext_message_type_alias_adapter": (
        "tests.unit.test_models_cqrs_full_coverage",
        "test_flext_message_type_alias_adapter",
    ),
    "test_flow_through_short_circuits_on_failure": (
        "tests.unit.test_result_additional",
        "test_flow_through_short_circuits_on_failure",
    ),
    "test_framework": ("tests.conftest", "test_framework"),
    "test_from_validation_and_to_model_paths": (
        "tests.unit.test_result_full_coverage",
        "test_from_validation_and_to_model_paths",
    ),
    "test_frozen_value_model_equality_and_hash": (
        "tests.unit.test_models_base_full_coverage",
        "test_frozen_value_model_equality_and_hash",
    ),
    "test_general_value_helpers_and_logger": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_general_value_helpers_and_logger",
    ),
    "test_generate_special_paths_and_dynamic_subclass": (
        "tests.unit.test_utilities_generators_full_coverage",
        "test_generate_special_paths_and_dynamic_subclass",
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
    "test_generators_additional_missed_paths": (
        "tests.unit.test_utilities_generators_full_coverage",
        "test_generators_additional_missed_paths",
    ),
    "test_generators_mapping_non_dict_normalization_path": (
        "tests.unit.test_utilities_generators_full_coverage",
        "test_generators_mapping_non_dict_normalization_path",
    ),
    "test_get_and_get_typed_resource_factory_paths": (
        "tests.unit.test_container_full_coverage",
        "test_get_and_get_typed_resource_factory_paths",
    ),
    "test_get_current_typings_from_pyproject_wrapper": (
        "tests.infra.unit.deps.test_detection_wrappers",
        "test_get_current_typings_from_pyproject_wrapper",
    ),
    "test_get_enum_values_returns_immutable_sequence": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_get_enum_values_returns_immutable_sequence",
    ),
    "test_get_logger_none_name_paths": (
        "tests.unit.test_runtime_full_coverage",
        "test_get_logger_none_name_paths",
    ),
    "test_get_plugin_and_register_metadata_and_list_items_exception": (
        "tests.unit.test_registry_full_coverage",
        "test_get_plugin_and_register_metadata_and_list_items_exception",
    ),
    "test_get_required_typings_wrapper": (
        "tests.infra.unit.deps.test_detection_wrappers",
        "test_get_required_typings_wrapper",
    ),
    "test_get_service_info": (
        "tests.unit.test_service_additional",
        "test_get_service_info",
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
    "test_group_sort_unique_slice_chunk_branches": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_group_sort_unique_slice_chunk_branches",
    ),
    "test_guard_in_has_empty_none_helpers": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_guard_in_has_empty_none_helpers",
    ),
    "test_guard_instance_attribute_access_warnings": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_guard_instance_attribute_access_warnings",
    ),
    "test_guards_bool_identity_branch_via_isinstance_fallback": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_guards_bool_identity_branch_via_isinstance_fallback",
    ),
    "test_guards_bool_shortcut_and_issubclass_typeerror": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_guards_bool_shortcut_and_issubclass_typeerror",
    ),
    "test_guards_handler_type_issubclass_typeerror_branch_direct": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_guards_handler_type_issubclass_typeerror_branch_direct",
    ),
    "test_guards_issubclass_success_when_callable_is_patched": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_guards_issubclass_success_when_callable_is_patched",
    ),
    "test_guards_issubclass_typeerror_when_class_not_treated_as_callable": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_guards_issubclass_typeerror_when_class_not_treated_as_callable",
    ),
    "test_handle_log_result_without_fallback_logger_and_non_dict_like_extra": (
        "tests.unit.test_decorators_full_coverage",
        "test_handle_log_result_without_fallback_logger_and_non_dict_like_extra",
    ),
    "test_handle_retry_exhaustion_falsey_exception_reaches_timeout_error": (
        "tests.unit.test_decorators_full_coverage",
        "test_handle_retry_exhaustion_falsey_exception_reaches_timeout_error",
    ),
    "test_handler_attribute_discovery": (
        "tests.unit.test_dispatcher_full_coverage",
        "test_handler_attribute_discovery",
    ),
    "test_handler_builder_fluent_methods": (
        "tests.unit.test_models_cqrs_full_coverage",
        "test_handler_builder_fluent_methods",
    ),
    "test_handler_type_literal_and_invalid": (
        "tests.unit.test_handlers_full_coverage",
        "test_handler_type_literal_and_invalid",
    ),
    "test_helper_consolidation_is_prechecked": (
        "tests.unit.test_refactor_policy_family_rules",
        "test_helper_consolidation_is_prechecked",
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
    "test_helpers_alias_is_reachable_pep621": (
        "tests.infra.unit.deps.test_path_sync_rewrite_pep621",
        "test_helpers_alias_is_reachable_pep621",
    ),
    "test_helpers_alias_is_reachable_poetry": (
        "tests.infra.unit.deps.test_path_sync_rewrite_poetry",
        "test_helpers_alias_is_reachable_poetry",
    ),
    "test_helpers_alias_is_reachable_project_obj": (
        "tests.infra.unit.deps.test_path_sync_main_project_obj",
        "test_helpers_alias_is_reachable_project_obj",
    ),
    "test_identifiable_unique_id_empty_rejected": (
        "tests.unit.test_models_base_full_coverage",
        "test_identifiable_unique_id_empty_rejected",
    ),
    "test_implements_decorator_helper_methods_and_static_wrappers": (
        "tests.unit.test_protocols_full_coverage",
        "test_implements_decorator_helper_methods_and_static_wrappers",
    ),
    "test_implements_decorator_validation_error_message": (
        "tests.unit.test_protocols_full_coverage",
        "test_implements_decorator_validation_error_message",
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
    "test_inactive_and_none_value_paths": (
        "tests.unit.test_context_full_coverage",
        "test_inactive_and_none_value_paths",
    ),
    "test_init_fallback_and_lazy_result_property": (
        "tests.unit.test_result_full_coverage",
        "test_init_fallback_and_lazy_result_property",
    ),
    "test_initialize_di_components_error_paths": (
        "tests.unit.test_container_full_coverage",
        "test_initialize_di_components_error_paths",
    ),
    "test_initialize_di_components_second_type_error_branch": (
        "tests.unit.test_container_full_coverage",
        "test_initialize_di_components_second_type_error_branch",
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
    "test_inject_sets_missing_dependency_from_container": (
        "tests.unit.test_decorators_full_coverage",
        "test_inject_sets_missing_dependency_from_container",
    ),
    "test_invalid_handler_mode_init_raises": (
        "tests.unit.test_handlers_full_coverage",
        "test_invalid_handler_mode_init_raises",
    ),
    "test_invalid_registration_attempts": (
        "tests.unit.test_dispatcher_full_coverage",
        "test_invalid_registration_attempts",
    ),
    "test_invert_and_json_conversion_branches": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_invert_and_json_conversion_branches",
    ),
    "test_is_flexible_value_covers_all_branches": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_is_flexible_value_covers_all_branches",
    ),
    "test_is_general_value_type_negative_paths_and_callable": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_is_general_value_type_negative_paths_and_callable",
    ),
    "test_is_handler_type_branches": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_is_handler_type_branches",
    ),
    "test_is_type_non_empty_unknown_and_tuple_and_fallback": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_is_type_non_empty_unknown_and_tuple_and_fallback",
    ),
    "test_is_type_protocol_fallback_branches": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_is_type_protocol_fallback_branches",
    ),
    "test_is_valid_handles_validation_exception": (
        "tests.unit.test_service_additional",
        "test_is_valid_handles_validation_exception",
    ),
    "test_lash_runtime_result_and_from_io_result_fallback": (
        "tests.unit.test_result_full_coverage",
        "test_lash_runtime_result_and_from_io_result_fallback",
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
    "test_log_operation_track_perf_exception_adds_duration": (
        "tests.unit.test_decorators_full_coverage",
        "test_log_operation_track_perf_exception_adds_duration",
    ),
    "test_loggings_bind_clear_level_error_paths": (
        "tests.unit.test_loggings_full_coverage",
        "test_loggings_bind_clear_level_error_paths",
    ),
    "test_loggings_context_and_factory_paths": (
        "tests.unit.test_loggings_full_coverage",
        "test_loggings_context_and_factory_paths",
    ),
    "test_loggings_exception_and_adapter_paths": (
        "tests.unit.test_loggings_full_coverage",
        "test_loggings_exception_and_adapter_paths",
    ),
    "test_loggings_instance_and_message_format_paths": (
        "tests.unit.test_loggings_full_coverage",
        "test_loggings_instance_and_message_format_paths",
    ),
    "test_loggings_remaining_branch_paths": (
        "tests.unit.test_loggings_full_coverage",
        "test_loggings_remaining_branch_paths",
    ),
    "test_loggings_source_and_log_error_paths": (
        "tests.unit.test_loggings_full_coverage",
        "test_loggings_source_and_log_error_paths",
    ),
    "test_loggings_uncovered_level_trace_path_and_exception_guards": (
        "tests.unit.test_loggings_full_coverage",
        "test_loggings_uncovered_level_trace_path_and_exception_guards",
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
    "test_map_error_identity_and_transform": (
        "tests.unit.test_result_additional",
        "test_map_error_identity_and_transform",
    ),
    "test_map_flags_collect_and_invert_branches": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_map_flags_collect_and_invert_branches",
    ),
    "test_map_flat_map_and_then_paths": (
        "tests.unit.test_result_full_coverage",
        "test_map_flat_map_and_then_paths",
    ),
    "test_members_uses_cache_on_second_call": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_members_uses_cache_on_second_call",
    ),
    "test_merge_defaults_and_dump_paths": (
        "tests.unit.test_utilities_model_full_coverage",
        "test_merge_defaults_and_dump_paths",
    ),
    "test_merge_metadata_context_paths": (
        "tests.unit.test_exceptions_full_coverage",
        "test_merge_metadata_context_paths",
    ),
    "test_metadata_attributes_accepts_basemodel_mapping": (
        "tests.unit.test_models_base_full_coverage",
        "test_metadata_attributes_accepts_basemodel_mapping",
    ),
    "test_metadata_attributes_accepts_none": (
        "tests.unit.test_models_base_full_coverage",
        "test_metadata_attributes_accepts_none",
    ),
    "test_metadata_attributes_accepts_t_dict_and_mapping": (
        "tests.unit.test_models_base_full_coverage",
        "test_metadata_attributes_accepts_t_dict_and_mapping",
    ),
    "test_metadata_attributes_rejects_basemodel_non_mapping_dump": (
        "tests.unit.test_models_base_full_coverage",
        "test_metadata_attributes_rejects_basemodel_non_mapping_dump",
    ),
    "test_metadata_attributes_rejects_non_mapping": (
        "tests.unit.test_models_base_full_coverage",
        "test_metadata_attributes_rejects_non_mapping",
    ),
    "test_migrate_makefile_not_found_non_dry_run": (
        "tests.infra.unit.test_infra_workspace_migrator_deps",
        "test_migrate_makefile_not_found_non_dry_run",
    ),
    "test_migrate_protocols_rewrites_references_with_p_alias": (
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_protocols_rewrites_references_with_p_alias",
    ),
    "test_migrate_pyproject_flext_core_non_dry_run": (
        "tests.infra.unit.test_infra_workspace_migrator_deps",
        "test_migrate_pyproject_flext_core_non_dry_run",
    ),
    "test_migrate_to_mro_inlines_alias_constant_into_constants_class": (
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_to_mro_inlines_alias_constant_into_constants_class",
    ),
    "test_migrate_to_mro_moves_constant_and_rewrites_reference": (
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_to_mro_moves_constant_and_rewrites_reference",
    ),
    "test_migrate_to_mro_moves_manual_uppercase_assignment": (
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_to_mro_moves_manual_uppercase_assignment",
    ),
    "test_migrate_to_mro_normalizes_facade_alias_to_c": (
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_to_mro_normalizes_facade_alias_to_c",
    ),
    "test_migrate_to_mro_rejects_unknown_target": (
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_to_mro_rejects_unknown_target",
    ),
    "test_migrate_typings_rewrites_references_with_t_alias": (
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_typings_rewrites_references_with_t_alias",
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
    "test_misc_unregistration_clear_and_reset": (
        "tests.unit.test_container_full_coverage",
        "test_misc_unregistration_clear_and_reset",
    ),
    "test_mixins_container_registration_and_logger_paths": (
        "tests.unit.test_mixins_full_coverage",
        "test_mixins_container_registration_and_logger_paths",
    ),
    "test_mixins_context_logging_and_cqrs_paths": (
        "tests.unit.test_mixins_full_coverage",
        "test_mixins_context_logging_and_cqrs_paths",
    ),
    "test_mixins_context_stack_pop_initializes_missing_stack_attr": (
        "tests.unit.test_mixins_full_coverage",
        "test_mixins_context_stack_pop_initializes_missing_stack_attr",
    ),
    "test_mixins_remaining_branch_paths": (
        "tests.unit.test_mixins_full_coverage",
        "test_mixins_remaining_branch_paths",
    ),
    "test_mixins_result_and_model_conversion_paths": (
        "tests.unit.test_mixins_full_coverage",
        "test_mixins_result_and_model_conversion_paths",
    ),
    "test_mixins_runtime_bootstrap_and_track_paths": (
        "tests.unit.test_mixins_full_coverage",
        "test_mixins_runtime_bootstrap_and_track_paths",
    ),
    "test_mixins_validation_and_protocol_paths": (
        "tests.unit.test_mixins_full_coverage",
        "test_mixins_validation_and_protocol_paths",
    ),
    "test_model_helpers_remaining_paths": (
        "tests.unit.test_runtime_full_coverage",
        "test_model_helpers_remaining_paths",
    ),
    "test_model_support_and_hash_compare_paths": (
        "tests.unit.test_runtime_full_coverage",
        "test_model_support_and_hash_compare_paths",
    ),
    "test_models_family_blocks_utilities_target": (
        "tests.unit.test_refactor_policy_family_rules",
        "test_models_family_blocks_utilities_target",
    ),
    "test_models_handler_branches": (
        "tests.unit.test_models_handler_full_coverage",
        "test_models_handler_branches",
    ),
    "test_models_handler_uncovered_mode_and_reset_paths": (
        "tests.unit.test_models_handler_full_coverage",
        "test_models_handler_uncovered_mode_and_reset_paths",
    ),
    "test_models_settings_branch_paths": (
        "tests.unit.test_models_settings_full_coverage",
        "test_models_settings_branch_paths",
    ),
    "test_models_settings_context_validator_and_non_standard_status_input": (
        "tests.unit.test_models_settings_full_coverage",
        "test_models_settings_context_validator_and_non_standard_status_input",
    ),
    "test_mro_checker_keeps_external_attribute_base": (
        "tests.infra.unit.refactor.test_infra_refactor_class_and_propagation",
        "test_mro_checker_keeps_external_attribute_base",
    ),
    "test_mro_redundancy_checker_removes_nested_attribute_inheritance": (
        "tests.infra.unit.refactor.test_infra_refactor_class_and_propagation",
        "test_mro_redundancy_checker_removes_nested_attribute_inheritance",
    ),
    "test_mro_resolver_accepts_expected_order": (
        "tests.integration.test_refactor_policy_mro",
        "test_mro_resolver_accepts_expected_order",
    ),
    "test_mro_resolver_rejects_wrong_order": (
        "tests.integration.test_refactor_policy_mro",
        "test_mro_resolver_rejects_wrong_order",
    ),
    "test_mro_scanner_includes_constants_variants_in_all_scopes": (
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_mro_scanner_includes_constants_variants_in_all_scopes",
    ),
    "test_multiple_instances_independent": (
        "tests.infra.unit.test_infra_templates",
        "test_multiple_instances_independent",
    ),
    "test_names_uses_cache_on_second_call": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_names_uses_cache_on_second_call",
    ),
    "test_namespace_enforce_cli_fails_on_manual_protocol_violation": (
        "tests.unit.test_refactor_cli_models_workflow",
        "test_namespace_enforce_cli_fails_on_manual_protocol_violation",
    ),
    "test_namespace_enforcer_apply_inserts_future_after_single_line_module_docstring": (
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_apply_inserts_future_after_single_line_module_docstring",
    ),
    "test_namespace_enforcer_apply_keeps_script_shebang_when_adding_future": (
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_apply_keeps_script_shebang_when_adding_future",
    ),
    "test_namespace_enforcer_apply_moves_manual_protocol_to_protocols_file": (
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_apply_moves_manual_protocol_to_protocols_file",
    ),
    "test_namespace_enforcer_creates_missing_facades_and_rewrites_imports": (
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_creates_missing_facades_and_rewrites_imports",
    ),
    "test_namespace_enforcer_detects_cyclic_imports_in_tests_directory": (
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_detects_cyclic_imports_in_tests_directory",
    ),
    "test_namespace_enforcer_detects_internal_private_imports": (
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_detects_internal_private_imports",
    ),
    "test_namespace_enforcer_detects_manual_protocol_outside_canonical_files": (
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_detects_manual_protocol_outside_canonical_files",
    ),
    "test_namespace_enforcer_detects_manual_typings_and_compat_aliases": (
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_detects_manual_typings_and_compat_aliases",
    ),
    "test_namespace_enforcer_detects_missing_runtime_alias_outside_src": (
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_detects_missing_runtime_alias_outside_src",
    ),
    "test_namespace_enforcer_does_not_rewrite_indented_import_aliases": (
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_does_not_rewrite_indented_import_aliases",
    ),
    "test_namespace_enforcer_does_not_rewrite_multiline_import_alias_blocks": (
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_does_not_rewrite_multiline_import_alias_blocks",
    ),
    "test_narrow_contextvar_invalid_inputs": (
        "tests.unit.test_context_full_coverage",
        "test_narrow_contextvar_invalid_inputs",
    ),
    "test_narrow_to_string_keyed_dict_and_mapping_paths": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_narrow_to_string_keyed_dict_and_mapping_paths",
    ),
    "test_nested_class_propagation_preserves_asname_and_rewrites_alias_usage": (
        "tests.unit.test_transformer_nested_class_propagation",
        "test_nested_class_propagation_preserves_asname_and_rewrites_alias_usage",
    ),
    "test_nested_class_propagation_updates_import_annotations_and_calls": (
        "tests.unit.test_transformer_nested_class_propagation",
        "test_nested_class_propagation_updates_import_annotations_and_calls",
    ),
    "test_non_empty_and_normalize_branches": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_non_empty_and_normalize_branches",
    ),
    "test_normalization_edge_branches": (
        "tests.unit.test_runtime_full_coverage",
        "test_normalization_edge_branches",
    ),
    "test_normalize_context_to_dict_error_paths": (
        "tests.unit.test_utilities_generators_full_coverage",
        "test_normalize_context_to_dict_error_paths",
    ),
    "test_normalize_to_list_passes_list_through": (
        "tests.unit.test_models_validation_full_coverage",
        "test_normalize_to_list_passes_list_through",
    ),
    "test_normalize_to_list_wraps_int": (
        "tests.unit.test_models_validation_full_coverage",
        "test_normalize_to_list_wraps_int",
    ),
    "test_normalize_to_list_wraps_scalar": (
        "tests.unit.test_models_validation_full_coverage",
        "test_normalize_to_list_wraps_scalar",
    ),
    "test_normalize_to_pydantic_dict_and_value_branches": (
        "tests.unit.test_utilities_model_full_coverage",
        "test_normalize_to_pydantic_dict_and_value_branches",
    ),
    "test_not_found_error_correlation_id_selection_and_extra_kwargs": (
        "tests.unit.test_exceptions_full_coverage",
        "test_not_found_error_correlation_id_selection_and_extra_kwargs",
    ),
    "test_object_dict_and_type_error_fallback_paths": (
        "tests.unit.test_utilities_checker_full_coverage",
        "test_object_dict_and_type_error_fallback_paths",
    ),
    "test_ok_raises_on_none": (
        "tests.unit.test_result_additional",
        "test_ok_raises_on_none",
    ),
    "test_operation_progress_start_operation_sets_runtime_fields": (
        "tests.unit.test_models_generic_full_coverage",
        "test_operation_progress_start_operation_sets_runtime_fields",
    ),
    "test_options_merge_conflict_paths_and_empty_merge_options": (
        "tests.unit.test_models_collections_full_coverage",
        "test_options_merge_conflict_paths_and_empty_merge_options",
    ),
    "test_pagination_response_string_fallbacks": (
        "tests.unit.test_utilities_pagination_full_coverage",
        "test_pagination_response_string_fallbacks",
    ),
    "test_parse_mapping_outer_exception": (
        "tests.unit.test_utilities_collection_full_coverage",
        "test_parse_mapping_outer_exception",
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
    "test_parser_convert_and_norm_branches": (
        "tests.unit.test_utilities_parser_full_coverage",
        "test_parser_convert_and_norm_branches",
    ),
    "test_parser_internal_helpers_additional_coverage": (
        "tests.unit.test_utilities_parser_full_coverage",
        "test_parser_internal_helpers_additional_coverage",
    ),
    "test_parser_parse_helpers_and_primitive_coercion_branches": (
        "tests.unit.test_utilities_parser_full_coverage",
        "test_parser_parse_helpers_and_primitive_coercion_branches",
    ),
    "test_parser_pipeline_and_pattern_branches": (
        "tests.unit.test_utilities_parser_full_coverage",
        "test_parser_pipeline_and_pattern_branches",
    ),
    "test_parser_remaining_branch_paths": (
        "tests.unit.test_utilities_parser_full_coverage",
        "test_parser_remaining_branch_paths",
    ),
    "test_parser_safe_length_and_parse_delimited_error_paths": (
        "tests.unit.test_utilities_parser_full_coverage",
        "test_parser_safe_length_and_parse_delimited_error_paths",
    ),
    "test_parser_split_and_normalize_exception_paths": (
        "tests.unit.test_utilities_parser_full_coverage",
        "test_parser_split_and_normalize_exception_paths",
    ),
    "test_parser_success_and_edge_paths_cover_major_branches": (
        "tests.unit.test_utilities_parser_full_coverage",
        "test_parser_success_and_edge_paths_cover_major_branches",
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
    "test_private_coerce_with_enum_and_string": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_private_coerce_with_enum_and_string",
    ),
    "test_private_getters_exception_paths": (
        "tests.unit.test_utilities_configuration_full_coverage",
        "test_private_getters_exception_paths",
    ),
    "test_private_is_member_by_name": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_private_is_member_by_name",
    ),
    "test_private_is_member_by_value": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_private_is_member_by_value",
    ),
    "test_private_parse_success_and_failure": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_private_parse_success_and_failure",
    ),
    "test_process_context_data_and_related_convenience": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_process_context_data_and_related_convenience",
    ),
    "test_process_outer_exception_and_coercion_branches": (
        "tests.unit.test_utilities_collection_full_coverage",
        "test_process_outer_exception_and_coercion_branches",
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
    "test_protocol_and_simple_guard_helpers": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_protocol_and_simple_guard_helpers",
    ),
    "test_protocol_base_name_methods_and_runtime_check_branch": (
        "tests.unit.test_protocols_full_coverage",
        "test_protocol_base_name_methods_and_runtime_check_branch",
    ),
    "test_protocol_meta_default_model_base_and_get_protocols_default": (
        "tests.unit.test_protocols_full_coverage",
        "test_protocol_meta_default_model_base_and_get_protocols_default",
    ),
    "test_protocol_model_and_settings_methods": (
        "tests.unit.test_protocols_full_coverage",
        "test_protocol_model_and_settings_methods",
    ),
    "test_protocol_name_and_builder": (
        "tests.unit.test_container_full_coverage",
        "test_protocol_name_and_builder",
    ),
    "test_protocol_name_and_narrow_contextvar_exception_branch": (
        "tests.unit.test_context_full_coverage",
        "test_protocol_name_and_narrow_contextvar_exception_branch",
    ),
    "test_provide_property_paths": (
        "tests.unit.test_container_full_coverage",
        "test_provide_property_paths",
    ),
    "test_publish_event_to_subscriber": (
        "tests.unit.test_dispatcher_minimal",
        "test_publish_event_to_subscriber",
    ),
    "test_publish_no_subscribers_succeeds": (
        "tests.unit.test_dispatcher_minimal",
        "test_publish_no_subscribers_succeeds",
    ),
    "test_query_resolve_pagination_wrapper_and_fallback": (
        "tests.unit.test_models_cqrs_full_coverage",
        "test_query_resolve_pagination_wrapper_and_fallback",
    ),
    "test_query_validate_pagination_dict_and_default": (
        "tests.unit.test_models_cqrs_full_coverage",
        "test_query_validate_pagination_dict_and_default",
    ),
    "test_railway_and_retry_additional_paths": (
        "tests.unit.test_decorators_full_coverage",
        "test_railway_and_retry_additional_paths",
    ),
    "test_rate_limiter_blocks_then_recovers": (
        "tests.unit.test_dispatcher_reliability",
        "test_rate_limiter_blocks_then_recovers",
    ),
    "test_rate_limiter_jitter_application": (
        "tests.unit.test_dispatcher_reliability",
        "test_rate_limiter_jitter_application",
    ),
    "test_reconfigure_and_reset_state_paths": (
        "tests.unit.test_runtime_full_coverage",
        "test_reconfigure_and_reset_state_paths",
    ),
    "test_recover_tap_and_tap_error_paths": (
        "tests.unit.test_result_full_coverage",
        "test_recover_tap_and_tap_error_paths",
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
    "test_refactor_utilities_iter_python_files_includes_examples_and_scripts": (
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_refactor_utilities_iter_python_files_includes_examples_and_scripts",
    ),
    "test_register_existing_providers_full_paths_and_misc_methods": (
        "tests.unit.test_container_full_coverage",
        "test_register_existing_providers_full_paths_and_misc_methods",
    ),
    "test_register_existing_providers_skips_and_register_core_fallback": (
        "tests.unit.test_container_full_coverage",
        "test_register_existing_providers_skips_and_register_core_fallback",
    ),
    "test_register_handler_as_event_subscriber": (
        "tests.unit.test_dispatcher_minimal",
        "test_register_handler_as_event_subscriber",
    ),
    "test_register_handler_with_can_handle": (
        "tests.unit.test_dispatcher_minimal",
        "test_register_handler_with_can_handle",
    ),
    "test_register_handler_with_message_type": (
        "tests.unit.test_dispatcher_minimal",
        "test_register_handler_with_message_type",
    ),
    "test_register_handler_without_route_fails": (
        "tests.unit.test_dispatcher_minimal",
        "test_register_handler_without_route_fails",
    ),
    "test_register_singleton_register_factory_and_bulk_register_paths": (
        "tests.unit.test_utilities_configuration_full_coverage",
        "test_register_singleton_register_factory_and_bulk_register_paths",
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
    "test_remaining_build_fields_construct_and_eq_paths": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_remaining_build_fields_construct_and_eq_paths",
    ),
    "test_remaining_uncovered_branches": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_remaining_uncovered_branches",
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
    "test_resolve_env_file_and_log_level": (
        "tests.unit.test_utilities_configuration_full_coverage",
        "test_resolve_env_file_and_log_level",
    ),
    "test_resolve_gates_maps_type_alias": (
        "tests.infra.unit.check.cli",
        "test_resolve_gates_maps_type_alias",
    ),
    "test_resolve_logger_prefers_logger_attribute": (
        "tests.unit.test_decorators_full_coverage",
        "test_resolve_logger_prefers_logger_attribute",
    ),
    "test_result_property_raises_on_failure": (
        "tests.unit.test_service_additional",
        "test_result_property_raises_on_failure",
    ),
    "test_results_internal_conflict_paths_and_combine": (
        "tests.unit.test_models_collections_full_coverage",
        "test_results_internal_conflict_paths_and_combine",
    ),
    "test_retry_policy_behavior": (
        "tests.unit.test_dispatcher_reliability",
        "test_retry_policy_behavior",
    ),
    "test_retry_unreachable_timeouterror_path": (
        "tests.unit.test_decorators_full_coverage",
        "test_retry_unreachable_timeouterror_path",
    ),
    "test_reuse_existing_runtime_coverage_branches": (
        "tests.unit.test_runtime_full_coverage",
        "test_reuse_existing_runtime_coverage_branches",
    ),
    "test_reuse_existing_runtime_scenarios": (
        "tests.unit.test_runtime_full_coverage",
        "test_reuse_existing_runtime_scenarios",
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
    "test_rules_merge_combines_model_dump_values": (
        "tests.unit.test_models_collections_full_coverage",
        "test_rules_merge_combines_model_dump_values",
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
    "test_run_pipeline_query_and_event_paths": (
        "tests.unit.test_handlers_full_coverage",
        "test_run_pipeline_query_and_event_paths",
    ),
    "test_run_raw_cases": (
        "tests.infra.unit.test_infra_subprocess_core",
        "test_run_raw_cases",
    ),
    "test_runtime_create_instance_failure_branch": (
        "tests.unit.test_runtime_full_coverage",
        "test_runtime_create_instance_failure_branch",
    ),
    "test_runtime_family_blocks_non_runtime_target": (
        "tests.unit.test_refactor_policy_family_rules",
        "test_runtime_family_blocks_non_runtime_target",
    ),
    "test_runtime_integration_tracking_paths": (
        "tests.unit.test_runtime_full_coverage",
        "test_runtime_integration_tracking_paths",
    ),
    "test_runtime_misc_remaining_paths": (
        "tests.unit.test_runtime_full_coverage",
        "test_runtime_misc_remaining_paths",
    ),
    "test_runtime_module_accessors_and_metadata": (
        "tests.unit.test_runtime_full_coverage",
        "test_runtime_module_accessors_and_metadata",
    ),
    "test_runtime_result_alias_compatibility": (
        "tests.unit.test_runtime_full_coverage",
        "test_runtime_result_alias_compatibility",
    ),
    "test_runtime_result_all_missed_branches": (
        "tests.unit.test_runtime_full_coverage",
        "test_runtime_result_all_missed_branches",
    ),
    "test_runtime_result_remaining_paths": (
        "tests.unit.test_runtime_full_coverage",
        "test_runtime_result_remaining_paths",
    ),
    "test_scope_data_validators_and_errors": (
        "tests.unit.test_models_context_full_coverage",
        "test_scope_data_validators_and_errors",
    ),
    "test_scoped_config_context_branches": (
        "tests.unit.test_container_full_coverage",
        "test_scoped_config_context_branches",
    ),
    "test_service_create_initial_runtime_prefers_custom_config_type_and_context_property": (
        "tests.unit.test_service_full_coverage",
        "test_service_create_initial_runtime_prefers_custom_config_type_and_context_property",
    ),
    "test_service_create_runtime_container_overrides_branch": (
        "tests.unit.test_service_full_coverage",
        "test_service_create_runtime_container_overrides_branch",
    ),
    "test_service_init_type_guards_and_properties": (
        "tests.unit.test_service_full_coverage",
        "test_service_init_type_guards_and_properties",
    ),
    "test_service_request_timeout_post_validator_messages": (
        "tests.unit.test_models_service_full_coverage",
        "test_service_request_timeout_post_validator_messages",
    ),
    "test_service_request_timeout_validator_branches": (
        "tests.unit.test_models_service_full_coverage",
        "test_service_request_timeout_validator_branches",
    ),
    "test_set_set_all_get_validation_and_error_paths": (
        "tests.unit.test_context_full_coverage",
        "test_set_set_all_get_validation_and_error_paths",
    ),
    "test_settings_materialize_and_context_overrides": (
        "tests.unit.test_settings_full_coverage",
        "test_settings_materialize_and_context_overrides",
    ),
    "test_shortcuts_delegate_to_primary_methods": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_shortcuts_delegate_to_primary_methods",
    ),
    "test_signature_propagation_removes_and_adds_keywords": (
        "tests.infra.unit.refactor.test_infra_refactor_class_and_propagation",
        "test_signature_propagation_removes_and_adds_keywords",
    ),
    "test_signature_propagation_renames_call_keyword": (
        "tests.infra.unit.refactor.test_infra_refactor_class_and_propagation",
        "test_signature_propagation_renames_call_keyword",
    ),
    "test_small_mapper_convenience_methods": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_small_mapper_convenience_methods",
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
    "test_statistics_and_custom_fields_validators": (
        "tests.unit.test_models_context_full_coverage",
        "test_statistics_and_custom_fields_validators",
    ),
    "test_statistics_from_dict_and_none_conflict_resolution": (
        "tests.unit.test_models_collections_full_coverage",
        "test_statistics_from_dict_and_none_conflict_resolution",
    ),
    "test_strict_registration_and_dispatch": (
        "tests.unit.test_dispatcher_full_coverage",
        "test_strict_registration_and_dispatch",
    ),
    "test_string_zero_return_value": (
        "tests.infra.unit.deps.test_main_dispatch",
        "test_string_zero_return_value",
    ),
    "test_strip_whitespace_preserves_clean": (
        "tests.unit.test_models_validation_full_coverage",
        "test_strip_whitespace_preserves_clean",
    ),
    "test_strip_whitespace_returns_empty_on_spaces": (
        "tests.unit.test_models_validation_full_coverage",
        "test_strip_whitespace_returns_empty_on_spaces",
    ),
    "test_strip_whitespace_trims_leading_trailing": (
        "tests.unit.test_models_validation_full_coverage",
        "test_strip_whitespace_trims_leading_trailing",
    ),
    "test_structlog_proxy_context_var_default_when_key_missing": (
        "tests.unit.test_models_context_full_coverage",
        "test_structlog_proxy_context_var_default_when_key_missing",
    ),
    "test_structlog_proxy_context_var_get_set_reset_paths": (
        "tests.unit.test_models_context_full_coverage",
        "test_structlog_proxy_context_var_get_set_reset_paths",
    ),
    "test_summary_error_paths_and_bindings_failures": (
        "tests.unit.test_registry_full_coverage",
        "test_summary_error_paths_and_bindings_failures",
    ),
    "test_summary_properties_and_subclass_storage_reset": (
        "tests.unit.test_registry_full_coverage",
        "test_summary_properties_and_subclass_storage_reset",
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
    "test_sync_config_namespace_paths": (
        "tests.unit.test_container_full_coverage",
        "test_sync_config_namespace_paths",
    ),
    "test_sync_config_registers_namespace_factories_and_fallbacks": (
        "tests.unit.test_container_full_coverage",
        "test_sync_config_registers_namespace_factories_and_fallbacks",
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
    "test_timeout_additional_success_and_reraise_timeout_paths": (
        "tests.unit.test_decorators_full_coverage",
        "test_timeout_additional_success_and_reraise_timeout_paths",
    ),
    "test_timeout_covers_exception_timeout_branch": (
        "tests.unit.test_decorators_full_coverage",
        "test_timeout_covers_exception_timeout_branch",
    ),
    "test_timeout_raises_when_successful_call_exceeds_limit": (
        "tests.unit.test_decorators_full_coverage",
        "test_timeout_raises_when_successful_call_exceeds_limit",
    ),
    "test_timeout_reraises_original_exception_when_within_limit": (
        "tests.unit.test_decorators_full_coverage",
        "test_timeout_reraises_original_exception_when_within_limit",
    ),
    "test_timestampable_timestamp_conversion_and_json_serializer": (
        "tests.unit.test_models_base_full_coverage",
        "test_timestampable_timestamp_conversion_and_json_serializer",
    ),
    "test_timestamped_model_and_alias_and_canonical_symbols": (
        "tests.unit.test_models_base_full_coverage",
        "test_timestamped_model_and_alias_and_canonical_symbols",
    ),
    "test_to_flexible_value_and_safe_list_branches": (
        "tests.unit.test_utilities_conversion_full_coverage",
        "test_to_flexible_value_and_safe_list_branches",
    ),
    "test_to_flexible_value_fallback_none_branch_for_unsupported_type": (
        "tests.unit.test_utilities_conversion_full_coverage",
        "test_to_flexible_value_fallback_none_branch_for_unsupported_type",
    ),
    "test_to_general_value_dict_removed": (
        "tests.unit.test_models_context_full_coverage",
        "test_to_general_value_dict_removed",
    ),
    "test_to_io_result_failure_path": (
        "tests.unit.test_result_additional",
        "test_to_io_result_failure_path",
    ),
    "test_track_performance_success_and_failure_paths": (
        "tests.unit.test_decorators_full_coverage",
        "test_track_performance_success_and_failure_paths",
    ),
    "test_transform_option_extract_and_step_helpers": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_transform_option_extract_and_step_helpers",
    ),
    "test_type_guards_and_narrowing_failures": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_type_guards_and_narrowing_failures",
    ),
    "test_type_guards_and_protocol_name": (
        "tests.unit.test_result_full_coverage",
        "test_type_guards_and_protocol_name",
    ),
    "test_ultrawork_models_cli_runs_dry_run_copy": (
        "tests.unit.test_refactor_cli_models_workflow",
        "test_ultrawork_models_cli_runs_dry_run_copy",
    ),
    "test_unwrap_item": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "test_unwrap_item",
    ),
    "test_unwrap_item_toml_item": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "test_unwrap_item_toml_item",
    ),
    "test_update_exception_path": (
        "tests.unit.test_utilities_model_full_coverage",
        "test_update_exception_path",
    ),
    "test_update_statistics_remove_hook_and_clone_false_result": (
        "tests.unit.test_context_full_coverage",
        "test_update_statistics_remove_hook_and_clone_false_result",
    ),
    "test_update_success_path_returns_ok_result": (
        "tests.unit.test_utilities_model_full_coverage",
        "test_update_success_path_returns_ok_result",
    ),
    "test_utilities_family_allows_utilities_target": (
        "tests.unit.test_refactor_policy_family_rules",
        "test_utilities_family_allows_utilities_target",
    ),
    "test_utilities_reliability_branches": (
        "tests.unit.test_utilities_reliability_full_coverage",
        "test_utilities_reliability_branches",
    ),
    "test_utilities_reliability_compose_returns_non_result_directly": (
        "tests.unit.test_utilities_reliability_full_coverage",
        "test_utilities_reliability_compose_returns_non_result_directly",
    ),
    "test_utilities_reliability_uncovered_retry_compose_and_sequence_paths": (
        "tests.unit.test_utilities_reliability_full_coverage",
        "test_utilities_reliability_uncovered_retry_compose_and_sequence_paths",
    ),
    "test_validate_config_dict_normalizes_dict": (
        "tests.unit.test_models_validation_full_coverage",
        "test_validate_config_dict_normalizes_dict",
    ),
    "test_validate_tags_list_from_string": (
        "tests.unit.test_models_validation_full_coverage",
        "test_validate_tags_list_from_string",
    ),
    "test_validate_tags_list_normalizes": (
        "tests.unit.test_models_validation_full_coverage",
        "test_validate_tags_list_normalizes",
    ),
    "test_validate_value_object_immutable_exception_and_no_setattr_branch": (
        "tests.unit.test_utilities_domain_full_coverage",
        "test_validate_value_object_immutable_exception_and_no_setattr_branch",
    ),
    "test_validation_like_error_structure": (
        "tests.unit.test_result_full_coverage",
        "test_validation_like_error_structure",
    ),
    "test_violation_analysis_counts_massive_patterns": (
        "tests.infra.unit.refactor.test_infra_refactor_analysis",
        "test_violation_analysis_counts_massive_patterns",
    ),
    "test_violation_analyzer_skips_non_utf8_files": (
        "tests.infra.unit.refactor.test_infra_refactor_analysis",
        "test_violation_analyzer_skips_non_utf8_files",
    ),
    "test_with_correlation_with_context_track_operation_and_factory": (
        "tests.unit.test_decorators_full_coverage",
        "test_with_correlation_with_context_track_operation_and_factory",
    ),
    "test_with_resource_cleanup_runs": (
        "tests.unit.test_result_additional",
        "test_with_resource_cleanup_runs",
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
    "u": ("tests.utilities", "u"),
    "v": ("tests.infra.unit.core.basemk_validator", "v"),
    "valid_hostnames": ("tests.conftest", "valid_hostnames"),
    "valid_port_numbers": ("tests.conftest", "valid_port_numbers"),
    "valid_ranges": ("tests.conftest", "valid_ranges"),
    "valid_strings": ("tests.conftest", "valid_strings"),
    "valid_uris": ("tests.conftest", "valid_uris"),
    "validation_scenarios": ("tests.conftest", "validation_scenarios"),
    "validator": ("tests.infra.unit.docs.validator_internals", "validator"),
    "whitespace_strings": ("tests.conftest", "whitespace_strings"),
    "workspace_root": (
        "tests.infra.unit.release.orchestrator_publish",
        "workspace_root",
    ),
    "x": ("tests.unit.test_automated_mixins", "TestAutomatedFlextMixins"),
}

__all__ = [
    "AlgarOudMigConstants",
    "AlgarOudMigModels",
    "AlgarOudMigProtocols",
    "AlgarOudMigTypes",
    "AlgarOudMigUtilities",
    "AssertionBuilder",
    "AssertionHelpers",
    "AttrObject",
    "AutoCommand",
    "AutoDiscoveryHandler",
    "BadBool",
    "BadMapping",
    "BadSingletonForTest",
    "BadString",
    "BrokenScenario",
    "CacheScenarios",
    "CheckProjectStub",
    "CoerceListValidatorScenario",
    "CoerceValidatorScenario",
    "CollectionScenarios",
    "ConcreteTestHandler",
    "ConcreteTestService",
    "ConfigTestCase",
    "ConfigTestFactories",
    "ConfigWithoutModelConfigForTest",
    "ContainerModelsScenarios",
    "CreateUserCommand",
    "CreateUserCommandHandler",
    "DataclassConfigForTest",
    "DependencyScenario",
    "DependencyScenarios",
    "DictHandler",
    "EchoHandler",
    "EmptyScenario",
    "EngineSafetyStub",
    "EnumScenarios",
    "EventHandler",
    "EventSubscriber",
    "ExplicitTypeHandler",
    "ExplodingGetattr",
    "ExplodingHandler",
    "ExplodingLenList",
    "ExtractPageParamsScenario",
    "FailingCommand",
    "FailingCommandHandler",
    "FailingOptionsForTest",
    "FailingService",
    "FailingServiceAuto",
    "FailingServiceAutoFactory",
    "FailingServiceFactory",
    "FixtureBuilder",
    "FlextCliConstants",
    "FlextCliModels",
    "FlextCliProtocols",
    "FlextCliTypes",
    "FlextCliUtilities",
    "FlextCommandId",
    "FlextCommandType",
    "FlextConsolidationContext",
    "FlextInfraTestConstants",
    "FlextInfraTestHelpers",
    "FlextInfraTestModels",
    "FlextInfraTestProtocols",
    "FlextInfraTestTypes",
    "FlextInfraTestUtilities",
    "FlextLdapConstants",
    "FlextLdapModels",
    "FlextLdapProtocols",
    "FlextLdapTypes",
    "FlextLdapUtilities",
    "FlextScenarioRunner",
    "FlextTestAutomationFramework",
    "FlextTestBuilder",
    "FullScenario",
    "FunctionalExternalService",
    "GenericHandler",
    "GenericModelFactory",
    "GetUserService",
    "GetUserServiceAuto",
    "GetUserServiceAutoFactory",
    "GetUserServiceFactory",
    "GitScenario",
    "GitScenarios",
    "GivenWhenThenBuilder",
    "IntHandler",
    "IsMemberScenario",
    "IsSubsetScenario",
    "LifecycleService",
    "MinimalScenario",
    "MissingType",
    "MockScanner",
    "MockScenario",
    "MultiOperationService",
    "NestedClassPropagationTransformer",
    "NoHandleMethod",
    "NonCallableHandle",
    "NotificationService",
    "ObjectHandler",
    "OptionsModelForTest",
    "PaginationScenarios",
    "ParameterizedTestBuilder",
    "ParseMappingScenario",
    "ParseOrDefaultScenario",
    "ParseScenario",
    "ParseSequenceScenario",
    "ParserScenario",
    "ParserScenarios",
    "PerformanceBenchmark",
    "PreparePaginationDataScenario",
    "Priority",
    "QueryHandler",
    "RailwayTestCase",
    "RealGitService",
    "RealSubprocessRunner",
    "ReliabilityScenario",
    "ReliabilityScenarios",
    "RunCallable",
    "RunStub",
    "RuntimeCloneService",
    "SampleCommand",
    "SampleEvent",
    "SampleHandler",
    "SampleQuery",
    "SendEmailService",
    "ServiceConfig",
    "ServiceFactoryRegistry",
    "ServiceTestCase",
    "ServiceTestCaseFactory",
    "ServiceTestCases",
    "SetupFn",
    "SimpleObj",
    "SingletonWithoutGetGlobalForTest",
    "SingletonWithoutModelDumpForTest",
    "Status",
    "StrictOptionsForTest",
    "StringHandler",
    "StringParserTestFactory",
    "SubprocessScenario",
    "SubprocessScenarios",
    "SuiteBuilder",
    "T",
    "TMessage",
    "T_co",
    "T_contra",
    "TestAdrHelpers",
    "TestAdvancedPatterns",
    "TestAggregateRoots",
    "TestAllDirectoriesScanned",
    "TestAllPatternsIntegration",
    "TestAltPropagatesException",
    "TestAssertExists",
    "TestAuditorBrokenLinks",
    "TestAuditorCore",
    "TestAuditorForbiddenTerms",
    "TestAuditorMainCli",
    "TestAuditorNormalize",
    "TestAuditorScope",
    "TestAuditorScopeFailure",
    "TestAuditorToMarkdown",
    "TestAutomatedFlextContainer",
    "TestAutomatedFlextContext",
    "TestAutomatedFlextDecorators",
    "TestAutomatedFlextDispatcher",
    "TestAutomatedFlextExceptions",
    "TestAutomatedFlextHandlers",
    "TestAutomatedFlextLoggings",
    "TestAutomatedFlextMixins",
    "TestAutomatedFlextRegistry",
    "TestAutomatedFlextRuntime",
    "TestAutomatedFlextService",
    "TestAutomatedFlextSettings",
    "TestAutomatedFlextUtilities",
    "TestAutomatedr",
    "TestBackwardCompatDiscardReturnValue",
    "TestBackwardCompatibility",
    "TestBaseMkGenerationFlow",
    "TestBaseMkValidatorCore",
    "TestBaseMkValidatorEdgeCases",
    "TestBaseMkValidatorSha256",
    "TestBatchOperations",
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
    "TestCleanText",
    "TestCloneContainer",
    "TestCloneRuntime",
    "TestClose",
    "TestCollectChanges",
    "TestCollectInternalDeps",
    "TestCollectInternalDepsEdgeCases",
    "TestCollectMarkdownFiles",
    "TestCommands",
    "TestCompleteFlextSystemIntegration",
    "TestComprehensiveIntegration",
    "TestConfig",
    "TestConfigConstants",
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
    "TestConfigMapDictOps",
    "TestConfigModels",
    "TestConfigServiceViaDI",
    "TestConsolidateGroupsPhase",
    "TestConstants",
    "TestConstantsQualityGateCLIDispatch",
    "TestConstantsQualityGateVerdict",
    "TestContainerDIRealExecution",
    "TestContainerInfo",
    "TestContainerIntegration",
    "TestContainerMemory",
    "TestContainerPerformance",
    "TestContainerStatus",
    "TestContext100Coverage",
    "TestContextDataModel",
    "TestContextServiceViaDI",
    "TestCoreModuleInit",
    "TestCorrelationDomain",
    "TestCoveragePush75Percent",
    "TestCreate",
    "TestCreateBranches",
    "TestCreateDatetimeProxy",
    "TestCreateDictProxy",
    "TestCreateFromCallableCarriesException",
    "TestCreateInStatic",
    "TestCreateStrProxy",
    "TestCreateTag",
    "TestCriticalReturnsResultBool",
    "TestCrossModuleIntegration",
    "TestDIBridgeRealExecution",
    "TestDataFactory",
    "TestDataGenerators",
    "TestDebugReturnsResultBool",
    "TestDependencyIntegrationRealExecution",
    "TestDetectMode",
    "TestDetectionUncoveredLines",
    "TestDetectorBasicDetection",
    "TestDetectorGitRunScenarios",
    "TestDetectorRepoNameExtraction",
    "TestDetectorReportFlags",
    "TestDetectorRunFailures",
    "TestDictMixinOperations",
    "TestDiscoverProjects",
    "TestDiscoveryDiscoverProjects",
    "TestDiscoveryFindAllPyprojectFiles",
    "TestDiscoveryIterPythonFiles",
    "TestDiscoveryProjectRoots",
    "TestDispatchPhase",
    "TestDispatcherDI",
    "TestDomainEvents",
    "TestDomainHashValue",
    "TestDomainLogger",
    "TestDomainResult",
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
    "TestEnterprisePatterns",
    "TestEntities",
    "TestEntityCoverageEdgeCases",
    "TestErrorOrPatternUnchanged",
    "TestErrorReporting",
    "TestErrorReturnsResultBool",
    "TestEventDrivenPatterns",
    "TestExceptionContext",
    "TestExceptionEdgeCases",
    "TestExceptionFactory",
    "TestExceptionIntegration",
    "TestExceptionLogging",
    "TestExceptionMetrics",
    "TestExceptionProperties",
    "TestExceptionPropertyAccess",
    "TestExceptionReturnsResultBool",
    "TestExceptionSerialization",
    "TestExcludedDirectories",
    "TestExcludedProjects",
    "TestExtractExports",
    "TestExtractInlineConstants",
    "TestExtractVersionExports",
    "TestFactories",
    "TestFactoriesHelpers",
    "TestFactoryDecoratorsDiscoveryHasFactories",
    "TestFactoryDecoratorsDiscoveryScanModule",
    "TestFactoryPatterns",
    "TestFailNoExceptionBackwardCompat",
    "TestFailWithException",
    "TestFileInfo",
    "TestFileInfoFromModels",
    "TestFixPyrelfyCLI",
    "TestFixabilityClassification",
    "TestFixerCore",
    "TestFixerMaybeFixLink",
    "TestFixerProcessFile",
    "TestFixerScope",
    "TestFixerToc",
    "TestFixtureFactory",
    "TestFlatMapPropagatesException",
    "TestFlextCommand",
    "TestFlextCommandHandler",
    "TestFlextCommandResults",
    "TestFlextConstants",
    "TestFlextContainer",
    "TestFlextContext",
    "TestFlextDecorators",
    "TestFlextExceptionsHierarchy",
    "TestFlextHandlers",
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
    "TestFlextInfraNamespaceValidator",
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
    "TestFlextLogLevel",
    "TestFlextLogger",
    "TestFlextLoggerIntegration",
    "TestFlextLoggerUsage",
    "TestFlextMixinsNestedClasses",
    "TestFlextModels",
    "TestFlextModelsAggregateRoot",
    "TestFlextModelsCollectionsCategories",
    "TestFlextModelsCollectionsOptions",
    "TestFlextModelsCollectionsResults",
    "TestFlextModelsCollectionsSettings",
    "TestFlextModelsCollectionsStatistics",
    "TestFlextModelsCommand",
    "TestFlextModelsContainer",
    "TestFlextModelsDomainEvent",
    "TestFlextModelsEdgeCases",
    "TestFlextModelsEntity",
    "TestFlextModelsIntegration",
    "TestFlextModelsQuery",
    "TestFlextModelsValue",
    "TestFlextProtocols",
    "TestFlextRegistry",
    "TestFlextRuntime",
    "TestFlextServiceIntegration",
    "TestFlextSettings",
    "TestFlextSettingsSingletonIntegration",
    "TestFlextTestsBuilders",
    "TestFlextTestsDocker",
    "TestFlextTestsDockerWorkerId",
    "TestFlextTestsDockerWorkspaceRoot",
    "TestFlextTestsDomains",
    "TestFlextTestsFactoriesModernAPI",
    "TestFlextTestsFiles",
    "TestFlextTestsFilesNewApi",
    "TestFlextTestsMatchers",
    "TestFlextTestsUtilitiesFactory",
    "TestFlextTestsUtilitiesResult",
    "TestFlextTestsUtilitiesResultCompat",
    "TestFlextTestsUtilitiesTestContext",
    "TestFlextTypings",
    "TestFlextUtilitiesArgs",
    "TestFlextUtilitiesConfiguration",
    "TestFlextUtilitiesModelNormalizeToMetadata",
    "TestFlextUtilitiesReliability",
    "TestFlextVersion",
    "TestFormatAppId",
    "TestFormattingRunRuffFix",
    "TestFromIOResultCarriesException",
    "TestFromValidationCarriesException",
    "TestFunction",
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
    "TestGlobalContextManagement",
    "TestGoFmtEmptyLinesInOutput",
    "TestHandleLazyInit",
    "TestHandlerDecoratorMetadata",
    "TestHandlerDiscoveryClass",
    "TestHandlerDiscoveryEdgeCases",
    "TestHandlerDiscoveryIntegration",
    "TestHandlerDiscoveryModule",
    "TestHandlerDiscoveryServiceIntegration",
    "TestHelperConsolidationTransformer",
    "TestHierarchicalExceptionSystem",
    "TestIdempotency",
    "TestInferOwnerFromOrigin",
    "TestInferPackage",
    "TestInfoReturnsResultBool",
    "TestInfoWithContentMeta",
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
    "TestInstanceCreation",
    "TestIntegrationWithRealCommandServices",
    "TestInventoryServiceCore",
    "TestInventoryServiceReports",
    "TestInventoryServiceScripts",
    "TestIsInternalPathDep",
    "TestIsRelativeTo",
    "TestIsWorkspaceMode",
    "TestIterMarkdownFiles",
    "TestIterWorkspacePythonModules",
    "TestJsonWriteFailure",
    "TestLashPropagatesException",
    "TestLevelBasedContextManagement",
    "TestLibraryIntegration",
    "TestLintAndFormatPublicMethods",
    "TestLoadAuditBudgets",
    "TestLoadDependencyLimits",
    "TestLogReturnsResultBool",
    "TestLoggerServiceViaDI",
    "TestLoggingIntegration",
    "TestLoggingMethods",
    "TestLoggingsErrorPaths",
    "TestMain",
    "TestMainBaseMkValidate",
    "TestMainCli",
    "TestMainCliRouting",
    "TestMainCommandDispatch",
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
    "TestMainStructlogConfiguration",
    "TestMainSubcommandDispatch",
    "TestMainSysArgvModification",
    "TestMainWithFlags",
    "TestMaintenanceMainEnforcer",
    "TestMaintenanceMainSuccess",
    "TestMapPropagatesException",
    "TestMapperBuildFlagsDict",
    "TestMapperCollectActiveKeys",
    "TestMapperFilterDict",
    "TestMapperInvertDict",
    "TestMapperMapDictKeys",
    "TestMapperTransformValues",
    "TestMarkdownReportEmptyGates",
    "TestMarkdownReportSkipsEmptyGates",
    "TestMarkdownReportWithErrors",
    "TestMaybeWriteTodo",
    "TestMerge",
    "TestMergeChildExports",
    "TestMetadata",
    "TestMigrationComplexity",
    "TestMigrationScenario1",
    "TestMigrationScenario2",
    "TestMigrationScenario4",
    "TestMigrationScenario5",
    "TestMigratorDryRun",
    "TestMigratorEdgeCases",
    "TestMigratorFlextCore",
    "TestMigratorInternalMakefile",
    "TestMigratorInternalPyproject",
    "TestMigratorPoetryDeps",
    "TestMigratorReadFailures",
    "TestMigratorWriteFailures",
    "TestModelIntegration",
    "TestModelSerialization",
    "TestModelValidation",
    "TestModernizerEdgeCases",
    "TestModernizerRunAndMain",
    "TestModernizerUncoveredLines",
    "TestModuleAndTypingsFlow",
    "TestModuleLevelWrappers",
    "TestMonadicOperationsUnchanged",
    "TestMroFacadeMethods",
    "TestMypyEmptyLinesInOutput",
    "TestNormalizeStringList",
    "TestOkNoneGuardStillRaises",
    "TestOrchestrate",
    "TestOrchestratorBasic",
    "TestOrchestratorFailures",
    "TestOutputSingletonConsistency",
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
    "TestPathResolverDiscoveryFlow",
    "TestPathSyncEdgeCases",
    "TestPattern1V1Explicit",
    "TestPattern2V2Property",
    "TestPattern3RailwayV1",
    "TestPattern4RailwayV2Property",
    "TestPattern5MonadicComposition",
    "TestPattern6ErrorHandling",
    "TestPattern7AutomaticInfrastructure",
    "TestPattern8MultipleOperations",
    "TestPerformanceAnalysis",
    "TestPerformanceBenchmarks",
    "TestPerformanceDomain",
    "TestPhase2FinalCoveragePush",
    "TestPhaseBuild",
    "TestPhasePublish",
    "TestPhaseValidate",
    "TestPhaseVersion",
    "TestPreviousTag",
    "TestProcessDirectory",
    "TestProcessFileReadError",
    "TestProjectLevelRefactor",
    "TestProjectResultProperties",
    "TestPropertyBasedPatterns",
    "TestProtocolComplianceStructlogLogger",
    "TestPushRelease",
    "TestPytestDiagExtractorCore",
    "TestPytestDiagLogParsing",
    "TestPytestDiagParseXml",
    "TestQueries",
    "TestReadDoc",
    "TestReadExistingDocstring",
    "TestReadRequiredMinor",
    "TestRealWiringScenarios",
    "TestRealWorldScenarios",
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
    "TestResult",
    "TestResultBasics",
    "TestResultCo",
    "TestResultTransformations",
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
    "TestRuntimeDictLike",
    "TestRuntimeTypeChecking",
    "TestSafeCarriesException",
    "TestSafeLoadYaml",
    "TestSafeString",
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
    "TestScopedContextManagement",
    "TestSelectedProjectNames",
    "TestSelectorFunction",
    "TestService",
    "TestService100Coverage",
    "TestServiceBootstrap",
    "TestServiceBootstrapWithDI",
    "TestServiceDomain",
    "TestServiceResultProperty",
    "TestServiceWithValidation",
    "TestServicerChaining",
    "TestServicesIntegrationViaDI",
    "TestShortAlias",
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
    "TestTextLogger",
    "TestTimeoutEnforcerCleanup",
    "TestTimeoutEnforcerEdgeCases",
    "TestTimeoutEnforcerExecutorManagement",
    "TestTimeoutEnforcerInitialization",
    "TestToIOChainsException",
    "TestToInfraValue",
    "TestTraceReturnsResultBool",
    "TestTraversePropagatesException",
    "TestTriggerRelease",
    "TestUpdateChangelog",
    "TestUser",
    "TestUtilitiesCoverage",
    "TestUtilitiesDomain",
    "TestValidateCore",
    "TestValidateGitRefEdgeCases",
    "TestValidateReport",
    "TestValidateScope",
    "TestValidateValueImmutable",
    "TestValidatorCallable",
    "TestValidatorMapMixin",
    "TestValues",
    "TestVersionFiles",
    "TestView",
    "TestViolationPattern",
    "TestWarningReturnsResultBool",
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
    "TestWorkspaceDetectionOrchestrationFlow",
    "TestWorkspaceLevelRefactor",
    "TestWorkspaceRoot",
    "TestWorkspaceRootFromEnv",
    "TestWorkspaceRootFromParents",
    "TestWriteJson",
    "TestWriteMarkdown",
    "TestWriteReport",
    "Teste",
    "Testr",
    "TestrCoverage",
    "TestsCore",
    "TestsFlextConstants",
    "TestsFlextModels",
    "TestsFlextProtocols",
    "TestsFlextServiceBase",
    "TestsFlextTestsFactoriesDict",
    "TestsFlextTestsFactoriesGeneric",
    "TestsFlextTestsFactoriesList",
    "TestsFlextTestsFactoriesModel",
    "TestsFlextTestsFactoriesRes",
    "TestsFlextTypes",
    "TestsFlextUtilities",
    "Testu",
    "TestuCacheClearObjectCache",
    "TestuCacheGenerateCacheKey",
    "TestuCacheHasCacheAttributes",
    "TestuCacheLogger",
    "TestuCacheNormalizeComponent",
    "TestuCacheSortDictKeys",
    "TestuCacheSortKey",
    "TestuCollectionBatch",
    "TestuCollectionChunk",
    "TestuCollectionCoerceDictValidator",
    "TestuCollectionCoerceListValidator",
    "TestuCollectionCount",
    "TestuCollectionFilter",
    "TestuCollectionFind",
    "TestuCollectionGroup",
    "TestuCollectionMap",
    "TestuCollectionMerge",
    "TestuCollectionParseMapping",
    "TestuCollectionParseSequence",
    "TestuCollectionProcess",
    "TestuDomain",
    "TestuEnumCoerceByNameValidator",
    "TestuEnumCoerceValidator",
    "TestuEnumIsMember",
    "TestuEnumIsSubset",
    "TestuEnumMetadata",
    "TestuEnumParse",
    "TestuEnumParseOrDefault",
    "TestuMapperAccessors",
    "TestuMapperAdvanced",
    "TestuMapperBuild",
    "TestuMapperConversions",
    "TestuMapperExtract",
    "TestuMapperUtils",
    "TestuPaginationBuildPaginationResponse",
    "TestuPaginationExtractPageParams",
    "TestuPaginationExtractPaginationConfig",
    "TestuPaginationPreparePaginationData",
    "TestuPaginationValidatePaginationParams",
    "TestuStringParser",
    "TestuTypeChecker",
    "TestuTypeGuardsIsDictNonEmpty",
    "TestuTypeGuardsIsListNonEmpty",
    "TestuTypeGuardsIsStringNonEmpty",
    "TestuTypeGuardsNormalizeToMetadataValue",
    "TextLike",
    "TextUtilityContract",
    "ThreadSafetyTest",
    "TimeoutEnforcerScenarios",
    "TypeGuardsScenarios",
    "UnknownHint",
    "UnregisteredCommand",
    "UpdateUserCommand",
    "UpdateUserCommandHandler",
    "User",
    "UserFactory",
    "UserQueryService",
    "UserServiceEntity",
    "ValidatePaginationParamsScenario",
    "ValidatingService",
    "ValidatingServiceAuto",
    "ValidatingServiceAutoFactory",
    "ValidatingServiceFactory",
    "ValidationScenario",
    "ValidationScenarios",
    "ValidationService",
    "WorkspaceFactory",
    "WorkspaceScenario",
    "WorkspaceScenarios",
    "arrange_act_assert",
    "assert_rejects",
    "assert_validates",
    "assertion_helpers",
    "auditor",
    "automation_framework",
    "builder",
    "c",
    "census",
    "clean_container",
    "consolidation_context",
    "create_compare_entities_cases",
    "create_compare_value_objects_cases",
    "create_hash_entity_cases",
    "create_hash_value_object_cases",
    "create_validate_entity_has_id_cases",
    "create_validate_value_object_immutable_cases",
    "d",
    "detector",
    "dispatcher",
    "doc",
    "e",
    "empty_strings",
    "engine",
    "fixer",
    "fixture_factory",
    "flext_result_failure",
    "flext_result_success",
    "gen",
    "generators_module",
    "get_memory_usage",
    "git_repo",
    "h",
    "handlers_module",
    "infra_git",
    "infra_git_repo",
    "infra_io",
    "infra_path",
    "infra_patterns",
    "infra_pr_manager",
    "infra_pr_workspace_manager",
    "infra_reporting",
    "infra_safe_command_output",
    "infra_selection",
    "infra_subprocess",
    "infra_templates",
    "infra_test_workspace",
    "infra_toml",
    "infra_workflow_linter",
    "infra_workflow_syncer",
    "invalid_hostnames",
    "invalid_port_numbers",
    "invalid_uris",
    "is_external",
    "m",
    "make_result_logger",
    "mapper",
    "mark_test_pattern",
    "mock_external_service",
    "normalize_link",
    "orchestrator",
    "out_of_range",
    "p",
    "parser_scenarios",
    "pyright_content",
    "pytestmark",
    "r",
    "rAssertionHelper",
    "real_docs_project",
    "real_entity",
    "real_git_repo",
    "real_makefile_project",
    "real_python_package",
    "real_toml_project",
    "real_value_object",
    "real_workspace",
    "reliability_scenarios",
    "reset_all_factories",
    "reset_global_container",
    "reset_runtime_state",
    "result_assertion_helper",
    "run_lint",
    "run_pr",
    "run_pr_workspace",
    "run_workflows",
    "runner",
    "runtime_cov_tests",
    "runtime_tests",
    "s",
    "sample_data",
    "scenario_runner",
    "service",
    "should_skip_target",
    "svc",
    "t",
    "temp_dir",
    "temp_directory",
    "temp_file",
    "test_accessor_take_pick_as_or_flat_and_agg_branches",
    "test_additional_container_branches_cover_fluent_and_lookup_paths",
    "test_additional_register_factory_and_unregister_paths",
    "test_aliases_are_available",
    "test_args_get_enum_params_annotated_unwrap_branch",
    "test_args_get_enum_params_branches",
    "test_array",
    "test_as_string_list",
    "test_as_string_list_toml_item",
    "test_async_log_writer_paths",
    "test_async_log_writer_shutdown_with_full_queue",
    "test_at_take_and_as_branches",
    "test_atomic_write_fail",
    "test_atomic_write_ok",
    "test_authentication_error_normalizes_extra_kwargs_into_context",
    "test_auto_value_lowercases_input",
    "test_bad_string_and_bad_bool_raise_value_error",
    "test_base_error_normalize_metadata_merges_existing_metadata_model",
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
    "test_basic_imports_work",
    "test_batch_fail_collect_flatten_and_progress",
    "test_bi_map_returns_forward_copy_and_inverse",
    "test_bind_operation_context_without_ensure_correlation_and_bind_failure",
    "test_build_apply_transform_and_process_error_paths",
    "test_build_impact_map_extracts_rename_entries",
    "test_build_impact_map_extracts_signature_entries",
    "test_build_options_invalid_only_kwargs_returns_base",
    "test_bump_version_invalid",
    "test_bump_version_result_type",
    "test_bump_version_valid",
    "test_callable_registration_with_attribute",
    "test_canonical_aliases_are_available",
    "test_canonical_dev_dependencies",
    "test_capture_cases",
    "test_categories_clear_and_symbols_are_available",
    "test_centralize_pydantic_cli_outputs_extended_metrics",
    "test_centralizer_converts_typed_dict_factory_to_model",
    "test_centralizer_does_not_touch_settings_module",
    "test_centralizer_moves_dict_alias_in_typings_without_keyword_name",
    "test_centralizer_moves_manual_type_aliases_to_models_file",
    "test_check_implements_protocol_false_non_runtime_protocol",
    "test_check_main_executes_real_cli",
    "test_checker_logger_and_safe_type_hints_fallback",
    "test_chk_exercises_missed_branches",
    "test_circuit_breaker_transitions_and_metrics",
    "test_class_nesting_appends_to_existing_namespace_and_removes_pass",
    "test_class_nesting_keeps_unmapped_top_level_classes",
    "test_class_nesting_moves_top_level_class_into_new_namespace",
    "test_class_nesting_refactor_single_file_end_to_end",
    "test_class_reconstructor_reorders_each_contiguous_method_block",
    "test_class_reconstructor_reorders_methods_by_config",
    "test_class_reconstructor_skips_interleaved_non_method_members",
    "test_clear_keys_values_items_and_validate_branches",
    "test_clear_operation_scope_and_handle_log_result_paths",
    "test_cli_result_by_project_root",
    "test_codegen_dir_returns_all_exports",
    "test_codegen_getattr_raises_attribute_error",
    "test_codegen_init_getattr_raises_attribute_error",
    "test_codegen_lazy_imports_work",
    "test_codegen_pipeline_end_to_end",
    "test_collection_batch_failure_error_capture_and_parse_sequence_outer_error",
    "test_combined_with_and_without_railway_uses_injection",
    "test_command_pagination_limit",
    "test_config_bridge_and_trace_context_and_http_validation",
    "test_config_context_properties_and_defaults",
    "test_config_hash_from_mapping_and_non_hashable",
    "test_configuration_mapping_and_dict_negative_branches",
    "test_configure_structlog_edge_paths",
    "test_configure_structlog_print_logger_factory_fallback",
    "test_configure_with_resource_register_and_factory_error_paths",
    "test_consolidate_groups_phase_apply_removes_old_groups",
    "test_consolidate_groups_phase_apply_with_empty_poetry_group",
    "test_constants_auto_enum_and_bimapping_paths",
    "test_construct_transform_and_deep_eq_branches",
    "test_container_and_service_domain_paths",
    "test_container_remaining_branch_paths_in_sync_factory_and_getters",
    "test_container_resource_registration_metadata_normalized",
    "test_context",
    "test_context_data_metadata_normalizer_removed",
    "test_context_data_normalize_and_json_checks",
    "test_context_data_validate_dict_serializable_error_paths",
    "test_context_data_validate_dict_serializable_none_and_mapping",
    "test_context_data_validate_dict_serializable_real_dicts",
    "test_context_export_serializable_and_validators",
    "test_context_export_statistics_validator_and_computed_fields",
    "test_context_export_validate_dict_serializable_mapping_and_models",
    "test_context_export_validate_dict_serializable_valid",
    "test_conversion_add_converted_and_error_metadata_append_paths",
    "test_conversion_add_skipped_skip_reason_upsert_paths",
    "test_conversion_add_warning_metadata_append_paths",
    "test_conversion_and_extract_success_branches",
    "test_conversion_start_and_complete_methods",
    "test_conversion_string_and_join_paths",
    "test_convert_default_fallback_matrix",
    "test_convert_sequence_branch_returns_tuple",
    "test_cqrs_query_resolve_deeper_and_int_pagination",
    "test_create_auto_discover_and_mode_mapping",
    "test_create_auto_register_factories_path",
    "test_create_auto_register_factory_wrapper_callable_and_non_callable",
    "test_create_discriminated_union_multiple_enums",
    "test_create_enum_executes_factory_path",
    "test_create_from_callable_and_repr",
    "test_create_from_callable_branches",
    "test_create_merges_metadata_dict_branch",
    "test_create_overloads_and_auto_correlation",
    "test_create_scoped_instance_and_scoped_additional_branches",
    "test_current_workspace_version",
    "test_data_alias_matches_value",
    "test_data_factory",
    "test_decorators_family_blocks_dispatcher_target",
    "test_dedupe_specs",
    "test_dep_name",
    "test_dependency_integration_and_wiring_paths",
    "test_dependency_registration_duplicate_guards",
    "test_deprecated_class_noop_init_branch",
    "test_deprecated_wrapper_emits_warning_and_returns_value",
    "test_detect_mode_with_nonexistent_path",
    "test_detect_mode_with_path_object",
    "test_discover_project_roots_without_nested_git_dirs",
    "test_discover_projects_wrapper",
    "test_discovery_narrowed_function_paths",
    "test_dispatch_after_handler_removal_fails",
    "test_dispatch_auto_discovery_handler",
    "test_dispatch_coerce_mode_with_enum_string_and_other_object",
    "test_dispatch_command_success",
    "test_dispatch_handler_exception_returns_failure",
    "test_dispatch_invalid_input_types",
    "test_dispatch_is_member_by_name_and_by_value",
    "test_dispatch_is_name_mode",
    "test_dispatch_no_handler_fails",
    "test_dispatch_parse_mode_with_enum_string_and_other_object",
    "test_dispatch_unknown_mode_raises",
    "test_dispatcher_family_blocks_models_target",
    "test_dispatcher_reliability_branch_paths",
    "test_engine_always_enables_class_nesting_file_rule",
    "test_engine_constants_shared",
    "test_enrich_and_ensure_trace_context_branches",
    "test_ensure_and_extract_array_index_helpers",
    "test_ensure_dict_branches",
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
    "test_ensure_trace_context_dict_conversion_paths",
    "test_ensure_utc_datetime_adds_tzinfo_when_naive",
    "test_ensure_utc_datetime_preserves_aware",
    "test_ensure_utc_datetime_returns_none_on_none",
    "test_entity_comparable_map_and_bulk_validation_paths",
    "test_event_publishing_strict",
    "test_exception_handling_in_dispatch",
    "test_exceptions_uncovered_metadata_paths",
    "test_execute_and_register_handler_failure_paths",
    "test_execute_retry_exponential_and_handle_exhaustion_raise_last_exception",
    "test_execute_retry_loop_covers_default_linear_and_never_ran",
    "test_export_paths_with_metadata_and_statistics",
    "test_extract_dep_name",
    "test_extract_error_paths_and_prop_accessor",
    "test_extract_field_value_and_ensure_variants",
    "test_extract_mapping_or_none_branches",
    "test_extract_message_type_annotation_and_dict_subclass_paths",
    "test_extract_message_type_from_handle_with_only_self",
    "test_extract_message_type_from_parameter_branches",
    "test_extract_requirement_name",
    "test_facade_binding_is_correct",
    "test_field_and_fields_multi_branches",
    "test_files_modified_tracks_affected_files",
    "test_filter_map_normalize_convert_helpers",
    "test_find_mapping_no_match_and_merge_error_paths",
    "test_fix_pyrefly_config_main_executes_real_cli_help",
    "test_flexcore_excluded_from_run",
    "test_flext_infra_pyproject_modernizer_find_pyproject_files",
    "test_flext_infra_pyproject_modernizer_process_file_invalid_toml",
    "test_flext_message_type_alias_adapter",
    "test_flow_through_short_circuits_on_failure",
    "test_framework",
    "test_from_validation_and_to_model_paths",
    "test_frozen_value_model_equality_and_hash",
    "test_general_value_helpers_and_logger",
    "test_generate_special_paths_and_dynamic_subclass",
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
    "test_generators_additional_missed_paths",
    "test_generators_mapping_non_dict_normalization_path",
    "test_get_and_get_typed_resource_factory_paths",
    "test_get_current_typings_from_pyproject_wrapper",
    "test_get_enum_values_returns_immutable_sequence",
    "test_get_logger_none_name_paths",
    "test_get_plugin_and_register_metadata_and_list_items_exception",
    "test_get_required_typings_wrapper",
    "test_get_service_info",
    "test_gitignore_entry_scenarios",
    "test_gitignore_sync_failure",
    "test_gitignore_write_failure",
    "test_group_sort_unique_slice_chunk_branches",
    "test_guard_in_has_empty_none_helpers",
    "test_guard_instance_attribute_access_warnings",
    "test_guards_bool_identity_branch_via_isinstance_fallback",
    "test_guards_bool_shortcut_and_issubclass_typeerror",
    "test_guards_handler_type_issubclass_typeerror_branch_direct",
    "test_guards_issubclass_success_when_callable_is_patched",
    "test_guards_issubclass_typeerror_when_class_not_treated_as_callable",
    "test_handle_log_result_without_fallback_logger_and_non_dict_like_extra",
    "test_handle_retry_exhaustion_falsey_exception_reaches_timeout_error",
    "test_handler_attribute_discovery",
    "test_handler_builder_fluent_methods",
    "test_handler_type_literal_and_invalid",
    "test_helper_consolidation_is_prechecked",
    "test_helpers_alias_exposed",
    "test_helpers_alias_is_reachable_helpers",
    "test_helpers_alias_is_reachable_main",
    "test_helpers_alias_is_reachable_pep621",
    "test_helpers_alias_is_reachable_poetry",
    "test_helpers_alias_is_reachable_project_obj",
    "test_identifiable_unique_id_empty_rejected",
    "test_implements_decorator_helper_methods_and_static_wrappers",
    "test_implements_decorator_validation_error_message",
    "test_import_modernizer_adds_c_when_existing_c_is_aliased",
    "test_import_modernizer_does_not_rewrite_function_parameter_shadow",
    "test_import_modernizer_does_not_rewrite_rebound_local_name_usage",
    "test_import_modernizer_partial_import_keeps_unmapped_symbols",
    "test_import_modernizer_partial_import_with_asname_keeps_unmapped_alias",
    "test_import_modernizer_skips_rewrite_when_runtime_alias_shadowed_in_function",
    "test_import_modernizer_skips_when_runtime_alias_name_is_blocked",
    "test_import_modernizer_updates_aliased_symbol_usage",
    "test_in_context_typevar_not_flagged",
    "test_inactive_and_none_value_paths",
    "test_init_fallback_and_lazy_result_property",
    "test_initialize_di_components_error_paths",
    "test_initialize_di_components_second_type_error_branch",
    "test_inject_comments_phase_apply_banner",
    "test_inject_comments_phase_apply_broken_group_section",
    "test_inject_comments_phase_apply_markers",
    "test_inject_comments_phase_apply_with_optional_dependencies_dev",
    "test_inject_sets_missing_dependency_from_container",
    "test_invalid_handler_mode_init_raises",
    "test_invalid_registration_attempts",
    "test_invert_and_json_conversion_branches",
    "test_is_flexible_value_covers_all_branches",
    "test_is_general_value_type_negative_paths_and_callable",
    "test_is_handler_type_branches",
    "test_is_type_non_empty_unknown_and_tuple_and_fallback",
    "test_is_type_protocol_fallback_branches",
    "test_is_valid_handles_validation_exception",
    "test_lash_runtime_result_and_from_io_result_fallback",
    "test_lazy_import_rule_hoists_import_to_module_level",
    "test_lazy_import_rule_uses_fix_action_for_hoist",
    "test_legacy_import_bypass_collapses_to_primary_import",
    "test_legacy_rule_uses_fix_action_remove_for_aliases",
    "test_legacy_wrapper_forwarding_keywords_is_inlined_as_alias",
    "test_legacy_wrapper_forwarding_varargs_is_inlined_as_alias",
    "test_legacy_wrapper_function_is_inlined_as_alias",
    "test_legacy_wrapper_non_passthrough_is_not_inlined",
    "test_log_operation_track_perf_exception_adds_duration",
    "test_loggings_bind_clear_level_error_paths",
    "test_loggings_context_and_factory_paths",
    "test_loggings_exception_and_adapter_paths",
    "test_loggings_instance_and_message_format_paths",
    "test_loggings_remaining_branch_paths",
    "test_loggings_source_and_log_error_paths",
    "test_loggings_uncovered_level_trace_path_and_exception_guards",
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
    "test_map_error_identity_and_transform",
    "test_map_flags_collect_and_invert_branches",
    "test_map_flat_map_and_then_paths",
    "test_members_uses_cache_on_second_call",
    "test_merge_defaults_and_dump_paths",
    "test_merge_metadata_context_paths",
    "test_metadata_attributes_accepts_basemodel_mapping",
    "test_metadata_attributes_accepts_none",
    "test_metadata_attributes_accepts_t_dict_and_mapping",
    "test_metadata_attributes_rejects_basemodel_non_mapping_dump",
    "test_metadata_attributes_rejects_non_mapping",
    "test_migrate_makefile_not_found_non_dry_run",
    "test_migrate_protocols_rewrites_references_with_p_alias",
    "test_migrate_pyproject_flext_core_non_dry_run",
    "test_migrate_to_mro_inlines_alias_constant_into_constants_class",
    "test_migrate_to_mro_moves_constant_and_rewrites_reference",
    "test_migrate_to_mro_moves_manual_uppercase_assignment",
    "test_migrate_to_mro_normalizes_facade_alias_to_c",
    "test_migrate_to_mro_rejects_unknown_target",
    "test_migrate_typings_rewrites_references_with_t_alias",
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
    "test_misc_unregistration_clear_and_reset",
    "test_mixins_container_registration_and_logger_paths",
    "test_mixins_context_logging_and_cqrs_paths",
    "test_mixins_context_stack_pop_initializes_missing_stack_attr",
    "test_mixins_remaining_branch_paths",
    "test_mixins_result_and_model_conversion_paths",
    "test_mixins_runtime_bootstrap_and_track_paths",
    "test_mixins_validation_and_protocol_paths",
    "test_model_helpers_remaining_paths",
    "test_model_support_and_hash_compare_paths",
    "test_models_family_blocks_utilities_target",
    "test_models_handler_branches",
    "test_models_handler_uncovered_mode_and_reset_paths",
    "test_models_settings_branch_paths",
    "test_models_settings_context_validator_and_non_standard_status_input",
    "test_mro_checker_keeps_external_attribute_base",
    "test_mro_redundancy_checker_removes_nested_attribute_inheritance",
    "test_mro_resolver_accepts_expected_order",
    "test_mro_resolver_rejects_wrong_order",
    "test_mro_scanner_includes_constants_variants_in_all_scopes",
    "test_multiple_instances_independent",
    "test_names_uses_cache_on_second_call",
    "test_namespace_enforce_cli_fails_on_manual_protocol_violation",
    "test_namespace_enforcer_apply_inserts_future_after_single_line_module_docstring",
    "test_namespace_enforcer_apply_keeps_script_shebang_when_adding_future",
    "test_namespace_enforcer_apply_moves_manual_protocol_to_protocols_file",
    "test_namespace_enforcer_creates_missing_facades_and_rewrites_imports",
    "test_namespace_enforcer_detects_cyclic_imports_in_tests_directory",
    "test_namespace_enforcer_detects_internal_private_imports",
    "test_namespace_enforcer_detects_manual_protocol_outside_canonical_files",
    "test_namespace_enforcer_detects_manual_typings_and_compat_aliases",
    "test_namespace_enforcer_detects_missing_runtime_alias_outside_src",
    "test_namespace_enforcer_does_not_rewrite_indented_import_aliases",
    "test_namespace_enforcer_does_not_rewrite_multiline_import_alias_blocks",
    "test_narrow_contextvar_invalid_inputs",
    "test_narrow_to_string_keyed_dict_and_mapping_paths",
    "test_nested_class_propagation_preserves_asname_and_rewrites_alias_usage",
    "test_nested_class_propagation_updates_import_annotations_and_calls",
    "test_non_empty_and_normalize_branches",
    "test_normalization_edge_branches",
    "test_normalize_context_to_dict_error_paths",
    "test_normalize_to_list_passes_list_through",
    "test_normalize_to_list_wraps_int",
    "test_normalize_to_list_wraps_scalar",
    "test_normalize_to_pydantic_dict_and_value_branches",
    "test_not_found_error_correlation_id_selection_and_extra_kwargs",
    "test_object_dict_and_type_error_fallback_paths",
    "test_ok_raises_on_none",
    "test_operation_progress_start_operation_sets_runtime_fields",
    "test_options_merge_conflict_paths_and_empty_merge_options",
    "test_pagination_response_string_fallbacks",
    "test_parse_mapping_outer_exception",
    "test_parse_semver_invalid",
    "test_parse_semver_result_type",
    "test_parse_semver_valid",
    "test_parser_convert_and_norm_branches",
    "test_parser_internal_helpers_additional_coverage",
    "test_parser_parse_helpers_and_primitive_coercion_branches",
    "test_parser_pipeline_and_pattern_branches",
    "test_parser_remaining_branch_paths",
    "test_parser_safe_length_and_parse_delimited_error_paths",
    "test_parser_split_and_normalize_exception_paths",
    "test_parser_success_and_edge_paths_cover_major_branches",
    "test_pattern_rule_converts_dict_annotations_to_mapping",
    "test_pattern_rule_keeps_dict_param_when_copy_used",
    "test_pattern_rule_keeps_dict_param_when_subscript_mutated",
    "test_pattern_rule_keeps_type_cast_when_not_nested_object_cast",
    "test_pattern_rule_optionally_converts_return_annotations_to_mapping",
    "test_pattern_rule_removes_configured_redundant_casts",
    "test_pattern_rule_removes_nested_type_object_cast_chain",
    "test_pattern_rule_skips_overload_signatures",
    "test_private_coerce_with_enum_and_string",
    "test_private_getters_exception_paths",
    "test_private_is_member_by_name",
    "test_private_is_member_by_value",
    "test_private_parse_success_and_failure",
    "test_process_context_data_and_related_convenience",
    "test_process_outer_exception_and_coercion_branches",
    "test_project_dev_groups",
    "test_project_dev_groups_missing_sections",
    "test_project_without_src_returns_empty",
    "test_protocol_and_simple_guard_helpers",
    "test_protocol_base_name_methods_and_runtime_check_branch",
    "test_protocol_meta_default_model_base_and_get_protocols_default",
    "test_protocol_model_and_settings_methods",
    "test_protocol_name_and_builder",
    "test_protocol_name_and_narrow_contextvar_exception_branch",
    "test_provide_property_paths",
    "test_publish_event_to_subscriber",
    "test_publish_no_subscribers_succeeds",
    "test_query_resolve_pagination_wrapper_and_fallback",
    "test_query_validate_pagination_dict_and_default",
    "test_railway_and_retry_additional_paths",
    "test_rate_limiter_blocks_then_recovers",
    "test_rate_limiter_jitter_application",
    "test_reconfigure_and_reset_state_paths",
    "test_recover_tap_and_tap_error_paths",
    "test_refactor_files_skips_non_python_inputs",
    "test_refactor_project_integrates_safety_manager",
    "test_refactor_project_scans_tests_and_scripts_dirs",
    "test_refactor_utilities_iter_python_files_includes_examples_and_scripts",
    "test_register_existing_providers_full_paths_and_misc_methods",
    "test_register_existing_providers_skips_and_register_core_fallback",
    "test_register_handler_as_event_subscriber",
    "test_register_handler_with_can_handle",
    "test_register_handler_with_message_type",
    "test_register_handler_without_route_fails",
    "test_register_singleton_register_factory_and_bulk_register_paths",
    "test_release_tag_from_branch_invalid",
    "test_release_tag_from_branch_result_type",
    "test_release_tag_from_branch_valid",
    "test_remaining_build_fields_construct_and_eq_paths",
    "test_remaining_uncovered_branches",
    "test_render_all_generates_large_makefile",
    "test_render_all_has_no_scripts_path_references",
    "test_render_failure",
    "test_render_success",
    "test_replace_project_version",
    "test_resolve_env_file_and_log_level",
    "test_resolve_gates_maps_type_alias",
    "test_resolve_logger_prefers_logger_attribute",
    "test_result_property_raises_on_failure",
    "test_results_internal_conflict_paths_and_combine",
    "test_retry_policy_behavior",
    "test_retry_unreachable_timeouterror_path",
    "test_reuse_existing_runtime_coverage_branches",
    "test_reuse_existing_runtime_scenarios",
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
    "test_rules_merge_combines_model_dump_values",
    "test_run_cases",
    "test_run_cli_run_returns_one_for_fail",
    "test_run_cli_run_returns_two_for_error",
    "test_run_cli_run_returns_zero_for_pass",
    "test_run_cli_with_fail_fast_flag",
    "test_run_cli_with_multiple_projects",
    "test_run_deptry_wrapper",
    "test_run_mypy_stub_hints_wrapper",
    "test_run_pip_check_wrapper",
    "test_run_pipeline_query_and_event_paths",
    "test_run_raw_cases",
    "test_runtime_create_instance_failure_branch",
    "test_runtime_family_blocks_non_runtime_target",
    "test_runtime_integration_tracking_paths",
    "test_runtime_misc_remaining_paths",
    "test_runtime_module_accessors_and_metadata",
    "test_runtime_result_alias_compatibility",
    "test_runtime_result_all_missed_branches",
    "test_runtime_result_remaining_paths",
    "test_scope_data_validators_and_errors",
    "test_scoped_config_context_branches",
    "test_service_create_initial_runtime_prefers_custom_config_type_and_context_property",
    "test_service_create_runtime_container_overrides_branch",
    "test_service_init_type_guards_and_properties",
    "test_service_request_timeout_post_validator_messages",
    "test_service_request_timeout_validator_branches",
    "test_set_set_all_get_validation_and_error_paths",
    "test_settings_materialize_and_context_overrides",
    "test_shortcuts_delegate_to_primary_methods",
    "test_signature_propagation_removes_and_adds_keywords",
    "test_signature_propagation_renames_call_keyword",
    "test_small_mapper_convenience_methods",
    "test_standalone_final_detected_as_fixable",
    "test_standalone_typealias_detected_as_fixable",
    "test_standalone_typevar_detected_as_fixable",
    "test_statistics_and_custom_fields_validators",
    "test_statistics_from_dict_and_none_conflict_resolution",
    "test_strict_registration_and_dispatch",
    "test_string_zero_return_value",
    "test_strip_whitespace_preserves_clean",
    "test_strip_whitespace_returns_empty_on_spaces",
    "test_strip_whitespace_trims_leading_trailing",
    "test_structlog_proxy_context_var_default_when_key_missing",
    "test_structlog_proxy_context_var_get_set_reset_paths",
    "test_summary_error_paths_and_bindings_failures",
    "test_summary_properties_and_subclass_storage_reset",
    "test_symbol_propagation_keeps_alias_reference_when_asname_used",
    "test_symbol_propagation_renames_import_and_local_references",
    "test_symbol_propagation_updates_mro_base_references",
    "test_sync_basemk_scenarios",
    "test_sync_config_namespace_paths",
    "test_sync_config_registers_namespace_factories_and_fallbacks",
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
    "test_timeout_additional_success_and_reraise_timeout_paths",
    "test_timeout_covers_exception_timeout_branch",
    "test_timeout_raises_when_successful_call_exceeds_limit",
    "test_timeout_reraises_original_exception_when_within_limit",
    "test_timestampable_timestamp_conversion_and_json_serializer",
    "test_timestamped_model_and_alias_and_canonical_symbols",
    "test_to_flexible_value_and_safe_list_branches",
    "test_to_flexible_value_fallback_none_branch_for_unsupported_type",
    "test_to_general_value_dict_removed",
    "test_to_io_result_failure_path",
    "test_track_performance_success_and_failure_paths",
    "test_transform_option_extract_and_step_helpers",
    "test_type_guards_and_narrowing_failures",
    "test_type_guards_and_protocol_name",
    "test_ultrawork_models_cli_runs_dry_run_copy",
    "test_unwrap_item",
    "test_unwrap_item_toml_item",
    "test_update_exception_path",
    "test_update_statistics_remove_hook_and_clone_false_result",
    "test_update_success_path_returns_ok_result",
    "test_utilities_family_allows_utilities_target",
    "test_utilities_reliability_branches",
    "test_utilities_reliability_compose_returns_non_result_directly",
    "test_utilities_reliability_uncovered_retry_compose_and_sequence_paths",
    "test_validate_config_dict_normalizes_dict",
    "test_validate_tags_list_from_string",
    "test_validate_tags_list_normalizes",
    "test_validate_value_object_immutable_exception_and_no_setattr_branch",
    "test_validation_like_error_structure",
    "test_violation_analysis_counts_massive_patterns",
    "test_violation_analyzer_skips_non_utf8_files",
    "test_with_correlation_with_context_track_operation_and_factory",
    "test_with_resource_cleanup_runs",
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
    "v",
    "valid_hostnames",
    "valid_port_numbers",
    "valid_ranges",
    "valid_strings",
    "valid_uris",
    "validation_scenarios",
    "validator",
    "whitespace_strings",
    "workspace_root",
    "x",
]


def __getattr__(name: str) -> Any:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
