# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Unit package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
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
        test_generator_normalize_config_with_basemk_config,
        test_generator_normalize_config_with_dict,
        test_generator_normalize_config_with_invalid_dict,
        test_generator_normalize_config_with_none,
        test_generator_validate_generated_output_handles_oserror,
        test_generator_write_creates_parent_directories,
        test_generator_write_fails_without_output_or_stream,
        test_generator_write_handles_file_permission_error,
        test_generator_write_to_file,
        test_generator_write_to_stream,
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
    from tests.infra.unit.container.test_infra_container import (
        TestInfraContainerFunctions,
        TestInfraMroPattern,
        TestInfraServiceRetrieval,
    )
    from tests.infra.unit.core.basemk_validator import (
        TestBaseMkValidatorCore,
        TestBaseMkValidatorEdgeCases,
        TestBaseMkValidatorSha256,
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
        TestDiagResult,
        TestPytestDiagExtractorCore,
        TestPytestDiagLogParsing,
        TestPytestDiagParseXml,
    )
    from tests.infra.unit.core.scanner import (
        TestScannerCore,
        TestScannerHelpers,
        TestScannerMultiFile,
        TestScannerValidation,
    )
    from tests.infra.unit.core.skill_validator import (
        TestNormalizeStringList,
        TestSafeLoadYaml,
        TestSkillValidatorAstGrepCount,
        TestSkillValidatorCore,
        TestSkillValidatorCustomCount,
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
    from tests.infra.unit.deps.internal_sync import (
        TestCollectInternalDeps,
        TestCollectInternalDepsEdgeCases,
        TestEnsureCheckout,
        TestEnsureCheckoutEdgeCases,
        TestEnsureSymlink,
        TestEnsureSymlinkEdgeCases,
        TestIsWorkspaceMode,
        TestParseGitmodules,
        TestParseRepoMap,
        TestSync,
        TestSyncMethodEdgeCases,
        TestWorkspaceRootFromEnv,
        TestWorkspaceRootFromParents,
    )
    from tests.infra.unit.deps.main import (
        TestMainExceptionHandling,
        TestMainModuleImport,
        TestMainStructlogConfiguration,
        TestMainSubcommandDispatch,
        TestMainSysArgvModification,
    )
    from tests.infra.unit.deps.path_sync import (
        TestDetectMode,
        TestExtractDepName,
        TestExtractRequirementName,
        TestFlextInfraDependencyPathSync,
        TestPathSyncEdgeCases,
        TestRewriteDepPaths,
        TestRewritePep621,
        TestRewritePoetry,
        TestTargetPath,
        test_detect_mode_with_path_object,
        test_extract_requirement_name_invalid,
        test_extract_requirement_name_simple,
        test_extract_requirement_name_with_path_dep,
        test_main_discovery_failure,
        test_main_no_changes_needed,
        test_main_project_obj_not_dict_first_loop,
        test_main_project_obj_not_dict_second_loop,
        test_main_with_changes_and_dry_run,
        test_main_with_changes_no_dry_run,
        test_rewrite_dep_paths_dry_run,
        test_rewrite_dep_paths_read_failure,
        test_rewrite_dep_paths_with_internal_names,
        test_rewrite_pep621_invalid_path_dep_regex,
        test_rewrite_pep621_no_project_table,
        test_rewrite_pep621_non_string_item,
        test_rewrite_poetry_no_poetry_table,
        test_rewrite_poetry_no_tool_table,
        test_rewrite_poetry_with_non_dict_value,
        test_target_path_standalone,
        test_target_path_workspace_root,
        test_target_path_workspace_subproject,
        test_workspace_root_fallback,
    )
    from tests.infra.unit.deps.test_detection_classify import (
        TestBuildProjectReport,
        TestClassifyIssues,
        test_helpers_alias_is_reachable,
    )
    from tests.infra.unit.deps.test_detection_deptry import (
        TestDiscoverProjects,
        TestRunDeptry,
        TestRunPipCheck,
    )
    from tests.infra.unit.deps.test_detection_models import (
        TestFlextInfraDependencyDetectionModels,
        TestFlextInfraDependencyDetectionModels as m,
        TestFlextInfraDependencyDetectionService,
        TestToInfraValue,
    )
    from tests.infra.unit.deps.test_detection_typings import (
        TestLoadDependencyLimits,
        TestModuleAndTypingsFlow,
        TestRunMypyStubHints,
    )
    from tests.infra.unit.deps.test_detection_wrappers import (
        TestDetectionUncoveredLines,
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
    from tests.infra.unit.deps.test_detector_init import (
        TestFlextInfraRuntimeDevDependencyDetectorInit,
    )
    from tests.infra.unit.deps.test_detector_main import (
        TestFlextInfraRuntimeDevDependencyDetectorRunTypings,
        TestMainFunction,
    )
    from tests.infra.unit.deps.test_detector_models import (
        TestFlextInfraDependencyDetectorModels,
        test_helpers_alias_available,
    )
    from tests.infra.unit.deps.test_detector_report import (
        TestFlextInfraRuntimeDevDependencyDetectorRunReport,
    )
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
        TestMain,
        TestSyncExtraPaths,
        TestSyncOneEdgeCases,
    )
    from tests.infra.unit.deps.test_init import TestFlextInfraDeps
    from tests.infra.unit.deps.test_internal_sync_resolve import (
        TestInferOwnerFromOrigin,
        TestResolveRef,
        TestSynthesizedRepoMap,
    )
    from tests.infra.unit.deps.test_internal_sync_validation import (
        TestFlextInfraInternalDependencySyncService,
        TestFlextInfraInternalDependencySyncService as s,
        TestIsInternalPathDep,
        TestIsRelativeTo,
        TestOwnerFromRemoteUrl,
        TestValidateGitRefEdgeCases,
    )
    from tests.infra.unit.deps.test_main import (
        TestMainHelpAndErrors,
        TestMainReturnValues,
        TestSubcommandMapping,
    )
    from tests.infra.unit.deps.test_modernizer_consolidate import (
        TestConsolidateGroupsPhase,
        test_consolidate_groups_phase_apply_removes_old_groups,
        test_consolidate_groups_phase_apply_with_empty_poetry_group,
    )
    from tests.infra.unit.deps.test_modernizer_helpers import (
        TestArray,
        TestAsStringList,
        TestCanonicalDevDependencies,
        TestDedupeSpecs,
        TestDepName,
        TestEnsureTable,
        TestProjectDevGroups,
        TestUnwrapItem,
        test_as_string_list_with_item_and_edge_values,
        test_ensure_table_with_non_table_value_uncovered,
        test_unwrap_item_with_item,
        test_unwrap_item_with_none,
    )
    from tests.infra.unit.deps.test_modernizer_pyrefly import (
        TestEnsurePyreflyConfigPhase,
        test_ensure_pyrefly_config_phase_apply_ignore_errors,
        test_ensure_pyrefly_config_phase_apply_python_version,
        test_ensure_pyrefly_config_phase_apply_search_path_and_errors,
    )
    from tests.infra.unit.deps.test_modernizer_pyright import (
        TestEnsurePyrightConfigPhase,
    )
    from tests.infra.unit.deps.test_modernizer_workspace import (
        TestParser,
        TestReadDoc,
        TestWorkspaceRoot,
        test_workspace_root_doc_construction,
    )
    from tests.infra.unit.discovery.test_infra_discovery import (
        TestFlextInfraDiscoveryService,
        TestFlextInfraDiscoveryServiceUncoveredLines,
    )
    from tests.infra.unit.docs.auditor import (
        TestAuditorBudgets,
        TestAuditorCore,
        TestAuditorNormalize,
    )
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
    from tests.infra.unit.docs.builder import TestBuilderCore
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
    )
    from tests.infra.unit.docs.init import TestFlextInfraDocs
    from tests.infra.unit.docs.main import TestRunAudit, TestRunFix
    from tests.infra.unit.docs.main_commands import (
        TestRunBuild,
        TestRunGenerate,
        TestRunValidate,
    )
    from tests.infra.unit.docs.main_entry import TestMainRouting, TestMainWithFlags
    from tests.infra.unit.docs.shared import (
        TestFlextInfraDocScope,
        TestFlextInfraDocsShared,
    )
    from tests.infra.unit.docs.validator import TestFlextInfraDocValidator
    from tests.infra.unit.github.linter import TestFlextInfraWorkflowLinter
    from tests.infra.unit.github.main import (
        TestRunLint,
        TestRunPrWorkspace,
        TestRunWorkflows,
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
    from tests.infra.unit.io.test_infra_json_io import TestFlextInfraJsonService
    from tests.infra.unit.io.test_infra_output import (
        TestInfraOutputEdgeCases,
        TestInfraOutputHeader,
        TestInfraOutputMessages,
        TestInfraOutputNoColor,
        TestInfraOutputProgress,
        TestInfraOutputStatus,
        TestInfraOutputSummary,
        TestMroFacadeMethods,
        TestShouldUseColor,
        TestShouldUseUnicode,
    )
    from tests.infra.unit.refactor.test_infra_refactor import (
        EngineSafetyStub,
        test_build_impact_map_extracts_rename_entries,
        test_build_impact_map_extracts_signature_entries,
        test_class_reconstructor_reorders_each_contiguous_method_block,
        test_class_reconstructor_reorders_methods_by_config,
        test_class_reconstructor_skips_interleaved_non_method_members,
        test_engine_always_enables_class_nesting_file_rule,
        test_ensure_future_annotations_after_docstring,
        test_ensure_future_annotations_moves_existing_import_to_top,
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
        test_legacy_import_bypass_collapses_to_primary_import,
        test_legacy_rule_uses_fix_action_remove_for_aliases,
        test_legacy_wrapper_forwarding_keywords_is_inlined_as_alias,
        test_legacy_wrapper_forwarding_varargs_is_inlined_as_alias,
        test_legacy_wrapper_function_is_inlined_as_alias,
        test_legacy_wrapper_non_passthrough_is_not_inlined,
        test_main_analyze_violations_is_read_only,
        test_main_analyze_violations_writes_json_report,
        test_mro_checker_keeps_external_attribute_base,
        test_mro_redundancy_checker_removes_nested_attribute_inheritance,
        test_pattern_rule_converts_dict_annotations_to_mapping,
        test_pattern_rule_keeps_dict_param_when_copy_used,
        test_pattern_rule_keeps_dict_param_when_subscript_mutated,
        test_pattern_rule_keeps_type_cast_when_not_nested_object_cast,
        test_pattern_rule_optionally_converts_return_annotations_to_mapping,
        test_pattern_rule_removes_configured_redundant_casts,
        test_pattern_rule_removes_nested_type_object_cast_chain,
        test_pattern_rule_skips_overload_signatures,
        test_refactor_files_skips_non_python_inputs,
        test_refactor_project_integrates_safety_manager,
        test_refactor_project_scans_tests_and_scripts_dirs,
        test_rule_dispatch_fails_on_invalid_pattern_rule_config,
        test_rule_dispatch_fails_on_unknown_rule_mapping,
        test_rule_dispatch_keeps_legacy_id_fallback_mapping,
        test_rule_dispatch_prefers_fix_action_metadata,
        test_signature_propagation_removes_and_adds_keywords,
        test_signature_propagation_renames_call_keyword,
        test_symbol_propagation_keeps_alias_reference_when_asname_used,
        test_symbol_propagation_renames_import_and_local_references,
        test_symbol_propagation_updates_mro_base_references,
        test_violation_analysis_counts_massive_patterns,
        test_violation_analyzer_skips_non_utf8_files,
    )
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
    from tests.infra.unit.test_infra_constants import (
        TestFlextInfraConstantsAlias,
        TestFlextInfraConstantsCheckNamespace,
        TestFlextInfraConstantsConsistency,
        TestFlextInfraConstantsEncodingNamespace,
        TestFlextInfraConstantsExcludedNamespace,
        TestFlextInfraConstantsFilesNamespace,
        TestFlextInfraConstantsGatesNamespace,
        TestFlextInfraConstantsGithubNamespace,
        TestFlextInfraConstantsImmutability,
        TestFlextInfraConstantsPathsNamespace,
        TestFlextInfraConstantsStatusNamespace,
    )
    from tests.infra.unit.test_infra_git import (
        TestFlextInfraGitService,
        TestPreviousTag,
        TestPushRelease,
        TestRemovedCompatibilityMethods,
    )
    from tests.infra.unit.test_infra_init_lazy import (
        TestFlextInfraInitLazyLoading,
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
    from tests.infra.unit.test_infra_maintenance_main import TestMaintenanceMain
    from tests.infra.unit.test_infra_maintenance_python_version import (
        TestFlextInfraPythonVersionEnforcer,
    )
    from tests.infra.unit.test_infra_paths import TestFlextInfraPathResolver
    from tests.infra.unit.test_infra_patterns import (
        TestFlextInfraPatternsEdgeCases,
        TestFlextInfraPatternsMarkdown,
        TestFlextInfraPatternsPatternTypes,
        TestFlextInfraPatternsPatternTypes as t,
        TestFlextInfraPatternsTooling,
    )
    from tests.infra.unit.test_infra_protocols import TestFlextInfraProtocolsImport
    from tests.infra.unit.test_infra_reporting import TestFlextInfraReportingService
    from tests.infra.unit.test_infra_selection import TestFlextInfraUtilitiesSelection
    from tests.infra.unit.test_infra_subprocess import TestFlextInfraCommandRunner
    from tests.infra.unit.test_infra_templates import (
        TestTemplateEngineConstants,
        TestTemplateEngineErrorHandling,
        TestTemplateEngineInstances,
        TestTemplateEngineRender,
    )
    from tests.infra.unit.test_infra_toml_io import TestFlextInfraTomlService
    from tests.infra.unit.test_infra_typings import TestFlextInfraTypesImport
    from tests.infra.unit.test_infra_utilities import TestFlextInfraUtilitiesImport
    from tests.infra.unit.test_infra_version import TestFlextInfraVersion
    from tests.infra.unit.test_infra_versioning import TestFlextInfraVersioningService
    from tests.infra.unit.test_infra_workspace_cli import (
        test_workspace_cli_migrate_command,
        test_workspace_cli_migrate_output_contains_summary,
    )
    from tests.infra.unit.test_infra_workspace_detector import (
        detector,
        test_detector_detects_standalone_mode_without_parent_git,
        test_detector_detects_standalone_with_non_flext_repo,
        test_detector_detects_workspace_mode_with_flext_repo,
        test_detector_detects_workspace_mode_with_parent_git,
        test_detector_execute_returns_failure,
        test_detector_extracts_repo_name_from_https_url,
        test_detector_extracts_repo_name_from_ssh_url,
        test_detector_extracts_repo_name_without_git_suffix,
        test_detector_handles_empty_origin_url,
        test_detector_handles_exception_during_detection,
        test_detector_handles_git_command_errors,
        test_detector_handles_git_command_failure,
        test_detector_handles_runner_failure,
        test_detector_returns_standalone_when_no_parent_git,
    )
    from tests.infra.unit.test_infra_workspace_init import TestFlextInfraWorkspace
    from tests.infra.unit.test_infra_workspace_main import (
        test_main_calls_sys_exit,
        test_main_detect_command,
        test_main_entry_point,
        test_main_migrate_command,
        test_main_migrate_dry_run,
        test_main_no_command,
        test_main_orchestrate_command,
        test_main_orchestrate_with_fail_fast,
        test_main_orchestrate_with_make_args,
        test_main_sync_command,
        test_main_sync_with_canonical_root,
        test_run_detect_failure,
        test_run_detect_success,
        test_run_migrate_failure,
        test_run_migrate_success,
        test_run_migrate_with_project_errors,
        test_run_orchestrate_failure,
        test_run_orchestrate_no_projects,
        test_run_orchestrate_success,
        test_run_orchestrate_with_failures,
        test_run_sync_failure,
        test_run_sync_success,
    )
    from tests.infra.unit.test_infra_workspace_migrator import (
        test_migrate_makefile_not_found_non_dry_run,
        test_migrate_pyproject_flext_core_non_dry_run,
        test_migrator_apply_updates_project_files,
        test_migrator_basemk_generation_failure,
        test_migrator_basemk_write_failure,
        test_migrator_discovery_failure,
        test_migrator_dry_run_reports_changes_without_writes,
        test_migrator_execute_returns_failure,
        test_migrator_flext_core_dry_run,
        test_migrator_flext_core_project_skipped,
        test_migrator_gitignore_already_normalized_dry_run,
        test_migrator_gitignore_read_failure,
        test_migrator_gitignore_write_failure,
        test_migrator_handles_missing_pyproject_gracefully,
        test_migrator_has_flext_core_dependency_in_poetry,
        test_migrator_has_flext_core_dependency_poetry_deps_not_table,
        test_migrator_has_flext_core_dependency_poetry_table_missing,
        test_migrator_makefile_not_found_dry_run,
        test_migrator_makefile_read_failure,
        test_migrator_makefile_write_failure,
        test_migrator_no_changes_needed,
        test_migrator_preserves_custom_makefile_content,
        test_migrator_pyproject_not_found_dry_run,
        test_migrator_pyproject_parse_failure,
        test_migrator_pyproject_write_failure,
        test_migrator_workspace_root_not_exists,
        test_migrator_workspace_root_project_detection,
        test_workspace_migrator_error_handling_on_invalid_workspace,
        test_workspace_migrator_makefile_not_found_dry_run,
        test_workspace_migrator_makefile_read_error,
        test_workspace_migrator_pyproject_write_error,
    )
    from tests.infra.unit.test_infra_workspace_orchestrator import (
        orchestrator,
        test_orchestrate_run_project_failure_with_fail_fast,
        test_orchestrate_with_project_execution_failure,
        test_orchestrate_with_runner_failure_fail_fast,
        test_orchestrator_captures_per_project_output,
        test_orchestrator_continues_on_failure_without_fail_fast,
        test_orchestrator_execute_returns_failure,
        test_orchestrator_executes_verb_across_projects,
        test_orchestrator_fail_fast_skips_remaining_projects,
        test_orchestrator_fail_fast_with_failure_result,
        test_orchestrator_handles_empty_project_list,
        test_orchestrator_handles_runner_exception,
        test_orchestrator_stops_on_first_failure_with_fail_fast,
        test_orchestrator_with_make_args,
    )
    from tests.infra.unit.test_infra_workspace_sync import (
        sync_service,
        test_main_returns_one_on_failure,
        test_main_returns_zero_on_success,
        test_sync_service_atomic_write_failure,
        test_sync_service_atomic_write_success,
        test_sync_service_basemk_generation_failure,
        test_sync_service_canonical_root_copy,
        test_sync_service_creates_base_mk_if_missing,
        test_sync_service_detects_changes_via_sha256,
        test_sync_service_ensure_gitignore_entries_all_present,
        test_sync_service_ensure_gitignore_entries_missing_entries,
        test_sync_service_ensure_gitignore_entries_write_failure,
        test_sync_service_execute_returns_failure,
        test_sync_service_generates_base_mk,
        test_sync_service_gitignore_sync_failure,
        test_sync_service_gitignore_update_failure,
        test_sync_service_lock_acquisition_failure,
        test_sync_service_main_cli,
        test_sync_service_project_root_not_exists,
        test_sync_service_project_root_required,
        test_sync_service_sha256_content,
        test_sync_service_sha256_file,
        test_sync_service_skips_write_when_content_unchanged,
        test_sync_service_sync_basemk_from_canonical,
        test_sync_service_sync_basemk_generation_failure,
        test_sync_service_sync_basemk_no_change_needed,
        test_sync_service_sync_basemk_with_canonical_root,
        test_sync_service_validates_gitignore_entries,
    )

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "EngineSafetyStub": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "EngineSafetyStub",
    ),
    "TestAllDirectoriesScanned": (
        "tests.infra.unit.codegen.lazy_init_tests",
        "TestAllDirectoriesScanned",
    ),
    "TestArray": ("tests.infra.unit.deps.test_modernizer_helpers", "TestArray"),
    "TestAsStringList": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "TestAsStringList",
    ),
    "TestAuditorBrokenLinks": (
        "tests.infra.unit.docs.auditor_links",
        "TestAuditorBrokenLinks",
    ),
    "TestAuditorBudgets": ("tests.infra.unit.docs.auditor", "TestAuditorBudgets"),
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
    "TestBuildSiblingExportIndex": (
        "tests.infra.unit.codegen.lazy_init",
        "TestBuildSiblingExportIndex",
    ),
    "TestBuilderCore": ("tests.infra.unit.docs.builder", "TestBuilderCore"),
    "TestBuilderScope": ("tests.infra.unit.docs.builder_scope", "TestBuilderScope"),
    "TestCanonicalDevDependencies": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "TestCanonicalDevDependencies",
    ),
    "TestCensusReportModel": (
        "tests.infra.unit.codegen.census",
        "TestCensusReportModel",
    ),
    "TestCensusViolationModel": (
        "tests.infra.unit.codegen.census",
        "TestCensusViolationModel",
    ),
    "TestCheckIssueFormatted": (
        "tests.infra.unit.check.extended",
        "TestCheckIssueFormatted",
    ),
    "TestCheckMainEntryPoint": (
        "tests.infra.unit.check.extended",
        "TestCheckMainEntryPoint",
    ),
    "TestCheckOnlyMode": (
        "tests.infra.unit.codegen.lazy_init_tests",
        "TestCheckOnlyMode",
    ),
    "TestCheckpoint": ("tests.infra.unit.github.pr_workspace", "TestCheckpoint"),
    "TestChecks": ("tests.infra.unit.github.pr", "TestChecks"),
    "TestClassifyIssues": (
        "tests.infra.unit.deps.test_detection_classify",
        "TestClassifyIssues",
    ),
    "TestClose": ("tests.infra.unit.github.pr", "TestClose"),
    "TestCollectInternalDeps": (
        "tests.infra.unit.deps.internal_sync",
        "TestCollectInternalDeps",
    ),
    "TestCollectInternalDepsEdgeCases": (
        "tests.infra.unit.deps.internal_sync",
        "TestCollectInternalDepsEdgeCases",
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
    "TestConsolidateGroupsPhase": (
        "tests.infra.unit.deps.test_modernizer_consolidate",
        "TestConsolidateGroupsPhase",
    ),
    "TestConstants": (
        "tests.infra.unit.deps.test_extra_paths_manager",
        "TestConstants",
    ),
    "TestCoreModuleInit": ("tests.infra.unit.core.init", "TestCoreModuleInit"),
    "TestCreate": ("tests.infra.unit.github.pr", "TestCreate"),
    "TestDedupeSpecs": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "TestDedupeSpecs",
    ),
    "TestDepName": ("tests.infra.unit.deps.test_modernizer_helpers", "TestDepName"),
    "TestDetectMode": ("tests.infra.unit.deps.path_sync", "TestDetectMode"),
    "TestDetectionUncoveredLines": (
        "tests.infra.unit.deps.test_detection_wrappers",
        "TestDetectionUncoveredLines",
    ),
    "TestDiagResult": ("tests.infra.unit.core.pytest_diag", "TestDiagResult"),
    "TestDiscoverProjects": (
        "tests.infra.unit.deps.test_detection_deptry",
        "TestDiscoverProjects",
    ),
    "TestEdgeCases": ("tests.infra.unit.codegen.lazy_init_tests", "TestEdgeCases"),
    "TestEnsureCheckout": ("tests.infra.unit.deps.internal_sync", "TestEnsureCheckout"),
    "TestEnsureCheckoutEdgeCases": (
        "tests.infra.unit.deps.internal_sync",
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
    "TestEnsureSymlink": ("tests.infra.unit.deps.internal_sync", "TestEnsureSymlink"),
    "TestEnsureSymlinkEdgeCases": (
        "tests.infra.unit.deps.internal_sync",
        "TestEnsureSymlinkEdgeCases",
    ),
    "TestEnsureTable": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "TestEnsureTable",
    ),
    "TestExcludedDirectories": (
        "tests.infra.unit.codegen.lazy_init_tests",
        "TestExcludedDirectories",
    ),
    "TestExcludedProjects": ("tests.infra.unit.codegen.census", "TestExcludedProjects"),
    "TestExtractDepName": ("tests.infra.unit.deps.path_sync", "TestExtractDepName"),
    "TestExtractExports": ("tests.infra.unit.codegen.lazy_init", "TestExtractExports"),
    "TestExtractInlineConstants": (
        "tests.infra.unit.codegen.lazy_init",
        "TestExtractInlineConstants",
    ),
    "TestExtractRequirementName": (
        "tests.infra.unit.deps.path_sync",
        "TestExtractRequirementName",
    ),
    "TestExtractVersionExports": (
        "tests.infra.unit.codegen.lazy_init",
        "TestExtractVersionExports",
    ),
    "TestFixPyrelfyCLI": ("tests.infra.unit.check.extended", "TestFixPyrelfyCLI"),
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
        "tests.infra.unit.codegen.lazy_init",
        "TestFlextInfraCodegenLazyInit",
    ),
    "TestFlextInfraCommandRunner": (
        "tests.infra.unit.test_infra_subprocess",
        "TestFlextInfraCommandRunner",
    ),
    "TestFlextInfraConfigFixer": (
        "tests.infra.unit.check.pyrefly",
        "TestFlextInfraConfigFixer",
    ),
    "TestFlextInfraConstantsAlias": (
        "tests.infra.unit.test_infra_constants",
        "TestFlextInfraConstantsAlias",
    ),
    "TestFlextInfraConstantsCheckNamespace": (
        "tests.infra.unit.test_infra_constants",
        "TestFlextInfraConstantsCheckNamespace",
    ),
    "TestFlextInfraConstantsConsistency": (
        "tests.infra.unit.test_infra_constants",
        "TestFlextInfraConstantsConsistency",
    ),
    "TestFlextInfraConstantsEncodingNamespace": (
        "tests.infra.unit.test_infra_constants",
        "TestFlextInfraConstantsEncodingNamespace",
    ),
    "TestFlextInfraConstantsExcludedNamespace": (
        "tests.infra.unit.test_infra_constants",
        "TestFlextInfraConstantsExcludedNamespace",
    ),
    "TestFlextInfraConstantsFilesNamespace": (
        "tests.infra.unit.test_infra_constants",
        "TestFlextInfraConstantsFilesNamespace",
    ),
    "TestFlextInfraConstantsGatesNamespace": (
        "tests.infra.unit.test_infra_constants",
        "TestFlextInfraConstantsGatesNamespace",
    ),
    "TestFlextInfraConstantsGithubNamespace": (
        "tests.infra.unit.test_infra_constants",
        "TestFlextInfraConstantsGithubNamespace",
    ),
    "TestFlextInfraConstantsImmutability": (
        "tests.infra.unit.test_infra_constants",
        "TestFlextInfraConstantsImmutability",
    ),
    "TestFlextInfraConstantsPathsNamespace": (
        "tests.infra.unit.test_infra_constants",
        "TestFlextInfraConstantsPathsNamespace",
    ),
    "TestFlextInfraConstantsStatusNamespace": (
        "tests.infra.unit.test_infra_constants",
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
        "tests.infra.unit.deps.path_sync",
        "TestFlextInfraDependencyPathSync",
    ),
    "TestFlextInfraDeps": ("tests.infra.unit.deps.test_init", "TestFlextInfraDeps"),
    "TestFlextInfraDiscoveryService": (
        "tests.infra.unit.discovery.test_infra_discovery",
        "TestFlextInfraDiscoveryService",
    ),
    "TestFlextInfraDiscoveryServiceUncoveredLines": (
        "tests.infra.unit.discovery.test_infra_discovery",
        "TestFlextInfraDiscoveryServiceUncoveredLines",
    ),
    "TestFlextInfraDocScope": (
        "tests.infra.unit.docs.shared",
        "TestFlextInfraDocScope",
    ),
    "TestFlextInfraDocValidator": (
        "tests.infra.unit.docs.validator",
        "TestFlextInfraDocValidator",
    ),
    "TestFlextInfraDocs": ("tests.infra.unit.docs.init", "TestFlextInfraDocs"),
    "TestFlextInfraDocsShared": (
        "tests.infra.unit.docs.shared",
        "TestFlextInfraDocsShared",
    ),
    "TestFlextInfraExtraPathsManager": (
        "tests.infra.unit.deps.test_extra_paths_manager",
        "TestFlextInfraExtraPathsManager",
    ),
    "TestFlextInfraGitService": (
        "tests.infra.unit.test_infra_git",
        "TestFlextInfraGitService",
    ),
    "TestFlextInfraInitLazyLoading": (
        "tests.infra.unit.test_infra_init_lazy",
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
        "tests.infra.unit.test_infra_patterns",
        "TestFlextInfraPatternsEdgeCases",
    ),
    "TestFlextInfraPatternsMarkdown": (
        "tests.infra.unit.test_infra_patterns",
        "TestFlextInfraPatternsMarkdown",
    ),
    "TestFlextInfraPatternsPatternTypes": (
        "tests.infra.unit.test_infra_patterns",
        "TestFlextInfraPatternsPatternTypes",
    ),
    "TestFlextInfraPatternsTooling": (
        "tests.infra.unit.test_infra_patterns",
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
    "TestFlextInfraPythonVersionEnforcer": (
        "tests.infra.unit.test_infra_maintenance_python_version",
        "TestFlextInfraPythonVersionEnforcer",
    ),
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
    "TestFlextInfraReportingService": (
        "tests.infra.unit.test_infra_reporting",
        "TestFlextInfraReportingService",
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
        "tests.infra.unit.test_infra_init_lazy",
        "TestFlextInfraSubmoduleInitLazyLoading",
    ),
    "TestFlextInfraTomlService": (
        "tests.infra.unit.test_infra_toml_io",
        "TestFlextInfraTomlService",
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
    "TestFlextInfraVersion": (
        "tests.infra.unit.test_infra_version",
        "TestFlextInfraVersion",
    ),
    "TestFlextInfraVersioningService": (
        "tests.infra.unit.test_infra_versioning",
        "TestFlextInfraVersioningService",
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
    "TestGithubInit": ("tests.infra.unit.github.pr", "TestGithubInit"),
    "TestGoFormatEmptyLineSkipping": (
        "tests.infra.unit.check.extended",
        "TestGoFormatEmptyLineSkipping",
    ),
    "TestGoFormatEmptyLines": (
        "tests.infra.unit.check.extended",
        "TestGoFormatEmptyLines",
    ),
    "TestGoFormatParsing": ("tests.infra.unit.check.extended", "TestGoFormatParsing"),
    "TestHandleLazyInit": ("tests.infra.unit.codegen.main", "TestHandleLazyInit"),
    "TestInferOwnerFromOrigin": (
        "tests.infra.unit.deps.test_internal_sync_resolve",
        "TestInferOwnerFromOrigin",
    ),
    "TestInferPackage": ("tests.infra.unit.codegen.lazy_init", "TestInferPackage"),
    "TestInfraContainerFunctions": (
        "tests.infra.unit.container.test_infra_container",
        "TestInfraContainerFunctions",
    ),
    "TestInfraMroPattern": (
        "tests.infra.unit.container.test_infra_container",
        "TestInfraMroPattern",
    ),
    "TestInfraOutputEdgeCases": (
        "tests.infra.unit.io.test_infra_output",
        "TestInfraOutputEdgeCases",
    ),
    "TestInfraOutputHeader": (
        "tests.infra.unit.io.test_infra_output",
        "TestInfraOutputHeader",
    ),
    "TestInfraOutputMessages": (
        "tests.infra.unit.io.test_infra_output",
        "TestInfraOutputMessages",
    ),
    "TestInfraOutputNoColor": (
        "tests.infra.unit.io.test_infra_output",
        "TestInfraOutputNoColor",
    ),
    "TestInfraOutputProgress": (
        "tests.infra.unit.io.test_infra_output",
        "TestInfraOutputProgress",
    ),
    "TestInfraOutputStatus": (
        "tests.infra.unit.io.test_infra_output",
        "TestInfraOutputStatus",
    ),
    "TestInfraOutputSummary": (
        "tests.infra.unit.io.test_infra_output",
        "TestInfraOutputSummary",
    ),
    "TestInfraServiceRetrieval": (
        "tests.infra.unit.container.test_infra_container",
        "TestInfraServiceRetrieval",
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
        "tests.infra.unit.deps.internal_sync",
        "TestIsWorkspaceMode",
    ),
    "TestJsonWriteFailure": ("tests.infra.unit.check.extended", "TestJsonWriteFailure"),
    "TestLintAndFormatPublicMethods": (
        "tests.infra.unit.check.extended",
        "TestLintAndFormatPublicMethods",
    ),
    "TestLoadDependencyLimits": (
        "tests.infra.unit.deps.test_detection_typings",
        "TestLoadDependencyLimits",
    ),
    "TestMain": ("tests.infra.unit.deps.test_extra_paths_sync", "TestMain"),
    "TestMainBaseMkValidate": ("tests.infra.unit.core.main", "TestMainBaseMkValidate"),
    "TestMainCliRouting": ("tests.infra.unit.core.main", "TestMainCliRouting"),
    "TestMainCommandDispatch": (
        "tests.infra.unit.codegen.main",
        "TestMainCommandDispatch",
    ),
    "TestMainEntryPoint": ("tests.infra.unit.codegen.main", "TestMainEntryPoint"),
    "TestMainExceptionHandling": (
        "tests.infra.unit.deps.main",
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
    "TestMainModuleImport": ("tests.infra.unit.deps.main", "TestMainModuleImport"),
    "TestMainReturnValues": ("tests.infra.unit.deps.test_main", "TestMainReturnValues"),
    "TestMainRouting": ("tests.infra.unit.docs.main_entry", "TestMainRouting"),
    "TestMainScan": ("tests.infra.unit.core.main", "TestMainScan"),
    "TestMainStructlogConfiguration": (
        "tests.infra.unit.deps.main",
        "TestMainStructlogConfiguration",
    ),
    "TestMainSubcommandDispatch": (
        "tests.infra.unit.deps.main",
        "TestMainSubcommandDispatch",
    ),
    "TestMainSysArgvModification": (
        "tests.infra.unit.deps.main",
        "TestMainSysArgvModification",
    ),
    "TestMainWithFlags": ("tests.infra.unit.docs.main_entry", "TestMainWithFlags"),
    "TestMaintenanceMain": (
        "tests.infra.unit.test_infra_maintenance_main",
        "TestMaintenanceMain",
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
    "TestMerge": ("tests.infra.unit.github.pr", "TestMerge"),
    "TestMergeChildExports": (
        "tests.infra.unit.codegen.lazy_init",
        "TestMergeChildExports",
    ),
    "TestModuleAndTypingsFlow": (
        "tests.infra.unit.deps.test_detection_typings",
        "TestModuleAndTypingsFlow",
    ),
    "TestModuleLevelWrappers": (
        "tests.infra.unit.deps.test_detection_wrappers",
        "TestModuleLevelWrappers",
    ),
    "TestMroFacadeMethods": (
        "tests.infra.unit.io.test_infra_output",
        "TestMroFacadeMethods",
    ),
    "TestMypyEmptyLineSkipping": (
        "tests.infra.unit.check.extended",
        "TestMypyEmptyLineSkipping",
    ),
    "TestMypyEmptyLines": ("tests.infra.unit.check.extended", "TestMypyEmptyLines"),
    "TestMypyJSONParsing": ("tests.infra.unit.check.extended", "TestMypyJSONParsing"),
    "TestNormalizeStringList": (
        "tests.infra.unit.core.skill_validator",
        "TestNormalizeStringList",
    ),
    "TestOrchestrate": ("tests.infra.unit.github.pr_workspace", "TestOrchestrate"),
    "TestOwnerFromRemoteUrl": (
        "tests.infra.unit.deps.test_internal_sync_validation",
        "TestOwnerFromRemoteUrl",
    ),
    "TestParseArgs": ("tests.infra.unit.github.pr", "TestParseArgs"),
    "TestParseGitmodules": (
        "tests.infra.unit.deps.internal_sync",
        "TestParseGitmodules",
    ),
    "TestParseRepoMap": ("tests.infra.unit.deps.internal_sync", "TestParseRepoMap"),
    "TestParseViolationInvalid": (
        "tests.infra.unit.codegen.census",
        "TestParseViolationInvalid",
    ),
    "TestParseViolationValid": (
        "tests.infra.unit.codegen.census",
        "TestParseViolationValid",
    ),
    "TestParser": ("tests.infra.unit.deps.test_modernizer_workspace", "TestParser"),
    "TestPathDepPathsPep621": (
        "tests.infra.unit.deps.test_extra_paths_pep621",
        "TestPathDepPathsPep621",
    ),
    "TestPathDepPathsPoetry": (
        "tests.infra.unit.deps.test_extra_paths_pep621",
        "TestPathDepPathsPoetry",
    ),
    "TestPathSyncEdgeCases": (
        "tests.infra.unit.deps.path_sync",
        "TestPathSyncEdgeCases",
    ),
    "TestPreviousTag": ("tests.infra.unit.test_infra_git", "TestPreviousTag"),
    "TestProcessDirectory": (
        "tests.infra.unit.codegen.lazy_init",
        "TestProcessDirectory",
    ),
    "TestProcessFileReadError": (
        "tests.infra.unit.check.extended",
        "TestProcessFileReadError",
    ),
    "TestProjectDevGroups": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "TestProjectDevGroups",
    ),
    "TestProjectResultProperties": (
        "tests.infra.unit.check.extended",
        "TestProjectResultProperties",
    ),
    "TestPushRelease": ("tests.infra.unit.test_infra_git", "TestPushRelease"),
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
        "tests.infra.unit.codegen.lazy_init",
        "TestReadExistingDocstring",
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
    "TestRemovedCompatibilityMethods": (
        "tests.infra.unit.test_infra_git",
        "TestRemovedCompatibilityMethods",
    ),
    "TestRenderTemplate": ("tests.infra.unit.github.workflows", "TestRenderTemplate"),
    "TestResolveAliases": ("tests.infra.unit.codegen.lazy_init", "TestResolveAliases"),
    "TestResolveRef": (
        "tests.infra.unit.deps.test_internal_sync_resolve",
        "TestResolveRef",
    ),
    "TestResolveVersionInteractive": (
        "tests.infra.unit.release.main",
        "TestResolveVersionInteractive",
    ),
    "TestRewriteDepPaths": ("tests.infra.unit.deps.path_sync", "TestRewriteDepPaths"),
    "TestRewritePep621": ("tests.infra.unit.deps.path_sync", "TestRewritePep621"),
    "TestRewritePoetry": ("tests.infra.unit.deps.path_sync", "TestRewritePoetry"),
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
    "TestRunAudit": ("tests.infra.unit.docs.main", "TestRunAudit"),
    "TestRunBuild": ("tests.infra.unit.docs.main_commands", "TestRunBuild"),
    "TestRunCLI": ("tests.infra.unit.check.extended", "TestRunCLI"),
    "TestRunDeptry": ("tests.infra.unit.deps.test_detection_deptry", "TestRunDeptry"),
    "TestRunFix": ("tests.infra.unit.docs.main", "TestRunFix"),
    "TestRunGenerate": ("tests.infra.unit.docs.main_commands", "TestRunGenerate"),
    "TestRunLint": ("tests.infra.unit.github.main", "TestRunLint"),
    "TestRunMypyStubHints": (
        "tests.infra.unit.deps.test_detection_typings",
        "TestRunMypyStubHints",
    ),
    "TestRunPipCheck": (
        "tests.infra.unit.deps.test_detection_deptry",
        "TestRunPipCheck",
    ),
    "TestRunPr": ("tests.infra.unit.github.pr_workspace", "TestRunPr"),
    "TestRunPrWorkspace": ("tests.infra.unit.github.main", "TestRunPrWorkspace"),
    "TestRunRuffFix": ("tests.infra.unit.codegen.lazy_init", "TestRunRuffFix"),
    "TestRunValidate": ("tests.infra.unit.docs.main_commands", "TestRunValidate"),
    "TestRunWorkflows": ("tests.infra.unit.github.main", "TestRunWorkflows"),
    "TestSafeLoadYaml": ("tests.infra.unit.core.skill_validator", "TestSafeLoadYaml"),
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
    "TestScannerCore": ("tests.infra.unit.core.scanner", "TestScannerCore"),
    "TestScannerHelpers": ("tests.infra.unit.core.scanner", "TestScannerHelpers"),
    "TestScannerMultiFile": ("tests.infra.unit.core.scanner", "TestScannerMultiFile"),
    "TestScannerValidation": ("tests.infra.unit.core.scanner", "TestScannerValidation"),
    "TestSelectorFunction": ("tests.infra.unit.github.pr", "TestSelectorFunction"),
    "TestShouldBubbleUp": ("tests.infra.unit.codegen.lazy_init", "TestShouldBubbleUp"),
    "TestShouldUseColor": (
        "tests.infra.unit.io.test_infra_output",
        "TestShouldUseColor",
    ),
    "TestShouldUseUnicode": (
        "tests.infra.unit.io.test_infra_output",
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
    "TestSkillValidatorCustomCount": (
        "tests.infra.unit.core.skill_validator",
        "TestSkillValidatorCustomCount",
    ),
    "TestSkillValidatorRenderTemplate": (
        "tests.infra.unit.core.skill_validator",
        "TestSkillValidatorRenderTemplate",
    ),
    "TestStaticMethods": ("tests.infra.unit.github.pr_workspace", "TestStaticMethods"),
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
    "TestSync": ("tests.infra.unit.deps.internal_sync", "TestSync"),
    "TestSyncExtraPaths": (
        "tests.infra.unit.deps.test_extra_paths_sync",
        "TestSyncExtraPaths",
    ),
    "TestSyncMethodEdgeCases": (
        "tests.infra.unit.deps.internal_sync",
        "TestSyncMethodEdgeCases",
    ),
    "TestSyncOne": ("tests.infra.unit.deps.test_extra_paths_manager", "TestSyncOne"),
    "TestSyncOneEdgeCases": (
        "tests.infra.unit.deps.test_extra_paths_sync",
        "TestSyncOneEdgeCases",
    ),
    "TestSyncOperation": ("tests.infra.unit.github.workflows", "TestSyncOperation"),
    "TestSyncProject": ("tests.infra.unit.github.workflows", "TestSyncProject"),
    "TestSynthesizedRepoMap": (
        "tests.infra.unit.deps.test_internal_sync_resolve",
        "TestSynthesizedRepoMap",
    ),
    "TestTargetPath": ("tests.infra.unit.deps.path_sync", "TestTargetPath"),
    "TestTemplateEngineConstants": (
        "tests.infra.unit.test_infra_templates",
        "TestTemplateEngineConstants",
    ),
    "TestTemplateEngineErrorHandling": (
        "tests.infra.unit.test_infra_templates",
        "TestTemplateEngineErrorHandling",
    ),
    "TestTemplateEngineInstances": (
        "tests.infra.unit.test_infra_templates",
        "TestTemplateEngineInstances",
    ),
    "TestTemplateEngineRender": (
        "tests.infra.unit.test_infra_templates",
        "TestTemplateEngineRender",
    ),
    "TestToInfraValue": (
        "tests.infra.unit.deps.test_detection_models",
        "TestToInfraValue",
    ),
    "TestTriggerRelease": ("tests.infra.unit.github.pr", "TestTriggerRelease"),
    "TestUnwrapItem": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "TestUnwrapItem",
    ),
    "TestValidateGitRefEdgeCases": (
        "tests.infra.unit.deps.test_internal_sync_validation",
        "TestValidateGitRefEdgeCases",
    ),
    "TestView": ("tests.infra.unit.github.pr", "TestView"),
    "TestViolationPattern": ("tests.infra.unit.codegen.census", "TestViolationPattern"),
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
    "TestWorkspaceRoot": (
        "tests.infra.unit.deps.test_modernizer_workspace",
        "TestWorkspaceRoot",
    ),
    "TestWorkspaceRootFromEnv": (
        "tests.infra.unit.deps.internal_sync",
        "TestWorkspaceRootFromEnv",
    ),
    "TestWorkspaceRootFromParents": (
        "tests.infra.unit.deps.internal_sync",
        "TestWorkspaceRootFromParents",
    ),
    "c": ("tests.infra.unit.codegen.lazy_init", "TestExtractInlineConstants"),
    "census": ("tests.infra.unit.codegen.census", "census"),
    "detector": ("tests.infra.unit.test_infra_workspace_detector", "detector"),
    "fixer": ("tests.infra.unit.codegen.autofix", "fixer"),
    "m": (
        "tests.infra.unit.deps.test_detection_models",
        "TestFlextInfraDependencyDetectionModels",
    ),
    "orchestrator": (
        "tests.infra.unit.test_infra_workspace_orchestrator",
        "orchestrator",
    ),
    "r": ("tests.infra.unit.check.extended", "TestWorkspaceCheckerBuildGateResult"),
    "run_lint": ("tests.infra.unit.github.main", "run_lint"),
    "run_pr": ("tests.infra.unit.github.main", "run_pr"),
    "run_pr_workspace": ("tests.infra.unit.github.main", "run_pr_workspace"),
    "run_workflows": ("tests.infra.unit.github.main", "run_workflows"),
    "s": (
        "tests.infra.unit.deps.test_internal_sync_validation",
        "TestFlextInfraInternalDependencySyncService",
    ),
    "sync_service": ("tests.infra.unit.test_infra_workspace_sync", "sync_service"),
    "t": ("tests.infra.unit.test_infra_patterns", "TestFlextInfraPatternsPatternTypes"),
    "test_as_string_list_with_item_and_edge_values": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "test_as_string_list_with_item_and_edge_values",
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
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_build_impact_map_extracts_rename_entries",
    ),
    "test_build_impact_map_extracts_signature_entries": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_build_impact_map_extracts_signature_entries",
    ),
    "test_check_main_executes_real_cli": (
        "tests.infra.unit.check.main",
        "test_check_main_executes_real_cli",
    ),
    "test_class_reconstructor_reorders_each_contiguous_method_block": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_class_reconstructor_reorders_each_contiguous_method_block",
    ),
    "test_class_reconstructor_reorders_methods_by_config": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_class_reconstructor_reorders_methods_by_config",
    ),
    "test_class_reconstructor_skips_interleaved_non_method_members": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_class_reconstructor_skips_interleaved_non_method_members",
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
    "test_consolidate_groups_phase_apply_removes_old_groups": (
        "tests.infra.unit.deps.test_modernizer_consolidate",
        "test_consolidate_groups_phase_apply_removes_old_groups",
    ),
    "test_consolidate_groups_phase_apply_with_empty_poetry_group": (
        "tests.infra.unit.deps.test_modernizer_consolidate",
        "test_consolidate_groups_phase_apply_with_empty_poetry_group",
    ),
    "test_detect_mode_with_path_object": (
        "tests.infra.unit.deps.path_sync",
        "test_detect_mode_with_path_object",
    ),
    "test_detector_detects_standalone_mode_without_parent_git": (
        "tests.infra.unit.test_infra_workspace_detector",
        "test_detector_detects_standalone_mode_without_parent_git",
    ),
    "test_detector_detects_standalone_with_non_flext_repo": (
        "tests.infra.unit.test_infra_workspace_detector",
        "test_detector_detects_standalone_with_non_flext_repo",
    ),
    "test_detector_detects_workspace_mode_with_flext_repo": (
        "tests.infra.unit.test_infra_workspace_detector",
        "test_detector_detects_workspace_mode_with_flext_repo",
    ),
    "test_detector_detects_workspace_mode_with_parent_git": (
        "tests.infra.unit.test_infra_workspace_detector",
        "test_detector_detects_workspace_mode_with_parent_git",
    ),
    "test_detector_execute_returns_failure": (
        "tests.infra.unit.test_infra_workspace_detector",
        "test_detector_execute_returns_failure",
    ),
    "test_detector_extracts_repo_name_from_https_url": (
        "tests.infra.unit.test_infra_workspace_detector",
        "test_detector_extracts_repo_name_from_https_url",
    ),
    "test_detector_extracts_repo_name_from_ssh_url": (
        "tests.infra.unit.test_infra_workspace_detector",
        "test_detector_extracts_repo_name_from_ssh_url",
    ),
    "test_detector_extracts_repo_name_without_git_suffix": (
        "tests.infra.unit.test_infra_workspace_detector",
        "test_detector_extracts_repo_name_without_git_suffix",
    ),
    "test_detector_handles_empty_origin_url": (
        "tests.infra.unit.test_infra_workspace_detector",
        "test_detector_handles_empty_origin_url",
    ),
    "test_detector_handles_exception_during_detection": (
        "tests.infra.unit.test_infra_workspace_detector",
        "test_detector_handles_exception_during_detection",
    ),
    "test_detector_handles_git_command_errors": (
        "tests.infra.unit.test_infra_workspace_detector",
        "test_detector_handles_git_command_errors",
    ),
    "test_detector_handles_git_command_failure": (
        "tests.infra.unit.test_infra_workspace_detector",
        "test_detector_handles_git_command_failure",
    ),
    "test_detector_handles_runner_failure": (
        "tests.infra.unit.test_infra_workspace_detector",
        "test_detector_handles_runner_failure",
    ),
    "test_detector_returns_standalone_when_no_parent_git": (
        "tests.infra.unit.test_infra_workspace_detector",
        "test_detector_returns_standalone_when_no_parent_git",
    ),
    "test_discover_projects_wrapper": (
        "tests.infra.unit.deps.test_detection_wrappers",
        "test_discover_projects_wrapper",
    ),
    "test_engine_always_enables_class_nesting_file_rule": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_engine_always_enables_class_nesting_file_rule",
    ),
    "test_ensure_future_annotations_after_docstring": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_ensure_future_annotations_after_docstring",
    ),
    "test_ensure_future_annotations_moves_existing_import_to_top": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_ensure_future_annotations_moves_existing_import_to_top",
    ),
    "test_ensure_pyrefly_config_phase_apply_ignore_errors": (
        "tests.infra.unit.deps.test_modernizer_pyrefly",
        "test_ensure_pyrefly_config_phase_apply_ignore_errors",
    ),
    "test_ensure_pyrefly_config_phase_apply_python_version": (
        "tests.infra.unit.deps.test_modernizer_pyrefly",
        "test_ensure_pyrefly_config_phase_apply_python_version",
    ),
    "test_ensure_pyrefly_config_phase_apply_search_path_and_errors": (
        "tests.infra.unit.deps.test_modernizer_pyrefly",
        "test_ensure_pyrefly_config_phase_apply_search_path_and_errors",
    ),
    "test_ensure_table_with_non_table_value_uncovered": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "test_ensure_table_with_non_table_value_uncovered",
    ),
    "test_extract_requirement_name_invalid": (
        "tests.infra.unit.deps.path_sync",
        "test_extract_requirement_name_invalid",
    ),
    "test_extract_requirement_name_simple": (
        "tests.infra.unit.deps.path_sync",
        "test_extract_requirement_name_simple",
    ),
    "test_extract_requirement_name_with_path_dep": (
        "tests.infra.unit.deps.path_sync",
        "test_extract_requirement_name_with_path_dep",
    ),
    "test_files_modified_tracks_affected_files": (
        "tests.infra.unit.codegen.autofix",
        "test_files_modified_tracks_affected_files",
    ),
    "test_fix_pyrefly_config_main_executes_real_cli_help": (
        "tests.infra.unit.check.fix_pyrefly_config",
        "test_fix_pyrefly_config_main_executes_real_cli_help",
    ),
    "test_flexcore_excluded_from_run": (
        "tests.infra.unit.codegen.autofix",
        "test_flexcore_excluded_from_run",
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
        "tests.infra.unit.basemk.generator",
        "test_generator_normalize_config_with_basemk_config",
    ),
    "test_generator_normalize_config_with_dict": (
        "tests.infra.unit.basemk.generator",
        "test_generator_normalize_config_with_dict",
    ),
    "test_generator_normalize_config_with_invalid_dict": (
        "tests.infra.unit.basemk.generator",
        "test_generator_normalize_config_with_invalid_dict",
    ),
    "test_generator_normalize_config_with_none": (
        "tests.infra.unit.basemk.generator",
        "test_generator_normalize_config_with_none",
    ),
    "test_generator_renders_with_config_override": (
        "tests.infra.unit.basemk.engine",
        "test_generator_renders_with_config_override",
    ),
    "test_generator_validate_generated_output_handles_oserror": (
        "tests.infra.unit.basemk.generator",
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
        "tests.infra.unit.basemk.generator",
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
        "tests.infra.unit.basemk.generator",
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
    "test_handle_constants_quality_gate_json_exits_with_int": (
        "tests.infra.unit.codegen.constants_quality_gate",
        "test_handle_constants_quality_gate_json_exits_with_int",
    ),
    "test_handle_constants_quality_gate_text_exits_with_int": (
        "tests.infra.unit.codegen.constants_quality_gate",
        "test_handle_constants_quality_gate_text_exits_with_int",
    ),
    "test_helpers_alias_available": (
        "tests.infra.unit.deps.test_detector_models",
        "test_helpers_alias_available",
    ),
    "test_helpers_alias_exposed": (
        "tests.infra.unit.deps.test_extra_paths_pep621",
        "test_helpers_alias_exposed",
    ),
    "test_helpers_alias_is_reachable": (
        "tests.infra.unit.deps.test_detection_classify",
        "test_helpers_alias_is_reachable",
    ),
    "test_import_modernizer_adds_c_when_existing_c_is_aliased": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_import_modernizer_adds_c_when_existing_c_is_aliased",
    ),
    "test_import_modernizer_does_not_rewrite_function_parameter_shadow": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_import_modernizer_does_not_rewrite_function_parameter_shadow",
    ),
    "test_import_modernizer_does_not_rewrite_rebound_local_name_usage": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_import_modernizer_does_not_rewrite_rebound_local_name_usage",
    ),
    "test_import_modernizer_partial_import_keeps_unmapped_symbols": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_import_modernizer_partial_import_keeps_unmapped_symbols",
    ),
    "test_import_modernizer_partial_import_with_asname_keeps_unmapped_alias": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_import_modernizer_partial_import_with_asname_keeps_unmapped_alias",
    ),
    "test_import_modernizer_skips_rewrite_when_runtime_alias_shadowed_in_function": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_import_modernizer_skips_rewrite_when_runtime_alias_shadowed_in_function",
    ),
    "test_import_modernizer_skips_when_runtime_alias_name_is_blocked": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_import_modernizer_skips_when_runtime_alias_name_is_blocked",
    ),
    "test_import_modernizer_updates_aliased_symbol_usage": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_import_modernizer_updates_aliased_symbol_usage",
    ),
    "test_in_context_typevar_not_flagged": (
        "tests.infra.unit.codegen.autofix",
        "test_in_context_typevar_not_flagged",
    ),
    "test_lazy_import_rule_hoists_import_to_module_level": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_lazy_import_rule_hoists_import_to_module_level",
    ),
    "test_lazy_import_rule_uses_fix_action_for_hoist": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_lazy_import_rule_uses_fix_action_for_hoist",
    ),
    "test_legacy_import_bypass_collapses_to_primary_import": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_legacy_import_bypass_collapses_to_primary_import",
    ),
    "test_legacy_rule_uses_fix_action_remove_for_aliases": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_legacy_rule_uses_fix_action_remove_for_aliases",
    ),
    "test_legacy_wrapper_forwarding_keywords_is_inlined_as_alias": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_legacy_wrapper_forwarding_keywords_is_inlined_as_alias",
    ),
    "test_legacy_wrapper_forwarding_varargs_is_inlined_as_alias": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_legacy_wrapper_forwarding_varargs_is_inlined_as_alias",
    ),
    "test_legacy_wrapper_function_is_inlined_as_alias": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_legacy_wrapper_function_is_inlined_as_alias",
    ),
    "test_legacy_wrapper_non_passthrough_is_not_inlined": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_legacy_wrapper_non_passthrough_is_not_inlined",
    ),
    "test_main_all_groups_defined": (
        "tests.infra.unit.test_infra_main",
        "test_main_all_groups_defined",
    ),
    "test_main_analyze_violations_is_read_only": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_main_analyze_violations_is_read_only",
    ),
    "test_main_analyze_violations_writes_json_report": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_main_analyze_violations_writes_json_report",
    ),
    "test_main_calls_sys_exit": (
        "tests.infra.unit.test_infra_workspace_main",
        "test_main_calls_sys_exit",
    ),
    "test_main_constants_quality_gate_dispatch": (
        "tests.infra.unit.codegen.constants_quality_gate",
        "test_main_constants_quality_gate_dispatch",
    ),
    "test_main_constants_quality_gate_parses_before_report": (
        "tests.infra.unit.codegen.constants_quality_gate",
        "test_main_constants_quality_gate_parses_before_report",
    ),
    "test_main_detect_command": (
        "tests.infra.unit.test_infra_workspace_main",
        "test_main_detect_command",
    ),
    "test_main_discovery_failure": (
        "tests.infra.unit.deps.path_sync",
        "test_main_discovery_failure",
    ),
    "test_main_entry_point": (
        "tests.infra.unit.test_infra_workspace_main",
        "test_main_entry_point",
    ),
    "test_main_group_modules_are_valid": (
        "tests.infra.unit.test_infra_main",
        "test_main_group_modules_are_valid",
    ),
    "test_main_help_flag_returns_zero": (
        "tests.infra.unit.test_infra_main",
        "test_main_help_flag_returns_zero",
    ),
    "test_main_migrate_command": (
        "tests.infra.unit.test_infra_workspace_main",
        "test_main_migrate_command",
    ),
    "test_main_migrate_dry_run": (
        "tests.infra.unit.test_infra_workspace_main",
        "test_main_migrate_dry_run",
    ),
    "test_main_no_changes_needed": (
        "tests.infra.unit.deps.path_sync",
        "test_main_no_changes_needed",
    ),
    "test_main_no_command": (
        "tests.infra.unit.test_infra_workspace_main",
        "test_main_no_command",
    ),
    "test_main_orchestrate_command": (
        "tests.infra.unit.test_infra_workspace_main",
        "test_main_orchestrate_command",
    ),
    "test_main_orchestrate_with_fail_fast": (
        "tests.infra.unit.test_infra_workspace_main",
        "test_main_orchestrate_with_fail_fast",
    ),
    "test_main_orchestrate_with_make_args": (
        "tests.infra.unit.test_infra_workspace_main",
        "test_main_orchestrate_with_make_args",
    ),
    "test_main_project_obj_not_dict_first_loop": (
        "tests.infra.unit.deps.path_sync",
        "test_main_project_obj_not_dict_first_loop",
    ),
    "test_main_project_obj_not_dict_second_loop": (
        "tests.infra.unit.deps.path_sync",
        "test_main_project_obj_not_dict_second_loop",
    ),
    "test_main_returns_error_when_no_args": (
        "tests.infra.unit.test_infra_main",
        "test_main_returns_error_when_no_args",
    ),
    "test_main_returns_one_on_failure": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_main_returns_one_on_failure",
    ),
    "test_main_returns_zero_on_success": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_main_returns_zero_on_success",
    ),
    "test_main_sync_command": (
        "tests.infra.unit.test_infra_workspace_main",
        "test_main_sync_command",
    ),
    "test_main_sync_with_canonical_root": (
        "tests.infra.unit.test_infra_workspace_main",
        "test_main_sync_with_canonical_root",
    ),
    "test_main_unknown_group_returns_error": (
        "tests.infra.unit.test_infra_main",
        "test_main_unknown_group_returns_error",
    ),
    "test_main_with_changes_and_dry_run": (
        "tests.infra.unit.deps.path_sync",
        "test_main_with_changes_and_dry_run",
    ),
    "test_main_with_changes_no_dry_run": (
        "tests.infra.unit.deps.path_sync",
        "test_main_with_changes_no_dry_run",
    ),
    "test_migrate_makefile_not_found_non_dry_run": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrate_makefile_not_found_non_dry_run",
    ),
    "test_migrate_pyproject_flext_core_non_dry_run": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrate_pyproject_flext_core_non_dry_run",
    ),
    "test_migrator_apply_updates_project_files": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_apply_updates_project_files",
    ),
    "test_migrator_basemk_generation_failure": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_basemk_generation_failure",
    ),
    "test_migrator_basemk_write_failure": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_basemk_write_failure",
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
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_flext_core_dry_run",
    ),
    "test_migrator_flext_core_project_skipped": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_flext_core_project_skipped",
    ),
    "test_migrator_gitignore_already_normalized_dry_run": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_gitignore_already_normalized_dry_run",
    ),
    "test_migrator_gitignore_read_failure": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_gitignore_read_failure",
    ),
    "test_migrator_gitignore_write_failure": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_gitignore_write_failure",
    ),
    "test_migrator_handles_missing_pyproject_gracefully": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_handles_missing_pyproject_gracefully",
    ),
    "test_migrator_has_flext_core_dependency_in_poetry": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_has_flext_core_dependency_in_poetry",
    ),
    "test_migrator_has_flext_core_dependency_poetry_deps_not_table": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_has_flext_core_dependency_poetry_deps_not_table",
    ),
    "test_migrator_has_flext_core_dependency_poetry_table_missing": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_has_flext_core_dependency_poetry_table_missing",
    ),
    "test_migrator_makefile_not_found_dry_run": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_makefile_not_found_dry_run",
    ),
    "test_migrator_makefile_read_failure": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_makefile_read_failure",
    ),
    "test_migrator_makefile_write_failure": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_makefile_write_failure",
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
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_pyproject_not_found_dry_run",
    ),
    "test_migrator_pyproject_parse_failure": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_pyproject_parse_failure",
    ),
    "test_migrator_pyproject_write_failure": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_migrator_pyproject_write_failure",
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
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_mro_checker_keeps_external_attribute_base",
    ),
    "test_mro_redundancy_checker_removes_nested_attribute_inheritance": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_mro_redundancy_checker_removes_nested_attribute_inheritance",
    ),
    "test_orchestrate_run_project_failure_with_fail_fast": (
        "tests.infra.unit.test_infra_workspace_orchestrator",
        "test_orchestrate_run_project_failure_with_fail_fast",
    ),
    "test_orchestrate_with_project_execution_failure": (
        "tests.infra.unit.test_infra_workspace_orchestrator",
        "test_orchestrate_with_project_execution_failure",
    ),
    "test_orchestrate_with_runner_failure_fail_fast": (
        "tests.infra.unit.test_infra_workspace_orchestrator",
        "test_orchestrate_with_runner_failure_fail_fast",
    ),
    "test_orchestrator_captures_per_project_output": (
        "tests.infra.unit.test_infra_workspace_orchestrator",
        "test_orchestrator_captures_per_project_output",
    ),
    "test_orchestrator_continues_on_failure_without_fail_fast": (
        "tests.infra.unit.test_infra_workspace_orchestrator",
        "test_orchestrator_continues_on_failure_without_fail_fast",
    ),
    "test_orchestrator_execute_returns_failure": (
        "tests.infra.unit.test_infra_workspace_orchestrator",
        "test_orchestrator_execute_returns_failure",
    ),
    "test_orchestrator_executes_verb_across_projects": (
        "tests.infra.unit.test_infra_workspace_orchestrator",
        "test_orchestrator_executes_verb_across_projects",
    ),
    "test_orchestrator_fail_fast_skips_remaining_projects": (
        "tests.infra.unit.test_infra_workspace_orchestrator",
        "test_orchestrator_fail_fast_skips_remaining_projects",
    ),
    "test_orchestrator_fail_fast_with_failure_result": (
        "tests.infra.unit.test_infra_workspace_orchestrator",
        "test_orchestrator_fail_fast_with_failure_result",
    ),
    "test_orchestrator_handles_empty_project_list": (
        "tests.infra.unit.test_infra_workspace_orchestrator",
        "test_orchestrator_handles_empty_project_list",
    ),
    "test_orchestrator_handles_runner_exception": (
        "tests.infra.unit.test_infra_workspace_orchestrator",
        "test_orchestrator_handles_runner_exception",
    ),
    "test_orchestrator_stops_on_first_failure_with_fail_fast": (
        "tests.infra.unit.test_infra_workspace_orchestrator",
        "test_orchestrator_stops_on_first_failure_with_fail_fast",
    ),
    "test_orchestrator_with_make_args": (
        "tests.infra.unit.test_infra_workspace_orchestrator",
        "test_orchestrator_with_make_args",
    ),
    "test_pattern_rule_converts_dict_annotations_to_mapping": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_pattern_rule_converts_dict_annotations_to_mapping",
    ),
    "test_pattern_rule_keeps_dict_param_when_copy_used": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_pattern_rule_keeps_dict_param_when_copy_used",
    ),
    "test_pattern_rule_keeps_dict_param_when_subscript_mutated": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_pattern_rule_keeps_dict_param_when_subscript_mutated",
    ),
    "test_pattern_rule_keeps_type_cast_when_not_nested_object_cast": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_pattern_rule_keeps_type_cast_when_not_nested_object_cast",
    ),
    "test_pattern_rule_optionally_converts_return_annotations_to_mapping": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_pattern_rule_optionally_converts_return_annotations_to_mapping",
    ),
    "test_pattern_rule_removes_configured_redundant_casts": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_pattern_rule_removes_configured_redundant_casts",
    ),
    "test_pattern_rule_removes_nested_type_object_cast_chain": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_pattern_rule_removes_nested_type_object_cast_chain",
    ),
    "test_pattern_rule_skips_overload_signatures": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_pattern_rule_skips_overload_signatures",
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
    "test_refactor_files_skips_non_python_inputs": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_refactor_files_skips_non_python_inputs",
    ),
    "test_refactor_project_integrates_safety_manager": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_refactor_project_integrates_safety_manager",
    ),
    "test_refactor_project_scans_tests_and_scripts_dirs": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_refactor_project_scans_tests_and_scripts_dirs",
    ),
    "test_render_all_generates_large_makefile": (
        "tests.infra.unit.basemk.engine",
        "test_render_all_generates_large_makefile",
    ),
    "test_render_all_has_no_scripts_path_references": (
        "tests.infra.unit.basemk.engine",
        "test_render_all_has_no_scripts_path_references",
    ),
    "test_resolve_gates_maps_type_alias": (
        "tests.infra.unit.check.cli",
        "test_resolve_gates_maps_type_alias",
    ),
    "test_rewrite_dep_paths_dry_run": (
        "tests.infra.unit.deps.path_sync",
        "test_rewrite_dep_paths_dry_run",
    ),
    "test_rewrite_dep_paths_read_failure": (
        "tests.infra.unit.deps.path_sync",
        "test_rewrite_dep_paths_read_failure",
    ),
    "test_rewrite_dep_paths_with_internal_names": (
        "tests.infra.unit.deps.path_sync",
        "test_rewrite_dep_paths_with_internal_names",
    ),
    "test_rewrite_pep621_invalid_path_dep_regex": (
        "tests.infra.unit.deps.path_sync",
        "test_rewrite_pep621_invalid_path_dep_regex",
    ),
    "test_rewrite_pep621_no_project_table": (
        "tests.infra.unit.deps.path_sync",
        "test_rewrite_pep621_no_project_table",
    ),
    "test_rewrite_pep621_non_string_item": (
        "tests.infra.unit.deps.path_sync",
        "test_rewrite_pep621_non_string_item",
    ),
    "test_rewrite_poetry_no_poetry_table": (
        "tests.infra.unit.deps.path_sync",
        "test_rewrite_poetry_no_poetry_table",
    ),
    "test_rewrite_poetry_no_tool_table": (
        "tests.infra.unit.deps.path_sync",
        "test_rewrite_poetry_no_tool_table",
    ),
    "test_rewrite_poetry_with_non_dict_value": (
        "tests.infra.unit.deps.path_sync",
        "test_rewrite_poetry_with_non_dict_value",
    ),
    "test_rule_dispatch_fails_on_invalid_pattern_rule_config": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_rule_dispatch_fails_on_invalid_pattern_rule_config",
    ),
    "test_rule_dispatch_fails_on_unknown_rule_mapping": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_rule_dispatch_fails_on_unknown_rule_mapping",
    ),
    "test_rule_dispatch_keeps_legacy_id_fallback_mapping": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_rule_dispatch_keeps_legacy_id_fallback_mapping",
    ),
    "test_rule_dispatch_prefers_fix_action_metadata": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_rule_dispatch_prefers_fix_action_metadata",
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
    "test_run_deptry_wrapper": (
        "tests.infra.unit.deps.test_detection_wrappers",
        "test_run_deptry_wrapper",
    ),
    "test_run_detect_failure": (
        "tests.infra.unit.test_infra_workspace_main",
        "test_run_detect_failure",
    ),
    "test_run_detect_success": (
        "tests.infra.unit.test_infra_workspace_main",
        "test_run_detect_success",
    ),
    "test_run_migrate_failure": (
        "tests.infra.unit.test_infra_workspace_main",
        "test_run_migrate_failure",
    ),
    "test_run_migrate_success": (
        "tests.infra.unit.test_infra_workspace_main",
        "test_run_migrate_success",
    ),
    "test_run_migrate_with_project_errors": (
        "tests.infra.unit.test_infra_workspace_main",
        "test_run_migrate_with_project_errors",
    ),
    "test_run_mypy_stub_hints_wrapper": (
        "tests.infra.unit.deps.test_detection_wrappers",
        "test_run_mypy_stub_hints_wrapper",
    ),
    "test_run_orchestrate_failure": (
        "tests.infra.unit.test_infra_workspace_main",
        "test_run_orchestrate_failure",
    ),
    "test_run_orchestrate_no_projects": (
        "tests.infra.unit.test_infra_workspace_main",
        "test_run_orchestrate_no_projects",
    ),
    "test_run_orchestrate_success": (
        "tests.infra.unit.test_infra_workspace_main",
        "test_run_orchestrate_success",
    ),
    "test_run_orchestrate_with_failures": (
        "tests.infra.unit.test_infra_workspace_main",
        "test_run_orchestrate_with_failures",
    ),
    "test_run_pip_check_wrapper": (
        "tests.infra.unit.deps.test_detection_wrappers",
        "test_run_pip_check_wrapper",
    ),
    "test_run_sync_failure": (
        "tests.infra.unit.test_infra_workspace_main",
        "test_run_sync_failure",
    ),
    "test_run_sync_success": (
        "tests.infra.unit.test_infra_workspace_main",
        "test_run_sync_success",
    ),
    "test_signature_propagation_removes_and_adds_keywords": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_signature_propagation_removes_and_adds_keywords",
    ),
    "test_signature_propagation_renames_call_keyword": (
        "tests.infra.unit.refactor.test_infra_refactor",
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
    "test_symbol_propagation_keeps_alias_reference_when_asname_used": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_symbol_propagation_keeps_alias_reference_when_asname_used",
    ),
    "test_symbol_propagation_renames_import_and_local_references": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_symbol_propagation_renames_import_and_local_references",
    ),
    "test_symbol_propagation_updates_mro_base_references": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_symbol_propagation_updates_mro_base_references",
    ),
    "test_sync_service_atomic_write_failure": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_service_atomic_write_failure",
    ),
    "test_sync_service_atomic_write_success": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_service_atomic_write_success",
    ),
    "test_sync_service_basemk_generation_failure": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_service_basemk_generation_failure",
    ),
    "test_sync_service_canonical_root_copy": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_service_canonical_root_copy",
    ),
    "test_sync_service_creates_base_mk_if_missing": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_service_creates_base_mk_if_missing",
    ),
    "test_sync_service_detects_changes_via_sha256": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_service_detects_changes_via_sha256",
    ),
    "test_sync_service_ensure_gitignore_entries_all_present": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_service_ensure_gitignore_entries_all_present",
    ),
    "test_sync_service_ensure_gitignore_entries_missing_entries": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_service_ensure_gitignore_entries_missing_entries",
    ),
    "test_sync_service_ensure_gitignore_entries_write_failure": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_service_ensure_gitignore_entries_write_failure",
    ),
    "test_sync_service_execute_returns_failure": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_service_execute_returns_failure",
    ),
    "test_sync_service_generates_base_mk": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_service_generates_base_mk",
    ),
    "test_sync_service_gitignore_sync_failure": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_service_gitignore_sync_failure",
    ),
    "test_sync_service_gitignore_update_failure": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_service_gitignore_update_failure",
    ),
    "test_sync_service_lock_acquisition_failure": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_service_lock_acquisition_failure",
    ),
    "test_sync_service_main_cli": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_service_main_cli",
    ),
    "test_sync_service_project_root_not_exists": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_service_project_root_not_exists",
    ),
    "test_sync_service_project_root_required": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_service_project_root_required",
    ),
    "test_sync_service_sha256_content": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_service_sha256_content",
    ),
    "test_sync_service_sha256_file": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_service_sha256_file",
    ),
    "test_sync_service_skips_write_when_content_unchanged": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_service_skips_write_when_content_unchanged",
    ),
    "test_sync_service_sync_basemk_from_canonical": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_service_sync_basemk_from_canonical",
    ),
    "test_sync_service_sync_basemk_generation_failure": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_service_sync_basemk_generation_failure",
    ),
    "test_sync_service_sync_basemk_no_change_needed": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_service_sync_basemk_no_change_needed",
    ),
    "test_sync_service_sync_basemk_with_canonical_root": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_service_sync_basemk_with_canonical_root",
    ),
    "test_sync_service_validates_gitignore_entries": (
        "tests.infra.unit.test_infra_workspace_sync",
        "test_sync_service_validates_gitignore_entries",
    ),
    "test_syntax_error_files_skipped": (
        "tests.infra.unit.codegen.autofix",
        "test_syntax_error_files_skipped",
    ),
    "test_target_path_standalone": (
        "tests.infra.unit.deps.path_sync",
        "test_target_path_standalone",
    ),
    "test_target_path_workspace_root": (
        "tests.infra.unit.deps.path_sync",
        "test_target_path_workspace_root",
    ),
    "test_target_path_workspace_subproject": (
        "tests.infra.unit.deps.path_sync",
        "test_target_path_workspace_subproject",
    ),
    "test_unwrap_item_with_item": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "test_unwrap_item_with_item",
    ),
    "test_unwrap_item_with_none": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "test_unwrap_item_with_none",
    ),
    "test_violation_analysis_counts_massive_patterns": (
        "tests.infra.unit.refactor.test_infra_refactor",
        "test_violation_analysis_counts_massive_patterns",
    ),
    "test_violation_analyzer_skips_non_utf8_files": (
        "tests.infra.unit.refactor.test_infra_refactor",
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
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_workspace_migrator_error_handling_on_invalid_workspace",
    ),
    "test_workspace_migrator_makefile_not_found_dry_run": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_workspace_migrator_makefile_not_found_dry_run",
    ),
    "test_workspace_migrator_makefile_read_error": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_workspace_migrator_makefile_read_error",
    ),
    "test_workspace_migrator_pyproject_write_error": (
        "tests.infra.unit.test_infra_workspace_migrator",
        "test_workspace_migrator_pyproject_write_error",
    ),
    "test_workspace_root_doc_construction": (
        "tests.infra.unit.deps.test_modernizer_workspace",
        "test_workspace_root_doc_construction",
    ),
    "test_workspace_root_fallback": (
        "tests.infra.unit.deps.path_sync",
        "test_workspace_root_fallback",
    ),
}

__all__ = [
    "EngineSafetyStub",
    "TestAllDirectoriesScanned",
    "TestArray",
    "TestAsStringList",
    "TestAuditorBrokenLinks",
    "TestAuditorBudgets",
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
    "TestBuildSiblingExportIndex",
    "TestBuilderCore",
    "TestBuilderScope",
    "TestCanonicalDevDependencies",
    "TestCensusReportModel",
    "TestCensusViolationModel",
    "TestCheckIssueFormatted",
    "TestCheckMainEntryPoint",
    "TestCheckOnlyMode",
    "TestCheckpoint",
    "TestChecks",
    "TestClassifyIssues",
    "TestClose",
    "TestCollectInternalDeps",
    "TestCollectInternalDepsEdgeCases",
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
    "TestConsolidateGroupsPhase",
    "TestConstants",
    "TestCoreModuleInit",
    "TestCreate",
    "TestDedupeSpecs",
    "TestDepName",
    "TestDetectMode",
    "TestDetectionUncoveredLines",
    "TestDiagResult",
    "TestDiscoverProjects",
    "TestEdgeCases",
    "TestEnsureCheckout",
    "TestEnsureCheckoutEdgeCases",
    "TestEnsurePyreflyConfigPhase",
    "TestEnsurePyrightConfigPhase",
    "TestEnsureSymlink",
    "TestEnsureSymlinkEdgeCases",
    "TestEnsureTable",
    "TestExcludedDirectories",
    "TestExcludedProjects",
    "TestExtractDepName",
    "TestExtractExports",
    "TestExtractInlineConstants",
    "TestExtractRequirementName",
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
    "TestFlextInfraCommandRunner",
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
    "TestFlextInfraDocValidator",
    "TestFlextInfraDocs",
    "TestFlextInfraDocsShared",
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
    "TestFlextInfraPythonVersionEnforcer",
    "TestFlextInfraReleaseOrchestrator",
    "TestFlextInfraReleaseOrchestratorChangeCollection",
    "TestFlextInfraReleaseOrchestratorDispatchPhase",
    "TestFlextInfraReleaseOrchestratorPhaseBuild",
    "TestFlextInfraReleaseOrchestratorPhaseVersion",
    "TestFlextInfraReportingService",
    "TestFlextInfraRuntimeDevDependencyDetectorInit",
    "TestFlextInfraRuntimeDevDependencyDetectorRunDetect",
    "TestFlextInfraRuntimeDevDependencyDetectorRunReport",
    "TestFlextInfraRuntimeDevDependencyDetectorRunTypings",
    "TestFlextInfraSubmoduleInitLazyLoading",
    "TestFlextInfraTomlService",
    "TestFlextInfraTypesImport",
    "TestFlextInfraUtilitiesImport",
    "TestFlextInfraUtilitiesSelection",
    "TestFlextInfraVersion",
    "TestFlextInfraVersioningService",
    "TestFlextInfraWorkflowLinter",
    "TestFlextInfraWorkflowSyncer",
    "TestFlextInfraWorkspace",
    "TestFlextInfraWorkspaceChecker",
    "TestGenerateFile",
    "TestGenerateTypeChecking",
    "TestGeneratedClassNamingConvention",
    "TestGeneratedFilesAreValidPython",
    "TestGeneratorCore",
    "TestGeneratorHelpers",
    "TestGeneratorScope",
    "TestGetDepPaths",
    "TestGithubInit",
    "TestGoFormatEmptyLineSkipping",
    "TestGoFormatEmptyLines",
    "TestGoFormatParsing",
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
    "TestInventoryServiceCore",
    "TestInventoryServiceReports",
    "TestInventoryServiceScripts",
    "TestIsInternalPathDep",
    "TestIsRelativeTo",
    "TestIsWorkspaceMode",
    "TestJsonWriteFailure",
    "TestLintAndFormatPublicMethods",
    "TestLoadDependencyLimits",
    "TestMain",
    "TestMainBaseMkValidate",
    "TestMainCliRouting",
    "TestMainCommandDispatch",
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
    "TestMaintenanceMain",
    "TestMarkdownLinting",
    "TestMarkdownReportSkipsEmptyGates",
    "TestMarkdownReportWithErrors",
    "TestMerge",
    "TestMergeChildExports",
    "TestModuleAndTypingsFlow",
    "TestModuleLevelWrappers",
    "TestMroFacadeMethods",
    "TestMypyEmptyLineSkipping",
    "TestMypyEmptyLines",
    "TestMypyJSONParsing",
    "TestNormalizeStringList",
    "TestOrchestrate",
    "TestOwnerFromRemoteUrl",
    "TestParseArgs",
    "TestParseGitmodules",
    "TestParseRepoMap",
    "TestParseViolationInvalid",
    "TestParseViolationValid",
    "TestParser",
    "TestPathDepPathsPep621",
    "TestPathDepPathsPoetry",
    "TestPathSyncEdgeCases",
    "TestPreviousTag",
    "TestProcessDirectory",
    "TestProcessFileReadError",
    "TestProjectDevGroups",
    "TestProjectResultProperties",
    "TestPushRelease",
    "TestPytestDiagExtractorCore",
    "TestPytestDiagLogParsing",
    "TestPytestDiagParseXml",
    "TestReadDoc",
    "TestReadExistingDocstring",
    "TestReleaseInit",
    "TestReleaseMainFlow",
    "TestReleaseMainParsing",
    "TestReleaseMainTagResolution",
    "TestReleaseMainVersionResolution",
    "TestRemovedCompatibilityMethods",
    "TestRenderTemplate",
    "TestResolveAliases",
    "TestResolveRef",
    "TestResolveVersionInteractive",
    "TestRewriteDepPaths",
    "TestRewritePep621",
    "TestRewritePoetry",
    "TestRuffFormatDeduplication",
    "TestRuffFormatDuplicateSkipping",
    "TestRuffFormatDuplicates",
    "TestRuffFormatEmptyLines",
    "TestRunAudit",
    "TestRunBuild",
    "TestRunCLI",
    "TestRunDeptry",
    "TestRunFix",
    "TestRunGenerate",
    "TestRunLint",
    "TestRunMypyStubHints",
    "TestRunPipCheck",
    "TestRunPr",
    "TestRunPrWorkspace",
    "TestRunRuffFix",
    "TestRunValidate",
    "TestRunWorkflows",
    "TestSafeLoadYaml",
    "TestScaffoldProjectCreatesSrcModules",
    "TestScaffoldProjectCreatesTestsModules",
    "TestScaffoldProjectIdempotency",
    "TestScaffoldProjectNoop",
    "TestScanAstPublicDefs",
    "TestScannerCore",
    "TestScannerHelpers",
    "TestScannerMultiFile",
    "TestScannerValidation",
    "TestSelectorFunction",
    "TestShouldBubbleUp",
    "TestShouldUseColor",
    "TestShouldUseUnicode",
    "TestSkillValidatorAstGrepCount",
    "TestSkillValidatorCore",
    "TestSkillValidatorCustomCount",
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
    "TestSyncExtraPaths",
    "TestSyncMethodEdgeCases",
    "TestSyncOne",
    "TestSyncOneEdgeCases",
    "TestSyncOperation",
    "TestSyncProject",
    "TestSynthesizedRepoMap",
    "TestTargetPath",
    "TestTemplateEngineConstants",
    "TestTemplateEngineErrorHandling",
    "TestTemplateEngineInstances",
    "TestTemplateEngineRender",
    "TestToInfraValue",
    "TestTriggerRelease",
    "TestUnwrapItem",
    "TestValidateGitRefEdgeCases",
    "TestView",
    "TestViolationPattern",
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
    "TestWorkspaceRoot",
    "TestWorkspaceRootFromEnv",
    "TestWorkspaceRootFromParents",
    "c",
    "census",
    "detector",
    "fixer",
    "m",
    "orchestrator",
    "r",
    "run_lint",
    "run_pr",
    "run_pr_workspace",
    "run_workflows",
    "s",
    "sync_service",
    "t",
    "test_as_string_list_with_item_and_edge_values",
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
    "test_check_main_executes_real_cli",
    "test_class_reconstructor_reorders_each_contiguous_method_block",
    "test_class_reconstructor_reorders_methods_by_config",
    "test_class_reconstructor_skips_interleaved_non_method_members",
    "test_codegen_dir_returns_all_exports",
    "test_codegen_getattr_raises_attribute_error",
    "test_codegen_init_getattr_raises_attribute_error",
    "test_codegen_lazy_imports_work",
    "test_codegen_pipeline_end_to_end",
    "test_consolidate_groups_phase_apply_removes_old_groups",
    "test_consolidate_groups_phase_apply_with_empty_poetry_group",
    "test_detect_mode_with_path_object",
    "test_detector_detects_standalone_mode_without_parent_git",
    "test_detector_detects_standalone_with_non_flext_repo",
    "test_detector_detects_workspace_mode_with_flext_repo",
    "test_detector_detects_workspace_mode_with_parent_git",
    "test_detector_execute_returns_failure",
    "test_detector_extracts_repo_name_from_https_url",
    "test_detector_extracts_repo_name_from_ssh_url",
    "test_detector_extracts_repo_name_without_git_suffix",
    "test_detector_handles_empty_origin_url",
    "test_detector_handles_exception_during_detection",
    "test_detector_handles_git_command_errors",
    "test_detector_handles_git_command_failure",
    "test_detector_handles_runner_failure",
    "test_detector_returns_standalone_when_no_parent_git",
    "test_discover_projects_wrapper",
    "test_engine_always_enables_class_nesting_file_rule",
    "test_ensure_future_annotations_after_docstring",
    "test_ensure_future_annotations_moves_existing_import_to_top",
    "test_ensure_pyrefly_config_phase_apply_ignore_errors",
    "test_ensure_pyrefly_config_phase_apply_python_version",
    "test_ensure_pyrefly_config_phase_apply_search_path_and_errors",
    "test_ensure_table_with_non_table_value_uncovered",
    "test_extract_requirement_name_invalid",
    "test_extract_requirement_name_simple",
    "test_extract_requirement_name_with_path_dep",
    "test_files_modified_tracks_affected_files",
    "test_fix_pyrefly_config_main_executes_real_cli_help",
    "test_flexcore_excluded_from_run",
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
    "test_handle_constants_quality_gate_json_exits_with_int",
    "test_handle_constants_quality_gate_text_exits_with_int",
    "test_helpers_alias_available",
    "test_helpers_alias_exposed",
    "test_helpers_alias_is_reachable",
    "test_import_modernizer_adds_c_when_existing_c_is_aliased",
    "test_import_modernizer_does_not_rewrite_function_parameter_shadow",
    "test_import_modernizer_does_not_rewrite_rebound_local_name_usage",
    "test_import_modernizer_partial_import_keeps_unmapped_symbols",
    "test_import_modernizer_partial_import_with_asname_keeps_unmapped_alias",
    "test_import_modernizer_skips_rewrite_when_runtime_alias_shadowed_in_function",
    "test_import_modernizer_skips_when_runtime_alias_name_is_blocked",
    "test_import_modernizer_updates_aliased_symbol_usage",
    "test_in_context_typevar_not_flagged",
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
    "test_main_calls_sys_exit",
    "test_main_constants_quality_gate_dispatch",
    "test_main_constants_quality_gate_parses_before_report",
    "test_main_detect_command",
    "test_main_discovery_failure",
    "test_main_entry_point",
    "test_main_group_modules_are_valid",
    "test_main_help_flag_returns_zero",
    "test_main_migrate_command",
    "test_main_migrate_dry_run",
    "test_main_no_changes_needed",
    "test_main_no_command",
    "test_main_orchestrate_command",
    "test_main_orchestrate_with_fail_fast",
    "test_main_orchestrate_with_make_args",
    "test_main_project_obj_not_dict_first_loop",
    "test_main_project_obj_not_dict_second_loop",
    "test_main_returns_error_when_no_args",
    "test_main_returns_one_on_failure",
    "test_main_returns_zero_on_success",
    "test_main_sync_command",
    "test_main_sync_with_canonical_root",
    "test_main_unknown_group_returns_error",
    "test_main_with_changes_and_dry_run",
    "test_main_with_changes_no_dry_run",
    "test_migrate_makefile_not_found_non_dry_run",
    "test_migrate_pyproject_flext_core_non_dry_run",
    "test_migrator_apply_updates_project_files",
    "test_migrator_basemk_generation_failure",
    "test_migrator_basemk_write_failure",
    "test_migrator_discovery_failure",
    "test_migrator_dry_run_reports_changes_without_writes",
    "test_migrator_execute_returns_failure",
    "test_migrator_flext_core_dry_run",
    "test_migrator_flext_core_project_skipped",
    "test_migrator_gitignore_already_normalized_dry_run",
    "test_migrator_gitignore_read_failure",
    "test_migrator_gitignore_write_failure",
    "test_migrator_handles_missing_pyproject_gracefully",
    "test_migrator_has_flext_core_dependency_in_poetry",
    "test_migrator_has_flext_core_dependency_poetry_deps_not_table",
    "test_migrator_has_flext_core_dependency_poetry_table_missing",
    "test_migrator_makefile_not_found_dry_run",
    "test_migrator_makefile_read_failure",
    "test_migrator_makefile_write_failure",
    "test_migrator_no_changes_needed",
    "test_migrator_preserves_custom_makefile_content",
    "test_migrator_pyproject_not_found_dry_run",
    "test_migrator_pyproject_parse_failure",
    "test_migrator_pyproject_write_failure",
    "test_migrator_workspace_root_not_exists",
    "test_migrator_workspace_root_project_detection",
    "test_mro_checker_keeps_external_attribute_base",
    "test_mro_redundancy_checker_removes_nested_attribute_inheritance",
    "test_orchestrate_run_project_failure_with_fail_fast",
    "test_orchestrate_with_project_execution_failure",
    "test_orchestrate_with_runner_failure_fail_fast",
    "test_orchestrator_captures_per_project_output",
    "test_orchestrator_continues_on_failure_without_fail_fast",
    "test_orchestrator_execute_returns_failure",
    "test_orchestrator_executes_verb_across_projects",
    "test_orchestrator_fail_fast_skips_remaining_projects",
    "test_orchestrator_fail_fast_with_failure_result",
    "test_orchestrator_handles_empty_project_list",
    "test_orchestrator_handles_runner_exception",
    "test_orchestrator_stops_on_first_failure_with_fail_fast",
    "test_orchestrator_with_make_args",
    "test_pattern_rule_converts_dict_annotations_to_mapping",
    "test_pattern_rule_keeps_dict_param_when_copy_used",
    "test_pattern_rule_keeps_dict_param_when_subscript_mutated",
    "test_pattern_rule_keeps_type_cast_when_not_nested_object_cast",
    "test_pattern_rule_optionally_converts_return_annotations_to_mapping",
    "test_pattern_rule_removes_configured_redundant_casts",
    "test_pattern_rule_removes_nested_type_object_cast_chain",
    "test_pattern_rule_skips_overload_signatures",
    "test_project_without_src_returns_empty",
    "test_quality_gate_real_workspace_run",
    "test_quality_gate_success_verdict_helper",
    "test_refactor_files_skips_non_python_inputs",
    "test_refactor_project_integrates_safety_manager",
    "test_refactor_project_scans_tests_and_scripts_dirs",
    "test_render_all_generates_large_makefile",
    "test_render_all_has_no_scripts_path_references",
    "test_resolve_gates_maps_type_alias",
    "test_rewrite_dep_paths_dry_run",
    "test_rewrite_dep_paths_read_failure",
    "test_rewrite_dep_paths_with_internal_names",
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
    "test_run_cli_run_returns_one_for_fail",
    "test_run_cli_run_returns_two_for_error",
    "test_run_cli_run_returns_zero_for_pass",
    "test_run_cli_with_fail_fast_flag",
    "test_run_cli_with_multiple_projects",
    "test_run_deptry_wrapper",
    "test_run_detect_failure",
    "test_run_detect_success",
    "test_run_migrate_failure",
    "test_run_migrate_success",
    "test_run_migrate_with_project_errors",
    "test_run_mypy_stub_hints_wrapper",
    "test_run_orchestrate_failure",
    "test_run_orchestrate_no_projects",
    "test_run_orchestrate_success",
    "test_run_orchestrate_with_failures",
    "test_run_pip_check_wrapper",
    "test_run_sync_failure",
    "test_run_sync_success",
    "test_signature_propagation_removes_and_adds_keywords",
    "test_signature_propagation_renames_call_keyword",
    "test_standalone_final_detected_as_fixable",
    "test_standalone_typealias_detected_as_fixable",
    "test_standalone_typevar_detected_as_fixable",
    "test_symbol_propagation_keeps_alias_reference_when_asname_used",
    "test_symbol_propagation_renames_import_and_local_references",
    "test_symbol_propagation_updates_mro_base_references",
    "test_sync_service_atomic_write_failure",
    "test_sync_service_atomic_write_success",
    "test_sync_service_basemk_generation_failure",
    "test_sync_service_canonical_root_copy",
    "test_sync_service_creates_base_mk_if_missing",
    "test_sync_service_detects_changes_via_sha256",
    "test_sync_service_ensure_gitignore_entries_all_present",
    "test_sync_service_ensure_gitignore_entries_missing_entries",
    "test_sync_service_ensure_gitignore_entries_write_failure",
    "test_sync_service_execute_returns_failure",
    "test_sync_service_generates_base_mk",
    "test_sync_service_gitignore_sync_failure",
    "test_sync_service_gitignore_update_failure",
    "test_sync_service_lock_acquisition_failure",
    "test_sync_service_main_cli",
    "test_sync_service_project_root_not_exists",
    "test_sync_service_project_root_required",
    "test_sync_service_sha256_content",
    "test_sync_service_sha256_file",
    "test_sync_service_skips_write_when_content_unchanged",
    "test_sync_service_sync_basemk_from_canonical",
    "test_sync_service_sync_basemk_generation_failure",
    "test_sync_service_sync_basemk_no_change_needed",
    "test_sync_service_sync_basemk_with_canonical_root",
    "test_sync_service_validates_gitignore_entries",
    "test_syntax_error_files_skipped",
    "test_target_path_standalone",
    "test_target_path_workspace_root",
    "test_target_path_workspace_subproject",
    "test_unwrap_item_with_item",
    "test_unwrap_item_with_none",
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
]


def __getattr__(name: str) -> Any:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
