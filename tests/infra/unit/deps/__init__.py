# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Tests for flext_infra.deps dependency management modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from tests.infra.unit.deps.test_detection_classify import (
        TestBuildProjectReport,
        TestClassifyIssues,
    )
    from tests.infra.unit.deps.test_detection_deptry import TestRunDeptry
    from tests.infra.unit.deps.test_detection_discover import TestDiscoverProjects
    from tests.infra.unit.deps.test_detection_models import (
        TestFlextInfraDependencyDetectionModels,
        TestFlextInfraDependencyDetectionModels as m,
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
        TestConstants as c,
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
        TestSyncExtraPaths,
        TestSyncOneEdgeCases,
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
        TestArray,
        TestAsStringList,
        TestCanonicalDevDependencies,
        TestDedupeSpecs,
        TestDepName,
        TestEnsureTable,
        TestProjectDevGroups,
        TestUnwrapItem,
        test_as_string_list_with_item,
        test_as_string_list_with_item_unwrap_returns_none,
        test_as_string_list_with_mapping,
        test_as_string_list_with_string,
        test_ensure_table_with_non_table_value_uncovered,
        test_unwrap_item_with_item,
        test_unwrap_item_with_none,
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
        TestWorkspaceRoot,
        test_workspace_root_doc_construction,
    )
    from tests.infra.unit.deps.test_path_sync_helpers import (
        TestExtractDepName,
        TestExtractRequirementName,
        TestTargetPath,
        test_extract_requirement_name_invalid,
        test_extract_requirement_name_simple,
        test_extract_requirement_name_with_path_dep,
        test_helpers_alias_is_reachable_helpers,
        test_target_path_standalone,
        test_target_path_workspace_root,
        test_target_path_workspace_subproject,
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

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "TestArray": ("tests.infra.unit.deps.test_modernizer_helpers", "TestArray"),
    "TestAsStringList": ("tests.infra.unit.deps.test_modernizer_helpers", "TestAsStringList"),
    "TestBuildProjectReport": ("tests.infra.unit.deps.test_detection_classify", "TestBuildProjectReport"),
    "TestCanonicalDevDependencies": ("tests.infra.unit.deps.test_modernizer_helpers", "TestCanonicalDevDependencies"),
    "TestClassifyIssues": ("tests.infra.unit.deps.test_detection_classify", "TestClassifyIssues"),
    "TestCollectInternalDeps": ("tests.infra.unit.deps.test_internal_sync_discovery", "TestCollectInternalDeps"),
    "TestCollectInternalDepsEdgeCases": ("tests.infra.unit.deps.test_internal_sync_discovery_edge", "TestCollectInternalDepsEdgeCases"),
    "TestConsolidateGroupsPhase": ("tests.infra.unit.deps.test_modernizer_consolidate", "TestConsolidateGroupsPhase"),
    "TestConstants": ("tests.infra.unit.deps.test_extra_paths_manager", "TestConstants"),
    "TestDedupeSpecs": ("tests.infra.unit.deps.test_modernizer_helpers", "TestDedupeSpecs"),
    "TestDepName": ("tests.infra.unit.deps.test_modernizer_helpers", "TestDepName"),
    "TestDetectMode": ("tests.infra.unit.deps.test_path_sync_init", "TestDetectMode"),
    "TestDetectionUncoveredLines": ("tests.infra.unit.deps.test_detection_uncovered", "TestDetectionUncoveredLines"),
    "TestDetectorReportFlags": ("tests.infra.unit.deps.test_detector_report_flags", "TestDetectorReportFlags"),
    "TestDetectorRunFailures": ("tests.infra.unit.deps.test_detector_detect_failures", "TestDetectorRunFailures"),
    "TestDiscoverProjects": ("tests.infra.unit.deps.test_detection_discover", "TestDiscoverProjects"),
    "TestEnsureCheckout": ("tests.infra.unit.deps.test_internal_sync_update", "TestEnsureCheckout"),
    "TestEnsureCheckoutEdgeCases": ("tests.infra.unit.deps.test_internal_sync_update_checkout_edge", "TestEnsureCheckoutEdgeCases"),
    "TestEnsurePyreflyConfigPhase": ("tests.infra.unit.deps.test_modernizer_pyrefly", "TestEnsurePyreflyConfigPhase"),
    "TestEnsurePyrightConfigPhase": ("tests.infra.unit.deps.test_modernizer_pyright", "TestEnsurePyrightConfigPhase"),
    "TestEnsurePytestConfigPhase": ("tests.infra.unit.deps.test_modernizer_pytest", "TestEnsurePytestConfigPhase"),
    "TestEnsureSymlink": ("tests.infra.unit.deps.test_internal_sync_update", "TestEnsureSymlink"),
    "TestEnsureSymlinkEdgeCases": ("tests.infra.unit.deps.test_internal_sync_update", "TestEnsureSymlinkEdgeCases"),
    "TestEnsureTable": ("tests.infra.unit.deps.test_modernizer_helpers", "TestEnsureTable"),
    "TestExtractDepName": ("tests.infra.unit.deps.test_path_sync_helpers", "TestExtractDepName"),
    "TestExtractRequirementName": ("tests.infra.unit.deps.test_path_sync_helpers", "TestExtractRequirementName"),
    "TestFlextInfraDependencyDetectionModels": ("tests.infra.unit.deps.test_detection_models", "TestFlextInfraDependencyDetectionModels"),
    "TestFlextInfraDependencyDetectionService": ("tests.infra.unit.deps.test_detection_models", "TestFlextInfraDependencyDetectionService"),
    "TestFlextInfraDependencyDetectorModels": ("tests.infra.unit.deps.test_detector_models", "TestFlextInfraDependencyDetectorModels"),
    "TestFlextInfraDependencyPathSync": ("tests.infra.unit.deps.test_path_sync_init", "TestFlextInfraDependencyPathSync"),
    "TestFlextInfraDeps": ("tests.infra.unit.deps.test_init", "TestFlextInfraDeps"),
    "TestFlextInfraExtraPathsManager": ("tests.infra.unit.deps.test_extra_paths_manager", "TestFlextInfraExtraPathsManager"),
    "TestFlextInfraInternalDependencySyncService": ("tests.infra.unit.deps.test_internal_sync_validation", "TestFlextInfraInternalDependencySyncService"),
    "TestFlextInfraPyprojectModernizer": ("tests.infra.unit.deps.test_modernizer_main", "TestFlextInfraPyprojectModernizer"),
    "TestFlextInfraRuntimeDevDependencyDetectorInit": ("tests.infra.unit.deps.test_detector_init", "TestFlextInfraRuntimeDevDependencyDetectorInit"),
    "TestFlextInfraRuntimeDevDependencyDetectorRunDetect": ("tests.infra.unit.deps.test_detector_detect", "TestFlextInfraRuntimeDevDependencyDetectorRunDetect"),
    "TestFlextInfraRuntimeDevDependencyDetectorRunReport": ("tests.infra.unit.deps.test_detector_report", "TestFlextInfraRuntimeDevDependencyDetectorRunReport"),
    "TestFlextInfraRuntimeDevDependencyDetectorRunTypings": ("tests.infra.unit.deps.test_detector_main", "TestFlextInfraRuntimeDevDependencyDetectorRunTypings"),
    "TestGetDepPaths": ("tests.infra.unit.deps.test_extra_paths_manager", "TestGetDepPaths"),
    "TestInferOwnerFromOrigin": ("tests.infra.unit.deps.test_internal_sync_resolve", "TestInferOwnerFromOrigin"),
    "TestInjectCommentsPhase": ("tests.infra.unit.deps.test_modernizer_comments", "TestInjectCommentsPhase"),
    "TestIsInternalPathDep": ("tests.infra.unit.deps.test_internal_sync_validation", "TestIsInternalPathDep"),
    "TestIsRelativeTo": ("tests.infra.unit.deps.test_internal_sync_validation", "TestIsRelativeTo"),
    "TestIsWorkspaceMode": ("tests.infra.unit.deps.test_internal_sync_workspace", "TestIsWorkspaceMode"),
    "TestLoadDependencyLimits": ("tests.infra.unit.deps.test_detection_typings", "TestLoadDependencyLimits"),
    "TestMain": ("tests.infra.unit.deps.test_path_sync_main", "TestMain"),
    "TestMainEdgeCases": ("tests.infra.unit.deps.test_path_sync_main_edges", "TestMainEdgeCases"),
    "TestMainExceptionHandling": ("tests.infra.unit.deps.test_main_dispatch", "TestMainExceptionHandling"),
    "TestMainFunction": ("tests.infra.unit.deps.test_detector_main", "TestMainFunction"),
    "TestMainHelpAndErrors": ("tests.infra.unit.deps.test_main", "TestMainHelpAndErrors"),
    "TestMainModuleImport": ("tests.infra.unit.deps.test_main_dispatch", "TestMainModuleImport"),
    "TestMainReturnValues": ("tests.infra.unit.deps.test_main", "TestMainReturnValues"),
    "TestMainStructlogConfiguration": ("tests.infra.unit.deps.test_main_dispatch", "TestMainStructlogConfiguration"),
    "TestMainSubcommandDispatch": ("tests.infra.unit.deps.test_main_dispatch", "TestMainSubcommandDispatch"),
    "TestMainSysArgvModification": ("tests.infra.unit.deps.test_main_dispatch", "TestMainSysArgvModification"),
    "TestModernizerEdgeCases": ("tests.infra.unit.deps.test_modernizer_main_extra", "TestModernizerEdgeCases"),
    "TestModernizerRunAndMain": ("tests.infra.unit.deps.test_modernizer_main", "TestModernizerRunAndMain"),
    "TestModernizerUncoveredLines": ("tests.infra.unit.deps.test_modernizer_main_extra", "TestModernizerUncoveredLines"),
    "TestModuleAndTypingsFlow": ("tests.infra.unit.deps.test_detection_typings_flow", "TestModuleAndTypingsFlow"),
    "TestModuleLevelWrappers": ("tests.infra.unit.deps.test_detection_wrappers", "TestModuleLevelWrappers"),
    "TestOwnerFromRemoteUrl": ("tests.infra.unit.deps.test_internal_sync_validation", "TestOwnerFromRemoteUrl"),
    "TestParseGitmodules": ("tests.infra.unit.deps.test_internal_sync_discovery", "TestParseGitmodules"),
    "TestParseRepoMap": ("tests.infra.unit.deps.test_internal_sync_discovery", "TestParseRepoMap"),
    "TestParser": ("tests.infra.unit.deps.test_modernizer_workspace", "TestParser"),
    "TestPathDepPathsPep621": ("tests.infra.unit.deps.test_extra_paths_pep621", "TestPathDepPathsPep621"),
    "TestPathDepPathsPoetry": ("tests.infra.unit.deps.test_extra_paths_pep621", "TestPathDepPathsPoetry"),
    "TestPathSyncEdgeCases": ("tests.infra.unit.deps.test_path_sync_init", "TestPathSyncEdgeCases"),
    "TestProjectDevGroups": ("tests.infra.unit.deps.test_modernizer_helpers", "TestProjectDevGroups"),
    "TestReadDoc": ("tests.infra.unit.deps.test_modernizer_workspace", "TestReadDoc"),
    "TestResolveRef": ("tests.infra.unit.deps.test_internal_sync_resolve", "TestResolveRef"),
    "TestRewriteDepPaths": ("tests.infra.unit.deps.test_path_sync_rewrite_deps", "TestRewriteDepPaths"),
    "TestRewritePep621": ("tests.infra.unit.deps.test_path_sync_rewrite_pep621", "TestRewritePep621"),
    "TestRewritePoetry": ("tests.infra.unit.deps.test_path_sync_rewrite_poetry", "TestRewritePoetry"),
    "TestRunDeptry": ("tests.infra.unit.deps.test_detection_deptry", "TestRunDeptry"),
    "TestRunMypyStubHints": ("tests.infra.unit.deps.test_detection_typings", "TestRunMypyStubHints"),
    "TestRunPipCheck": ("tests.infra.unit.deps.test_detection_pip_check", "TestRunPipCheck"),
    "TestSubcommandMapping": ("tests.infra.unit.deps.test_main", "TestSubcommandMapping"),
    "TestSync": ("tests.infra.unit.deps.test_internal_sync_sync", "TestSync"),
    "TestSyncExtraPaths": ("tests.infra.unit.deps.test_extra_paths_sync", "TestSyncExtraPaths"),
    "TestSyncMethodEdgeCases": ("tests.infra.unit.deps.test_internal_sync_sync_edge", "TestSyncMethodEdgeCases"),
    "TestSyncMethodEdgeCasesMore": ("tests.infra.unit.deps.test_internal_sync_sync_edge_more", "TestSyncMethodEdgeCasesMore"),
    "TestSyncOne": ("tests.infra.unit.deps.test_extra_paths_manager", "TestSyncOne"),
    "TestSyncOneEdgeCases": ("tests.infra.unit.deps.test_extra_paths_sync", "TestSyncOneEdgeCases"),
    "TestSynthesizedRepoMap": ("tests.infra.unit.deps.test_internal_sync_resolve", "TestSynthesizedRepoMap"),
    "TestTargetPath": ("tests.infra.unit.deps.test_path_sync_helpers", "TestTargetPath"),
    "TestToInfraValue": ("tests.infra.unit.deps.test_detection_models", "TestToInfraValue"),
    "TestUnwrapItem": ("tests.infra.unit.deps.test_modernizer_helpers", "TestUnwrapItem"),
    "TestValidateGitRefEdgeCases": ("tests.infra.unit.deps.test_internal_sync_validation", "TestValidateGitRefEdgeCases"),
    "TestWorkspaceRoot": ("tests.infra.unit.deps.test_modernizer_workspace", "TestWorkspaceRoot"),
    "TestWorkspaceRootFromEnv": ("tests.infra.unit.deps.test_internal_sync_workspace", "TestWorkspaceRootFromEnv"),
    "TestWorkspaceRootFromParents": ("tests.infra.unit.deps.test_internal_sync_workspace", "TestWorkspaceRootFromParents"),
    "c": ("tests.infra.unit.deps.test_extra_paths_manager", "TestConstants"),
    "m": ("tests.infra.unit.deps.test_detection_models", "TestFlextInfraDependencyDetectionModels"),
    "s": ("tests.infra.unit.deps.test_detection_models", "TestFlextInfraDependencyDetectionService"),
    "test_as_string_list_with_item": ("tests.infra.unit.deps.test_modernizer_helpers", "test_as_string_list_with_item"),
    "test_as_string_list_with_item_unwrap_returns_none": ("tests.infra.unit.deps.test_modernizer_helpers", "test_as_string_list_with_item_unwrap_returns_none"),
    "test_as_string_list_with_mapping": ("tests.infra.unit.deps.test_modernizer_helpers", "test_as_string_list_with_mapping"),
    "test_as_string_list_with_string": ("tests.infra.unit.deps.test_modernizer_helpers", "test_as_string_list_with_string"),
    "test_consolidate_groups_phase_apply_removes_old_groups": ("tests.infra.unit.deps.test_modernizer_consolidate", "test_consolidate_groups_phase_apply_removes_old_groups"),
    "test_consolidate_groups_phase_apply_with_empty_poetry_group": ("tests.infra.unit.deps.test_modernizer_consolidate", "test_consolidate_groups_phase_apply_with_empty_poetry_group"),
    "test_detect_mode_with_nonexistent_path": ("tests.infra.unit.deps.test_path_sync_init", "test_detect_mode_with_nonexistent_path"),
    "test_detect_mode_with_path_object": ("tests.infra.unit.deps.test_path_sync_init", "test_detect_mode_with_path_object"),
    "test_discover_projects_wrapper": ("tests.infra.unit.deps.test_detection_wrappers", "test_discover_projects_wrapper"),
    "test_ensure_pyrefly_config_phase_apply_errors": ("tests.infra.unit.deps.test_modernizer_pyrefly", "test_ensure_pyrefly_config_phase_apply_errors"),
    "test_ensure_pyrefly_config_phase_apply_ignore_errors": ("tests.infra.unit.deps.test_modernizer_pyrefly", "test_ensure_pyrefly_config_phase_apply_ignore_errors"),
    "test_ensure_pyrefly_config_phase_apply_python_version": ("tests.infra.unit.deps.test_modernizer_pyrefly", "test_ensure_pyrefly_config_phase_apply_python_version"),
    "test_ensure_pyrefly_config_phase_apply_search_path": ("tests.infra.unit.deps.test_modernizer_pyrefly", "test_ensure_pyrefly_config_phase_apply_search_path"),
    "test_ensure_pytest_config_phase_apply_markers": ("tests.infra.unit.deps.test_modernizer_pytest", "test_ensure_pytest_config_phase_apply_markers"),
    "test_ensure_pytest_config_phase_apply_minversion": ("tests.infra.unit.deps.test_modernizer_pytest", "test_ensure_pytest_config_phase_apply_minversion"),
    "test_ensure_pytest_config_phase_apply_python_classes": ("tests.infra.unit.deps.test_modernizer_pytest", "test_ensure_pytest_config_phase_apply_python_classes"),
    "test_ensure_table_with_non_table_value_uncovered": ("tests.infra.unit.deps.test_modernizer_helpers", "test_ensure_table_with_non_table_value_uncovered"),
    "test_extract_requirement_name_invalid": ("tests.infra.unit.deps.test_path_sync_helpers", "test_extract_requirement_name_invalid"),
    "test_extract_requirement_name_simple": ("tests.infra.unit.deps.test_path_sync_helpers", "test_extract_requirement_name_simple"),
    "test_extract_requirement_name_with_path_dep": ("tests.infra.unit.deps.test_path_sync_helpers", "test_extract_requirement_name_with_path_dep"),
    "test_flext_infra_pyproject_modernizer_find_pyproject_files": ("tests.infra.unit.deps.test_modernizer_main_extra", "test_flext_infra_pyproject_modernizer_find_pyproject_files"),
    "test_flext_infra_pyproject_modernizer_process_file_invalid_toml": ("tests.infra.unit.deps.test_modernizer_main_extra", "test_flext_infra_pyproject_modernizer_process_file_invalid_toml"),
    "test_get_current_typings_from_pyproject_wrapper": ("tests.infra.unit.deps.test_detection_wrappers", "test_get_current_typings_from_pyproject_wrapper"),
    "test_get_required_typings_wrapper": ("tests.infra.unit.deps.test_detection_wrappers", "test_get_required_typings_wrapper"),
    "test_helpers_alias_exposed": ("tests.infra.unit.deps.test_extra_paths_pep621", "test_helpers_alias_exposed"),
    "test_helpers_alias_is_reachable_helpers": ("tests.infra.unit.deps.test_path_sync_helpers", "test_helpers_alias_is_reachable_helpers"),
    "test_helpers_alias_is_reachable_main": ("tests.infra.unit.deps.test_path_sync_main", "test_helpers_alias_is_reachable_main"),
    "test_helpers_alias_is_reachable_pep621": ("tests.infra.unit.deps.test_path_sync_rewrite_pep621", "test_helpers_alias_is_reachable_pep621"),
    "test_helpers_alias_is_reachable_poetry": ("tests.infra.unit.deps.test_path_sync_rewrite_poetry", "test_helpers_alias_is_reachable_poetry"),
    "test_helpers_alias_is_reachable_project_obj": ("tests.infra.unit.deps.test_path_sync_main_project_obj", "test_helpers_alias_is_reachable_project_obj"),
    "test_inject_comments_phase_apply_banner": ("tests.infra.unit.deps.test_modernizer_comments", "test_inject_comments_phase_apply_banner"),
    "test_inject_comments_phase_apply_broken_group_section": ("tests.infra.unit.deps.test_modernizer_comments", "test_inject_comments_phase_apply_broken_group_section"),
    "test_inject_comments_phase_apply_markers": ("tests.infra.unit.deps.test_modernizer_comments", "test_inject_comments_phase_apply_markers"),
    "test_inject_comments_phase_apply_with_optional_dependencies_dev": ("tests.infra.unit.deps.test_modernizer_comments", "test_inject_comments_phase_apply_with_optional_dependencies_dev"),
    "test_main_discovery_failure": ("tests.infra.unit.deps.test_path_sync_main_more", "test_main_discovery_failure"),
    "test_main_no_changes_needed": ("tests.infra.unit.deps.test_path_sync_main_more", "test_main_no_changes_needed"),
    "test_main_project_invalid_toml": ("tests.infra.unit.deps.test_path_sync_main_more", "test_main_project_invalid_toml"),
    "test_main_project_no_name": ("tests.infra.unit.deps.test_path_sync_main_more", "test_main_project_no_name"),
    "test_main_project_non_string_name": ("tests.infra.unit.deps.test_path_sync_main_more", "test_main_project_non_string_name"),
    "test_main_project_obj_not_dict_first_loop": ("tests.infra.unit.deps.test_path_sync_main_project_obj", "test_main_project_obj_not_dict_first_loop"),
    "test_main_project_obj_not_dict_second_loop": ("tests.infra.unit.deps.test_path_sync_main_project_obj", "test_main_project_obj_not_dict_second_loop"),
    "test_main_with_changes_and_dry_run": ("tests.infra.unit.deps.test_path_sync_main_more", "test_main_with_changes_and_dry_run"),
    "test_main_with_changes_no_dry_run": ("tests.infra.unit.deps.test_path_sync_main_more", "test_main_with_changes_no_dry_run"),
    "test_rewrite_dep_paths_dry_run": ("tests.infra.unit.deps.test_path_sync_rewrite_deps", "test_rewrite_dep_paths_dry_run"),
    "test_rewrite_dep_paths_read_failure": ("tests.infra.unit.deps.test_path_sync_rewrite_deps", "test_rewrite_dep_paths_read_failure"),
    "test_rewrite_dep_paths_with_internal_names": ("tests.infra.unit.deps.test_path_sync_rewrite_deps", "test_rewrite_dep_paths_with_internal_names"),
    "test_rewrite_dep_paths_with_no_deps": ("tests.infra.unit.deps.test_path_sync_rewrite_deps", "test_rewrite_dep_paths_with_no_deps"),
    "test_rewrite_pep621_invalid_path_dep_regex": ("tests.infra.unit.deps.test_path_sync_rewrite_pep621", "test_rewrite_pep621_invalid_path_dep_regex"),
    "test_rewrite_pep621_no_project_table": ("tests.infra.unit.deps.test_path_sync_rewrite_pep621", "test_rewrite_pep621_no_project_table"),
    "test_rewrite_pep621_non_string_item": ("tests.infra.unit.deps.test_path_sync_rewrite_pep621", "test_rewrite_pep621_non_string_item"),
    "test_rewrite_poetry_no_poetry_table": ("tests.infra.unit.deps.test_path_sync_rewrite_poetry", "test_rewrite_poetry_no_poetry_table"),
    "test_rewrite_poetry_no_tool_table": ("tests.infra.unit.deps.test_path_sync_rewrite_poetry", "test_rewrite_poetry_no_tool_table"),
    "test_rewrite_poetry_with_non_dict_value": ("tests.infra.unit.deps.test_path_sync_rewrite_poetry", "test_rewrite_poetry_with_non_dict_value"),
    "test_run_deptry_wrapper": ("tests.infra.unit.deps.test_detection_wrappers", "test_run_deptry_wrapper"),
    "test_run_mypy_stub_hints_wrapper": ("tests.infra.unit.deps.test_detection_wrappers", "test_run_mypy_stub_hints_wrapper"),
    "test_run_pip_check_wrapper": ("tests.infra.unit.deps.test_detection_wrappers", "test_run_pip_check_wrapper"),
    "test_string_zero_return_value": ("tests.infra.unit.deps.test_main_dispatch", "test_string_zero_return_value"),
    "test_target_path_standalone": ("tests.infra.unit.deps.test_path_sync_helpers", "test_target_path_standalone"),
    "test_target_path_workspace_root": ("tests.infra.unit.deps.test_path_sync_helpers", "test_target_path_workspace_root"),
    "test_target_path_workspace_subproject": ("tests.infra.unit.deps.test_path_sync_helpers", "test_target_path_workspace_subproject"),
    "test_unwrap_item_with_item": ("tests.infra.unit.deps.test_modernizer_helpers", "test_unwrap_item_with_item"),
    "test_unwrap_item_with_none": ("tests.infra.unit.deps.test_modernizer_helpers", "test_unwrap_item_with_none"),
    "test_workspace_root_doc_construction": ("tests.infra.unit.deps.test_modernizer_workspace", "test_workspace_root_doc_construction"),
    "test_workspace_root_fallback": ("tests.infra.unit.deps.test_path_sync_main_more", "test_workspace_root_fallback"),
}

__all__ = [
    "TestArray",
    "TestAsStringList",
    "TestBuildProjectReport",
    "TestCanonicalDevDependencies",
    "TestClassifyIssues",
    "TestCollectInternalDeps",
    "TestCollectInternalDepsEdgeCases",
    "TestConsolidateGroupsPhase",
    "TestConstants",
    "TestDedupeSpecs",
    "TestDepName",
    "TestDetectMode",
    "TestDetectionUncoveredLines",
    "TestDetectorReportFlags",
    "TestDetectorRunFailures",
    "TestDiscoverProjects",
    "TestEnsureCheckout",
    "TestEnsureCheckoutEdgeCases",
    "TestEnsurePyreflyConfigPhase",
    "TestEnsurePyrightConfigPhase",
    "TestEnsurePytestConfigPhase",
    "TestEnsureSymlink",
    "TestEnsureSymlinkEdgeCases",
    "TestEnsureTable",
    "TestExtractDepName",
    "TestExtractRequirementName",
    "TestFlextInfraDependencyDetectionModels",
    "TestFlextInfraDependencyDetectionService",
    "TestFlextInfraDependencyDetectorModels",
    "TestFlextInfraDependencyPathSync",
    "TestFlextInfraDeps",
    "TestFlextInfraExtraPathsManager",
    "TestFlextInfraInternalDependencySyncService",
    "TestFlextInfraPyprojectModernizer",
    "TestFlextInfraRuntimeDevDependencyDetectorInit",
    "TestFlextInfraRuntimeDevDependencyDetectorRunDetect",
    "TestFlextInfraRuntimeDevDependencyDetectorRunReport",
    "TestFlextInfraRuntimeDevDependencyDetectorRunTypings",
    "TestGetDepPaths",
    "TestInferOwnerFromOrigin",
    "TestInjectCommentsPhase",
    "TestIsInternalPathDep",
    "TestIsRelativeTo",
    "TestIsWorkspaceMode",
    "TestLoadDependencyLimits",
    "TestMain",
    "TestMainEdgeCases",
    "TestMainExceptionHandling",
    "TestMainFunction",
    "TestMainHelpAndErrors",
    "TestMainModuleImport",
    "TestMainReturnValues",
    "TestMainStructlogConfiguration",
    "TestMainSubcommandDispatch",
    "TestMainSysArgvModification",
    "TestModernizerEdgeCases",
    "TestModernizerRunAndMain",
    "TestModernizerUncoveredLines",
    "TestModuleAndTypingsFlow",
    "TestModuleLevelWrappers",
    "TestOwnerFromRemoteUrl",
    "TestParseGitmodules",
    "TestParseRepoMap",
    "TestParser",
    "TestPathDepPathsPep621",
    "TestPathDepPathsPoetry",
    "TestPathSyncEdgeCases",
    "TestProjectDevGroups",
    "TestReadDoc",
    "TestResolveRef",
    "TestRewriteDepPaths",
    "TestRewritePep621",
    "TestRewritePoetry",
    "TestRunDeptry",
    "TestRunMypyStubHints",
    "TestRunPipCheck",
    "TestSubcommandMapping",
    "TestSync",
    "TestSyncExtraPaths",
    "TestSyncMethodEdgeCases",
    "TestSyncMethodEdgeCasesMore",
    "TestSyncOne",
    "TestSyncOneEdgeCases",
    "TestSynthesizedRepoMap",
    "TestTargetPath",
    "TestToInfraValue",
    "TestUnwrapItem",
    "TestValidateGitRefEdgeCases",
    "TestWorkspaceRoot",
    "TestWorkspaceRootFromEnv",
    "TestWorkspaceRootFromParents",
    "c",
    "m",
    "s",
    "test_as_string_list_with_item",
    "test_as_string_list_with_item_unwrap_returns_none",
    "test_as_string_list_with_mapping",
    "test_as_string_list_with_string",
    "test_consolidate_groups_phase_apply_removes_old_groups",
    "test_consolidate_groups_phase_apply_with_empty_poetry_group",
    "test_detect_mode_with_nonexistent_path",
    "test_detect_mode_with_path_object",
    "test_discover_projects_wrapper",
    "test_ensure_pyrefly_config_phase_apply_errors",
    "test_ensure_pyrefly_config_phase_apply_ignore_errors",
    "test_ensure_pyrefly_config_phase_apply_python_version",
    "test_ensure_pyrefly_config_phase_apply_search_path",
    "test_ensure_pytest_config_phase_apply_markers",
    "test_ensure_pytest_config_phase_apply_minversion",
    "test_ensure_pytest_config_phase_apply_python_classes",
    "test_ensure_table_with_non_table_value_uncovered",
    "test_extract_requirement_name_invalid",
    "test_extract_requirement_name_simple",
    "test_extract_requirement_name_with_path_dep",
    "test_flext_infra_pyproject_modernizer_find_pyproject_files",
    "test_flext_infra_pyproject_modernizer_process_file_invalid_toml",
    "test_get_current_typings_from_pyproject_wrapper",
    "test_get_required_typings_wrapper",
    "test_helpers_alias_exposed",
    "test_helpers_alias_is_reachable_helpers",
    "test_helpers_alias_is_reachable_main",
    "test_helpers_alias_is_reachable_pep621",
    "test_helpers_alias_is_reachable_poetry",
    "test_helpers_alias_is_reachable_project_obj",
    "test_inject_comments_phase_apply_banner",
    "test_inject_comments_phase_apply_broken_group_section",
    "test_inject_comments_phase_apply_markers",
    "test_inject_comments_phase_apply_with_optional_dependencies_dev",
    "test_main_discovery_failure",
    "test_main_no_changes_needed",
    "test_main_project_invalid_toml",
    "test_main_project_no_name",
    "test_main_project_non_string_name",
    "test_main_project_obj_not_dict_first_loop",
    "test_main_project_obj_not_dict_second_loop",
    "test_main_with_changes_and_dry_run",
    "test_main_with_changes_no_dry_run",
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
    "test_run_deptry_wrapper",
    "test_run_mypy_stub_hints_wrapper",
    "test_run_pip_check_wrapper",
    "test_string_zero_return_value",
    "test_target_path_standalone",
    "test_target_path_workspace_root",
    "test_target_path_workspace_subproject",
    "test_unwrap_item_with_item",
    "test_unwrap_item_with_none",
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
