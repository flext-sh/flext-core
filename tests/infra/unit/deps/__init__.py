# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Tests for flext_infra.deps dependency management modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
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

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "TestArray": ("tests.infra.unit.deps.test_modernizer_helpers", "TestArray"),
    "TestAsStringList": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "TestAsStringList",
    ),
    "TestBuildProjectReport": (
        "tests.infra.unit.deps.test_detection_classify",
        "TestBuildProjectReport",
    ),
    "TestCanonicalDevDependencies": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "TestCanonicalDevDependencies",
    ),
    "TestClassifyIssues": (
        "tests.infra.unit.deps.test_detection_classify",
        "TestClassifyIssues",
    ),
    "TestCollectInternalDeps": (
        "tests.infra.unit.deps.internal_sync",
        "TestCollectInternalDeps",
    ),
    "TestCollectInternalDepsEdgeCases": (
        "tests.infra.unit.deps.internal_sync",
        "TestCollectInternalDepsEdgeCases",
    ),
    "TestConsolidateGroupsPhase": (
        "tests.infra.unit.deps.test_modernizer_consolidate",
        "TestConsolidateGroupsPhase",
    ),
    "TestConstants": (
        "tests.infra.unit.deps.test_extra_paths_manager",
        "TestConstants",
    ),
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
    "TestDiscoverProjects": (
        "tests.infra.unit.deps.test_detection_deptry",
        "TestDiscoverProjects",
    ),
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
    "TestExtractDepName": ("tests.infra.unit.deps.path_sync", "TestExtractDepName"),
    "TestExtractRequirementName": (
        "tests.infra.unit.deps.path_sync",
        "TestExtractRequirementName",
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
    "TestFlextInfraExtraPathsManager": (
        "tests.infra.unit.deps.test_extra_paths_manager",
        "TestFlextInfraExtraPathsManager",
    ),
    "TestFlextInfraInternalDependencySyncService": (
        "tests.infra.unit.deps.test_internal_sync_validation",
        "TestFlextInfraInternalDependencySyncService",
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
    "TestGetDepPaths": (
        "tests.infra.unit.deps.test_extra_paths_manager",
        "TestGetDepPaths",
    ),
    "TestInferOwnerFromOrigin": (
        "tests.infra.unit.deps.test_internal_sync_resolve",
        "TestInferOwnerFromOrigin",
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
    "TestLoadDependencyLimits": (
        "tests.infra.unit.deps.test_detection_typings",
        "TestLoadDependencyLimits",
    ),
    "TestMain": ("tests.infra.unit.deps.test_extra_paths_sync", "TestMain"),
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
    "TestMainModuleImport": ("tests.infra.unit.deps.main", "TestMainModuleImport"),
    "TestMainReturnValues": ("tests.infra.unit.deps.test_main", "TestMainReturnValues"),
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
    "TestModuleAndTypingsFlow": (
        "tests.infra.unit.deps.test_detection_typings",
        "TestModuleAndTypingsFlow",
    ),
    "TestModuleLevelWrappers": (
        "tests.infra.unit.deps.test_detection_wrappers",
        "TestModuleLevelWrappers",
    ),
    "TestOwnerFromRemoteUrl": (
        "tests.infra.unit.deps.test_internal_sync_validation",
        "TestOwnerFromRemoteUrl",
    ),
    "TestParseGitmodules": (
        "tests.infra.unit.deps.internal_sync",
        "TestParseGitmodules",
    ),
    "TestParseRepoMap": ("tests.infra.unit.deps.internal_sync", "TestParseRepoMap"),
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
    "TestProjectDevGroups": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "TestProjectDevGroups",
    ),
    "TestReadDoc": ("tests.infra.unit.deps.test_modernizer_workspace", "TestReadDoc"),
    "TestResolveRef": (
        "tests.infra.unit.deps.test_internal_sync_resolve",
        "TestResolveRef",
    ),
    "TestRewriteDepPaths": ("tests.infra.unit.deps.path_sync", "TestRewriteDepPaths"),
    "TestRewritePep621": ("tests.infra.unit.deps.path_sync", "TestRewritePep621"),
    "TestRewritePoetry": ("tests.infra.unit.deps.path_sync", "TestRewritePoetry"),
    "TestRunDeptry": ("tests.infra.unit.deps.test_detection_deptry", "TestRunDeptry"),
    "TestRunMypyStubHints": (
        "tests.infra.unit.deps.test_detection_typings",
        "TestRunMypyStubHints",
    ),
    "TestRunPipCheck": (
        "tests.infra.unit.deps.test_detection_deptry",
        "TestRunPipCheck",
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
    "TestSynthesizedRepoMap": (
        "tests.infra.unit.deps.test_internal_sync_resolve",
        "TestSynthesizedRepoMap",
    ),
    "TestTargetPath": ("tests.infra.unit.deps.path_sync", "TestTargetPath"),
    "TestToInfraValue": (
        "tests.infra.unit.deps.test_detection_models",
        "TestToInfraValue",
    ),
    "TestUnwrapItem": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "TestUnwrapItem",
    ),
    "TestValidateGitRefEdgeCases": (
        "tests.infra.unit.deps.test_internal_sync_validation",
        "TestValidateGitRefEdgeCases",
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
    "c": ("tests.infra.unit.deps.test_extra_paths_manager", "TestConstants"),
    "m": (
        "tests.infra.unit.deps.test_detection_models",
        "TestFlextInfraDependencyDetectionModels",
    ),
    "s": (
        "tests.infra.unit.deps.test_internal_sync_validation",
        "TestFlextInfraInternalDependencySyncService",
    ),
    "test_as_string_list_with_item_and_edge_values": (
        "tests.infra.unit.deps.test_modernizer_helpers",
        "test_as_string_list_with_item_and_edge_values",
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
    "test_discover_projects_wrapper": (
        "tests.infra.unit.deps.test_detection_wrappers",
        "test_discover_projects_wrapper",
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
    "test_get_current_typings_from_pyproject_wrapper": (
        "tests.infra.unit.deps.test_detection_wrappers",
        "test_get_current_typings_from_pyproject_wrapper",
    ),
    "test_get_required_typings_wrapper": (
        "tests.infra.unit.deps.test_detection_wrappers",
        "test_get_required_typings_wrapper",
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
    "test_main_discovery_failure": (
        "tests.infra.unit.deps.path_sync",
        "test_main_discovery_failure",
    ),
    "test_main_no_changes_needed": (
        "tests.infra.unit.deps.path_sync",
        "test_main_no_changes_needed",
    ),
    "test_main_project_obj_not_dict_first_loop": (
        "tests.infra.unit.deps.path_sync",
        "test_main_project_obj_not_dict_first_loop",
    ),
    "test_main_project_obj_not_dict_second_loop": (
        "tests.infra.unit.deps.path_sync",
        "test_main_project_obj_not_dict_second_loop",
    ),
    "test_main_with_changes_and_dry_run": (
        "tests.infra.unit.deps.path_sync",
        "test_main_with_changes_and_dry_run",
    ),
    "test_main_with_changes_no_dry_run": (
        "tests.infra.unit.deps.path_sync",
        "test_main_with_changes_no_dry_run",
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
    "TestDiscoverProjects",
    "TestEnsureCheckout",
    "TestEnsureCheckoutEdgeCases",
    "TestEnsurePyreflyConfigPhase",
    "TestEnsurePyrightConfigPhase",
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
    "TestFlextInfraRuntimeDevDependencyDetectorInit",
    "TestFlextInfraRuntimeDevDependencyDetectorRunDetect",
    "TestFlextInfraRuntimeDevDependencyDetectorRunReport",
    "TestFlextInfraRuntimeDevDependencyDetectorRunTypings",
    "TestGetDepPaths",
    "TestInferOwnerFromOrigin",
    "TestIsInternalPathDep",
    "TestIsRelativeTo",
    "TestIsWorkspaceMode",
    "TestLoadDependencyLimits",
    "TestMain",
    "TestMainExceptionHandling",
    "TestMainFunction",
    "TestMainHelpAndErrors",
    "TestMainModuleImport",
    "TestMainReturnValues",
    "TestMainStructlogConfiguration",
    "TestMainSubcommandDispatch",
    "TestMainSysArgvModification",
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
    "test_as_string_list_with_item_and_edge_values",
    "test_consolidate_groups_phase_apply_removes_old_groups",
    "test_consolidate_groups_phase_apply_with_empty_poetry_group",
    "test_detect_mode_with_path_object",
    "test_discover_projects_wrapper",
    "test_ensure_pyrefly_config_phase_apply_ignore_errors",
    "test_ensure_pyrefly_config_phase_apply_python_version",
    "test_ensure_pyrefly_config_phase_apply_search_path_and_errors",
    "test_ensure_table_with_non_table_value_uncovered",
    "test_extract_requirement_name_invalid",
    "test_extract_requirement_name_simple",
    "test_extract_requirement_name_with_path_dep",
    "test_get_current_typings_from_pyproject_wrapper",
    "test_get_required_typings_wrapper",
    "test_helpers_alias_available",
    "test_helpers_alias_exposed",
    "test_helpers_alias_is_reachable",
    "test_main_discovery_failure",
    "test_main_no_changes_needed",
    "test_main_project_obj_not_dict_first_loop",
    "test_main_project_obj_not_dict_second_loop",
    "test_main_with_changes_and_dry_run",
    "test_main_with_changes_no_dry_run",
    "test_rewrite_dep_paths_dry_run",
    "test_rewrite_dep_paths_read_failure",
    "test_rewrite_dep_paths_with_internal_names",
    "test_rewrite_pep621_invalid_path_dep_regex",
    "test_rewrite_pep621_no_project_table",
    "test_rewrite_pep621_non_string_item",
    "test_rewrite_poetry_no_poetry_table",
    "test_rewrite_poetry_no_tool_table",
    "test_rewrite_poetry_with_non_dict_value",
    "test_run_deptry_wrapper",
    "test_run_mypy_stub_hints_wrapper",
    "test_run_pip_check_wrapper",
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
