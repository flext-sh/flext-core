"""Lazy export map part 03."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map

TESTS_FLEXT_CORE_UNIT_LAZY_IMPORTS_PART_03 = build_lazy_import_map({
    ".test_result": ("test_result",),
    ".test_result_callables_fold": ("TestsFlextResultCallablesFold",),
    ".test_result_chain_helpers": ("TestsFlextResultChainHelpers",),
    ".test_result_exception_failures": ("TestsFlextResultExceptionFailures",),
    ".test_result_exception_mapping": ("TestsFlextResultExceptionMapping",),
    ".test_result_exception_safe_callable": ("TestsFlextResultExceptionSafeCallable",),
    ".test_result_exception_traverse_validation": (
        "TestsFlextResultExceptionTraverseValidation",
    ),
    ".test_result_laws": ("TestsFlextResultLaws",),
    ".test_result_operations": ("TestsFlextResultOperations",),
    ".test_result_recent_behaviors": ("TestsFlextResultRecentBehaviors",),
    ".test_result_transforms": ("TestsFlextResultTransforms",),
    ".test_result_traverse_resource": ("TestsFlextResultTraverseResource",),
    ".test_runtime": ("TestsFlextRuntime",),
    ".test_service": ("TestsFlextService",),
    ".test_service_bootstrap": ("TestsFlextServiceBootstrap",),
    ".test_settings": ("TestsFlextSettings",),
    ".test_typings_aliases": ("TestsFlextTypesAliases",),
    ".test_typings_containers": ("TestsFlextTypesContainers",),
    ".test_typings_new": ("test_typings_new",),
    ".test_typings_validation_numbers": ("TestsFlextTypesValidationNumbers",),
    ".test_typings_validation_scalars": ("TestsFlextTypesValidationScalars",),
    ".test_utilities": ("TestsFlextUtilitiesSmoke",),
    ".test_utilities_collection_coverage_100": ("TestsFlextCoreUtilitiesCollection",),
    ".test_utilities_coverage": ("TestsFlextCoreUtilitiesCoverage",),
    ".test_utilities_domain": ("TestsFlextUtilitiesDomain",),
    ".test_utilities_generators_full_coverage": ("TestsFlextCoreUtilitiesGenerators",),
    ".test_utilities_project_metadata": ("test_utilities_project_metadata",),
    ".test_utilities_project_metadata_config": (
        "TestsFlextUtilitiesProjectMetadataConfig",
    ),
    ".test_utilities_project_metadata_read": (
        "TestsFlextUtilitiesProjectMetadataRead",
    ),
    ".test_utilities_pydantic_coverage_100": ("TestsFlextUtilitiesPydantic",),
    ".test_utilities_reliability": ("TestsFlextUtilitiesReliability",),
    ".test_utilities_runtime_violation_registry_coverage_100": (
        "TestsFlextCoreUtilitiesRuntimeViolationRegistry",
    ),
    ".test_utilities_settings_coverage_100": ("TestsFlextCoreUtilitiesSettings",),
    ".test_utilities_text_full_coverage": ("TestsFlextUtilitiesText",),
    ".test_utilities_type_guards_coverage_100": ("TestsFlextCoreUtilitiesTypeGuards",),
    ".test_version": ("TestsFlextVersion",),
})

__all__: list[str] = ["TESTS_FLEXT_CORE_UNIT_LAZY_IMPORTS_PART_03"]
