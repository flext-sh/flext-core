"""Lazy export map part 05."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map

TESTS_FLEXT_CORE_LAZY_IMPORTS_PART_05 = build_lazy_import_map({
    ".unit.test_result_operations": ("TestsFlextResultOperations",),
    ".unit.test_result_recent_behaviors": ("TestsFlextResultRecentBehaviors",),
    ".unit.test_result_transforms": ("TestsFlextResultTransforms",),
    ".unit.test_result_traverse_resource": ("TestsFlextResultTraverseResource",),
    ".unit.test_runtime": ("TestsFlextRuntime",),
    ".unit.test_service": ("TestsFlextService",),
    ".unit.test_service_bootstrap": ("TestsFlextServiceBootstrap",),
    ".unit.test_settings": ("TestsFlextSettings",),
    ".unit.test_typings_aliases": ("TestsFlextTypesAliases",),
    ".unit.test_typings_containers": ("TestsFlextTypesContainers",),
    ".unit.test_typings_validation_numbers": ("TestsFlextTypesValidationNumbers",),
    ".unit.test_typings_validation_scalars": ("TestsFlextTypesValidationScalars",),
    ".unit.test_utilities": ("TestsFlextUtilitiesSmoke",),
    ".unit.test_utilities_collection_coverage_100": ("TestsFlextUtilitiesCollection",),
    ".unit.test_utilities_coverage": ("TestsFlextUtilitiesCoverage",),
    ".unit.test_utilities_domain": ("TestsFlextUtilitiesDomain",),
    ".unit.test_utilities_generators_full_coverage": ("TestsFlextUtilitiesGenerators",),
    ".unit.test_utilities_project_metadata_config": (
        "TestsFlextUtilitiesProjectMetadataConfig",
    ),
    ".unit.test_utilities_project_metadata_read": (
        "TestsFlextUtilitiesProjectMetadataRead",
    ),
    ".unit.test_utilities_pydantic_coverage_100": ("TestsFlextUtilitiesPydantic",),
    ".unit.test_utilities_reliability": ("TestsFlextUtilitiesReliability",),
    ".unit.test_utilities_runtime_violation_registry_coverage_100": (
        "TestsFlextRuntimeViolationRegistry",
    ),
    ".unit.test_utilities_settings_coverage_100": (
        "TestsFlextUtilitiesSettings",
        "TestsFlextUtilitiesSettingsEnvFile",
        "TestsFlextUtilitiesSettingsRegisterFactory",
    ),
    ".unit.test_utilities_text_full_coverage": ("TestsFlextUtilitiesText",),
    ".unit.test_utilities_type_guards_coverage_100": ("TestsFlextUtilitiesTypeGuards",),
    ".unit.test_version": ("TestsFlextVersion",),
    ".utilities": (
        "TestsFlextUtilities",
        "u",
    ),
    "flext_tests": (
        "d",
        "e",
        "h",
        "r",
        "td",
        "tf",
        "tk",
        "tm",
        "tv",
        "x",
    ),
})

__all__: list[str] = ["TESTS_FLEXT_CORE_LAZY_IMPORTS_PART_05"]
