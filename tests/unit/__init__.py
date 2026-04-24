# AUTO-GENERATED FILE — Regenerate with: make gen
"""Unit package."""

from __future__ import annotations

from flext_core.lazy import (
    build_lazy_import_map,
    install_lazy_exports,
    merge_lazy_imports,
)

_LAZY_IMPORTS = merge_lazy_imports(
    (
        "._models",
        "._utilities",
    ),
    build_lazy_import_map(
        {
            "._models.test_base": ("TestModelsBase",),
            "._models.test_cqrs": ("TestModelsCQRS",),
            "._models.test_entity": ("TestModelsEntity",),
            "._models.test_exception_params": ("TestFlextModelsExceptionParams",),
            "._utilities.test_guards": ("TestUtilitiesGuards",),
            "._utilities.test_mapper": ("TestUtilitiesMapper",),
            ".base": ("TestsFlextCoreServiceBase",),
            ".test_beartype_engine": (
                "TestAliasContainsAny",
                "TestBeartypeClawCompatibility",
                "TestBeartypeConf",
                "TestContainsAny",
                "TestCountUnionMembers",
                "TestFacadeAccessibility",
                "TestForbiddenCollectionOrigin",
                "TestMatchesStrNoneUnion",
            ),
            ".test_constants_new": ("TestFlextConstants",),
            ".test_constants_project_metadata": (
                "TestsFlextCoreConstantsProjectMetadata",
            ),
            ".test_container": ("TestsFlextCoreContainer",),
            ".test_context": ("TestsFlextCoreContext",),
            ".test_coverage_exceptions": ("TestCoverageExceptions",),
            ".test_coverage_loggings": ("TestCoverageLoggings",),
            ".test_decorators": ("TestFlextDecorators",),
            ".test_decorators_discovery_full_coverage": (
                "TestsFlextCoreDecoratorsDiscovery",
            ),
            ".test_decorators_full_coverage": ("TestDecoratorsFullCoverage",),
            ".test_deprecation_warnings": ("TestDeprecationWarnings",),
            ".test_dispatcher_di": ("TestDispatcherDI",),
            ".test_dispatcher_minimal": ("TestDispatcherMinimal",),
            ".test_dispatcher_reliability": ("TestDispatcherReliability",),
            ".test_enforcement": (
                "TestAccessorMethodBan",
                "TestBaseModelCoverage",
                "TestClassPrefixScope",
                "TestConstantsLayerRules",
                "TestDetailSubstitution",
                "TestEnforcementMode",
                "TestFalsePositiveSkips",
                "TestFieldRules",
                "TestHasNestedNamespaceViaMro",
                "TestModelClassRules",
                "TestNamespaceInheritance",
                "TestProjectDiscovery",
                "TestProjectPrefixOverrides",
                "TestProtocolsLayerRules",
                "TestReportApi",
                "TestSettingsInheritance",
                "TestTypesLayerRules",
                "TestUtilitiesLayerRules",
            ),
            ".test_enforcement_catalog": ("TestsFlextCoreEnforcementCatalog",),
            ".test_enforcement_integration": (
                "TestBadModuleFiresExpectedRules",
                "TestCleanModuleEmitsNothing",
            ),
            ".test_enum_utilities_coverage_100": ("TestEnumUtilitiesCoverage",),
            ".test_exceptions": ("TestExceptions",),
            ".test_handler_decorator_discovery": ("TestHandlerDecoratorDiscovery",),
            ".test_handlers": ("TestsFlextCoreFlextHandlers",),
            ".test_lazy_exports": ("TestsFlextCoreLazy",),
            ".test_loggings_full_coverage": ("TestsFlextCoreLoggings",),
            ".test_mixins": ("TestFlextMixinsNestedClasses",),
            ".test_models": ("TestsFlextCoreModelsUnit",),
            ".test_models_base_full_coverage": (
                "TestsFlextCoreModelsBaseFullCoverage",
            ),
            ".test_models_container": ("TestModelsContainer",),
            ".test_models_cqrs_full_coverage": ("TestModelsCqrsFullCoverage",),
            ".test_models_project_metadata": ("TestModelsProjectMetadata",),
            ".test_project_metadata_facade_access": ("TestFacadeFlatSsotAccess",),
            ".test_protocols_project_metadata": (
                "TestProjectClassStemDeriverProtocol",
                "TestProjectMetadataReaderProtocol",
                "TestProjectTierFacadeNamerProtocol",
            ),
            ".test_registry": ("TestRegistry",),
            ".test_registry_full_coverage": ("TestRegistryFullCoverage",),
            ".test_result": ("Testr",),
            ".test_result_exception_carrying": (
                "TestsFlextCoreResultExceptionCarrying",
            ),
            ".test_runtime": ("TestFlextRuntime",),
            ".test_service": (
                "TestService",
                "TestsFlextCoreServiceUserData",
                "TestsFlextCoreServiceUserService",
            ),
            ".test_service_bootstrap": ("TestsFlextCoreServiceBootstrap",),
            ".test_service_coverage_100": ("TestService100Coverage",),
            ".test_settings": ("TestFlextSettings",),
            ".test_settings_coverage": ("TestFlextSettingsCoverage",),
            ".test_typings_new": ("TestFlextTypes",),
            ".test_utilities": ("TestUtilitiesSmoke",),
            ".test_utilities_collection_coverage_100": (
                "TestUtilitiesCollectionCoverage",
            ),
            ".test_utilities_coverage": ("TestUtilitiesCoverage",),
            ".test_utilities_domain": ("TestUtilitiesDomain",),
            ".test_utilities_generators_full_coverage": (
                "TestUtilitiesGeneratorsFullCoverage",
            ),
            ".test_utilities_project_metadata": (
                "TestsFlextCoreUtilitiesProjectMetadata",
            ),
            ".test_utilities_reliability": ("TestFlextUtilitiesReliability",),
            ".test_utilities_settings_coverage_100": ("TestFlextUtilitiesSettings",),
            ".test_utilities_text_full_coverage": ("TestUtilitiesTextFullCoverage",),
            ".test_utilities_type_guards_coverage_100": (
                "TestUtilitiesTypeGuardsCoverage100",
            ),
            ".test_version": ("TestsFlextCoreVersion",),
        },
    ),
    exclude_names=(
        "cleanup_submodule_namespace",
        "install_lazy_exports",
        "lazy_getattr",
        "logger",
        "merge_lazy_imports",
        "output",
        "output_reporting",
    ),
    module_name=__name__,
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
