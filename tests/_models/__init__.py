# AUTO-GENERATED FILE — Regenerate with: make gen
"""Models package."""

from __future__ import annotations

from flext_core.lazy import (
    build_lazy_import_map,
    install_lazy_exports,
    merge_lazy_imports,
)

_LAZY_IMPORTS = merge_lazy_imports(
    ("._mixins",),
    build_lazy_import_map(
        {
            "._mixins": ("_mixins",),
            "._mixins.container": ("TestsFlextModelsContainerMixin",),
            "._mixins.core": ("TestsFlextModelsCoreMixin",),
            "._mixins.core_errors": ("TestsFlextModelsCoreErrorsMixin",),
            "._mixins.core_public": ("TestsFlextModelsCorePublicMixin",),
            "._mixins.core_state": ("TestsFlextModelsCoreStateMixin",),
            "._mixins.domain": ("TestsFlextModelsDomainMixin",),
            "._mixins.fixture_payloads": ("TestsFlextModelsFixturePayloadsMixin",),
            "._mixins.fixture_suite": ("TestsFlextModelsFixtureSuiteMixin",),
            "._mixins.fixtures": ("TestsFlextModelsFixtureDictsMixin",),
            "._mixins.guards_mapper": ("TestsFlextModelsGuardsMapperMixin",),
            "._mixins.service_case_core": ("TestsFlextModelsServiceCaseCoreMixin",),
            "._mixins.service_case_reliability": (
                "TestsFlextModelsServiceCaseReliabilityMixin",
            ),
            "._mixins.service_case_validation": (
                "TestsFlextModelsServiceCaseValidationMixin",
            ),
            "._mixins.service_cases": ("TestsFlextModelsServiceCasesMixin",),
            "._mixins.test_data": ("TestsFlextModelsTestDataMixin",),
            "._mixins.test_data_identity": ("TestsFlextModelsTestDataIdentityMixin",),
            "._mixins.test_data_values": ("TestsFlextModelsTestDataValuesMixin",),
            ".mixins": ("TestsFlextModelsMixins",),
            "flext_tests": (
                "c",
                "d",
                "e",
                "h",
                "m",
                "p",
                "r",
                "s",
                "t",
                "td",
                "tf",
                "tk",
                "tm",
                "tv",
                "u",
                "x",
            ),
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
        "pytest_addoption",
        "pytest_collect_file",
        "pytest_collection_modifyitems",
        "pytest_configure",
        "pytest_runtest_setup",
        "pytest_runtest_teardown",
        "pytest_sessionfinish",
        "pytest_sessionstart",
        "pytest_terminal_summary",
        "pytest_warning_recorded",
    ),
    module_name=__name__,
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
