# AUTO-GENERATED FILE — Regenerate with: make gen
"""Models package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import (
    build_lazy_import_map,
    install_lazy_exports,
    merge_lazy_imports,
)

if TYPE_CHECKING:
    from flext_tests import (
        c as c,
        d as d,
        e as e,
        h as h,
        m as m,
        p as p,
        r as r,
        s as s,
        t as t,
        td as td,
        tf as tf,
        tk as tk,
        tm as tm,
        tv as tv,
        u as u,
        x as x,
    )

    from tests._models._mixins.container import (
        TestsFlextModelsContainerMixin as TestsFlextModelsContainerMixin,
    )
    from tests._models._mixins.core import (
        TestsFlextModelsCoreMixin as TestsFlextModelsCoreMixin,
    )
    from tests._models._mixins.core_errors import (
        TestsFlextModelsCoreErrorsMixin as TestsFlextModelsCoreErrorsMixin,
    )
    from tests._models._mixins.core_public import (
        TestsFlextModelsCorePublicMixin as TestsFlextModelsCorePublicMixin,
    )
    from tests._models._mixins.core_state import (
        TestsFlextModelsCoreStateMixin as TestsFlextModelsCoreStateMixin,
    )
    from tests._models._mixins.domain import (
        TestsFlextModelsDomainMixin as TestsFlextModelsDomainMixin,
    )
    from tests._models._mixins.fixture_payloads import (
        TestsFlextModelsFixturePayloadsMixin as TestsFlextModelsFixturePayloadsMixin,
    )
    from tests._models._mixins.fixture_suite import (
        TestsFlextModelsFixtureSuiteMixin as TestsFlextModelsFixtureSuiteMixin,
    )
    from tests._models._mixins.fixtures import (
        TestsFlextModelsFixtureDictsMixin as TestsFlextModelsFixtureDictsMixin,
    )
    from tests._models._mixins.guards_mapper import (
        TestsFlextModelsGuardsMapperMixin as TestsFlextModelsGuardsMapperMixin,
    )
    from tests._models._mixins.service_case_core import (
        TestsFlextModelsServiceCaseCoreMixin as TestsFlextModelsServiceCaseCoreMixin,
    )
    from tests._models._mixins.service_case_reliability import (
        TestsFlextModelsServiceCaseReliabilityMixin as TestsFlextModelsServiceCaseReliabilityMixin,
    )
    from tests._models._mixins.service_case_validation import (
        TestsFlextModelsServiceCaseValidationMixin as TestsFlextModelsServiceCaseValidationMixin,
    )
    from tests._models._mixins.service_cases import (
        TestsFlextModelsServiceCasesMixin as TestsFlextModelsServiceCasesMixin,
    )
    from tests._models._mixins.test_data import (
        TestsFlextModelsTestDataMixin as TestsFlextModelsTestDataMixin,
    )
    from tests._models._mixins.test_data_identity import (
        TestsFlextModelsTestDataIdentityMixin as TestsFlextModelsTestDataIdentityMixin,
    )
    from tests._models._mixins.test_data_values import (
        TestsFlextModelsTestDataValuesMixin as TestsFlextModelsTestDataValuesMixin,
    )
    from tests._models.mixins import TestsFlextModelsMixins as TestsFlextModelsMixins
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
