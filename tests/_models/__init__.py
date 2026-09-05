# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests. Models package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import _mixins as _mixins
    from flext_tests import c, d, e, h, m, p, r, s, t, td, tf, tk, tm, tv, u, x

    from ._mixins.container import TestsFlextModelsContainerMixin
    from ._mixins.core import TestsFlextModelsCoreMixin
    from ._mixins.core_errors import TestsFlextModelsCoreErrorsMixin
    from ._mixins.core_public import TestsFlextModelsCorePublicMixin
    from ._mixins.core_state import TestsFlextModelsCoreStateMixin
    from ._mixins.domain import TestsFlextModelsDomainMixin
    from ._mixins.fixture_payloads import TestsFlextModelsFixturePayloadsMixin
    from ._mixins.fixture_suite import TestsFlextModelsFixtureSuiteMixin
    from ._mixins.fixtures import TestsFlextModelsFixtureDictsMixin
    from ._mixins.guards_mapper import TestsFlextModelsGuardsMapperMixin
    from ._mixins.service_case_core import TestsFlextModelsServiceCaseCoreMixin
    from ._mixins.service_case_reliability import (
        TestsFlextModelsServiceCaseReliabilityMixin,
    )
    from ._mixins.service_case_validation import (
        TestsFlextModelsServiceCaseValidationMixin,
    )
    from ._mixins.service_cases import TestsFlextModelsServiceCasesMixin
    from .mixins import TestsFlextModelsMixins
__all__: tuple[str, ...] = (
    "TestsFlextModelsContainerMixin",
    "TestsFlextModelsCoreErrorsMixin",
    "TestsFlextModelsCoreMixin",
    "TestsFlextModelsCorePublicMixin",
    "TestsFlextModelsCoreStateMixin",
    "TestsFlextModelsDomainMixin",
    "TestsFlextModelsFixtureDictsMixin",
    "TestsFlextModelsFixturePayloadsMixin",
    "TestsFlextModelsFixtureSuiteMixin",
    "TestsFlextModelsGuardsMapperMixin",
    "TestsFlextModelsMixins",
    "TestsFlextModelsServiceCaseCoreMixin",
    "TestsFlextModelsServiceCaseReliabilityMixin",
    "TestsFlextModelsServiceCaseValidationMixin",
    "TestsFlextModelsServiceCasesMixin",
    "_mixins",
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
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
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
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
