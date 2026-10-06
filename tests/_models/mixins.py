"""Mixins module.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from tests._models._mixins.container import TestsFlextModelsContainerMixin
from tests._models._mixins.core import TestsFlextModelsCoreMixin
from tests._models._mixins.domain import TestsFlextModelsDomainMixin
from tests._models._mixins.fixtures import TestsFlextModelsFixtureDictsMixin
from tests._models._mixins.guards_mapper import TestsFlextModelsGuardsMapperMixin
from tests._models._mixins.service_cases import TestsFlextModelsServiceCasesMixin
from tests._models._mixins.test_data import TestsFlextModelsTestDataMixin


class TestsFlextModelsNamespace:
    """Canonical namespace owner."""

    class TestsFlextModelsMixins:
        """Canonical namespace owner."""

        class TestsFlextModelsMixins(
            TestsFlextModelsContainerMixin,
            TestsFlextModelsCoreMixin,
            TestsFlextModelsDomainMixin,
            TestsFlextModelsFixtureDictsMixin,
            TestsFlextModelsGuardsMapperMixin,
            TestsFlextModelsServiceCasesMixin,
            TestsFlextModelsTestDataMixin,
        ):
            """flext-core test models namespace."""

    @staticmethod
    def _populate_container_scenarios() -> None:
        """Attach the scenario tables to their canonical containers.

        The scenario values need the fully defined namespace classes, so the
        wiring runs once at import instead of patching class attributes at
        module scope.
        """
        mixin = (
            TestsFlextModelsNamespace.TestsFlextModelsMixins
        )
        mixin.ContainerScenarios.SERVICE_SCENARIOS = [
            mixin.ServiceScenario(
                name="test_service",
                service="test_service_value",
                description="Simple string service",
            ),
            mixin.ServiceScenario(
                name="service_instance",
                service=42,
                description="Integer service instance",
            ),
            mixin.ServiceScenario(
                name="string_service",
                service="test_value",
                description="String service",
            ),
        ]
        mixin.ContainerScenarios.TYPED_RETRIEVAL_SCENARIOS = [
            mixin.TypedRetrievalScenario(
                name="dict_service",
                service="test_dict_service",
                expected_type=str,
                should_pass=True,
                description="String service",
            ),
            mixin.TypedRetrievalScenario(
                name="string_service",
                service="test_string",
                expected_type=str,
                should_pass=True,
                description="String service",
            ),
            mixin.TypedRetrievalScenario(
                name="list_service",
                service=123,
                expected_type=int,
                should_pass=True,
                description="Integer service for typed retrieval",
            ),
        ]

    m = TestsFlextModelsNamespace.TestsFlextModelsMixins

    # Flat name restored (the consumers and the tests lazy map resolve it):
    # commit 0958021717 had accidentally double-prefixed the namespace owner.
    TestsFlextModelsMixins = TestsFlextModelsNamespace.TestsFlextModelsMixins


_populate_container_scenarios()

m = TestsFlextModelsNamespace.TestsFlextModelsMixins

# Flat name restored (the consumers and the tests lazy map resolve it):
# commit 0958021717 had accidentally double-prefixed the namespace owner.
TestsFlextModelsMixins = TestsFlextModelsNamespace.TestsFlextModelsMixins

__all__: list[str] = ["TestsFlextModelsNamespace", "TestsFlextModelsMixins", "m"]
