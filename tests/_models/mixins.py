"""Mixins module.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from tests._models._mixins.container import TestsFlextModelsContainerMixin
from tests._models._mixins.core import TestsFlextModelsCoreMixin
from tests._models._mixins.domain import TestsFlextModelsDomainMixin
from tests._models._mixins.fixtures import TestsFlextModelsFixtureDictsMixin
from tests._models._mixins.guards_mapper import TestsFlextModelsGuardsMapperMixin
from tests._models._mixins.service_cases import TestsFlextModelsServiceCasesMixin
from tests._models._mixins.test_data import TestsFlextModelsTestDataMixin

if TYPE_CHECKING:
    from tests.typings import t


class TestsFlextFlextModelsMixins:
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

    # Populate ContainerScenarios after class is fully defined to allow forward references
    _svc_scenarios: ClassVar[t.SequenceOf[TestsFlextModelsMixins.ServiceScenario]] = [
        TestsFlextModelsMixins.ServiceScenario(
            name="test_service",
            service="test_service_value",
            description="Simple string service",
        ),
        TestsFlextModelsMixins.ServiceScenario(
            name="service_instance",
            service=42,
            description="Integer service instance",
        ),
        TestsFlextModelsMixins.ServiceScenario(
            name="string_service",
            service="test_value",
            description="String service",
        ),
    ]

    _typed_scenarios: ClassVar[
        t.SequenceOf[TestsFlextModelsMixins.TypedRetrievalScenario]
    ] = [
        TestsFlextModelsMixins.TypedRetrievalScenario(
            name="dict_service",
            service="test_dict_service",
            expected_type=str,
            should_pass=True,
            description="String service",
        ),
        TestsFlextModelsMixins.TypedRetrievalScenario(
            name="string_service",
            service="test_string",
            expected_type=str,
            should_pass=True,
            description="String service",
        ),
        TestsFlextModelsMixins.TypedRetrievalScenario(
            name="list_service",
            service=123,
            expected_type=int,
            should_pass=True,
            description="Integer service for typed retrieval",
        ),
    ]


TestsFlextFlextModelsMixins.TestsFlextModelsMixins.ContainerScenarios.SERVICE_SCENARIOS = TestsFlextFlextModelsMixins._svc_scenarios
TestsFlextFlextModelsMixins.TestsFlextModelsMixins.ContainerScenarios.TYPED_RETRIEVAL_SCENARIOS = TestsFlextFlextModelsMixins._typed_scenarios

m = TestsFlextFlextModelsMixins.TestsFlextModelsMixins

__all__: list[str] = ["TestsFlextFlextModelsMixins", "m"]
