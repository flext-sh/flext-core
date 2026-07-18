from __future__ import annotations


from tests._models._mixins.container import TestsFlextModelsContainerMixin
from tests._models._mixins.core import TestsFlextModelsCoreMixin
from tests._models._mixins.domain import TestsFlextModelsDomainMixin
from tests._models._mixins.fixtures import TestsFlextModelsFixtureDictsMixin
from tests._models._mixins.guards_mapper import TestsFlextModelsGuardsMapperMixin
from tests._models._mixins.service_cases import TestsFlextModelsServiceCasesMixin
from tests._models._mixins.test_data import TestsFlextModelsTestDataMixin

from tests import t


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
_svc_scenarios: t.SequenceOf[TestsFlextModelsMixins.ServiceScenario] = [
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
TestsFlextModelsMixins.ContainerScenarios.SERVICE_SCENARIOS = _svc_scenarios
_typed_scenarios: t.SequenceOf[TestsFlextModelsMixins.TypedRetrievalScenario] = [
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
TestsFlextModelsMixins.ContainerScenarios.TYPED_RETRIEVAL_SCENARIOS = _typed_scenarios

__all__: list[str] = ["TestsFlextModelsMixins"]
