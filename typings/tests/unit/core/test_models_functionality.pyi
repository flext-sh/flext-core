from collections.abc import Callable as Callable

import pytest
from _typeshed import Incomplete

from flext_core import FlextEntity, FlextValue

from ...conftest import (
    AssertHelpers as AssertHelpers,
    PerformanceMonitor as PerformanceMonitor,
    TestCase as TestCase,
    TestScenario as TestScenario,
    assert_performance as assert_performance,
)

pytestmark: Incomplete

class TestFlextModelEnumsAdvanced:
    @pytest.fixture
    def enum_test_cases(self) -> list[TestCase[str]]: ...
    @pytest.mark.parametrize_advanced
    def test_enum_values_structured(
        self, enum_test_cases: list[TestCase[str]]
    ) -> None: ...
    def test_enum_completeness(
        self, enum_type: str, enum_values: list[object], context: str
    ) -> None: ...

class TestFlextModelValidationAdvanced:
    @pytest.fixture
    def validation_test_cases(self) -> list[TestCase[dict[str, object]]]: ...
    def test_base_model_with_fixtures(
        self,
        test_builder: object,
        assert_helpers: object,
        validation_test_cases: list[TestCase[dict[str, object]]],
    ) -> None: ...
    @pytest.mark.hypothesis
    def test_model_validation_property_based(self, name: str, value: int) -> None: ...
    @pytest.mark.performance
    def test_model_creation_performance(
        self,
        performance_monitor: PerformanceMonitor,
        performance_threshold: dict[str, float],
    ) -> None: ...

class TestModelInheritanceBehaviorAdvanced:
    @pytest.fixture
    def inheritance_test_cases(self) -> list[TestCase]: ...
    @pytest.mark.parametrize_advanced
    def test_model_inheritance_structured(
        self,
        inheritance_test_cases: list[TestCase],
        test_builder: object,
        snapshot_manager: object,
    ) -> None: ...
    @pytest.mark.hypothesis
    def test_inheritance_properties(self, name: str, value: int) -> None: ...
    @pytest.mark.performance
    def test_inheritance_performance(
        self, performance_monitor: PerformanceMonitor
    ) -> None: ...

class TestFlextEntityAdvanced:
    def test_domain_entity_with_factory_fixture(
        self,
        entity_factory: Callable[[str, dict[str, object]], FlextEntity],
        assert_helpers: AssertHelpers,
    ) -> None: ...
    def test_domain_entity_events_parametrized(
        self,
        entity_id: str,
        initial_events: list[dict[str, object]],
        expected_event_count: int,
        test_builder: object,
    ) -> None: ...
    @pytest.mark.hypothesis
    def test_domain_entity_properties(self, entity_id: str) -> None: ...
    @pytest.mark.performance
    def test_domain_entity_performance(
        self,
        performance_monitor: PerformanceMonitor,
        performance_threshold: dict[str, float],
    ) -> None: ...
    @pytest.mark.snapshot
    def test_domain_entity_snapshot(self, snapshot_manager: object) -> None: ...

class TestFlextValueAdvanced:
    def test_value_object_with_factory_fixture(
        self,
        value_object_factory: Callable[[dict[str, object]], FlextValue],
        assert_helpers: AssertHelpers,
    ) -> None: ...
    def test_value_object_equality_parametrized(
        self,
        metadata: dict[str, object],
        expected_hash_fields: list[str],
        should_equal: list[dict[str, object]],
    ) -> None: ...
    @pytest.mark.hypothesis
    def test_value_object_immutability_properties(
        self, metadata: dict[str, object]
    ) -> None: ...
    @pytest.mark.performance
    def test_value_object_performance(
        self, performance_monitor: PerformanceMonitor
    ) -> None: ...

class TestModelFactoryFunctionsAdvanced:
    @pytest.fixture
    def factory_test_cases(self) -> list[TestCase]: ...
    @pytest.mark.parametrize_advanced
    def test_factory_functions_structured(
        self, factory_test_cases: list[TestCase]
    ) -> None: ...
    @pytest.mark.performance
    def test_factory_performance(
        self, performance_monitor: PerformanceMonitor
    ) -> None: ...

class TestDatabaseOracleModelsAdvanced:
    def test_database_model_with_fixtures(self, test_builder: object) -> None: ...
    def test_oracle_model_parametrized(
        self,
        oracle_config: dict[str, object],
        expected_connection: str,
        *,
        should_validate: bool,
        assert_helpers: AssertHelpers,
    ) -> None: ...
    @pytest.mark.performance
    def test_model_creation_performance(
        self, performance_monitor: PerformanceMonitor
    ) -> None: ...

class TestUtilityFunctionsAdvanced:
    def test_model_to_dict_safe_parametrized(
        self,
        input_data: object,
        expected_result: dict[str, object],
        test_description: str,
    ) -> None: ...
    @pytest.mark.hypothesis
    def test_validate_all_models_properties(self, models_count: int) -> None: ...
    @pytest.mark.performance
    def test_utility_functions_performance(
        self, performance_monitor: PerformanceMonitor
    ) -> None: ...
    @pytest.mark.snapshot
    def test_model_structure_snapshot(self, snapshot_manager: object) -> None: ...

class TestModelsIntegrationAdvanced:
    def test_complete_model_workflow(
        self,
        test_builder: object,
        entity_factory: Callable[[str, dict[str, object]], FlextEntity],
        value_object_factory: Callable[[dict[str, object]], FlextValue],
        performance_monitor: PerformanceMonitor,
    ) -> None: ...
    @pytest.mark.boundary
    def test_model_edge_cases_integration(
        self, assert_helpers: AssertHelpers
    ) -> None: ...
    @pytest.mark.async_integration
    async def test_async_model_operations(self, async_client: object) -> None: ...
