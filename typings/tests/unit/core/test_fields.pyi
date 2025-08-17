from collections.abc import Callable as Callable

import pytest
from _typeshed import Incomplete

from flext_core import FlextFieldType

from ...conftest import (
    TestCase as TestCase,
    TestScenario as TestScenario,
    assert_performance as assert_performance,
)

pytestmark: Incomplete

class TestFlextFieldCoreAdvanced:
    @pytest.fixture
    def field_creation_test_cases(self) -> list[TestCase[dict[str, object]]]: ...
    @pytest.mark.parametrize_advanced
    def test_field_creation_scenarios(
        self,
        field_creation_test_cases: list[TestCase[dict[str, object]]],
        assert_helpers: object,
    ) -> None: ...
    @pytest.fixture
    def validation_test_cases(self) -> list[TestCase[dict[str, object]]]: ...
    @pytest.mark.parametrize_advanced
    def test_field_validation_scenarios(
        self,
        validation_test_cases: list[TestCase[dict[str, object]]],
        assert_helpers: object,
    ) -> None: ...
    def test_field_type_validation_matrix(
        self,
        field_type: FlextFieldType,
        test_values: list[object],
        expected_valid: list[bool],
        assert_helpers: object,
    ) -> None: ...

class TestFlextFieldCorePropertyBased:
    @pytest.mark.hypothesis
    def test_string_field_length_properties(
        self, field_name: str, min_length: int, max_length: int
    ) -> None: ...
    @pytest.mark.hypothesis
    def test_integer_field_range_properties(
        self, min_value: int, max_value: int, test_value: int
    ) -> None: ...
    @pytest.mark.hypothesis
    def test_allowed_values_property(
        self, allowed_values: list[str], test_value: str
    ) -> None: ...

class TestFlextFieldCorePerformance:
    @pytest.mark.benchmark
    def test_field_creation_performance(
        self,
        performance_monitor: Callable[[Callable[[], object]], dict[str, object]],
        performance_threshold: dict[str, float],
    ) -> None: ...
    @pytest.mark.benchmark
    def test_validation_performance(
        self, performance_monitor: Callable[[Callable[[], object]], dict[str, object]]
    ) -> None: ...
    @pytest.mark.benchmark
    def test_registry_performance(
        self, performance_monitor: Callable[[Callable[[], object]], dict[str, object]]
    ) -> None: ...

class TestFlextFieldCoreWithFixtures:
    def test_fields_with_test_builder(
        self, test_builder: Callable[[], object], assert_helpers: object
    ) -> None: ...
    def test_fields_with_sample_data(
        self,
        sample_data: dict[str, object],
        validators: dict[str, Callable[[str], bool]],
    ) -> None: ...
    def test_fields_with_mock_factory(
        self, mock_factory: Callable[[str], object]
    ) -> None: ...

class TestFlextFieldCoreSnapshot:
    @pytest.mark.snapshot
    def test_comprehensive_field_snapshot(
        self, snapshot_manager: Callable[[str, object], None]
    ) -> None: ...
    @pytest.mark.snapshot
    def test_field_registry_snapshot(
        self, snapshot_manager: Callable[[str, object], None]
    ) -> None: ...

class TestFlextFieldCoreIntegration:
    def test_complete_field_workflow_integration(
        self,
        test_builder: Callable[[], object],
        assert_helpers: object,
        performance_monitor: Callable[[Callable[[], object]], dict[str, object]],
    ) -> None: ...
    @pytest.mark.integration
    def test_factory_methods_integration(self, assert_helpers: object) -> None: ...

class TestFlextFieldCoreEdgeCases:
    @pytest.mark.boundary
    def test_field_type_edge_values(
        self,
        field_type: FlextFieldType,
        edge_values: list[object],
        descriptions: list[str],
    ) -> None: ...
    @pytest.mark.boundary
    def test_field_constraints_boundary_conditions(self) -> None: ...

class TestFlextFieldCoreBackwardCompatibility:
    def test_legacy_function_compatibility(
        self,
        legacy_function: Callable[[], object],
        modern_method: Callable[[], object],
        args: dict[str, object],
    ) -> None: ...
    def test_field_metadata_backward_compatibility(self) -> None: ...
