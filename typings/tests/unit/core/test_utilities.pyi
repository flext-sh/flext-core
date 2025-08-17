from collections.abc import Callable
from unittest.mock import Mock

import pytest
from _typeshed import Incomplete

from ...conftest import (
    AssertHelpers as AssertHelpers,
    PerformanceMetrics as PerformanceMetrics,
    TestCase as TestCase,
    TestDataBuilder as TestDataBuilder,
    TestScenario as TestScenario,
)

pytestmark: Incomplete

class TestFlextUtilitiesParametrized:
    @pytest.fixture
    def constant_test_cases(self) -> list[TestCase[dict[str, object], int]]: ...
    @pytest.mark.parametrize_advanced
    def test_utility_constants(
        self, constant_test_cases: list[TestCase[dict[str, object], int]]
    ) -> None: ...
    def test_generator_methods_structure(
        self, generator_method: str, prefix: str, min_length: int
    ) -> None: ...
    def test_truncate_parametrized(
        self, text: str, max_length: int, expected_length: int, should_end_with: str
    ) -> None: ...

class TestFlextUtilitiesPropertyBased:
    test_truncate_properties: Incomplete
    test_format_duration_properties: Incomplete
    test_type_guard_properties: Incomplete

class TestFlextUtilitiesPerformance:
    @pytest.mark.benchmark
    def test_id_generation_performance(
        self, performance_monitor: Callable[[Callable[[], object]], PerformanceMetrics]
    ) -> None: ...
    @pytest.mark.benchmark
    def test_truncate_performance(
        self, performance_monitor: Callable[[Callable[[], object]], PerformanceMetrics]
    ) -> None: ...
    @pytest.mark.benchmark
    def test_performance_tracking_overhead(self) -> None: ...

class TestFlextUtilitiesWithFixtures:
    def test_utilities_with_mock_factory(self, mock_factory: Mock) -> None: ...
    def test_utilities_with_test_builder(
        self, test_builder: type[TestDataBuilder[object]]
    ) -> None: ...
    def test_utilities_with_sample_data(
        self,
        sample_data: dict[str, object],
        validators: dict[str, Callable[[object], bool]],
    ) -> None: ...
    def test_utilities_with_error_context(
        self, error_context: dict[str, str]
    ) -> None: ...

class TestFlextUtilitiesErrorHandling:
    @pytest.fixture
    def error_test_cases(self) -> list[TestCase[dict[str, object], None]]: ...
    @pytest.mark.error_path
    @pytest.mark.parametrize_advanced
    def test_cli_error_handling(
        self, error_test_cases: list[TestCase[dict[str, object], None]]
    ) -> None: ...
    @pytest.mark.happy_path
    def test_cli_success_handling(self) -> None: ...

class TestFlextUtilitiesSnapshot:
    @pytest.mark.snapshot
    def test_performance_metrics_snapshot(
        self, snapshot_manager: Callable[[str, object], None]
    ) -> None: ...
    @pytest.mark.snapshot
    def test_generator_output_snapshot(
        self, snapshot_manager: Callable[[str, object], None]
    ) -> None: ...

class TestFlextUtilitiesIntegration:
    def test_complete_workflow_integration(
        self,
        test_builder: type[TestDataBuilder[object]],
        assert_helpers: AssertHelpers,
        performance_monitor: Callable[[Callable[[], object]], PerformanceMetrics],
    ) -> None: ...
    @pytest.mark.integration
    def test_performance_tracking_integration(self) -> None: ...

class TestFlextUtilitiesEdgeCases:
    @pytest.mark.boundary
    def test_truncate_edge_cases(
        self, text: str, max_length: int, suffix: str, description: str
    ) -> None: ...
    @pytest.mark.boundary
    def test_duration_formatting_edge_cases(self) -> None: ...
    @pytest.mark.boundary
    def test_type_guards_edge_cases(self) -> None: ...

class TestFlextUtilitiesBackwardCompatibility:
    type GenFunc = Callable[[], str]
    type TruncFunc = Callable[[str, int], str]
    type PredOld = Callable[[object], bool]
    type PredNew = Callable[[object | None], bool]
    type BackwardCase = (
        tuple[GenFunc, GenFunc, tuple[()]]
        | tuple[TruncFunc, TruncFunc, tuple[str, int]]
        | tuple[PredOld, PredNew, tuple[object]]
    )
    def test_backward_compatibility_equivalence(self) -> None: ...
