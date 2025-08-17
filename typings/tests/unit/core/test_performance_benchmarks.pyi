from collections.abc import Callable as Callable

import pytest

class TestPerformanceBenchmarks:
    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_flext_result_performance(self, benchmark: object) -> None: ...
    @pytest.mark.performance
    @pytest.mark.benchmark
    def test_container_performance(self, benchmark: object) -> None: ...
    @pytest.mark.performance
    @pytest.mark.slow
    def test_logging_performance_bulk(self) -> None: ...
    @pytest.mark.performance
    @pytest.mark.architecture
    def test_handler_chain_performance(self) -> None: ...

class TestStressTests:
    @pytest.mark.performance
    @pytest.mark.slow
    def test_container_stress_test(self) -> None: ...
    @pytest.mark.performance
    @pytest.mark.slow
    def test_result_chain_stress_test(self) -> None: ...

class TestMemoryPerformance:
    @pytest.mark.performance
    def test_logger_factory_memory_efficiency(self) -> None: ...
    @pytest.mark.performance
    @pytest.mark.architecture
    def test_core_singleton_performance(self) -> None: ...

class TestConcurrencyPerformance:
    @pytest.mark.performance
    def test_result_thread_safety(self) -> None: ...
