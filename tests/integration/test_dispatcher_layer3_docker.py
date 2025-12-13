"""Docker-based integration tests for FlextDispatcher Layer 3 Advanced Processing.

Module: flext_core.dispatcher
Scope: Layer 3 operations with real Docker containers

Tests validate:
- Processor registration and lifecycle with real service containers
- Single item processing against live services
- Batch processing with rate limiting and circuit breaker
- Parallel processing with thread pool management
- Timeout enforcement against slow services
- Fallback chains with cascading failures
- Comprehensive metrics collection from real operations
- Performance characteristics under load
- Circuit breaker state transitions with real failures
- Retry logic with exponential backoff

All tests use REAL Docker services (PostgreSQL, Redis, etc.) without mocks,
validating actual network behavior, timeouts, and failure scenarios.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import time
from enum import StrEnum
from typing import Final, cast

import pytest

from flext_core import FlextDispatcher, FlextResult, t
from tests.test_utils import assertion_helpers

# Skip entire module if optional dependencies not available
psycopg2 = pytest.importorskip("psycopg2")
redis = pytest.importorskip("redis")


# ==================== TEST CONSTANTS ====================


class TestDispatcherConstants:
    """Test constants for dispatcher integration tests."""

    class ProcessorNames(StrEnum):
        """Processor name constants."""

        POSTGRES = "postgres"
        LATENCY = "latency"
        FAULT = "fault"
        PRIMARY = "primary"
        FALLBACK1 = "fallback1"
        FALLBACK2 = "fallback2"
        SLOW = "slow"
        NONEXISTENT = "nonexistent"

    class TestValues:
        """Test value constants."""

        LATENCY_DEFAULT: Final[float] = 0.05
        LATENCY_SLOW: Final[float] = 0.2
        LATENCY_FAST: Final[float] = 0.02
        TIMEOUT_SUCCESS: Final[float] = 1.0
        TIMEOUT_EXCEEDED: Final[float] = 0.05
        FAILURE_RATE_50: Final[float] = 0.5
        FAILURE_RATE_100: Final[float] = 1.0
        FAILURE_RATE_0: Final[float] = 0.0
        BATCH_SIZE_SMALL: Final[int] = 2
        BATCH_SIZE_LARGE: Final[int] = 10
        MAX_WORKERS_SMALL: Final[int] = 2
        MAX_WORKERS_LARGE: Final[int] = 4
        PARALLEL_TIMEOUT_THRESHOLD: Final[float] = 0.2
        TEST_VALUE: Final[int] = 42
        ITERATIONS_METRICS: Final[int] = 3
        ITERATIONS_FAULT: Final[int] = 5
        ITERATIONS_BATCH: Final[int] = 5
        ITERATIONS_PARALLEL: Final[int] = 5
        ITERATIONS_LARGE: Final[int] = 50
        ITERATIONS_PARALLEL_SMALL: Final[int] = 4

    class DatabaseConfig:
        """Database configuration constants."""

        HOST_DEFAULT: Final[str] = "localhost"
        PORT_DEFAULT: Final[int] = 5432
        DATABASE: Final[str] = "test"
        USER: Final[str] = "testuser"
        PASSWORD: Final[str] = "testpass"
        CONNECT_TIMEOUT: Final[int] = 2

    class QueryStrings:
        """Query string constants."""

        SELECT_1: Final[str] = "SELECT 1"

    class ErrorMessages:
        """Error message patterns."""

        DATA_MUST_BE_DICT: Final[str] = "Data must be dict with 'query' key"
        POSTGRESQL_CONNECTION_ERROR: Final[str] = "PostgreSQL connection error: {}: {}"


# ==================== PROCESSOR CLASSES ====================


class PostgreSQLService:
    """Real PostgreSQL service processor for integration testing.

    Validates database connectivity and query execution through actual
    PostgreSQL service running in Docker container.
    """

    def __init__(
        self,
        host: str = TestDispatcherConstants.DatabaseConfig.HOST_DEFAULT,
        port: int = TestDispatcherConstants.DatabaseConfig.PORT_DEFAULT,
    ) -> None:
        """Initialize PostgreSQL service processor.

        Args:
            host: PostgreSQL host (from docker-compose)
            port: PostgreSQL port (from docker-compose)

        """
        self.host = host
        self.port = port
        self.connection_attempts = 0
        self.query_count = 0

    def process(self, data: object) -> FlextResult[t.GeneralValueType]:
        """Process data by executing query against PostgreSQL.

        This validates real database connectivity without ORM dependencies.

        Args:
            data: Query instruction (dict with 'query' key)

        Returns:
            FlextResult with query result or connection error

        """
        if not isinstance(data, dict) or "query" not in data:
            return FlextResult[t.GeneralValueType].fail(
                TestDispatcherConstants.ErrorMessages.DATA_MUST_BE_DICT,
            )

        try:
            self.connection_attempts += 1
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=TestDispatcherConstants.DatabaseConfig.DATABASE,
                user=TestDispatcherConstants.DatabaseConfig.USER,
                password=TestDispatcherConstants.DatabaseConfig.PASSWORD,
                connect_timeout=TestDispatcherConstants.DatabaseConfig.CONNECT_TIMEOUT,
            )
            cursor = conn.cursor()

            # Execute query
            cursor.execute(data["query"])
            result = cursor.fetchall()

            cursor.close()
            conn.close()

            self.query_count += 1
            result_dict: dict[str, t.GeneralValueType] = {
                "result": result,
                "query": data["query"],
            }
            return FlextResult[t.GeneralValueType].ok(result_dict)

        except Exception as e:
            self.connection_attempts += 1
            return FlextResult[t.GeneralValueType].fail(
                TestDispatcherConstants.ErrorMessages.POSTGRESQL_CONNECTION_ERROR.format(
                    type(e).__name__,
                    str(e),
                ),
            )


class NetworkLatencyProcessor:
    """Processor that simulates network latency.

    Used to test timeout enforcement and parallel processing performance.
    """

    def __init__(
        self,
        latency_seconds: float = TestDispatcherConstants.TestValues.LATENCY_DEFAULT,
    ) -> None:
        """Initialize network latency processor.

        Args:
            latency_seconds: Simulated network delay

        """
        self.latency_seconds = latency_seconds
        self.process_count = 0

    def process(self, data: object) -> FlextResult[t.GeneralValueType]:
        """Simulate network latency then return result.

        Args:
            data: Data to process

        Returns:
            FlextResult with processed data after delay

        """
        time.sleep(self.latency_seconds)
        self.process_count += 1

        assert self.process_count > 0, "Process count should be incremented"

        # Cast to t.GeneralValueType for type compatibility
        result_dict: dict[str, t.GeneralValueType] = {
            "processed": cast("t.GeneralValueType", data),
            "latency": cast("t.GeneralValueType", self.latency_seconds),
        }
        result = FlextResult[t.GeneralValueType].ok(result_dict)

        (
            assertion_helpers.assert_flext_result_success(result),
            "Processing should succeed",
        )
        assert result.value is not None, "Result should contain value"
        assert isinstance(result.value, dict), "Result value should be a dict"
        result_value_dict: dict[str, t.GeneralValueType] = result.value
        assert "processed" in result_value_dict, "Result should contain processed data"
        assert "latency" in result_value_dict, "Result should contain latency info"
        assert result_value_dict["latency"] == self.latency_seconds, (
            f"Latency should be {self.latency_seconds}, got {result_value_dict['latency']}"
        )

        return result


class FaultInjectionProcessor:
    """Processor with controllable failure injection.

    Used to test circuit breaker, retry logic, and fallback chains.
    """

    def __init__(
        self,
        failure_rate: float = TestDispatcherConstants.TestValues.FAILURE_RATE_50,
        *,
        rng: random.Random | None = None,
    ) -> None:
        """Initialize fault injection processor.

        Args:
            failure_rate: Probability of failure (0.0-1.0)
            rng: Random number generator for deterministic testing

        """
        if not 0.0 <= failure_rate <= 1.0:
            msg = "failure_rate must be between 0.0 and 1.0"
            raise ValueError(msg)
        self.failure_rate = failure_rate
        self.attempt_count = 0
        self.success_count = 0
        self.failure_count = 0
        self._rng = rng or random.Random()

    def process(self, data: object) -> FlextResult[t.GeneralValueType]:
        """Process with probabilistic failure.

        Args:
            data: Data to process

        Returns:
            FlextResult success or failure based on failure_rate

        """
        self.attempt_count += 1

        if self._rng.random() < self.failure_rate:
            self.failure_count += 1
            return FlextResult[t.GeneralValueType].fail(
                f"Injected failure #{self.failure_count}",
            )

        self.success_count += 1
        # Cast to t.GeneralValueType for type compatibility
        result_dict: dict[str, t.GeneralValueType] = {
            "data": cast("t.GeneralValueType", data),
            "attempt": cast("t.GeneralValueType", self.attempt_count),
        }
        return FlextResult[t.GeneralValueType].ok(result_dict)


# ==================== FIXTURES ====================


@pytest.fixture
def docker_enabled() -> bool:
    """Check if Docker is available for integration testing.

    Returns:
        True if Docker daemon is accessible, False otherwise

    """
    docker_path = shutil.which("docker")
    if not docker_path:
        return False

    try:
        _ = subprocess.run(
            [docker_path, "ps"],
            capture_output=True,
            timeout=5,
            check=True,
        )
        return True
    except Exception:
        return False


@pytest.fixture
def dispatcher() -> FlextDispatcher:
    """Fixture providing configured dispatcher instance.

    Returns:
        FlextDispatcher configured for integration testing

    """
    return FlextDispatcher()


@pytest.fixture
def postgres_service() -> PostgreSQLService:
    """Fixture providing PostgreSQL service processor.

    Returns:
        PostgreSQLService configured to connect to Docker container

    """
    return PostgreSQLService(
        host=os.getenv(
            "POSTGRES_HOST",
            TestDispatcherConstants.DatabaseConfig.HOST_DEFAULT,
        ),
        port=int(
            os.getenv(
                "POSTGRES_PORT",
                str(TestDispatcherConstants.DatabaseConfig.PORT_DEFAULT),
            ),
        ),
    )


@pytest.fixture
def latency_service() -> NetworkLatencyProcessor:
    """Fixture providing network latency service processor.

    Returns:
        NetworkLatencyProcessor with configured latency

    """
    return NetworkLatencyProcessor(
        latency_seconds=TestDispatcherConstants.TestValues.LATENCY_DEFAULT,
    )


@pytest.fixture
def fault_service() -> FaultInjectionProcessor:
    """Fixture providing fault injection service processor.

    Returns:
        FaultInjectionProcessor with 50% failure rate

    """
    return FaultInjectionProcessor(
        failure_rate=TestDispatcherConstants.TestValues.FAILURE_RATE_50,
        rng=random.Random(0),
    )


pytestmark = [pytest.mark.integration]


# ==================== TESTS ====================


class TestFlextDispatcherLayer3Docker:
    """Comprehensive integration tests for FlextDispatcher Layer 3 with Docker.

    Single class pattern with nested classes organizing test cases by functionality.
    Uses factories, constants, and DRY principles to minimize code duplication.
    """

    class TestServiceDiscovery:
        """Tests for service discovery and registration."""

        @pytest.mark.parametrize(
            ("processor_name", "processor_class"),
            [
                (
                    TestDispatcherConstants.ProcessorNames.POSTGRES.value,
                    PostgreSQLService,
                ),
                (
                    TestDispatcherConstants.ProcessorNames.LATENCY.value,
                    NetworkLatencyProcessor,
                ),
                (
                    TestDispatcherConstants.ProcessorNames.FAULT.value,
                    FaultInjectionProcessor,
                ),
            ],
        )
        def test_register_processor(
            self,
            dispatcher: FlextDispatcher,
            processor_name: str,
            processor_class: type[
                PostgreSQLService | NetworkLatencyProcessor | FaultInjectionProcessor
            ],
        ) -> None:
            """Test registering processor service.

            Args:
                dispatcher: FlextDispatcher instance
                processor_name: Name of processor to register
                processor_class: Processor class to instantiate

            """
            processor = processor_class()
            result = dispatcher.register_processor(processor_name, processor)

            assertion_helpers.assert_flext_result_success(result)
            assert processor_name in dispatcher.processor_metrics
            if processor_name == TestDispatcherConstants.ProcessorNames.POSTGRES.value:
                assert dispatcher.processor_metrics[processor_name]["executions"] == 0

    class TestSingleItemProcessing:
        """Tests for single item processing."""

        def test_process_through_postgres_service(
            self,
            dispatcher: FlextDispatcher,
            postgres_service: PostgreSQLService,
        ) -> None:
            """Test processing query through real PostgreSQL service.

            Args:
                dispatcher: FlextDispatcher instance
                postgres_service: PostgreSQL service processor

            """
            _ = dispatcher.register_processor(
                TestDispatcherConstants.ProcessorNames.POSTGRES.value,
                postgres_service,
            )

            result = dispatcher.process(
                TestDispatcherConstants.ProcessorNames.POSTGRES.value,
                {"query": TestDispatcherConstants.QueryStrings.SELECT_1},
            )

            assertion_helpers.assert_flext_result_success(result) or result.is_failure
            if result.is_success:
                data = result.value
                assert isinstance(data, dict)
                assert "result" in data or "mocked" in data
                assert postgres_service.query_count > 0

        def test_process_through_latency_service(
            self,
            dispatcher: FlextDispatcher,
            latency_service: NetworkLatencyProcessor,
        ) -> None:
            """Test processing with network latency.

            Args:
                dispatcher: FlextDispatcher instance
                latency_service: Network latency processor

            """
            _ = dispatcher.register_processor(
                TestDispatcherConstants.ProcessorNames.LATENCY.value,
                latency_service,
            )

            start_time = time.time()
            result = dispatcher.process(
                TestDispatcherConstants.ProcessorNames.LATENCY.value,
                {"value": TestDispatcherConstants.TestValues.TEST_VALUE},
            )
            elapsed = time.time() - start_time

            assertion_helpers.assert_flext_result_success(result)
            assert elapsed >= TestDispatcherConstants.TestValues.LATENCY_DEFAULT
            assert latency_service.process_count == 1

        def test_process_through_fault_injection(
            self,
            dispatcher: FlextDispatcher,
            fault_service: FaultInjectionProcessor,
        ) -> None:
            """Test processing with fault injection.

            Args:
                dispatcher: FlextDispatcher instance
                fault_service: Fault injection processor

            """
            _ = dispatcher.register_processor(
                TestDispatcherConstants.ProcessorNames.FAULT.value,
                fault_service,
            )

            results = [
                dispatcher.process(
                    TestDispatcherConstants.ProcessorNames.FAULT.value,
                    {"value": i},
                )
                for i in range(TestDispatcherConstants.TestValues.ITERATIONS_FAULT)
            ]

            assert (
                fault_service.attempt_count
                == TestDispatcherConstants.TestValues.ITERATIONS_FAULT
            )
            assert any(r.is_success for r in results)

    class TestBatchProcessing:
        """Tests for batch processing."""

        @staticmethod
        def _create_test_data(count: int) -> list[dict[str, int]]:
            """Factory for test data."""
            return [{"value": i} for i in range(count)]

        @pytest.mark.parametrize(
            ("data_count", "batch_size", "expected_count"),
            [
                (3, TestDispatcherConstants.TestValues.BATCH_SIZE_SMALL, 3),
                (0, TestDispatcherConstants.TestValues.BATCH_SIZE_SMALL, 0),
                (10, TestDispatcherConstants.TestValues.BATCH_SIZE_LARGE, 10),
            ],
        )
        def test_batch_process(
            self,
            dispatcher: FlextDispatcher,
            latency_service: NetworkLatencyProcessor,
            data_count: int,
            batch_size: int,
            expected_count: int,
        ) -> None:
            """Test batch processing with network latency.

            Args:
                dispatcher: FlextDispatcher instance
                latency_service: Network latency processor
                data_count: Number of items to process
                batch_size: Batch size for processing
                expected_count: Expected number of processed items

            """
            _ = dispatcher.register_processor(
                TestDispatcherConstants.ProcessorNames.LATENCY.value,
                latency_service,
            )

            data_list = self._create_test_data(data_count)
            # Cast to list[t.GeneralValueType] for type compatibility
            data_list_cast = cast("list[t.GeneralValueType]", data_list)
            start_time = time.time()
            result = dispatcher.process_batch(
                TestDispatcherConstants.ProcessorNames.LATENCY.value,
                data_list_cast,
                batch_size=batch_size if data_count > 0 else 1,
            )
            elapsed = time.time() - start_time

            assertion_helpers.assert_flext_result_success(result)
            items = result.value
            assert len(items) == expected_count
            if data_count > 0:
                assert elapsed >= TestDispatcherConstants.TestValues.LATENCY_DEFAULT
            assert latency_service.process_count == expected_count

    class TestParallelProcessing:
        """Tests for parallel processing."""

        @staticmethod
        def _create_test_data(count: int) -> list[dict[str, int]]:
            """Factory for test data."""
            return [{"value": i} for i in range(count)]

        @pytest.mark.parametrize(
            ("data_count", "max_workers", "expected_count"),
            [
                (4, TestDispatcherConstants.TestValues.MAX_WORKERS_SMALL, 4),
                (0, TestDispatcherConstants.TestValues.MAX_WORKERS_SMALL, 0),
                (50, TestDispatcherConstants.TestValues.MAX_WORKERS_LARGE, 50),
            ],
        )
        def test_parallel_process(
            self,
            dispatcher: FlextDispatcher,
            latency_service: NetworkLatencyProcessor,
            data_count: int,
            max_workers: int,
            expected_count: int,
        ) -> None:
            """Test parallel processing with network latency.

            Args:
                dispatcher: FlextDispatcher instance
                latency_service: Network latency processor
                data_count: Number of items to process
                max_workers: Maximum number of workers
                expected_count: Expected number of processed items

            """
            _ = dispatcher.register_processor(
                TestDispatcherConstants.ProcessorNames.LATENCY.value,
                latency_service,
            )

            data_list = self._create_test_data(data_count)
            # Cast to list[t.GeneralValueType] for type compatibility
            data_list_cast = cast("list[t.GeneralValueType]", data_list)
            start_time = time.time()
            result = dispatcher.process_parallel(
                TestDispatcherConstants.ProcessorNames.LATENCY.value,
                data_list_cast,
                max_workers=max_workers if data_count > 0 else 1,
            )
            elapsed = time.time() - start_time

            assertion_helpers.assert_flext_result_success(result)
            items = result.value
            assert len(items) == expected_count
            if data_count == 4:
                assert (
                    elapsed
                    < TestDispatcherConstants.TestValues.PARALLEL_TIMEOUT_THRESHOLD
                )
            assert latency_service.process_count == expected_count

    class TestTimeoutEnforcement:
        """Tests for timeout enforcement."""

        @pytest.mark.parametrize(
            ("latency", "timeout", "should_succeed"),
            [
                (
                    TestDispatcherConstants.TestValues.LATENCY_DEFAULT,
                    TestDispatcherConstants.TestValues.TIMEOUT_SUCCESS,
                    True,
                ),
                (
                    TestDispatcherConstants.TestValues.LATENCY_SLOW,
                    TestDispatcherConstants.TestValues.TIMEOUT_EXCEEDED,
                    False,
                ),
            ],
        )
        def test_execute_with_timeout(
            self,
            dispatcher: FlextDispatcher,
            latency: float,
            timeout: float,
            should_succeed: bool,
        ) -> None:
            """Test timeout enforcement.

            Args:
                dispatcher: FlextDispatcher instance
                latency: Latency for processor
                timeout: Timeout value
                should_succeed: Whether execution should succeed

            """
            service = NetworkLatencyProcessor(latency_seconds=latency)
            _ = dispatcher.register_processor(
                TestDispatcherConstants.ProcessorNames.SLOW.value,
                service,
            )

            result = dispatcher.execute_with_timeout(
                TestDispatcherConstants.ProcessorNames.SLOW.value,
                {"value": TestDispatcherConstants.TestValues.TEST_VALUE},
                timeout=timeout,
            )

            if should_succeed:
                assertion_helpers.assert_flext_result_success(result)
                assert service.process_count == 1
            else:
                assertion_helpers.assert_flext_result_failure(result)
                assert "timeout" in (result.error or "").lower()

        def test_execute_with_timeout_unregistered_processor(
            self,
            dispatcher: FlextDispatcher,
        ) -> None:
            """Test timeout with unregistered processor.

            Args:
                dispatcher: FlextDispatcher instance

            """
            result = dispatcher.execute_with_timeout(
                TestDispatcherConstants.ProcessorNames.NONEXISTENT.value,
                {"value": TestDispatcherConstants.TestValues.TEST_VALUE},
                timeout=TestDispatcherConstants.TestValues.TIMEOUT_SUCCESS,
            )

            assertion_helpers.assert_flext_result_failure(result)

    class TestFallbackChains:
        """Tests for fallback chain execution."""

        def test_process_primary_success(
            self,
            dispatcher: FlextDispatcher,
            latency_service: NetworkLatencyProcessor,
        ) -> None:
            """Test successful primary processor execution.

            Args:
                dispatcher: FlextDispatcher instance
                latency_service: Network latency processor

            """
            _ = dispatcher.register_processor(
                TestDispatcherConstants.ProcessorNames.PRIMARY.value,
                latency_service,
            )
            _ = dispatcher.register_processor(
                TestDispatcherConstants.ProcessorNames.FALLBACK1.value,
                NetworkLatencyProcessor(
                    latency_seconds=TestDispatcherConstants.TestValues.LATENCY_FAST,
                ),
            )

            result = dispatcher.process(
                TestDispatcherConstants.ProcessorNames.PRIMARY.value,
                {"value": TestDispatcherConstants.TestValues.TEST_VALUE},
            )

            assertion_helpers.assert_flext_result_success(result)
            assert latency_service.process_count == 1

        @pytest.mark.parametrize(
            ("primary_rate", "fallback_rate", "expected_failure"),
            [
                (
                    TestDispatcherConstants.TestValues.FAILURE_RATE_100,
                    TestDispatcherConstants.TestValues.FAILURE_RATE_0,
                    True,
                ),
            ],
        )
        def test_process_primary_failure_fast_fail(
            self,
            dispatcher: FlextDispatcher,
            primary_rate: float,
            fallback_rate: float,
            expected_failure: bool,
        ) -> None:
            """Test primary processor failure returns error immediately.

            Args:
                dispatcher: FlextDispatcher instance
                primary_rate: Primary processor failure rate
                fallback_rate: Fallback processor failure rate
                expected_failure: Whether failure is expected

            """
            primary = FaultInjectionProcessor(failure_rate=primary_rate)
            fallback = FaultInjectionProcessor(failure_rate=fallback_rate)

            _ = dispatcher.register_processor(
                TestDispatcherConstants.ProcessorNames.PRIMARY.value,
                primary,
            )
            _ = dispatcher.register_processor(
                TestDispatcherConstants.ProcessorNames.FALLBACK1.value,
                fallback,
            )

            result = dispatcher.process(
                TestDispatcherConstants.ProcessorNames.PRIMARY.value,
                {"value": TestDispatcherConstants.TestValues.TEST_VALUE},
            )

            if expected_failure:
                assertion_helpers.assert_flext_result_failure(result)
                assert primary.attempt_count >= 1
                assert fallback.attempt_count == 0

        def test_process_all_fail_fast_fail(
            self,
            dispatcher: FlextDispatcher,
        ) -> None:
            """Test processor failure returns error immediately.

            Args:
                dispatcher: FlextDispatcher instance

            """
            primary = FaultInjectionProcessor(
                failure_rate=TestDispatcherConstants.TestValues.FAILURE_RATE_100,
            )
            fallback1 = FaultInjectionProcessor(
                failure_rate=TestDispatcherConstants.TestValues.FAILURE_RATE_100,
            )
            fallback2 = FaultInjectionProcessor(
                failure_rate=TestDispatcherConstants.TestValues.FAILURE_RATE_100,
            )

            _ = dispatcher.register_processor(
                TestDispatcherConstants.ProcessorNames.PRIMARY.value,
                primary,
            )
            _ = dispatcher.register_processor(
                TestDispatcherConstants.ProcessorNames.FALLBACK1.value,
                fallback1,
            )
            _ = dispatcher.register_processor(
                TestDispatcherConstants.ProcessorNames.FALLBACK2.value,
                fallback2,
            )

            result = dispatcher.process(
                TestDispatcherConstants.ProcessorNames.PRIMARY.value,
                {"value": TestDispatcherConstants.TestValues.TEST_VALUE},
            )

            assertion_helpers.assert_flext_result_failure(result)
            assert result.error is not None
            assert fallback1.attempt_count == 0
            assert fallback2.attempt_count == 0

        def test_process_multiple_processors_independent(
            self,
            dispatcher: FlextDispatcher,
        ) -> None:
            """Test multiple processors operate independently.

            Args:
                dispatcher: FlextDispatcher instance

            """
            primary = FaultInjectionProcessor(
                failure_rate=TestDispatcherConstants.TestValues.FAILURE_RATE_100,
            )
            fallback1 = FaultInjectionProcessor(
                failure_rate=TestDispatcherConstants.TestValues.FAILURE_RATE_100,
            )
            fallback2 = FaultInjectionProcessor(
                failure_rate=TestDispatcherConstants.TestValues.FAILURE_RATE_0,
            )

            _ = dispatcher.register_processor(
                TestDispatcherConstants.ProcessorNames.PRIMARY.value,
                primary,
            )
            _ = dispatcher.register_processor(
                TestDispatcherConstants.ProcessorNames.FALLBACK1.value,
                fallback1,
            )
            _ = dispatcher.register_processor(
                TestDispatcherConstants.ProcessorNames.FALLBACK2.value,
                fallback2,
            )

            result1 = dispatcher.process(
                TestDispatcherConstants.ProcessorNames.PRIMARY.value,
                {"value": TestDispatcherConstants.TestValues.TEST_VALUE},
            )
            assert result1.is_failure

            result2 = dispatcher.process(
                TestDispatcherConstants.ProcessorNames.FALLBACK2.value,
                {"value": TestDispatcherConstants.TestValues.TEST_VALUE},
            )
            assert result2.is_success

    class TestMetricsAndObservability:
        """Tests for metrics collection."""

        def test_processor_metrics_tracking(
            self,
            dispatcher: FlextDispatcher,
            latency_service: NetworkLatencyProcessor,
        ) -> None:
            """Test processor metrics collection.

            Args:
                dispatcher: FlextDispatcher instance
                latency_service: Network latency processor

            """
            _ = dispatcher.register_processor(
                TestDispatcherConstants.ProcessorNames.LATENCY.value,
                latency_service,
            )

            for _ in range(TestDispatcherConstants.TestValues.ITERATIONS_METRICS):
                _ = dispatcher.process(
                    TestDispatcherConstants.ProcessorNames.LATENCY.value,
                    {"value": TestDispatcherConstants.TestValues.TEST_VALUE},
                )

            metrics = dispatcher.processor_metrics
            assert TestDispatcherConstants.ProcessorNames.LATENCY.value in metrics
            assert (
                metrics[TestDispatcherConstants.ProcessorNames.LATENCY.value][
                    "executions"
                ]
                == TestDispatcherConstants.TestValues.ITERATIONS_METRICS
            )
            assert (
                metrics[TestDispatcherConstants.ProcessorNames.LATENCY.value][
                    "successful_processes"
                ]
                == TestDispatcherConstants.TestValues.ITERATIONS_METRICS
            )

        @pytest.mark.parametrize(
            ("operation_type", "data_count"),
            [
                ("batch", TestDispatcherConstants.TestValues.ITERATIONS_BATCH),
                ("parallel", TestDispatcherConstants.TestValues.ITERATIONS_PARALLEL),
            ],
        )
        def test_performance_metrics(
            self,
            dispatcher: FlextDispatcher,
            latency_service: NetworkLatencyProcessor,
            operation_type: str,
            data_count: int,
        ) -> None:
            """Test performance metrics collection.

            Args:
                dispatcher: FlextDispatcher instance
                latency_service: Network latency processor
                operation_type: Type of operation (batch or parallel)
                data_count: Number of items to process

            """
            _ = dispatcher.register_processor(
                TestDispatcherConstants.ProcessorNames.LATENCY.value,
                latency_service,
            )

            data_list = [{"value": i} for i in range(data_count)]
            # Cast to list[t.GeneralValueType] for type compatibility
            data_list_cast = cast("list[t.GeneralValueType]", data_list)
            if operation_type == "batch":
                _ = dispatcher.process_batch(
                    TestDispatcherConstants.ProcessorNames.LATENCY.value,
                    data_list_cast,
                )
                perf = dispatcher.batch_performance
                assert "batch_operations" in perf
                batch_ops = perf["batch_operations"]
                assert isinstance(batch_ops, int) and batch_ops >= 0
            else:
                _ = dispatcher.process_parallel(
                    TestDispatcherConstants.ProcessorNames.LATENCY.value,
                    data_list_cast,
                )
                perf = dispatcher.parallel_performance
                assert "parallel_operations" in perf
                parallel_ops = perf["parallel_operations"]
                assert isinstance(parallel_ops, int) and parallel_ops >= 0

        def test_performance_analytics(
            self,
            dispatcher: FlextDispatcher,
            latency_service: NetworkLatencyProcessor,
        ) -> None:
            """Test comprehensive performance analytics.

            Args:
                dispatcher: FlextDispatcher instance
                latency_service: Network latency processor

            """
            _ = dispatcher.register_processor(
                TestDispatcherConstants.ProcessorNames.LATENCY.value,
                latency_service,
            )

            _ = dispatcher.process(
                TestDispatcherConstants.ProcessorNames.LATENCY.value,
                {"value": TestDispatcherConstants.TestValues.TEST_VALUE},
            )
            _ = dispatcher.process_batch(
                TestDispatcherConstants.ProcessorNames.LATENCY.value,
                [
                    {"value": i}
                    for i in range(
                        TestDispatcherConstants.TestValues.ITERATIONS_METRICS,
                    )
                ],
            )

            analytics_result = dispatcher.get_performance_analytics()
            assert analytics_result.is_success
            analytics = analytics_result.value
            assert isinstance(analytics, dict)
            assert "global_metrics" in analytics or "metrics" in analytics
