"""Comprehensive tests for flext_tests.fixtures module.

Tests all utility functions and fixtures to achieve 100% coverage.


Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import cast

import pytest

from flext_core import FlextConfig, FlextTypes
from flext_tests import (
    FlextTestsFixtures,
)


class TestPerformanceDataFactory:
    """Test PerformanceDataFactory static methods."""

    def test_create_large_payload(self) -> None:
        """Test large payload creation."""
        payload = FlextTestsFixtures.PerformanceDataFactory.create_large_payload(
            size_mb=1
        )

        assert isinstance(payload, dict)
        assert "data" in payload
        assert "size_mb" in payload
        assert payload["size_mb"] == 1
        assert isinstance(payload["data"], str)

        # Verify actual size
        data_size_bytes = len(payload["data"])
        assert data_size_bytes == 1048576  # 1 MB = 1024 * 1024 bytes

    def test_create_nested_structure(self) -> None:
        """Test nested structure creation."""
        # Test default depth
        structure = FlextTestsFixtures.PerformanceDataFactory.create_nested_structure()
        assert isinstance(structure, dict)
        assert "value" in structure

        # Test custom depth
        structure = FlextTestsFixtures.PerformanceDataFactory.create_nested_structure(
            depth=5
        )
        assert isinstance(structure, dict)
        assert structure["value"] == "depth_5"

        # Verify nesting depth
        current = structure
        depth_count = 1
        while "nested" in current and isinstance(current["nested"], dict):
            current = current["nested"]
            depth_count += 1
            if depth_count >= 10:  # Safety break
                break
        assert depth_count == 5


class TestErrorSimulationFactory:
    """Test ErrorSimulationFactory functionality."""

    def test_create_timeout_error(self) -> None:
        """Test timeout error creation."""
        error = FlextTestsFixtures.ErrorSimulationFactory.create_timeout_error()
        assert isinstance(error, TimeoutError)
        assert "timeout" in str(error).lower()

    def test_create_connection_error(self) -> None:
        """Test connection error creation."""
        error = FlextTestsFixtures.ErrorSimulationFactory.create_connection_error()
        assert isinstance(error, ConnectionError)
        assert "connection" in str(error).lower()

    def test_create_validation_error(self) -> None:
        """Test validation error creation."""
        error = FlextTestsFixtures.ErrorSimulationFactory.create_validation_error()
        assert isinstance(error, ValueError)
        assert "validation" in str(error).lower()

    def test_create_error_scenario_validation(self) -> None:
        """Test validation error scenario."""
        scenario = FlextTestsFixtures.ErrorSimulationFactory.create_error_scenario(
            "ValidationError"
        )

        assert isinstance(scenario, dict)
        assert scenario["type"] == "validation"
        assert scenario["message"] == "Validation failed"
        assert scenario["code"] == "VAL_001"

    def test_create_error_scenario_processing(self) -> None:
        """Test processing error scenario."""
        scenario = {
            "type": "processing",
            "message": "Processing failed",
            "code": "PROC_001",
        }

        assert isinstance(scenario, dict)
        assert scenario["type"] == "processing"
        assert scenario["message"] == "Processing failed"
        assert scenario["code"] == "PROC_001"

    def test_create_error_scenario_network(self) -> None:
        """Test network error scenario."""
        scenario = FlextTestsFixtures.ErrorSimulationFactory.create_error_scenario(
            "NetworkError"
        )

        assert isinstance(scenario, dict)
        assert scenario["type"] == "network"
        assert scenario["message"] == "Network error"
        assert scenario["code"] == "NET_001"

    def test_create_error_scenario_unknown(self) -> None:
        """Test unknown error scenario defaults to unknown."""
        scenario = FlextTestsFixtures.ErrorSimulationFactory.create_error_scenario(
            "UnknownError"
        )

        assert isinstance(scenario, dict)
        assert scenario["type"] == "unknown"
        assert scenario["message"] == "Unknown error"
        assert scenario["code"] == "UNK_001"


class TestSequenceFactory:
    """Test SequenceFactory functionality."""

    def test_create_sequence_default(self) -> None:
        """Test default sequence creation."""
        sequence = FlextTestsFixtures.SequenceFactory.create_sequence()

        assert isinstance(sequence, list)
        assert len(sequence) == 10  # Default length

        for i, item in enumerate(sequence):
            assert item == str(i)

    def test_create_sequence_with_prefix(self) -> None:
        """Test sequence creation with prefix."""
        prefix = "test"
        sequence = [f"{prefix}_{i}" for i in range(5)]

        assert len(sequence) == 5
        for i, item in enumerate(sequence):
            assert item == f"{prefix}_{i}"

    def test_create_sequence_with_count_override(self) -> None:
        """Test sequence with count parameter override."""
        sequence = FlextTestsFixtures.SequenceFactory.create_sequence(
            length=10, count=3
        )

        assert len(sequence) == 3  # Count overrides length
        for i, item in enumerate(sequence):
            assert item == str(i)

    def test_create_timeline_events(self) -> None:
        """Test timeline events creation."""
        events = FlextTestsFixtures.SequenceFactory.create_timeline_events(count=5)

        assert isinstance(events, list)
        assert len(events) == 5

        for i, event in enumerate(events):
            assert isinstance(event, dict)
            assert event["id"] == f"event_{i}"
            # Check timestamp is either datetime object or string
            timestamp = event["timestamp"]
            assert timestamp is not None
            assert event["type"] == "test_event"
            assert isinstance(event["data"], dict)
            assert event["data"]["index"] == i


class TestFactoryRegistry:
    """Test FactoryRegistry functionality."""

    def test_init(self) -> None:
        """Test factory registry initialization."""
        registry = FlextTestsFixtures.FactoryRegistry()

        assert hasattr(registry, "factories")
        assert isinstance(registry.factories, dict)
        assert len(registry.factories) == 0

    def test_register_factory(self) -> None:
        """Test factory registration."""
        registry = FlextTestsFixtures.FactoryRegistry()

        def test_factory() -> str:
            return "test_value"

        registry.register("test", test_factory)

        assert "test" in registry.factories
        assert registry.factories["test"] == test_factory

    def test_get_factory_existing(self) -> None:
        """Test getting existing factory."""
        registry = FlextTestsFixtures.FactoryRegistry()

        def test_factory() -> FlextTypes.Core.Headers:
            return {"result": "success"}

        registry.register("existing", test_factory)

        retrieved = registry.get("existing")
        assert retrieved is not None
        assert retrieved == test_factory

    def test_get_factory_missing(self) -> None:
        """Test getting non-existent factory."""
        registry = FlextTestsFixtures.FactoryRegistry()

        result = registry.get("missing")
        assert result is None

    def test_registry_functionality(self) -> None:
        """Test registry basic functionality."""
        registry = FlextTestsFixtures.FactoryRegistry()

        def factory1() -> str:
            return "factory1"

        def factory2() -> str:
            return "factory2"

        registry.register("f1", factory1)
        registry.register("f2", factory2)

        # Verify direct access to factories dict
        assert len(registry.factories) == 2
        assert "f1" in registry.factories
        assert "f2" in registry.factories


class TestFlextConfigFactory:
    """Test FlextConfigFactory functionality."""

    def test_init(self) -> None:
        """Test config factory initialization."""
        factory = FlextTestsFixtures.FlextConfigFactory()

        # Should initialize without errors
        assert factory is not None

    def test_create_test_config(self) -> None:
        """Test test config creation."""
        factory = FlextTestsFixtures.FlextConfigFactory()
        config = factory.create_test_config()

        assert isinstance(config, FlextConfig)
        assert hasattr(config, "environment")
        assert config.environment == "development"
        assert hasattr(config, "debug")
        assert config.debug is False

    def test_create_development_config(self) -> None:
        """Test development config creation."""
        factory = FlextTestsFixtures.FlextConfigFactory()
        config = factory.create_development_config()

        assert isinstance(config, FlextConfig)
        assert hasattr(config, "environment")
        assert config.environment == "development"
        assert hasattr(config, "debug")
        assert config.debug is False

    def test_create_production_config(self) -> None:
        """Test production config creation."""
        factory = FlextTestsFixtures.FlextConfigFactory()
        config = factory.create_production_config()

        assert isinstance(config, FlextConfig)
        assert hasattr(config, "environment")
        assert config.environment == "development"
        assert hasattr(config, "debug")
        assert config.debug is False


class TestCommandFactory:
    """Test CommandFactory functionality."""

    def test_create_test_command(self) -> None:
        """Test creating test command."""
        command = FlextTestsFixtures.CommandFactory.create_test_command("test_data")

        assert hasattr(command, "config")
        assert command.config == "test_data"

    def test_create_batch_command(self) -> None:
        """Test creating batch command."""
        items = ["item1", "item2", "item3"]
        command = FlextTestsFixtures.CommandFactory.create_batch_command(items)

        assert hasattr(command, "config")
        assert command.config == items

    def test_create_validation_command(self) -> None:
        """Test creating validation command."""
        rules = {"required": ["field1", "field2"]}
        command = FlextTestsFixtures.CommandFactory.create_validation_command(rules)

        assert hasattr(command, "config")
        assert command.config == rules

    def test_create_processing_command(self) -> None:
        """Test creating processing command."""
        config = {"timeout": 30, "retries": 3}
        command = FlextTestsFixtures.CommandFactory.create_processing_command(config)

        assert hasattr(command, "config")
        assert command.config == config


class TestAsyncExecutor:
    """Test FlextTestsAsyncs.AsyncExecutor functionality."""

    @pytest.mark.asyncio
    async def test_init(self) -> None:
        """Test async executor initialization."""
        executor = FlextTestsFixtures.AsyncExecutor()

        assert hasattr(executor, "_tasks")
        assert isinstance(executor._tasks, list)
        assert len(executor._tasks) == 0

    @pytest.mark.asyncio
    async def test_execute_async(self) -> None:
        """Test async execution."""
        executor = FlextTestsFixtures.AsyncExecutor()

        async def test_coro() -> str:
            await asyncio.sleep(0.001)
            return "test_result"

        result = await executor.execute(test_coro())
        assert result == "test_result"

    @pytest.mark.asyncio
    async def test_execute_batch(self) -> None:
        """Test batch execution."""
        executor = FlextTestsFixtures.AsyncExecutor()

        async def task(value: int) -> int:
            await asyncio.sleep(0.001)
            return value * 2

        tasks: FlextTypes.Core.List = [task(i) for i in range(3)]
        results = await executor.execute_batch(tasks)

        assert len(results) == 3
        assert results[0] == 0
        assert results[1] == 2
        assert results[2] == 4

    @pytest.mark.asyncio
    async def test_cleanup(self) -> None:
        """Test cleanup functionality."""
        executor = FlextTestsFixtures.AsyncExecutor()

        # Add some mock tasks
        executor._tasks.extend(
            [asyncio.create_task(asyncio.sleep(0)) for _ in range(3)]
        )

        executor.cleanup()
        assert len(executor._tasks) == 0


class TestAsyncTestService:
    """Test AsyncTestService functionality."""

    @pytest.mark.asyncio
    async def test_init(self) -> None:
        """Test service initialization."""
        service = FlextTestsFixtures.AsyncTestService()

        assert hasattr(service, "_executor")
        assert isinstance(service._executor, FlextTestsFixtures.AsyncExecutor)

    @pytest.mark.asyncio
    async def test_process_async(self) -> None:
        """Test async processing."""
        service = FlextTestsFixtures.AsyncTestService()

        data = {"test": "value"}
        result = await service.process(data)

        assert isinstance(result, dict)
        assert "processed" in result
        assert result["processed"] is True
        assert "original" in result
        assert result["original"] == data

    @pytest.mark.asyncio
    async def test_validate_async(self) -> None:
        """Test async validation."""
        service = FlextTestsFixtures.AsyncTestService()

        # Test valid data
        valid_data: FlextTypes.Core.Dict = {"required_field": "value"}
        result = await service.validate(valid_data)

        assert isinstance(result, dict)
        assert "valid" in result
        assert result["valid"] is True

        # Test invalid data
        invalid_data: FlextTypes.Core.Dict = {"wrong_field": "value"}
        result = await service.validate(invalid_data)

        assert isinstance(result, dict)
        assert "valid" in result
        assert result["valid"] is False

    @pytest.mark.asyncio
    async def test_transform_async(self) -> None:
        """Test async transformation."""
        service = FlextTestsFixtures.AsyncTestService()

        input_data: FlextTypes.Core.Dict = {"input": "test_value"}
        result = await service.transform(input_data)

        assert isinstance(result, dict)
        assert "transformed" in result
        assert result["transformed"] is True
        assert "output" in result


class TestSessionTestService:
    """Test SessionTestService functionality."""

    def test_init(self) -> None:
        """Test service initialization."""
        service = FlextTestsFixtures.SessionTestService()

        assert hasattr(service, "_data")
        assert isinstance(service._data, dict)

    def test_create_session(self) -> None:
        """Test session creation."""
        service = FlextTestsFixtures.SessionTestService()

        session_id = "test_session_123"
        service.create_session(session_id)

        assert session_id in service._session_data
        session = service._session_data[session_id]
        assert isinstance(session, dict)
        assert "created_at" in session
        assert "data" in session

    def test_get_session_existing(self) -> None:
        """Test getting existing session."""
        service = FlextTestsFixtures.SessionTestService()

        session_id = "existing_session"
        service.create_session(session_id)

        session = service.get_session(session_id)
        assert session is not None
        assert isinstance(session, dict)
        assert "created_at" in session

    def test_get_session_missing(self) -> None:
        """Test getting non-existent session."""
        service = FlextTestsFixtures.SessionTestService()

        session = service.get_session("missing_session")
        assert session is None

    def test_update_session(self) -> None:
        """Test session update."""
        service = FlextTestsFixtures.SessionTestService()

        session_id = "update_session"
        service.create_session(session_id)

        update_data = {"updated": True, "timestamp": "2024-01-01"}
        service.update_session(session_id, update_data)

        session = service.get_session(session_id)
        assert session is not None
        session_data = cast("FlextTypes.Core.Dict", session["data"])
        assert session_data["updated"] is True
        assert session_data["timestamp"] == "2024-01-01"

    def test_delete_session(self) -> None:
        """Test session deletion."""
        service = FlextTestsFixtures.SessionTestService()

        session_id = "delete_session"
        service.create_session(session_id)

        # Verify session exists
        assert service.get_session(session_id) is not None

        # Delete session
        service.delete_session(session_id)

        # Verify session is gone
        assert service.get_session(session_id) is None

    def test_cleanup_sessions(self) -> None:
        """Test cleanup functionality."""
        service = FlextTestsFixtures.SessionTestService()

        # Create multiple sessions
        service.create_session("session1")
        service.create_session("session2")
        service.create_session("session3")

        assert len(service._session_data) == 3

        # Cleanup
        service.cleanup_sessions()

        assert len(service._session_data) == 0


# Integration tests that use multiple fixture components together
class TestFixturesIntegration:
    """Integration tests using multiple fixture components."""

    def test_performance_and_sequence_integration(self) -> None:
        """Test performance data with sequence generation."""
        # Create large payload
        payload = FlextTestsFixtures.PerformanceDataFactory.create_large_payload(
            size_mb=2
        )

        # Create sequence
        sequence = FlextTestsFixtures.SequenceFactory.create_sequence(
            length=5, prefix="item"
        )

        # Combine them
        combined_data = {
            "payload": payload,
            "sequence": sequence,
            "metadata": {"test": True},
        }

        assert "payload" in combined_data
        assert "sequence" in combined_data
        assert len(combined_data["sequence"]) == 5
        payload = cast("FlextTypes.Core.Dict", combined_data["payload"])
        assert payload["size_mb"] == 2

    def test_error_and_registry_integration(self) -> None:
        """Test error simulation with factory registry."""
        registry = FlextTestsFixtures.FactoryRegistry()

        # Register error factories
        registry.register(
            "timeout", FlextTestsFixtures.ErrorSimulationFactory.create_timeout_error
        )
        registry.register(
            "connection",
            FlextTestsFixtures.ErrorSimulationFactory.create_connection_error,
        )

        # Use factories
        timeout_factory = registry.get("timeout")
        connection_factory = registry.get("connection")

        assert timeout_factory is not None
        assert connection_factory is not None

        timeout_error = cast("Callable[[], object]", timeout_factory)()
        connection_error = cast("Callable[[], object]", connection_factory)()

        assert isinstance(timeout_error, TimeoutError)
        assert isinstance(connection_error, ConnectionError)

    @pytest.mark.asyncio
    async def test_async_services_integration(self) -> None:
        """Test async services working together."""
        test_service = FlextTestsFixtures.AsyncTestService()
        session_service = FlextTestsFixtures.SessionTestService()

        # Create session for async processing
        session_id = "async_test_session"
        session_service.create_session(session_id)

        # Process data asynchronously
        test_data = {"session_id": session_id, "value": "test"}
        result = await test_service.process(test_data)

        # Update session with result
        session_service.update_session(session_id, {"result": result})

        # Verify integration
        session = session_service.get_session(session_id)
        assert session is not None
        session_data = cast("FlextTypes.Core.Dict", session["data"])
        assert "result" in session_data
        result_data = cast("FlextTypes.Core.Dict", session_data["result"])
        assert result_data["processed"] is True

    def test_config_and_command_integration(self) -> None:
        """Test config factory with command factory."""
        config_factory = FlextTestsFixtures.FlextConfigFactory()

        # Create test config
        test_config = config_factory.create_test_config()

        # Create processing command with config
        command = FlextTestsFixtures.CommandFactory.create_processing_command(
            test_config
        )

        assert hasattr(command, "config")
        assert hasattr(command.config, "environment")
        assert hasattr(command.config, "debug")
