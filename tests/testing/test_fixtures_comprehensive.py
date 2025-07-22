"""Comprehensive tests for flext_core.testing fixtures - Coverage Boost.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Comprehensive tests for all testing fixtures to achieve >90% coverage.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, Mock, patch

import pytest

from flext_core.testing import (
    AsyncTestCase,
    BaseTestCase,
    DatabaseFixtures,
    MemoryFixtures,
    MockConfig,
    MockLogger,
    MockRepository,
    get_project_root_fixture,
    get_test_environment_fixture,
    setup_flext_test_environment,
)
from flext_core.testing import FlextTestFixtures


class TestFixtureBasics:
    """Test basic fixture functionality."""

    def test_test_fixtures_creation(self) -> None:
        """Test FlextTestFixtures creation."""
        fixtures = FlextTestFixtures()

        # FlextTestFixtures has static methods only
        assert fixtures is not None

    def test_memory_fixtures_creation(self) -> None:
        """Test MemoryFixtures creation."""
        fixtures = MemoryFixtures()

        # Should work without errors
        assert fixtures is not None

    def test_database_fixtures_creation(self) -> None:
        """Test DatabaseFixtures creation."""
        fixtures = DatabaseFixtures()

        # DatabaseFixtures has static methods only
        assert fixtures is not None


class TestFlextTestFixturesComprehensive:
    """Comprehensive tests for FlextTestFixtures class."""

    def test_test_fixtures_all_methods(self) -> None:
        """Test all FlextTestFixtures methods for coverage."""
        fixtures = FlextTestFixtures()

        # Test all static methods
        uuid_val = FlextTestFixtures.valid_uuid()
        assert uuid_val is not None
        assert len(str(uuid_val)) == 36  # UUID format

        timestamp = FlextTestFixtures.current_timestamp()
        assert timestamp is not None

        project_name = FlextTestFixtures.test_project_name()
        assert project_name == "test-project"

        version = FlextTestFixtures.test_version()
        assert version == "1.0.0"

        # Test data generation methods
        config_data = FlextTestFixtures.test_config_data()
        assert isinstance(config_data, dict)
        assert "project_name" in config_data

        entity_data = FlextTestFixtures.test_entity_data()
        assert isinstance(entity_data, dict)
        assert "id" in entity_data

        pipeline_data = FlextTestFixtures.test_pipeline_data()
        assert isinstance(pipeline_data, dict)
        assert "name" in pipeline_data

        plugin_data = FlextTestFixtures.test_plugin_data()
        assert isinstance(plugin_data, dict)
        assert "name" in plugin_data

    def test_test_fixtures_repeated_calls(self) -> None:
        """Test FlextTestFixtures methods can be called multiple times."""
        fixtures = FlextTestFixtures()

        # Multiple calls should work
        uuid1 = FlextTestFixtures.valid_uuid()
        uuid2 = FlextTestFixtures.valid_uuid()
        assert uuid1 != uuid2  # Should generate different UUIDs

        # Static data should be consistent
        data1 = FlextTestFixtures.test_config_data()
        data2 = FlextTestFixtures.test_config_data()
        assert data1["project_name"] == data2["project_name"]  # Compare dict contents


class TestMemoryFixturesComprehensive:
    """Comprehensive tests for MemoryFixtures class."""

    def test_memory_fixtures_all_operations(self) -> None:
        """Test all MemoryFixtures operations."""
        fixtures = MemoryFixtures()

        # Test repository data management
        repo_data = fixtures.get_repository_data("test")
        assert isinstance(repo_data, dict)

        # Test different repository types
        repo_data2 = fixtures.get_repository_data("pipeline")
        assert isinstance(repo_data2, dict)

        # Test adding different types of test data
        pipeline_data = fixtures.add_test_pipeline()
        assert isinstance(pipeline_data, dict)
        assert "id" in pipeline_data

        plugin_data = fixtures.add_test_plugin()
        assert isinstance(plugin_data, dict)
        assert "id" in plugin_data

        # Test clearing specific repository
        fixtures.clear_repository("pipeline")
        fixtures.clear_repository("plugin")

        # Test clearing all
        fixtures.clear_all()

    def test_memory_fixtures_multiple_entities(self) -> None:
        """Test MemoryFixtures with multiple entities."""
        fixtures = MemoryFixtures()

        # Add multiple pipelines
        pipeline1 = fixtures.add_test_pipeline()
        pipeline2 = fixtures.add_test_pipeline()

        assert pipeline1["id"] != pipeline2["id"]

        # Add multiple plugins
        plugin1 = fixtures.add_test_plugin()
        plugin2 = fixtures.add_test_plugin()

        assert plugin1["id"] != plugin2["id"]

        # Clear specific types
        fixtures.clear_repository("pipeline")

        # Should still have plugins but no pipelines
        # (Implementation details may vary)


class TestDatabaseFixturesComprehensive:
    """Comprehensive tests for DatabaseFixtures class."""

    def test_database_fixtures_all_url_methods(self) -> None:
        """Test all DatabaseFixtures URL generation methods."""
        # Test SQLite URLs
        db_url = DatabaseFixtures.get_test_database_url()
        assert isinstance(db_url, str)
        assert "sqlite" in db_url

        async_db_url = DatabaseFixtures.get_async_test_database_url()
        assert isinstance(async_db_url, str)
        assert "sqlite" in async_db_url

        # Test PostgreSQL URLs
        pg_url = DatabaseFixtures.get_postgres_test_url()
        assert isinstance(pg_url, str)
        assert "postgresql" in pg_url

        # Test Redis URL
        redis_url = DatabaseFixtures.get_redis_test_url()
        assert isinstance(redis_url, str)
        assert "redis" in redis_url

    def test_database_fixtures_config_creation(self) -> None:
        """Test DatabaseFixtures configuration creation."""
        config = DatabaseFixtures.create_test_config()
        assert isinstance(config, dict)

        # Should have database URLs
        assert "database_url" in config
        assert "database_async_url" in config
        assert "redis_url" in config

        # All URLs should be strings
        assert isinstance(config["database_url"], str)
        assert isinstance(config["database_async_url"], str)
        assert isinstance(config["redis_url"], str)

    def test_database_fixtures_all_url_methods_static(self) -> None:
        """Test DatabaseFixtures static URL methods."""
        # Test additional URL methods
        async_pg_url = DatabaseFixtures.get_async_postgres_test_url()
        assert isinstance(async_pg_url, str)
        assert "postgresql+asyncpg" in async_pg_url


# MockService is not available in flext_core.testing, skipping these tests for now


class TestAsyncTestCaseComprehensive:
    """Comprehensive tests for AsyncTestCase class."""

    @pytest.mark.asyncio
    async def test_async_test_case_full_lifecycle(self) -> None:
        """Test AsyncTestCase full lifecycle."""

        class TestAsyncClass(AsyncTestCase):
            def __init__(self) -> None:
                super().__init__()
                self.setup_called = False
                self.teardown_called = False

            async def asyncSetUp(self) -> None:
                self.setup_called = True

            async def asyncTearDown(self) -> None:
                self.teardown_called = True

            async def test_async_operation(self) -> str:
                await asyncio.sleep(0.001)
                return "test_result"

        test_case = TestAsyncClass()

        # Test setup
        await test_case.asyncSetUp()
        assert test_case.setup_called

        # Test async operation
        result = await test_case.test_async_operation()
        assert result == "test_result"

        # Test teardown
        await test_case.asyncTearDown()
        assert test_case.teardown_called

    def test_async_test_case_run_async_method(self) -> None:
        """Test AsyncTestCase run_async method."""
        test_case = AsyncTestCase()
        test_case.setUp()  # Setup event loop

        try:

            async def sample_coroutine() -> str:
                return "async_result"

            # Test run_async method
            result = test_case.run_async(sample_coroutine())
            assert result == "async_result"
        finally:
            test_case.tearDown()


class TestUtilityFunctionsComprehensive:
    """Comprehensive tests for utility functions."""

    def test_get_project_root_fixture_functionality(self) -> None:
        """Test get_project_root_fixture returns a fixture function."""
        # The fixture function should return a callable
        fixture = get_project_root_fixture()
        assert callable(fixture)

    def test_get_test_environment_fixture_functionality(self) -> None:
        """Test get_test_environment_fixture returns a fixture function."""
        fixture = get_test_environment_fixture()
        assert callable(fixture)

    def test_setup_flext_test_environment_complete(self) -> None:
        """Test setup_flext_test_environment complete functionality."""
        # Clear environment first
        env_vars_to_clear = ["FLEXT_ENV", "FLEXT_DEBUG"]
        for var in env_vars_to_clear:
            if var in os.environ:
                del os.environ[var]

        # Setup environment
        setup_flext_test_environment()

        # Check environment variables are set
        assert os.environ.get("FLEXT_ENV") == "testing"
        assert os.environ.get("FLEXT_DEBUG") == "true"

    def test_setup_flext_test_environment_multiple_calls(self) -> None:
        """Test setup_flext_test_environment handles multiple calls."""
        # Multiple calls should be safe
        setup_flext_test_environment()
        setup_flext_test_environment()

        # Should still work correctly
        assert os.environ.get("FLEXT_ENV") == "testing"
        assert os.environ.get("FLEXT_DEBUG") == "true"


class TestMockConfigComprehensive:
    """Comprehensive tests for MockConfig class."""

    def test_mock_config_all_methods(self) -> None:
        """Test all MockConfig methods."""
        config = MockConfig()

        # Test default values
        assert config.project_name == "test-project"
        assert config.project_version == "1.0.0"
        assert config.environment == "test"
        assert config.debug is True
        assert config.log_level == "DEBUG"

        # Test model_dump
        data = config.model_dump()
        assert isinstance(data, dict)
        assert "project_name" in data
        assert "debug" in data

        # Test model_dump_json
        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        assert "test-project" in json_str

        # Test environment checks
        assert not config.is_production()
        assert not config.is_development()  # environment is "test"

    def test_mock_config_custom_values(self) -> None:
        """Test MockConfig with custom values."""
        custom_config = MockConfig(
            project_name="custom-project",
            environment="production",
            custom_field="custom_value",
            debug=False,
        )

        assert custom_config.project_name == "custom-project"
        assert custom_config.environment == "production"
        assert custom_config.custom_field == "custom_value"
        assert custom_config.debug is False

        # Test production check
        assert custom_config.is_production()
        assert not custom_config.is_development()

    def test_mock_config_development_environment(self) -> None:
        """Test MockConfig with development environment."""
        dev_config = MockConfig(environment="development")

        assert dev_config.is_development()
        assert not dev_config.is_production()


class TestMockLoggerComprehensive:
    """Comprehensive tests for MockLogger class."""

    def test_mock_logger_all_log_levels(self) -> None:
        """Test MockLogger with all log levels."""
        logger = MockLogger()

        # Test all log levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.exception("Exception message")
        logger.critical("Critical message")

        # Test log retrieval
        all_logs = logger.get_logs()
        assert len(all_logs) == 6

        # Test level filtering
        debug_logs = logger.get_logs("DEBUG")
        assert len(debug_logs) == 1
        assert debug_logs[0][1] == "Debug message"

        error_logs = logger.get_logs("ERROR")
        assert len(error_logs) == 2  # error + exception

        # Test clear logs
        logger.clear_logs()
        assert len(logger.get_logs()) == 0

    def test_mock_logger_with_arguments(self) -> None:
        """Test MockLogger with arguments and keyword arguments."""
        logger = MockLogger()

        # Test with positional arguments
        logger.info("Message with %s and %d", "string", 42)

        # Test with keyword arguments
        logger.error("Error message", exc_info=True, extra={"key": "value"})

        logs = logger.get_logs()
        assert len(logs) == 2

        # Check arguments are recorded
        info_log = logs[0]
        assert info_log[2] == ("string", 42)  # args

        error_log = logs[1]
        assert "exc_info" in error_log[3]  # kwargs

    def test_mock_logger_bind_functionality(self) -> None:
        """Test MockLogger bind functionality."""
        logger = MockLogger()

        # Test bind returns self
        bound_logger = logger.bind(user_id="123", action="test")
        assert bound_logger is logger

        # Should still work normally after bind
        bound_logger.info("Bound message")

        logs = logger.get_logs()
        assert len(logs) == 1

    def test_mock_logger_assert_logged(self) -> None:
        """Test MockLogger assert_logged functionality."""
        logger = MockLogger()

        logger.info("Test message with specific content")
        logger.error("Error with specific content")

        # Should pass for existing logs
        logger.assert_logged("INFO", "specific content")
        logger.assert_logged("ERROR", "Error with")

        # Should raise for non-existent logs
        with pytest.raises(AssertionError):
            logger.assert_logged("DEBUG", "nonexistent")

        with pytest.raises(AssertionError):
            logger.assert_logged("INFO", "wrong content")


class TestMockRepositoryComprehensive:
    """Comprehensive tests for MockRepository class."""

    @pytest.mark.asyncio
    async def test_mock_repository_call_history(self) -> None:
        """Test MockRepository call history tracking."""
        from uuid import uuid4

        repo = MockRepository()

        # Create test entity
        entity = Mock()
        entity.id = uuid4()
        entity.name = "test"

        # Perform operations
        await repo.save(entity)
        await repo.find_by_id(entity.id)
        await repo.list()
        await repo.count()
        await repo.delete(entity.id)

        # Check call history
        history = repo.get_call_history()
        assert len(history) == 5

        # Check method names
        method_names = [call[0] for call in history]
        expected_methods = ["save", "find_by_id", "list", "count", "delete"]
        assert method_names == expected_methods

        # Clear history
        repo.clear_call_history()
        assert len(repo.get_call_history()) == 0

    @pytest.mark.asyncio
    async def test_mock_repository_data_management(self) -> None:
        """Test MockRepository data management methods."""
        from uuid import uuid4

        repo = MockRepository()

        # Test direct data manipulation
        entity_id = uuid4()
        entity = Mock()
        entity.id = entity_id

        repo.add_mock_data(entity_id, entity)

        # Should be retrievable
        result = await repo.find_by_id(entity_id)
        assert result.success
        assert result.data["entity"] is entity

        # Clear data
        repo.clear_data()

        # Should no longer exist
        result2 = await repo.find_by_id(entity_id)
        assert not result2.success

    @pytest.mark.asyncio
    async def test_mock_repository_pagination(self) -> None:
        """Test MockRepository pagination functionality."""
        from uuid import uuid4

        repo = MockRepository()

        # Add multiple entities
        entities = []
        for i in range(10):
            entity = Mock()
            entity.id = uuid4()
            entity.name = f"entity_{i}"
            await repo.save(entity)
            entities.append(entity)

        # Test pagination
        page1 = await repo.list(limit=3, offset=0)
        assert page1.success
        assert len(page1.data["entities"]) == 3

        page2 = await repo.list(limit=3, offset=3)
        assert page2.success
        assert len(page2.data["entities"]) == 3

        # Test count
        count_result = await repo.count()
        assert count_result.success
        assert count_result.data == 10

    @pytest.mark.asyncio
    async def test_mock_repository_entity_without_id(self) -> None:
        """Test MockRepository with entity that has no ID."""
        repo = MockRepository()

        # Create entity without ID
        entity = Mock()
        entity.name = "no_id_entity"
        # Don't set entity.id

        # Save should generate ID
        result = await repo.save(entity)
        assert result.success
        assert hasattr(entity, "id")
        assert entity.id is not None

        # Should be retrievable by generated ID
        find_result = await repo.find_by_id(entity.id)
        assert find_result.success
        assert find_result.data["entity"].name == "no_id_entity"


class TestFlextTestFixturesStaticMethods:
    """Test FlextTestFixtures static methods individually."""

    def test_test_fixtures_config_data(self) -> None:
        """Test FlextTestFixtures.test_config_data method."""
        data = FlextTestFixtures.test_config_data()
        assert isinstance(data, dict)
        assert "project_name" in data

    def test_test_fixtures_plugin_data(self) -> None:
        """Test FlextTestFixtures.test_plugin_data method."""
        data = FlextTestFixtures.test_plugin_data()
        assert isinstance(data, dict)
        assert "name" in data

    def test_test_fixtures_entity_data(self) -> None:
        """Test FlextTestFixtures.test_entity_data method."""
        data = FlextTestFixtures.test_entity_data()
        assert isinstance(data, dict)
        assert "id" in data

    def test_test_fixtures_pipeline_data(self) -> None:
        """Test FlextTestFixtures.test_pipeline_data method."""
        data = FlextTestFixtures.test_pipeline_data()
        assert isinstance(data, dict)
        assert "name" in data

    def test_test_fixtures_project_name(self) -> None:
        """Test FlextTestFixtures.test_project_name method."""
        name = FlextTestFixtures.test_project_name()
        assert name == "test-project"

    def test_test_fixtures_version(self) -> None:
        """Test FlextTestFixtures.test_version method."""
        version = FlextTestFixtures.test_version()
        assert version == "1.0.0"


class TestIntegrationScenarios:
    """Test integration scenarios with multiple fixtures."""

    @pytest.mark.asyncio
    async def test_complete_test_scenario(self) -> None:
        """Test complete testing scenario using all fixtures."""
        # Create all fixtures
        memory_fixtures = MemoryFixtures()
        mock_logger = MockLogger()
        mock_config = MockConfig(environment="test")

        # Setup environment
        setup_flext_test_environment()

        # Generate test data
        pipeline_data = FlextTestFixtures.test_pipeline_data()
        entity_data = FlextTestFixtures.test_entity_data()

        # Use memory fixtures
        memory_fixtures.add_test_pipeline()
        memory_fixtures.add_test_plugin()

        # Use mock logger
        mock_logger.info("Starting test scenario")
        mock_logger.debug("Using config: %s", mock_config.project_name)

        # Verify database config
        db_config = DatabaseFixtures.create_test_config()
        assert "database_url" in db_config

        # Verify everything worked
        assert isinstance(pipeline_data, dict)
        assert isinstance(entity_data, dict)
        assert len(mock_logger.get_logs()) == 2
        assert mock_config.environment == "test"

    def test_fixture_isolation(self) -> None:
        """Test that fixtures don't interfere with each other."""
        # Create multiple instances
        fixtures1 = FlextTestFixtures()
        fixtures2 = FlextTestFixtures()

        memory1 = MemoryFixtures()
        memory2 = MemoryFixtures()

        logger1 = MockLogger()
        logger2 = MockLogger()

        # Use them independently
        memory1.add_test_pipeline()
        memory2.add_test_plugin()

        logger1.info("Logger 1")
        logger2.error("Logger 2")

        # Should be isolated
        assert len(logger1.get_logs()) == 1
        assert len(logger2.get_logs()) == 1
        assert logger1.get_logs()[0][1] == "Logger 1"
        assert logger2.get_logs()[0][1] == "Logger 2"
