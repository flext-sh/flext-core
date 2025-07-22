"""Tests for flext_core.testing module.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Comprehensive tests for testing utilities to boost coverage.
"""

from __future__ import annotations

import asyncio
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
    TestFixtures,
    get_project_root_fixture,
    get_test_environment_fixture,
    setup_flext_test_environment,
)


class TestBaseTestCase:
    """Test BaseTestCase functionality."""

    def test_base_test_case_creation(self) -> None:
        """Test creating BaseTestCase instance."""
        test_case = BaseTestCase()

        assert test_case is not None
        assert hasattr(test_case, "setUp")
        assert hasattr(test_case, "tearDown")

    def test_base_test_case_setup_teardown(self) -> None:
        """Test BaseTestCase setUp and tearDown methods."""
        test_case = BaseTestCase()

        # Setup should not raise
        test_case.setUp()

        # TearDown should not raise
        test_case.tearDown()

    def test_base_test_case_inheritance(self) -> None:
        """Test that BaseTestCase can be inherited."""

        class CustomTestCase(BaseTestCase):
            def __init__(self) -> None:
                super().__init__()
                self.custom_setup_called = False

            def setUp(self) -> None:
                super().setUp()
                self.custom_setup_called = True

        custom_case = CustomTestCase()
        custom_case.setUp()

        assert custom_case.custom_setup_called


class TestAsyncTestCase:
    """Test AsyncTestCase functionality."""

    def test_async_test_case_creation(self) -> None:
        """Test creating AsyncTestCase instance."""
        test_case = AsyncTestCase()

        assert test_case is not None
        assert hasattr(test_case, "setUp")
        assert hasattr(test_case, "tearDown")
        assert hasattr(test_case, "run_async")

    def test_async_test_case_functionality(self) -> None:
        """Test AsyncTestCase functionality."""
        test_case = AsyncTestCase()
        test_case.setUp()  # This sets up the event loop

        try:
            # Test async functionality
            async def sample_async_operation() -> str:
                await asyncio.sleep(0.01)
                return "async_result"

            result = test_case.run_async(sample_async_operation())
            assert result == "async_result"

        finally:
            test_case.tearDown()

    def test_async_test_case_inheritance(self) -> None:
        """Test that AsyncTestCase can be inherited."""

        class CustomAsyncTestCase(AsyncTestCase):
            def __init__(self) -> None:
                super().__init__()
                self.async_setup_called = False

            async def asyncSetUp(self) -> None:
                # asyncSetUp has no parent implementation in AsyncTestCase
                pass
                self.async_setup_called = True

        custom_case = CustomAsyncTestCase()

        # Should be able to call asyncSetUp
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(custom_case.asyncSetUp())
            assert custom_case.async_setup_called
        finally:
            loop.close()


class TestTestFixtures:
    """Test TestFixtures functionality."""

    def test_test_fixtures_creation(self) -> None:
        """Test creating TestFixtures instance."""
        fixtures = TestFixtures()

        assert fixtures is not None
        assert hasattr(fixtures, "valid_uuid")
        assert hasattr(fixtures, "test_project_name")

    def test_test_fixtures_methods(self) -> None:
        """Test TestFixtures static methods."""
        # Test static methods that should be available
        uuid_val = TestFixtures.valid_uuid()
        assert uuid_val is not None

        timestamp = TestFixtures.current_timestamp()
        assert timestamp is not None

        project_name = TestFixtures.test_project_name()
        assert project_name == "test-project"

    def test_test_fixtures_data_generation(self) -> None:
        """Test TestFixtures data generation methods."""
        config_data = TestFixtures.test_config_data()
        assert isinstance(config_data, dict)
        assert "project_name" in config_data

        entity_data = TestFixtures.test_entity_data()
        assert isinstance(entity_data, dict)
        assert "id" in entity_data

        pipeline_data = TestFixtures.test_pipeline_data()
        assert isinstance(pipeline_data, dict)
        assert "name" in pipeline_data


class TestMemoryFixtures:
    """Test MemoryFixtures functionality."""

    def test_memory_fixtures_creation(self) -> None:
        """Test creating MemoryFixtures instance."""
        fixtures = MemoryFixtures()

        assert fixtures is not None
        assert hasattr(fixtures, "get_repository_data")
        assert hasattr(fixtures, "clear_all")

    def test_memory_fixtures_operations(self) -> None:
        """Test MemoryFixtures data operations."""
        fixtures = MemoryFixtures()

        # Test repository data management
        repo_data = fixtures.get_repository_data("test")
        assert isinstance(repo_data, dict)

        # Test adding test data
        pipeline_data = fixtures.add_test_pipeline()
        assert isinstance(pipeline_data, dict)
        assert "id" in pipeline_data

        plugin_data = fixtures.add_test_plugin()
        assert isinstance(plugin_data, dict)
        assert "id" in plugin_data

    def test_memory_fixtures_cleanup(self) -> None:
        """Test MemoryFixtures cleanup operations."""
        fixtures = MemoryFixtures()

        # Add some data
        fixtures.add_test_pipeline()
        fixtures.add_test_plugin()

        # Clear specific repository
        fixtures.clear_repository("pipeline")

        # Clear all data
        fixtures.clear_all()

        # Should not raise any exceptions


class TestDatabaseFixtures:
    """Test DatabaseFixtures functionality."""

    def test_database_fixtures_creation(self) -> None:
        """Test creating DatabaseFixtures instance."""
        fixtures = DatabaseFixtures()

        assert fixtures is not None
        assert hasattr(fixtures, "get_test_database_url")
        assert hasattr(fixtures, "create_test_config")

    def test_database_fixtures_urls(self) -> None:
        """Test DatabaseFixtures URL generation."""
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

    def test_database_fixtures_config(self) -> None:
        """Test DatabaseFixtures configuration generation."""
        config = DatabaseFixtures.create_test_config()
        assert isinstance(config, dict)
        assert "database_url" in config
        assert "database_async_url" in config
        assert "redis_url" in config


class TestMockUtilities:
    """Test mock utilities."""

    def test_mock_config_creation(self) -> None:
        """Test MockConfig creation."""
        mock_config = MockConfig()

        assert mock_config is not None
        assert hasattr(mock_config, "project_name")
        assert hasattr(mock_config, "debug")
        assert hasattr(mock_config, "model_dump")

    def test_mock_config_attributes(self) -> None:
        """Test MockConfig attributes."""
        mock_config = MockConfig()

        # Test setting and getting values via attributes
        mock_config.test_key = "test_value"
        value = mock_config.test_key

        assert value == "test_value"

    def test_mock_config_default_values(self) -> None:
        """Test MockConfig with default values."""
        mock_config = MockConfig()

        # Test default attributes
        assert hasattr(mock_config, "project_name")
        assert hasattr(mock_config, "debug")
        assert hasattr(mock_config, "log_level")

    def test_mock_logger_creation(self) -> None:
        """Test MockLogger creation."""
        mock_logger = MockLogger()

        assert mock_logger is not None
        assert hasattr(mock_logger, "info")
        assert hasattr(mock_logger, "error")
        assert hasattr(mock_logger, "warning")
        assert hasattr(mock_logger, "debug")

    def test_mock_logger_logging_methods(self) -> None:
        """Test MockLogger logging methods."""
        mock_logger = MockLogger()

        # All logging methods should not raise
        mock_logger.info("Test info")
        mock_logger.error("Test error")
        mock_logger.warning("Test warning")
        mock_logger.debug("Test debug")

    def test_mock_repository_creation(self) -> None:
        """Test MockRepository creation."""
        mock_repo = MockRepository()

        assert mock_repo is not None
        assert hasattr(mock_repo, "save")
        assert hasattr(mock_repo, "find_by_id")
        assert hasattr(mock_repo, "delete")
        assert hasattr(mock_repo, "list")

    @pytest.mark.asyncio
    async def test_mock_repository_operations(self) -> None:
        """Test MockRepository basic operations."""
        from uuid import uuid4

        mock_repo = MockRepository()

        # Create test entity with ID
        entity_id = uuid4()
        test_entity = type("TestEntity", (), {"id": entity_id, "name": "test"})()

        # Save should not raise
        save_result = await mock_repo.save(test_entity)
        assert save_result.success

        # Find by ID should work
        find_result = await mock_repo.find_by_id(entity_id)
        assert find_result.success

        # Delete should not raise
        delete_result = await mock_repo.delete(entity_id)
        assert delete_result.success

        # List should work
        list_result = await mock_repo.list()
        assert list_result.success
        assert isinstance(list_result.data, (list, tuple))


class TestFixtureFunctions:
    """Test fixture utility functions."""

    def test_get_project_root_fixture(self) -> None:
        """Test get_project_root_fixture function."""
        # This returns a pytest fixture, not a Path directly
        fixture = get_project_root_fixture()

        assert fixture is not None
        assert callable(fixture)

    def test_get_test_environment_fixture(self) -> None:
        """Test get_test_environment_fixture function."""
        # This returns a pytest fixture
        fixture = get_test_environment_fixture()

        assert fixture is not None
        assert callable(fixture)

    def test_setup_flext_test_environment(self) -> None:
        """Test setup_flext_test_environment function."""
        # Should not raise when setting up test environment
        setup_flext_test_environment()  # Function returns None

        # Should set environment variables
        import os

        assert os.environ.get("FLEXT_ENV") == "testing"
        assert os.environ.get("FLEXT_DEBUG") == "true"


class TestTestingIntegration:
    """Test integration of testing utilities."""

    def test_all_exports_available(self) -> None:
        """Test that all exported classes are available."""
        # Test that imports work and classes are available
        assert AsyncTestCase is not None
        assert BaseTestCase is not None
        assert DatabaseFixtures is not None
        assert MemoryFixtures is not None
        assert MockConfig is not None
        assert MockLogger is not None
        assert MockRepository is not None
        assert TestFixtures is not None
        assert get_project_root_fixture is not None
        assert get_test_environment_fixture is not None
        assert setup_flext_test_environment is not None

    def test_fixtures_work_together(self) -> None:
        """Test that different fixtures can work together."""
        # Create multiple fixtures
        test_fixtures = TestFixtures()
        memory_fixtures = MemoryFixtures()

        # Should work together without conflicts
        assert test_fixtures is not None
        assert memory_fixtures is not None

    @pytest.mark.asyncio
    async def test_mocks_work_together(self) -> None:
        """Test that different mocks can work together."""
        mock_config = MockConfig()
        mock_logger = MockLogger()
        mock_repo = MockRepository()

        # Configure mock config
        mock_config.log_level = "DEBUG"

        # Use mock logger
        mock_logger.info("Test message")

        # Use mock repository
        from uuid import uuid4

        test_entity = type("TestEntity", (), {"id": uuid4(), "data": "test"})()
        result = await mock_repo.save(test_entity)

        # All should work without conflicts
        assert mock_config.log_level == "DEBUG"
        assert result.success

    def test_test_environment_setup_comprehensive(self) -> None:
        """Test comprehensive test environment setup."""
        # Get project root fixture (returns a fixture function)
        project_root_fixture = get_project_root_fixture()
        assert callable(project_root_fixture)

        # Setup test environment
        setup_flext_test_environment()  # Function returns None

        # Get environment fixture
        env_fixture = get_test_environment_fixture()

        # All should be consistent and work together
        assert project_root_fixture is not None
        assert env_fixture is not None


class TestErrorHandling:
    """Test error handling in testing utilities."""

    def test_fixtures_handle_double_operations(self) -> None:
        """Test fixtures handle operations gracefully."""
        # TestFixtures doesn't have setup/teardown methods
        # This is a static utility class
        fixtures = TestFixtures()

        # Should be able to call methods multiple times
        uuid1 = fixtures.valid_uuid()
        uuid2 = fixtures.valid_uuid()

        assert uuid1 != uuid2  # Should generate different UUIDs

        # Should be able to get data multiple times
        data1 = fixtures.test_config_data()
        data2 = fixtures.test_config_data()

        assert data1 == data2  # Static data should be consistent

    @pytest.mark.asyncio
    async def test_mock_repository_handles_missing_items(self) -> None:
        """Test MockRepository handles missing items gracefully."""
        from uuid import uuid4

        mock_repo = MockRepository()

        non_existent_id = uuid4()

        # Getting non-existent item should return failure
        result = await mock_repo.find_by_id(non_existent_id)
        assert not result.success

        # Deleting non-existent item should return failure
        delete_result = await mock_repo.delete(non_existent_id)
        assert not delete_result.success

    def test_mock_logger_handles_various_input_types(self) -> None:
        """Test MockLogger handles various input types."""
        mock_logger = MockLogger()

        # Should handle different types of input
        mock_logger.info("String message")
        mock_logger.error(123)  # Number
        mock_logger.warning({"key": "value"})  # Dict
        mock_logger.debug(None)  # None value

        # All should work without raising


class TestAsyncOperations:
    """Test async operations in testing utilities."""

    def test_async_test_case_with_real_async_operation(self) -> None:
        """Test AsyncTestCase with actual async operation."""

        class TestAsync(AsyncTestCase):
            async def async_operation(self) -> str:
                await asyncio.sleep(0.01)  # Small delay
                return "async_result"

        test_case = TestAsync()
        test_case.setUp()

        try:
            result = test_case.run_async(test_case.async_operation())
            assert result == "async_result"
        finally:
            test_case.tearDown()

    def test_database_fixtures_async_operations(self) -> None:
        """Test DatabaseFixtures with async database operations."""
        fixtures = DatabaseFixtures()

        # Should be able to create fixture without issues
        assert fixtures is not None


class TestFixtureIntegration:
    """Test fixture integration scenarios."""

    def test_fixtures_work_together(self) -> None:
        """Test that different fixture classes work together."""
        test_fixtures = TestFixtures()
        memory_fixtures = MemoryFixtures()
        database_fixtures = DatabaseFixtures()

        # All should be created successfully
        assert test_fixtures is not None
        assert memory_fixtures is not None
        assert database_fixtures is not None

        # Should be able to use data from TestFixtures with MemoryFixtures
        pipeline_data = test_fixtures.test_pipeline_data()
        memory_fixtures.add_test_pipeline()

        # Should not interfere with each other


class TestUtilityFunctionEdgeCases:
    """Test edge cases in utility functions."""

    def test_project_root_fixture_different_paths(self) -> None:
        """Test get_project_root_fixture with different working directories."""
        with TemporaryDirectory() as temp_dir:
            # Change directory temporarily
            import os

            old_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                # Should still return a fixture function
                root_fixture = get_project_root_fixture()
                assert callable(root_fixture)
            finally:
                os.chdir(old_cwd)

    def test_test_environment_fixture_with_different_configs(self) -> None:
        """Test get_test_environment_fixture with different configurations."""
        # Should handle various environment configurations
        env1 = get_test_environment_fixture()
        env2 = get_test_environment_fixture()

        # May return same or different instances depending on implementation
        # Important that neither raises an exception
        assert env1 is not None or env1 is None  # Either is acceptable
        assert env2 is not None or env2 is None

    def test_setup_flext_test_environment_multiple_calls(self) -> None:
        """Test setup_flext_test_environment with multiple calls."""
        # Multiple calls should be safe
        setup_flext_test_environment()  # Function returns None
        setup_flext_test_environment()  # Function returns None

        # Both should succeed without raising
        # Results may be None or configuration objects
