"""Tests for enhanced BaseTap enterprise features.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import pytest

from flext_core.base.tap_base import BaseTap, TapMetrics, ValidationResult

if TYPE_CHECKING:
    from collections.abc import Iterator


class TestTapMetrics:
    """Test TapMetrics class."""

    def test_init_default_values(self) -> None:
        """Test TapMetrics initialization with default values."""
        metrics = TapMetrics()

        assert metrics.records_extracted == 0
        assert metrics.records_transformed == 0
        assert metrics.bytes_processed == 0
        assert metrics.execution_time == 0.0
        assert metrics.errors_count == 0
        assert isinstance(metrics.start_time, float)
        assert metrics.start_time > 0

    def test_add_record(self) -> None:
        """Test adding record to metrics."""
        metrics = TapMetrics()

        metrics.add_record(100)

        assert metrics.records_extracted == 1
        assert metrics.bytes_processed == 100

    def test_add_record_without_size(self) -> None:
        """Test adding record without size parameter."""
        metrics = TapMetrics()

        metrics.add_record()

        assert metrics.records_extracted == 1
        assert metrics.bytes_processed == 0

    def test_add_error(self) -> None:
        """Test adding error to metrics."""
        metrics = TapMetrics()

        metrics.add_error()

        assert metrics.errors_count == 1

    def test_finalize(self) -> None:
        """Test finalizing metrics calculation."""
        metrics = TapMetrics()

        # Simulate some work
        time.sleep(0.01)

        metrics.finalize()

        assert metrics.execution_time > 0
        # Check that execution time is reasonable (within expected bounds)
        expected_min_time = 0.01  # At least the sleep time
        expected_max_time = 0.1  # Reasonable upper bound
        assert expected_min_time <= metrics.execution_time <= expected_max_time


class TestValidationResult:
    """Test ValidationResult class."""

    def test_init_valid_default(self) -> None:
        """Test ValidationResult initialization as valid by default."""
        result = ValidationResult(is_valid=True)

        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_init_with_errors(self) -> None:
        """Test ValidationResult initialization with errors and warnings."""
        errors = ["Error 1", "Error 2"]
        warnings = ["Warning 1"]

        result = ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        assert result.is_valid is False
        assert result.errors == errors
        assert result.warnings == warnings


class MockTap(BaseTap):
    """Mock tap implementation for testing."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize mock tap."""
        self._required_config_fields = ["host", "port"]
        super().__init__(config)

    def _get_schema(self) -> dict[str, dict[str, Any]]:
        """Return mock schema."""
        return {
            "streams": {
                "users": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "name": {"type": "string"},
                    },
                },
                "orders": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "user_id": {"type": "integer"},
                    },
                },
            }
        }

    def _extract_data(self, config: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Return mock data."""
        yield {"id": 1, "name": "John"}
        yield {"id": 2, "name": "Jane"}


class TestEnhancedBaseTap:
    """Test enhanced BaseTap features."""

    @pytest.fixture
    def valid_config(self) -> dict[str, Any]:
        """Create valid tap configuration."""
        return {
            "name": "test_tap",
            "version": "1.0.0",
            "host": "localhost",
            "port": 5432,
        }

    @pytest.fixture
    def invalid_config(self) -> dict[str, Any]:
        """Create invalid tap configuration."""
        return {
            "name": "test_tap"
            # Missing required fields: host, port
        }

    @pytest.fixture
    def mock_tap(self, valid_config: dict[str, Any]) -> MockTap:
        """Create mock tap instance."""
        return MockTap(valid_config)

    def test_init_with_enhanced_features(self, valid_config: dict[str, Any]) -> None:
        """Test BaseTap initialization with enhanced features."""
        tap = MockTap(valid_config)

        assert tap.config == valid_config
        assert isinstance(tap.metrics, TapMetrics)
        assert tap.logger.name == "MockTap"

    def test_collect_metrics_initial_state(self, mock_tap: MockTap) -> None:
        """Test collect_metrics with initial state."""
        metrics = mock_tap.collect_metrics()

        assert metrics["records_extracted"] == 0
        assert metrics["records_transformed"] == 0
        assert metrics["bytes_processed"] == 0
        assert metrics["execution_time"] >= 0
        assert metrics["errors_count"] == 0
        assert metrics["records_per_second"] == 0
        assert metrics["bytes_per_second"] == 0

    def test_collect_metrics_with_data(self, mock_tap: MockTap) -> None:
        """Test collect_metrics after processing data."""
        # Simulate data processing
        mock_tap.metrics.add_record(100)
        mock_tap.metrics.add_record(200)
        mock_tap.metrics.add_error()

        # Wait a bit for execution time
        time.sleep(0.01)

        metrics = mock_tap.collect_metrics()

        assert metrics["records_extracted"] == 2
        assert metrics["bytes_processed"] == 300
        assert metrics["errors_count"] == 1
        assert metrics["execution_time"] > 0
        assert metrics["records_per_second"] > 0
        assert metrics["bytes_per_second"] > 0

    def test_validate_config_success(self, mock_tap: MockTap) -> None:
        """Test successful configuration validation."""
        result = mock_tap.validate_config()

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_validate_config_missing_required_fields(
        self, invalid_config: dict[str, Any]
    ) -> None:
        """Test configuration validation with missing required fields."""
        tap = MockTap(invalid_config)
        result = tap.validate_config()

        assert result.is_valid is False
        assert "Required field missing: host" in result.errors
        assert "Required field missing: port" in result.errors

    def test_validate_config_missing_optional_fields(self) -> None:
        """Test configuration validation with missing optional fields."""
        config = {"host": "localhost", "port": 5432}  # Missing name and version
        tap = MockTap(config)
        result = tap.validate_config()

        assert result.is_valid is True  # Still valid despite missing optional fields
        assert "Tap name not specified" in result.warnings
        assert "Tap version not specified" in result.warnings

    def test_health_check_healthy(self, mock_tap: MockTap) -> None:
        """Test health check with healthy tap."""
        result = mock_tap.health_check()

        assert result.is_success is True
        health_info = result.data
        assert health_info is not None
        assert health_info["status"] == "healthy"
        assert health_info["tap_name"] == "MockTap"
        assert health_info["config_valid"] is True
        assert health_info["schema_available"] is True
        assert health_info["stream_count"] == 2

    def test_health_check_unhealthy_config(
        self, invalid_config: dict[str, Any]
    ) -> None:
        """Test health check with invalid configuration."""
        tap = MockTap(invalid_config)
        result = tap.health_check()

        assert result.is_success is True  # Health check succeeds but reports unhealthy
        health_info = result.data
        assert health_info is not None
        assert health_info["status"] == "unhealthy"
        assert health_info["config_valid"] is False
        assert "config_errors" in health_info

    def test_get_tap_info(self, mock_tap: MockTap) -> None:
        """Test get_tap_info method."""
        info = mock_tap.get_tap_info()

        assert info["type"] == "MockTap"
        assert set(info["config_keys"]) == {"name", "version", "host", "port"}
        assert info["available_streams"] == ["users", "orders"]
        assert "metrics" in info
        assert info["capabilities"]["discovery"] is True
        assert info["capabilities"]["sync"] is True
        assert info["capabilities"]["metrics"] is True
        assert info["capabilities"]["health_check"] is True
        assert info["capabilities"]["validation"] is True

    def test_sync_with_metrics(self, mock_tap: MockTap) -> None:
        """Test sync method updates metrics correctly."""
        records = list(mock_tap.sync())

        assert len(records) == 2
        assert records[0] == {"id": 1, "name": "John"}
        assert records[1] == {"id": 2, "name": "Jane"}

        # Check metrics were updated
        assert mock_tap.metrics.records_extracted == 2
        assert mock_tap.metrics.bytes_processed > 0

    def test_sync_with_custom_config(self, mock_tap: MockTap) -> None:
        """Test sync method with custom configuration."""
        custom_config = {"custom": "value"}

        # This should work because MockTap doesn't validate custom config strictly
        records = list(mock_tap.sync(custom_config))

        assert len(records) == 2

    def test_get_stream_names(self, mock_tap: MockTap) -> None:
        """Test get_stream_names method."""
        stream_names = mock_tap.get_stream_names()

        assert set(stream_names) == {"users", "orders"}

    def test_get_stream_schema_existing(self, mock_tap: MockTap) -> None:
        """Test get_stream_schema for existing stream."""
        schema = mock_tap.get_stream_schema("users")

        assert schema is not None
        assert schema["type"] == "object"
        assert "id" in schema["properties"]
        assert "name" in schema["properties"]

    def test_get_stream_schema_nonexistent(self, mock_tap: MockTap) -> None:
        """Test get_stream_schema for non-existent stream."""
        schema = mock_tap.get_stream_schema("nonexistent")

        assert schema is None


class TestBaseTapErrorHandling:
    """Test BaseTap error handling scenarios."""

    class ErrorTap(BaseTap):
        """Tap that raises errors for testing."""

        def _get_schema(self) -> dict[str, dict[str, Any]]:
            """Raise an error."""
            raise ValueError("Schema error")

        def _extract_data(self, config: dict[str, Any]) -> Iterator[dict[str, Any]]:
            """Raise an error."""
            raise OSError("Data extraction error")

    def test_discover_with_value_error(self) -> None:
        """Test discover method with ValueError."""
        tap = self.ErrorTap({})
        result = tap.discover()
        assert result.is_success is False
        assert result.error is not None
        assert "Discovery failed: Schema error" in result.error

    def test_health_check_with_schema_error(self) -> None:
        """Test health check when schema discovery fails."""
        tap = self.ErrorTap({})
        result = tap.health_check()

        assert result.is_success is True
        health_info = result.data
        assert health_info is not None
        assert health_info["status"] == "degraded"
        assert health_info["schema_available"] is False
        assert "schema_error" in health_info

    def test_validate_config_with_exception(self) -> None:
        """Test validate_config when validation raises exception."""

        class BadTap(BaseTap):
            def __init__(self, config: dict[str, Any]) -> None:
                """Initialize without calling parent validation."""
                self.config = config
                self.metrics = TapMetrics()
                self.logger = logging.getLogger(self.__class__.__name__)
                # Skip parent validation that would raise exception

            def _validate_config(self) -> None:
                raise RuntimeError("Validation error")

            def _get_schema(self) -> dict[str, dict[str, Any]]:
                return {}

            def _extract_data(self, config: dict[str, Any]) -> Iterator[dict[str, Any]]:
                return iter([])

        tap = BadTap({})
        result = tap.validate_config()

        assert result.is_valid is False
        assert "Configuration validation failed: Validation error" in result.errors
