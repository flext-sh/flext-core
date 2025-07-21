"""Tests for BaseTarget class.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from flext_core.base.target_base import BaseTarget
from flext_core.domain.types import ServiceResult


class MockTarget(BaseTarget):
    """Mock implementation of BaseTarget for testing."""

    def __init__(
        self,
        config: dict[str, Any],
        should_fail_write: bool = False,
        should_fail_flush: bool = False,
    ) -> None:
        """Initialize mock target."""
        self.should_fail_write = should_fail_write
        self.should_fail_flush = should_fail_flush
        self.written_records: list[dict[str, Any]] = []
        self.flush_called = False
        super().__init__(config)

    def _write_records(self, records: list[dict[str, Any]]) -> None:
        """Mock implementation of write records."""
        if self.should_fail_write:
            raise ValueError("Mock write failure")
        self.written_records.extend(records)

    def _flush_buffers(self) -> None:
        """Mock implementation of flush buffers."""
        if self.should_fail_flush:
            raise OSError("Mock flush failure")
        self.flush_called = True


class TestBaseTarget:
    """Test BaseTarget functionality."""

    def test_initialization_valid_config(self) -> None:
        """Test target initialization with valid configuration."""
        config = {"host": "localhost", "port": 8080}
        target = MockTarget(config)

        assert target.config == config

    def test_initialization_invalid_config_type(self) -> None:
        """Test target initialization with invalid configuration type."""
        with pytest.raises(TypeError, match="Config must be a dictionary"):
            MockTarget("not a dict")  # type: ignore[arg-type]

    def test_write_batch_success(self) -> None:
        """Test successful batch write."""
        config = {"test": "config"}
        target = MockTarget(config)
        records = [{"id": 1, "name": "test1"}, {"id": 2, "name": "test2"}]

        result = target.write_batch(records)

        assert result.is_success
        assert result.data is None
        assert target.written_records == records

    def test_write_batch_empty_list(self) -> None:
        """Test batch write with empty list."""
        config = {"test": "config"}
        target = MockTarget(config)
        records: list[dict[str, Any]] = []

        result = target.write_batch(records)

        assert result.is_success
        assert result.data is None
        assert target.written_records == []

    def test_write_batch_invalid_records_type(self) -> None:
        """Test batch write with invalid records type."""
        config = {"test": "config"}
        target = MockTarget(config)

        result = target.write_batch("not a list")  # type: ignore[arg-type]

        assert result.is_failure
        assert result.error is not None
        assert "Write validation failed" in result.error
        assert result.error is not None
        assert "Records must be a list" in result.error

    def test_write_batch_invalid_record_type(self) -> None:
        """Test batch write with invalid individual record type."""
        config = {"test": "config"}
        target = MockTarget(config)
        records = [
            {"id": 1},
            "invalid_record",
            {"id": 3},
        ]  # Invalid record type intentionally

        result = target.write_batch(records)  # type: ignore[arg-type]

        assert result.is_failure
        assert result.error is not None
        assert "Write validation failed" in result.error
        assert result.error is not None
        assert "Each record must be a dictionary" in result.error

    def test_write_batch_write_failure(self) -> None:
        """Test batch write with write implementation failure."""
        config = {"test": "config"}
        target = MockTarget(config, should_fail_write=True)
        records = [{"id": 1, "name": "test"}]

        result = target.write_batch(records)

        assert result.is_failure
        assert result.error is not None
        assert "Write validation failed" in result.error
        assert "Mock write failure" in result.error

    def test_write_batch_io_error(self) -> None:
        """Test batch write with I/O error."""
        config = {"test": "config"}
        target = MockTarget(config)
        records = [{"id": 1, "name": "test"}]

        # Mock the _write_records method to raise OSError
        original_write = target._write_records
        target._write_records = Mock(side_effect=OSError("Disk full"))  # type: ignore[method-assign]

        result = target.write_batch(records)

        assert result.is_failure
        assert result.error is not None
        assert "Write I/O error" in result.error
        assert "Disk full" in result.error

        # Restore original method
        target._write_records = original_write  # type: ignore[method-assign]

    def test_write_batch_unexpected_error(self) -> None:
        """Test batch write with unexpected error."""
        config = {"test": "config"}
        target = MockTarget(config)
        records = [{"id": 1, "name": "test"}]

        # Mock the _write_records method to raise unexpected error
        target._write_records = Mock(side_effect=RuntimeError("Unexpected error"))  # type: ignore[method-assign]

        result = target.write_batch(records)

        assert result.is_failure
        assert "Unexpected error during write" in result.error

    def test_write_record_success(self) -> None:
        """Test successful single record write."""
        config = {"test": "config"}
        target = MockTarget(config)
        record = {"id": 1, "name": "test"}

        result = target.write_record(record)

        assert result.is_success
        assert result.data is None
        assert target.written_records == [record]

    def test_write_record_failure(self) -> None:
        """Test single record write failure."""
        config = {"test": "config"}
        target = MockTarget(config, should_fail_write=True)
        record = {"id": 1, "name": "test"}

        result = target.write_record(record)

        assert result.is_failure
        assert result.error is not None
        assert "Write validation failed" in result.error
        assert "Mock write failure" in result.error

    def test_flush_success(self) -> None:
        """Test successful flush."""
        config = {"test": "config"}
        target = MockTarget(config)

        result = target.flush()

        assert result.is_success
        assert result.data is None
        assert target.flush_called is True

    def test_flush_value_error(self) -> None:
        """Test flush with ValueError."""
        config = {"test": "config"}
        target = MockTarget(config)

        # Mock the _flush_buffers method to raise ValueError
        target._flush_buffers = Mock(side_effect=ValueError("Invalid flush state"))  # type: ignore[method-assign]

        result = target.flush()

        assert result.is_failure
        assert result.error is not None
        assert "Flush failed" in result.error
        assert "Invalid flush state" in result.error

    def test_flush_os_error(self) -> None:
        """Test flush with OSError."""
        config = {"test": "config"}
        target = MockTarget(config, should_fail_flush=True)

        result = target.flush()

        assert result.is_failure
        assert result.error is not None
        assert "Flush failed" in result.error
        assert "Mock flush failure" in result.error

    def test_flush_unexpected_error(self) -> None:
        """Test flush with unexpected error."""
        config = {"test": "config"}
        target = MockTarget(config)

        # Mock the _flush_buffers method to raise unexpected error
        target._flush_buffers = Mock(side_effect=KeyError("Unexpected key error"))  # type: ignore[method-assign]

        result = target.flush()

        assert result.is_failure
        assert "Unexpected error during flush" in result.error

    def test_get_target_info(self) -> None:
        """Test getting target information."""
        config = {"host": "localhost", "port": 8080, "database": "test"}
        target = MockTarget(config)

        info = target.get_target_info()

        assert info["type"] == "MockTarget"
        assert set(info["config_keys"]) == {"host", "port", "database"}

    def test_get_target_info_empty_config(self) -> None:
        """Test getting target information with empty config."""
        config: dict[str, Any] = {}
        target = MockTarget(config)

        info = target.get_target_info()

        assert info["type"] == "MockTarget"
        assert info["config_keys"] == []

    def test_validate_records_with_mixed_types(self) -> None:
        """Test record validation with mixed valid/invalid types."""
        config = {"test": "config"}
        target = MockTarget(config)

        # This should work - all valid records
        valid_records = [{"id": 1}, {"id": 2}, {"id": 3}]
        result = target.write_batch(valid_records)
        assert result.is_success

        # This should fail - mixed types
        invalid_records = [{"id": 1}, None, {"id": 3}]
        result = target.write_batch(invalid_records)  # type: ignore[arg-type]
        assert result.is_failure
        assert result.error is not None
        assert "Each record must be a dictionary" in result.error

    def test_abstract_methods_not_implemented(self) -> None:
        """Test that abstract methods raise NotImplementedError."""
        # This test verifies the abstract methods themselves
        with pytest.raises(TypeError):
            BaseTarget({"test": "config"})  # type: ignore[abstract]


class TestBaseTargetErrorHandling:
    """Test BaseTarget error handling scenarios."""

    def test_config_validation_preserves_type_error(self) -> None:
        """Test that config validation preserves type error information."""
        with pytest.raises(TypeError, match="Config must be a dictionary"):
            MockTarget(123)  # type: ignore[arg-type]

    def test_record_validation_preserves_error_context(self) -> None:
        """Test that record validation preserves error context."""
        config = {"test": "config"}
        target = MockTarget(config)

        # Test with non-list input
        result = target.write_batch({"not": "list"})  # type: ignore[arg-type]
        assert result.is_failure
        assert result.error is not None
        assert "Records must be a list" in result.error

        # Test with invalid record in list
        result = target.write_batch([123])  # type: ignore[list-item]
        assert result.is_failure
        assert result.error is not None
        assert "Each record must be a dictionary" in result.error

    def test_service_result_error_propagation(self) -> None:
        """Test that ServiceResult properly propagates errors."""
        config = {"test": "config"}
        target = MockTarget(config, should_fail_write=True)

        result = target.write_batch([{"test": "record"}])

        assert result.is_failure
        assert result.data is None
        assert result.error is not None
        assert "Mock write failure" in result.error

    def test_exception_chain_preservation(self) -> None:
        """Test that errors are properly handled and logged."""
        config = {"test": "config"}
        target = MockTarget(config)

        # Mock to raise a chained exception
        target._write_records = Mock(side_effect=RuntimeError("Runtime error"))  # type: ignore[method-assign]

        result = target.write_batch([{"test": "record"}])

        assert result.is_failure
        assert "Unexpected error during write" in result.error
        assert "Runtime error" in result.error
