"""Comprehensive unit tests for domain value objects.

Adapted from flx-meltano-enterprise with proper imports.
"""

from __future__ import annotations

import pytest
from flx_core.domain.business_types import (
    EmailAddress,
    HostAddress,
    NetworkPort,
    Username,
)
from flx_core.domain.value_objects import (
    Duration,
    ExecutionId,
    ExecutionStatus,
    PipelineId,
    PipelineName,
    PluginId,
    PluginType,
)
from pydantic import ValidationError

# Python 3.13 type aliases
type TestResult = bool
type TestMessage = str

# Constants
SECONDS_IN_MINUTE = 60
SECONDS_IN_HOUR = 3600
SECONDS_IN_DAY = 86400
MILLISECONDS = 500
VALID_PORT = 5432
LOCALHOST = "localhost"
INVALID_PORT_HIGH = 70000
INVALID_PORT_LOW = 0


class TestDurationValueObject:
    """Test Duration value object."""

    def test_duration_creation_from_seconds(self) -> TestResult:
        """Test duration creation from seconds."""
        duration = Duration(seconds=SECONDS_IN_MINUTE)
        assert duration.total_seconds == SECONDS_IN_MINUTE
        return True

    def test_duration_creation_from_milliseconds(self) -> TestResult:
        """Test duration creation from milliseconds."""
        duration = Duration(milliseconds=MILLISECONDS)
        assert duration.total_seconds == 0.5
        return True

    def test_duration_creation_from_minutes(self) -> TestResult:
        """Test duration creation from minutes."""
        duration = Duration(minutes=1)
        assert duration.total_seconds == SECONDS_IN_MINUTE
        return True

    def test_duration_creation_from_hours(self) -> TestResult:
        """Test duration creation from hours."""
        duration = Duration(hours=1)
        assert duration.total_seconds == SECONDS_IN_HOUR
        return True

    def test_duration_creation_from_days(self) -> TestResult:
        """Test duration creation from days."""
        duration = Duration(days=1)
        assert duration.total_seconds == SECONDS_IN_DAY
        return True

    def test_duration_creation_mixed(self) -> TestResult:
        """Test duration creation with mixed units."""
        duration = Duration(minutes=1, seconds=30)
        assert duration.total_seconds == 90
        return True

    def test_duration_invalid_creation(self) -> TestResult:
        """Test that creating a duration with no arguments raises an error."""
        with pytest.raises(
            ValueError, match="Duration must be initialized with at least one unit"
        ):
            Duration()
        return True

    def test_duration_properties(self) -> TestResult:
        """Test duration properties."""
        duration = Duration(seconds=SECONDS_IN_DAY + SECONDS_IN_HOUR)
        assert duration.days == 1
        assert duration.hours == 1
        assert duration.minutes == 0
        assert duration.seconds == 0
        return True

    def test_duration_string_representation(self) -> TestResult:
        """Test the string representation of a duration."""
        duration = Duration(days=1, hours=2, minutes=3, seconds=4)
        assert str(duration) == "1d 2h 3m 4s"
        return True

    def test_duration_addition(self) -> TestResult:
        """Test duration addition."""
        duration1 = Duration(seconds=10)
        duration2 = Duration(minutes=1)
        result = duration1 + duration2
        assert result.total_seconds == 70
        return True

    def test_duration_comparison(self) -> TestResult:
        """Test duration comparison."""
        duration1 = Duration(seconds=60)
        duration2 = Duration(minutes=1)
        duration3 = Duration(seconds=30)

        assert duration1 == duration2
        assert duration1 > duration3
        assert duration3 < duration1
        assert duration1 >= duration2
        assert duration3 <= duration1
        return True


class TestExecutionStatus:
    """Test ExecutionStatus enum."""

    def test_execution_status_values(self) -> None:
        """Test execution status enum values."""
        assert ExecutionStatus.PENDING.value == "pending"
        assert ExecutionStatus.RUNNING.value == "running"
        assert ExecutionStatus.COMPLETED.value == "completed"
        assert ExecutionStatus.FAILED.value == "failed"
        assert ExecutionStatus.CANCELLED.value == "cancelled"

    def test_execution_status_is_terminal(self) -> None:
        """Test is_terminal property."""
        assert not ExecutionStatus.PENDING.is_terminal
        assert not ExecutionStatus.RUNNING.is_terminal
        assert ExecutionStatus.COMPLETED.is_terminal
        assert ExecutionStatus.FAILED.is_terminal
        assert ExecutionStatus.CANCELLED.is_terminal

    def test_execution_status_is_active(self) -> None:
        """Test is_active property."""
        assert not ExecutionStatus.PENDING.is_active
        assert ExecutionStatus.RUNNING.is_active
        assert not ExecutionStatus.COMPLETED.is_active
        assert not ExecutionStatus.FAILED.is_active
        assert not ExecutionStatus.CANCELLED.is_active


class TestIdentifiers:
    """Test identifier value objects."""

    def test_pipeline_id_creation(self) -> None:
        """Test PipelineId creation."""
        # Default creation
        pipeline_id1 = PipelineId()
        assert pipeline_id1.value is not None
        assert len(str(pipeline_id1.value)) == 36  # UUID length

        # Creation with specific value
        test_uuid = "12345678-1234-5678-1234-567812345678"
        pipeline_id2 = PipelineId(value=test_uuid)
        assert str(pipeline_id2.value) == test_uuid

    def test_execution_id_creation(self) -> None:
        """Test ExecutionId creation."""
        execution_id = ExecutionId()
        assert execution_id.value is not None
        assert len(str(execution_id.value)) == 36

    def test_plugin_id_creation(self) -> None:
        """Test PluginId creation."""
        plugin_id = PluginId()
        assert plugin_id.value is not None
        assert len(str(plugin_id.value)) == 36


class TestPipelineName:
    """Test PipelineName value object."""

    def test_pipeline_name_valid(self) -> None:
        """Test valid pipeline name creation."""
        name = PipelineName(value="valid_pipeline_name")
        assert name.value == "valid_pipeline_name"
        assert str(name) == "valid_pipeline_name"

    def test_pipeline_name_with_numbers(self) -> None:
        """Test pipeline name with numbers."""
        name = PipelineName(value="pipeline_123")
        assert name.value == "pipeline_123"

    def test_pipeline_name_with_hyphens(self) -> None:
        """Test pipeline name with hyphens."""
        name = PipelineName(value="my-pipeline-name")
        assert name.value == "my-pipeline-name"

    def test_pipeline_name_empty(self) -> None:
        """Test empty pipeline name raises validation error."""
        with pytest.raises(ValidationError):
            PipelineName(value="")

    def test_pipeline_name_too_long(self) -> None:
        """Test pipeline name exceeding max length."""
        with pytest.raises(ValidationError):
            PipelineName(value="a" * 256)

    def test_pipeline_name_invalid_characters(self) -> None:
        """Test pipeline name with invalid characters."""
        with pytest.raises(ValidationError):
            PipelineName(value="pipeline name with spaces")

        with pytest.raises(ValidationError):
            PipelineName(value="pipeline@name")

        with pytest.raises(ValidationError):
            PipelineName(value="pipeline#name")


class TestPluginType:
    """Test PluginType enum."""

    def test_plugin_type_values(self) -> None:
        """Test plugin type enum values."""
        assert PluginType.EXTRACTOR.value == "extractor"
        assert PluginType.LOADER.value == "loader"
        assert PluginType.TRANSFORMER.value == "transformer"
        assert PluginType.ORCHESTRATOR.value == "orchestrator"
        assert PluginType.UTILITY.value == "utility"

    def test_plugin_type_from_string(self) -> None:
        """Test creating plugin type from string."""
        assert PluginType("extractor") == PluginType.EXTRACTOR
        assert PluginType("loader") == PluginType.LOADER
        assert PluginType("transformer") == PluginType.TRANSFORMER


class TestBusinessTypes:
    """Test business type value objects."""

    def test_email_address_valid(self) -> None:
        """Test valid email address."""
        email = EmailAddress(value="test@example.com")
        assert email.value == "test@example.com"
        assert str(email) == "test@example.com"

    def test_email_address_invalid(self) -> None:
        """Test invalid email address."""
        with pytest.raises(ValidationError):
            EmailAddress(value="invalid-email")

        with pytest.raises(ValidationError):
            EmailAddress(value="@example.com")

        with pytest.raises(ValidationError):
            EmailAddress(value="test@")

    def test_username_valid(self) -> None:
        """Test valid username."""
        username = Username(value="valid_user123")
        assert username.value == "valid_user123"

    def test_username_invalid(self) -> None:
        """Test invalid username."""
        with pytest.raises(ValidationError):
            Username(value="ab")  # Too short

        with pytest.raises(ValidationError):
            Username(value="a" * 51)  # Too long

        with pytest.raises(ValidationError):
            Username(value="user@name")  # Invalid character

    def test_network_port_valid(self) -> None:
        """Test valid network port."""
        port = NetworkPort(value=VALID_PORT)
        assert port.value == VALID_PORT
        assert int(port) == VALID_PORT

    def test_network_port_invalid(self) -> None:
        """Test invalid network port."""
        with pytest.raises(ValidationError):
            NetworkPort(value=INVALID_PORT_LOW)

        with pytest.raises(ValidationError):
            NetworkPort(value=INVALID_PORT_HIGH)

    def test_host_address_valid(self) -> None:
        """Test valid host address."""
        host = HostAddress(value=LOCALHOST)
        assert host.value == LOCALHOST
        assert str(host) == LOCALHOST

        # IP address
        host_ip = HostAddress(value="192.168.1.1")
        assert host_ip.value == "192.168.1.1"

        # Domain name
        host_domain = HostAddress(value="example.com")
        assert host_domain.value == "example.com"
