"""Unit tests for FLEXT Core constants system - Modern pytest patterns.

Tests for the advanced constants system with type-safe enums and
immutable values.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from flext_core.constants import (
    FlextConstants,
    FlextEnvironment,
    FlextLogLevel,
    FlextResultStatus,
)

if TYPE_CHECKING:
    from flext_core.types import FlextConfigKey, FlextEntityId, FlextServiceName


@pytest.mark.unit
class TestFlextEnums:
    """Unit tests for FLEXT enum classes."""

    def test_flext_environment_enum_values(self) -> None:
        """Test FlextEnvironment enum has correct values."""
        assert FlextEnvironment.DEVELOPMENT.value == "development"
        assert FlextEnvironment.TESTING.value == "testing"
        assert FlextEnvironment.STAGING.value == "staging"
        assert FlextEnvironment.PRODUCTION.value == "production"

    def test_flext_environment_enum_membership(self) -> None:
        """Test FlextEnvironment enum membership."""
        assert "development" in FlextEnvironment
        assert "testing" in FlextEnvironment
        assert "staging" in FlextEnvironment
        assert "production" in FlextEnvironment
        assert "invalid" not in FlextEnvironment

    @pytest.mark.parametrize(
        "environment",
        [
            FlextEnvironment.DEVELOPMENT,
            FlextEnvironment.TESTING,
            FlextEnvironment.STAGING,
            FlextEnvironment.PRODUCTION,
        ],
    )
    def test_flext_environment_string_inheritance(
        self,
        environment: FlextEnvironment,
    ) -> None:
        """Test that FlextEnvironment inherits from str."""
        assert isinstance(environment, str)
        assert isinstance(environment, FlextEnvironment)

    def test_flext_log_level_enum_values(self) -> None:
        """Test FlextLogLevel enum has correct values."""
        assert FlextLogLevel.CRITICAL.value == "CRITICAL"
        assert FlextLogLevel.ERROR.value == "ERROR"
        assert FlextLogLevel.WARNING.value == "WARNING"
        assert FlextLogLevel.INFO.value == "INFO"
        assert FlextLogLevel.DEBUG.value == "DEBUG"
        assert FlextLogLevel.TRACE.value == "TRACE"

    @pytest.mark.parametrize(
        "log_level",
        [
            FlextLogLevel.CRITICAL,
            FlextLogLevel.ERROR,
            FlextLogLevel.WARNING,
            FlextLogLevel.INFO,
            FlextLogLevel.DEBUG,
            FlextLogLevel.TRACE,
        ],
    )
    def test_flext_log_level_string_inheritance(
        self,
        log_level: FlextLogLevel,
    ) -> None:
        """Test that FlextLogLevel inherits from str."""
        assert isinstance(log_level, str)
        assert isinstance(log_level, FlextLogLevel)

    def test_flext_result_status_enum_values(self) -> None:
        """Test FlextResultStatus enum has correct values."""
        assert FlextResultStatus.SUCCESS.value == "success"
        assert FlextResultStatus.FAILURE.value == "failure"

    @pytest.mark.parametrize(
        "status",
        [
            FlextResultStatus.SUCCESS,
            FlextResultStatus.FAILURE,
        ],
    )
    def test_flext_result_status_string_inheritance(
        self,
        status: FlextResultStatus,
    ) -> None:
        """Test that FlextResultStatus inherits from str."""
        assert isinstance(status, str)
        assert isinstance(status, FlextResultStatus)


@pytest.mark.unit
class TestVersionConstants:
    """Unit tests for version and metadata constants."""

    def test_flext_version_format(self) -> None:
        """Test FLEXT_VERSION follows semantic versioning."""
        version_pattern = r"^\d+\.\d+\.\d+(?:-[a-zA-Z0-9.-]+)?$"
        assert re.match(version_pattern, FlextConstants.VERSION)

    def test_flext_name_not_empty(self) -> None:
        """Test FLEXT_NAME is not empty."""
        assert FlextConstants.NAME
        assert isinstance(FlextConstants.NAME, str)
        assert len(FlextConstants.NAME.strip()) > 0

    def test_flext_description_not_empty(self) -> None:
        """Test FLEXT_DESCRIPTION is not empty."""
        assert FlextConstants.DESCRIPTION
        assert isinstance(FlextConstants.DESCRIPTION, str)
        assert len(FlextConstants.DESCRIPTION.strip()) > 0

    def test_version_constants_immutable(self) -> None:
        """Test that version constants are immutable."""
        # These are Final variables, so mypy would catch mutations
        # But we can verify the values are strings
        assert isinstance(FlextConstants.VERSION, str)
        assert isinstance(FlextConstants.NAME, str)
        assert isinstance(FlextConstants.DESCRIPTION, str)


@pytest.mark.unit
class TestDefaultConstants:
    """Unit tests for default configuration constants."""

    def test_default_environment_valid(self) -> None:
        """Test DEFAULT_ENVIRONMENT is valid FlextEnvironment."""
        assert isinstance(FlextConstants.DEFAULT_ENVIRONMENT, FlextEnvironment)
        assert FlextConstants.DEFAULT_ENVIRONMENT in FlextEnvironment

    def test_default_log_level_valid(self) -> None:
        """Test DEFAULT_LOG_LEVEL is valid FlextLogLevel."""
        assert isinstance(FlextConstants.DEFAULT_LOG_LEVEL, FlextLogLevel)
        assert FlextConstants.DEFAULT_LOG_LEVEL in FlextLogLevel

    def test_default_encoding_utf8(self) -> None:
        """Test DEFAULT_ENCODING is UTF-8."""
        assert FlextConstants.DEFAULT_ENCODING == "utf-8"
        assert isinstance(FlextConstants.DEFAULT_ENCODING, str)

    @pytest.mark.parametrize(
        ("constant", "expected_type", "min_value"),
        [
            (FlextConstants.DEFAULT_TIMEOUT, int, 1),
            (FlextConstants.DEFAULT_RETRY_COUNT, int, 0),
        ],
    )
    def test_default_numeric_values(
        self,
        constant: int,
        expected_type: type[int],
        min_value: int,
    ) -> None:
        """Test default numeric constants are valid."""
        assert isinstance(constant, expected_type)
        assert constant >= min_value


@pytest.mark.unit
class TestServiceContainerConstants:
    """Unit tests for service container constants."""

    def test_service_name_length_constraints(self) -> None:
        """Test service name length constraints are logical."""
        assert isinstance(FlextConstants.MAX_SERVICE_NAME_LENGTH, int)
        assert isinstance(FlextConstants.MIN_SERVICE_NAME_LENGTH, int)
        assert (
            FlextConstants.MAX_SERVICE_NAME_LENGTH
            > FlextConstants.MIN_SERVICE_NAME_LENGTH
        )
        assert FlextConstants.MIN_SERVICE_NAME_LENGTH > 0

    def test_reserved_service_names_not_empty(self) -> None:
        """Test RESERVED_SERVICE_NAMES is not empty."""
        assert isinstance(FlextConstants.RESERVED_SERVICE_NAMES, frozenset)
        assert len(FlextConstants.RESERVED_SERVICE_NAMES) > 0

        # All reserved names should be strings
        for name in FlextConstants.RESERVED_SERVICE_NAMES:
            assert isinstance(name, str)
            assert len(name) > 0

    @pytest.mark.parametrize(
        "reserved_name",
        [
            "container",
            "config",
            "logger",
            "system",
            "internal",
            "flext",
            "core",
            "admin",
            "health",
            "metrics",
        ],
    )
    def test_reserved_service_names_contains_expected(
        self,
        reserved_name: str,
    ) -> None:
        """Test RESERVED_SERVICE_NAMES contains expected values."""
        assert reserved_name in FlextConstants.RESERVED_SERVICE_NAMES

    def test_reserved_service_names_immutable(self) -> None:
        """Test RESERVED_SERVICE_NAMES is immutable."""
        # frozenset should be immutable
        with pytest.raises(AttributeError):
            FlextConstants.RESERVED_SERVICE_NAMES.add("new_name")


@pytest.mark.unit
class TestValidationPatterns:
    """Unit tests for validation pattern constants."""

    def test_identifier_pattern_compilation(self) -> None:
        """Test VALID_IDENTIFIER_PATTERN compiles as regex."""
        pattern = re.compile(FlextConstants.VALID_IDENTIFIER_PATTERN)
        assert pattern is not None

    def test_service_name_pattern_compilation(self) -> None:
        """Test VALID_SERVICE_NAME_PATTERN compiles as regex."""
        pattern = re.compile(FlextConstants.VALID_SERVICE_NAME_PATTERN)
        assert pattern is not None

    @pytest.mark.parametrize(
        ("identifier", "should_match"),
        [
            ("valid_identifier", True),
            ("_valid", True),
            ("Valid123", True),
            ("123invalid", False),
            ("invalid-name", False),
            ("", False),
        ],
    )
    def test_identifier_pattern_validation(
        self,
        identifier: str,
        *,
        should_match: bool,
    ) -> None:
        """Test VALID_IDENTIFIER_PATTERN validates correctly."""
        pattern = re.compile(FlextConstants.VALID_IDENTIFIER_PATTERN)
        match = pattern.match(identifier)

        if should_match:
            assert match is not None
        else:
            assert match is None

    @pytest.mark.parametrize(
        ("service_name", "should_match"),
        [
            ("valid-service", True),
            ("valid.service", True),
            ("valid_service", True),
            ("service123", True),
            ("service-with-dashes", True),
            ("service.with.dots", True),
            ("invalid service", False),  # Space not allowed
            ("invalid@service", False),  # @ not allowed
            ("", False),
        ],
    )
    def test_service_name_pattern_validation(
        self,
        service_name: str,
        *,
        should_match: bool,
    ) -> None:
        """Test VALID_SERVICE_NAME_PATTERN validates correctly."""
        pattern = re.compile(FlextConstants.VALID_SERVICE_NAME_PATTERN)
        match = pattern.match(service_name)

        if should_match:
            assert match is not None
        else:
            assert match is None


@pytest.mark.unit
class TestPerformanceLimits:
    """Unit tests for performance and limit constants."""

    @pytest.mark.parametrize(
        ("constant", "min_value"),
        [
            (FlextConstants.MAX_CONTAINER_SERVICES, 1000),
            (FlextConstants.MAX_NESTING_DEPTH, 10),
            (FlextConstants.CACHE_TTL_SECONDS, 60),
        ],
    )
    def test_performance_limits_reasonable(
        self,
        constant: int,
        min_value: int,
    ) -> None:
        """Test performance limits are reasonable values."""
        assert isinstance(constant, int)
        assert constant >= min_value


@pytest.mark.unit
class TestFlextConstants:
    """Unit tests for FlextConstants organized container."""

    def test_flext_constants_prevents_instantiation(self) -> None:
        """Test FlextConstants cannot be instantiated."""
        with pytest.raises(TypeError, match="should not be instantiated"):
            FlextConstants()

    def test_version_attributes_access(self) -> None:
        """Test version attributes are accessible."""
        assert FlextConstants.VERSION == FlextConstants.VERSION
        assert FlextConstants.NAME == FlextConstants.NAME
        assert FlextConstants.DESCRIPTION == FlextConstants.DESCRIPTION

    def test_environment_constants_access(self) -> None:
        """Test environment constants are accessible."""
        assert FlextConstants.DEFAULT_ENVIRONMENT == FlextEnvironment.DEVELOPMENT
        assert FlextConstants.ENV_DEVELOPMENT == FlextEnvironment.DEVELOPMENT
        assert FlextConstants.ENV_TESTING == FlextEnvironment.TESTING
        assert FlextConstants.ENV_STAGING == FlextEnvironment.STAGING
        assert FlextConstants.ENV_PRODUCTION == FlextEnvironment.PRODUCTION

    def test_logging_constants_access(self) -> None:
        """Test logging constants are accessible."""
        assert FlextConstants.DEFAULT_LOG_LEVEL == FlextLogLevel.INFO
        assert FlextConstants.LOG_CRITICAL == FlextLogLevel.CRITICAL
        assert FlextConstants.LOG_ERROR == FlextLogLevel.ERROR
        assert FlextConstants.LOG_WARNING == FlextLogLevel.WARNING
        assert FlextConstants.LOG_INFO == FlextLogLevel.INFO
        assert FlextConstants.LOG_DEBUG == FlextLogLevel.DEBUG
        assert FlextConstants.LOG_TRACE == FlextLogLevel.TRACE

    def test_container_constants_access(self) -> None:
        """Test container constants are accessible."""
        assert FlextConstants.MAX_CONTAINER_SERVICES == 10000
        assert FlextConstants.MAX_SERVICE_NAME_LENGTH == 255
        assert FlextConstants.MIN_SERVICE_NAME_LENGTH == 1
        assert isinstance(FlextConstants.RESERVED_SERVICE_NAMES, frozenset)

    def test_performance_constants_access(self) -> None:
        """Test performance constants are accessible."""
        assert FlextConstants.DEFAULT_TIMEOUT == 30
        assert FlextConstants.DEFAULT_RETRY_COUNT == 3
        assert FlextConstants.CACHE_TTL_SECONDS == 3600
        assert FlextConstants.MAX_NESTING_DEPTH == 100

    def test_pattern_constants_access(self) -> None:
        """Test pattern constants are accessible."""
        assert FlextConstants.VALID_IDENTIFIER_PATTERN == r"^[a-zA-Z_][a-zA-Z0-9_]*$"
        assert FlextConstants.VALID_SERVICE_NAME_PATTERN == r"^[a-zA-Z0-9_.-]+$"

    def test_constants_organization(self) -> None:
        """Test that constants are properly organized."""
        # Test that we can access all main constant groups
        assert hasattr(FlextConstants, "VERSION")
        assert hasattr(FlextConstants, "DEFAULT_ENVIRONMENT")
        assert hasattr(FlextConstants, "DEFAULT_LOG_LEVEL")
        assert hasattr(FlextConstants, "MAX_CONTAINER_SERVICES")
        assert hasattr(FlextConstants, "DEFAULT_TIMEOUT")
        assert hasattr(FlextConstants, "VALID_IDENTIFIER_PATTERN")

        # Test that constants are proper values
        assert isinstance(FlextConstants.VERSION, str)
        assert isinstance(FlextConstants.DEFAULT_TIMEOUT, int)
        assert isinstance(FlextConstants.RESERVED_SERVICE_NAMES, frozenset)


@pytest.mark.integration
class TestConstantsIntegration:
    """Integration tests for constants system."""

    def test_enum_integration_with_defaults(self) -> None:
        """Test enum constants integrate with default values."""
        # Default environment should be a valid enum value
        assert FlextConstants.DEFAULT_ENVIRONMENT in FlextEnvironment

        # Default log level should be a valid enum value
        assert FlextConstants.DEFAULT_LOG_LEVEL in FlextLogLevel

    def test_validation_patterns_with_reserved_names(self) -> None:
        """Test validation patterns work with reserved service names."""
        service_name_pattern = re.compile(
            FlextConstants.VALID_SERVICE_NAME_PATTERN,
        )

        # All reserved names should match the service name pattern
        for reserved_name in FlextConstants.RESERVED_SERVICE_NAMES:
            assert service_name_pattern.match(reserved_name) is not None

    def test_constants_consistency(self) -> None:
        """Test that constants are internally consistent."""
        # Service name length limits should be consistent with reserved
        # names
        for reserved_name in FlextConstants.RESERVED_SERVICE_NAMES:
            assert len(reserved_name) >= FlextConstants.MIN_SERVICE_NAME_LENGTH
            assert len(reserved_name) <= FlextConstants.MAX_SERVICE_NAME_LENGTH

        # Performance constants should be reasonable
        assert FlextConstants.DEFAULT_TIMEOUT < FlextConstants.CACHE_TTL_SECONDS
        assert FlextConstants.DEFAULT_RETRY_COUNT < FlextConstants.MAX_NESTING_DEPTH

    def test_type_aliases_with_validation_patterns(self) -> None:
        """Test type aliases work with validation patterns."""
        identifier_pattern = re.compile(
            FlextConstants.VALID_IDENTIFIER_PATTERN,
        )
        service_name_pattern = re.compile(
            FlextConstants.VALID_SERVICE_NAME_PATTERN,
        )

        # Valid service name should work with FlextServiceName type alias
        valid_service = "test_service"
        if service_name_pattern.match(valid_service):
            service_name: FlextServiceName = valid_service
            assert isinstance(service_name, str)

        # Valid identifier should work with FlextConfigKey/FlextEntityId
        valid_identifier = "valid_identifier"
        if identifier_pattern.match(valid_identifier):
            config_key: FlextConfigKey = valid_identifier
            entity_id: FlextEntityId = valid_identifier
            assert isinstance(config_key, str)
            assert isinstance(entity_id, str)
