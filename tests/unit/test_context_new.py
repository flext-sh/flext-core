"""Test suite for context_new module."""

from flext_core.constants import FlextConstants
from flext_core.context_new import FlextContextConfig, FlextContextCore
from flext_core.context_new.headers import FlextContextHeaders


class TestFlextContextConfig:
    """Test suite for FlextContextConfig."""

    def test_default_config(self) -> None:
        """Test default context configuration."""
        config = FlextContextConfig()

        assert config.environment == FlextConstants.Config.ConfigEnvironment.DEVELOPMENT
        assert config.log_level == FlextConstants.Config.LogLevel.DEBUG
        assert config.enable_correlation_tracking is True
        assert config.enable_service_context is True
        assert config.enable_performance_tracking is True
        assert config.context_propagation_enabled is True
        assert config.max_context_depth == 20
        assert config.context_serialization_enabled is True
        assert config.context_cleanup_enabled is True
        assert config.enable_nested_contexts is True

    def test_config_with_values(self) -> None:
        """Test context configuration with values."""
        config = FlextContextConfig(
            environment=FlextConstants.Config.ConfigEnvironment.PRODUCTION,
            log_level=FlextConstants.Config.LogLevel.ERROR,
            enable_correlation_tracking=False,
            max_context_depth=10,
        )

        assert config.environment == FlextConstants.Config.ConfigEnvironment.PRODUCTION
        assert config.log_level == FlextConstants.Config.LogLevel.ERROR
        assert config.enable_correlation_tracking is False
        assert config.max_context_depth == 10
        # Other fields should have defaults
        assert config.enable_service_context is True

    def test_config_validation(self) -> None:
        """Test context configuration validation."""
        # Valid config
        config = FlextContextConfig(
            environment=FlextConstants.Config.ConfigEnvironment.STAGING,
            max_context_depth=5,
        )
        assert config.environment == FlextConstants.Config.ConfigEnvironment.STAGING
        assert config.max_context_depth == 5

    def test_config_model_dump(self) -> None:
        """Test dumping config to dict."""
        config = FlextContextConfig(
            environment=FlextConstants.Config.ConfigEnvironment.TEST,
            enable_nested_contexts=False,
        )

        data = config.model_dump(exclude_unset=True)
        assert data["environment"] == FlextConstants.Config.ConfigEnvironment.TEST
        assert data["enable_nested_contexts"] is False

        # All fields should be in full dump
        full_data = config.model_dump()
        assert "log_level" in full_data
        assert "enable_correlation_tracking" in full_data

    def test_config_model_dump_json(self) -> None:
        """Test dumping config to JSON."""
        config = FlextContextConfig(log_level=FlextConstants.Config.LogLevel.WARNING)

        json_str = config.model_dump_json()
        assert "WARNING" in json_str or "warning" in json_str


class TestFlextContextHeaders:
    """Test suite for FlextContextHeaders."""

    def test_header_constants(self) -> None:
        """Test header constant values."""
        assert FlextContextHeaders.CORRELATION_ID == "X-Correlation-Id"
        assert FlextContextHeaders.PARENT_CORRELATION_ID == "X-Parent-Correlation-Id"
        assert FlextContextHeaders.SERVICE_NAME == "X-Service-Name"
        assert FlextContextHeaders.USER_ID == "X-User-Id"


class TestFlextContextCore:
    """Test suite for FlextContextCore."""

    def test_default_config(self) -> None:
        """Test getting default configuration."""
        config = FlextContextCore.default_config()

        assert isinstance(config, FlextContextConfig)
        assert config.environment == FlextConstants.Config.ConfigEnvironment.DEVELOPMENT
        assert config.enable_correlation_tracking is True

    def test_validate_config_success(self) -> None:
        """Test successful config validation."""
        config_dict = {
            "environment": FlextConstants.Config.ConfigEnvironment.PRODUCTION,
            "log_level": FlextConstants.Config.LogLevel.INFO,
            "enable_correlation_tracking": True,
            "max_context_depth": 15,
        }

        result = FlextContextCore.validate_config(config_dict)

        assert result.success
        config = result.unwrap()
        assert config.environment == FlextConstants.Config.ConfigEnvironment.PRODUCTION
        assert config.log_level == FlextConstants.Config.LogLevel.INFO
        assert config.enable_correlation_tracking is True
        assert config.max_context_depth == 15

    def test_validate_config_with_defaults(self) -> None:
        """Test config validation with partial input."""
        config_dict = {"environment": FlextConstants.Config.ConfigEnvironment.STAGING}

        result = FlextContextCore.validate_config(config_dict)

        assert result.success
        config = result.unwrap()
        assert config.environment == FlextConstants.Config.ConfigEnvironment.STAGING
        # Defaults should be set
        assert config.log_level == FlextConstants.Config.LogLevel.DEBUG
        assert config.enable_correlation_tracking is True

    def test_validate_config_failure(self) -> None:
        """Test config validation failure."""
        config_dict = {
            "max_context_depth": -1,  # Invalid: must be >= 0
        }

        result = FlextContextCore.validate_config(config_dict)

        assert result.failure
        assert "Invalid context config" in str(result.error)

    def test_validate_config_with_invalid_type(self) -> None:
        """Test config validation with wrong types."""
        config_dict = {
            "enable_correlation_tracking": "not_a_bool",  # Wrong type
        }

        result = FlextContextCore.validate_config(config_dict)

        assert result.failure
        assert "Invalid context config" in str(result.error)

    def test_to_header_context(self) -> None:
        """Test converting context to headers."""
        context = {
            "correlation_id": "corr123",
            "parent_correlation_id": "parent456",
            "service_name": "test-service",
            "user_id": "user789",
            "extra_field": "ignored",  # Should be ignored
        }

        headers = FlextContextCore.to_header_context(context)

        assert headers["X-Correlation-Id"] == "corr123"
        assert headers["X-Parent-Correlation-Id"] == "parent456"
        assert headers["X-Service-Name"] == "test-service"
        assert headers["X-User-Id"] == "user789"
        assert "extra_field" not in headers

    def test_to_header_context_with_non_string_values(self) -> None:
        """Test converting context with non-string values."""
        context = {
            "correlation_id": 123,  # Non-string, should be ignored
            "service_name": "valid",
            "user_id": None,  # None, should be ignored
        }

        headers = FlextContextCore.to_header_context(context)

        assert "X-Correlation-Id" not in headers
        assert headers["X-Service-Name"] == "valid"
        assert "X-User-Id" not in headers

    def test_from_header_context(self) -> None:
        """Test converting headers to context."""
        headers = {
            "X-Correlation-Id": "corr123",
            "X-Parent-Correlation-Id": "parent456",
            "X-Service-Name": "test-service",
            "X-User-Id": "user789",
            "Other-Header": "ignored",  # Should be ignored
        }

        context = FlextContextCore.from_header_context(headers)

        assert context["correlation_id"] == "corr123"
        assert context["parent_correlation_id"] == "parent456"
        assert context["service_name"] == "test-service"
        assert context["user_id"] == "user789"
        assert "Other-Header" not in context

    def test_from_header_context_empty(self) -> None:
        """Test converting empty headers."""
        headers: dict[str, str] = {}
        context = FlextContextCore.from_header_context(headers)

        assert context == {}

    def test_from_header_context_partial(self) -> None:
        """Test converting partial headers."""
        headers = {"X-Correlation-Id": "corr123", "X-Service-Name": "service"}

        context = FlextContextCore.from_header_context(headers)

        assert context["correlation_id"] == "corr123"
        assert context["service_name"] == "service"
        assert "parent_correlation_id" not in context
        assert "user_id" not in context

    def test_round_trip_conversion(self) -> None:
        """Test round-trip conversion between context and headers."""
        original_context = {
            "correlation_id": "corr123",
            "parent_correlation_id": "parent456",
            "service_name": "test-service",
            "user_id": "user789",
        }

        # Convert to headers
        headers = FlextContextCore.to_header_context(original_context)

        # Convert back to context
        recovered_context = FlextContextCore.from_header_context(headers)

        # Should match original (for supported fields)
        assert recovered_context["correlation_id"] == original_context["correlation_id"]
        assert (
            recovered_context["parent_correlation_id"]
            == original_context["parent_correlation_id"]
        )
        assert recovered_context["service_name"] == original_context["service_name"]
        assert recovered_context["user_id"] == original_context["user_id"]

    def test_integration_flow(self) -> None:
        """Test complete integration flow."""
        # Create config
        config = FlextContextConfig(
            environment=FlextConstants.Config.ConfigEnvironment.PRODUCTION,
            log_level=FlextConstants.Config.LogLevel.WARNING,
            enable_correlation_tracking=True,
            max_context_depth=10,
        )

        # Convert to dict
        config_dict = config.model_dump()

        # Validate config
        validation_result = FlextContextCore.validate_config(config_dict)
        assert validation_result.success

        # Create context with correlation info
        context = {
            "correlation_id": "corr123",
            "service_name": "api-gateway",
            "user_id": "user456",
        }

        # Convert to headers
        headers = FlextContextCore.to_header_context(context)
        assert headers["X-Correlation-Id"] == "corr123"
        assert headers["X-Service-Name"] == "api-gateway"
        assert headers["X-User-Id"] == "user456"

        # Simulate receiving headers
        received_context = FlextContextCore.from_header_context(headers)

        # Check received context
        assert received_context["correlation_id"] == "corr123"
        assert received_context["service_name"] == "api-gateway"
        assert received_context["user_id"] == "user456"
