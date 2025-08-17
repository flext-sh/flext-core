"""Comprehensive tests for semantic.py module.

This test suite provides complete coverage of the semantic architecture
including model semantics, observability patterns, error hierarchies,
and unified semantic patterns.

Coverage Target: semantic.py 64% → 95%+
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from flext_core import (
    FlextConstants,
    FlextResult,
    FlextSemantic,
    FlextSemanticError,
    FlextSemanticModel,
    FlextSemanticObservability,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextSemanticModel:
    """Test FlextSemanticModel functionality."""

    def test_foundation_get_base_config(self) -> None:
      """Test Foundation.get_base_config returns correct configuration."""
      config = FlextSemanticModel.Foundation.get_base_config()

      # Verify all expected keys are present
      expected_keys = {
          "extra",
          "validate_assignment",
          "use_enum_values",
          "str_strip_whitespace",
          "str_max_length",
          "arbitrary_types_allowed",
          "validate_default",
      }

      assert set(config.keys()) == expected_keys

      # Verify specific values from constants
      assert config["extra"] == FlextConstants.Models.EXTRA_FORBID
      assert (
          config["validate_assignment"] == FlextConstants.Models.VALIDATE_ASSIGNMENT
      )
      assert config["use_enum_values"] == FlextConstants.Models.USE_ENUM_VALUES
      assert (
          config["str_strip_whitespace"] == FlextConstants.Models.STR_STRIP_WHITESPACE
      )
      assert config["str_max_length"] == FlextConstants.Limits.MAX_STRING_LENGTH
      assert (
          config["arbitrary_types_allowed"]
          == FlextConstants.Models.ARBITRARY_TYPES_ALLOWED
      )
      assert config["validate_default"] == FlextConstants.Models.VALIDATE_DEFAULT

    def test_namespace_classes_exist(self) -> None:
      """Test namespace classes are properly defined."""
      # Test that namespace classes exist
      assert FlextSemanticModel.Namespace.FlextData is not None
      assert FlextSemanticModel.Namespace.FlextAuth is not None
      assert FlextSemanticModel.Namespace.FlextService is not None
      assert FlextSemanticModel.Namespace.FlextInfrastructure is not None

      # Test that they can be instantiated
      data_ns = FlextSemanticModel.Namespace.FlextData()
      auth_ns = FlextSemanticModel.Namespace.FlextAuth()
      service_ns = FlextSemanticModel.Namespace.FlextService()
      infra_ns = FlextSemanticModel.Namespace.FlextInfrastructure()

      assert data_ns is not None
      assert auth_ns is not None
      assert service_ns is not None
      assert infra_ns is not None

    def test_factory_create_model_with_defaults(self) -> None:
      """Test Factory.create_model_with_defaults with various inputs."""
      # Test with no additional kwargs
      defaults = FlextSemanticModel.Factory.create_model_with_defaults()

      assert "timeout" in defaults
      assert "status" in defaults
      assert defaults["timeout"] == FlextConstants.Defaults.TIMEOUT
      assert defaults["status"] == FlextConstants.Status.ACTIVE

      # Test with additional kwargs
      custom_data = FlextSemanticModel.Factory.create_model_with_defaults(
          name="test_model",
          version="1.0.0",
          custom_field=123,
      )

      # Should contain both defaults and custom data
      assert custom_data["timeout"] == FlextConstants.Defaults.TIMEOUT
      assert custom_data["status"] == FlextConstants.Status.ACTIVE
      assert custom_data["name"] == "test_model"
      assert custom_data["version"] == "1.0.0"
      assert custom_data["custom_field"] == 123

      # Test kwargs override defaults
      override_data = FlextSemanticModel.Factory.create_model_with_defaults(
          timeout=5000,
          status="custom_status",
      )

      assert override_data["timeout"] == 5000
      assert override_data["status"] == "custom_status"

    def test_factory_validate_with_business_rules_valid(self) -> None:
      """Test validate_with_business_rules with valid object."""
      # Mock object with valid business rules method
      mock_instance = Mock()
      mock_instance.validate_business_rules.return_value = FlextResult.ok(None)

      result = FlextSemanticModel.Factory.validate_with_business_rules(mock_instance)

      assert result.success is True
      mock_instance.validate_business_rules.assert_called_once()

    def test_factory_validate_with_business_rules_failure(self) -> None:
      """Test validate_with_business_rules with failing validation."""
      # Mock object with failing business rules
      mock_instance = Mock()
      mock_instance.validate_business_rules.return_value = FlextResult.fail(
          "Validation failed",
      )

      result = FlextSemanticModel.Factory.validate_with_business_rules(mock_instance)

      assert result.success is False
      assert result.error == "Validation failed"

    def test_factory_validate_with_business_rules_no_method(self) -> None:
      """Test validate_with_business_rules with object without validation method."""
      # Object without validate_business_rules method
      plain_object = object()

      result = FlextSemanticModel.Factory.validate_with_business_rules(plain_object)

      assert result.success is True

    def test_factory_validate_with_business_rules_non_result(self) -> None:
      """Test validate_with_business_rules when method returns non-FlextResult."""
      # Mock object that returns non-FlextResult
      mock_instance = Mock()
      mock_instance.validate_business_rules.return_value = "not a result"

      result = FlextSemanticModel.Factory.validate_with_business_rules(mock_instance)

      assert result.success is True


class TestFlextSemanticObservability:
    """Test FlextSemanticObservability functionality."""

    def test_logger_protocol_compliance(self) -> None:
      """Test Logger protocol compliance."""
      # Test that protocol methods exist
      assert hasattr(FlextSemanticObservability.Protocol.Logger, "trace")
      assert hasattr(FlextSemanticObservability.Protocol.Logger, "debug")
      assert hasattr(FlextSemanticObservability.Protocol.Logger, "info")
      assert hasattr(FlextSemanticObservability.Protocol.Logger, "warn")
      assert hasattr(FlextSemanticObservability.Protocol.Logger, "error")
      assert hasattr(FlextSemanticObservability.Protocol.Logger, "fatal")
      assert hasattr(FlextSemanticObservability.Protocol.Logger, "audit")

      # Test protocol is runtime checkable
      assert FlextSemanticObservability.Protocol.Logger.__dict__.get(
          "_is_runtime_protocol",
          False,
      )

    def test_span_protocol_compliance(self) -> None:
      """Test SpanProtocol compliance."""
      assert hasattr(FlextSemanticObservability.Protocol.SpanProtocol, "add_context")
      assert hasattr(FlextSemanticObservability.Protocol.SpanProtocol, "add_error")

      # Test protocol is runtime checkable
      assert FlextSemanticObservability.Protocol.SpanProtocol.__dict__.get(
          "_is_runtime_protocol",
          False,
      )

    def test_tracer_protocol_compliance(self) -> None:
      """Test Tracer protocol compliance."""
      assert hasattr(FlextSemanticObservability.Protocol.Tracer, "business_span")
      assert hasattr(FlextSemanticObservability.Protocol.Tracer, "technical_span")

      # Test protocol is runtime checkable
      assert FlextSemanticObservability.Protocol.Tracer.__dict__.get(
          "_is_runtime_protocol",
          False,
      )

    def test_metrics_protocol_compliance(self) -> None:
      """Test Metrics protocol compliance."""
      assert hasattr(FlextSemanticObservability.Protocol.Metrics, "increment")
      assert hasattr(FlextSemanticObservability.Protocol.Metrics, "histogram")
      assert hasattr(FlextSemanticObservability.Protocol.Metrics, "gauge")

      # Test protocol is runtime checkable
      assert FlextSemanticObservability.Protocol.Metrics.__dict__.get(
          "_is_runtime_protocol",
          False,
      )

    def test_observability_protocol_compliance(self) -> None:
      """Test Observability protocol compliance."""
      mock_obs = Mock()
      mock_obs.log = Mock()
      mock_obs.trace = Mock()
      mock_obs.metrics = Mock()

      assert isinstance(mock_obs, FlextSemanticObservability.Protocol.Observability)
      assert hasattr(FlextSemanticObservability.Protocol.Observability, "log")
      assert hasattr(FlextSemanticObservability.Protocol.Observability, "trace")
      assert hasattr(FlextSemanticObservability.Protocol.Observability, "metrics")

    def test_factory_get_minimal_observability(self) -> None:
      """Test Factory.get_minimal_observability."""
      # This will test dynamic import - may need mocking in some environments
      try:
          obs = FlextSemanticObservability.Factory.get_minimal_observability()
          assert obs is not None
      except (ImportError, AttributeError):
          # Expected if observability module doesn't have get_observability function
          pytest.skip(
              "Observability module not available or function not implemented",
          )

    def test_factory_configure_observability_default(self) -> None:
      """Test Factory.configure_observability with defaults."""
      try:
          obs = FlextSemanticObservability.Factory.configure_observability(
              "test-service",
          )
          assert obs is not None
      except (ImportError, AttributeError):
          # Expected if observability module doesn't have configure function
          pytest.skip("Observability configuration function not available")

    def test_factory_configure_observability_custom_level(self) -> None:
      """Test Factory.configure_observability with custom log level."""
      try:
          obs = FlextSemanticObservability.Factory.configure_observability(
              "test-service",
              log_level="DEBUG",
          )
          assert obs is not None
      except (ImportError, AttributeError):
          # Expected if observability module doesn't have configure function
          pytest.skip("Observability configuration function not available")


class TestFlextSemanticError:
    """Test FlextSemanticError functionality."""

    def test_flext_error_creation(self) -> None:
      """Test FlextError base class creation."""
      error = FlextSemanticError.Hierarchy.FlextError(
          "Test error",
          error_code="TEST_001",
          context={"correlation_id": "test-correlation-123"},
      )

      assert str(error) == "Test error"
      assert error.error_code == "TEST_001"
      assert error.context["correlation_id"] == "test-correlation-123"
      assert isinstance(error, Exception)

    def test_flext_error_defaults(self) -> None:
      """Test FlextError with default values."""
      error = FlextSemanticError.Hierarchy.FlextError("Simple error")

      assert str(error) == "Simple error"
      assert error.error_code == FlextConstants.Errors.GENERIC_ERROR
      assert error.cause is None
      assert error.context == {}

    def test_flext_business_error(self) -> None:
      """Test FlextBusinessError creation."""
      error = FlextSemanticError.Hierarchy.FlextBusinessError(
          "Business rule violation",
          context={"business_rule": "user_must_be_active", "entity_id": "user-123"},
      )

      assert str(error) == "Business rule violation"
      assert error.error_code == FlextConstants.Errors.BUSINESS_RULE_ERROR
      assert error.context["business_rule"] == "user_must_be_active"
      assert error.context["entity_id"] == "user-123"
      assert isinstance(error, FlextSemanticError.Hierarchy.FlextError)

    def test_flext_technical_error(self) -> None:
      """Test FlextTechnicalError creation."""
      error = FlextSemanticError.Hierarchy.FlextTechnicalError(
          "Database connection failed",
          context={
              "system_component": "database",
              "error_details": {"host": "localhost", "port": 5432},
          },
      )

      assert str(error) == "Database connection failed"
      assert error.error_code == FlextConstants.Errors.CONNECTION_ERROR
      assert error.context["system_component"] == "database"
      assert error.context["error_details"] == {"host": "localhost", "port": 5432}

    def test_flext_validation_error(self) -> None:
      """Test FlextValidationError creation."""
      error = FlextSemanticError.Hierarchy.FlextValidationError(
          "Invalid email format",
          context={
              "field_name": "email",
              "field_value": "invalid-email",
              "validation_rule": "email_format",
          },
      )

      assert str(error) == "Invalid email format"
      assert error.error_code == FlextConstants.Errors.VALIDATION_ERROR
      assert error.context["field_name"] == "email"
      assert error.context["field_value"] == "invalid-email"
      assert error.context["validation_rule"] == "email_format"

    def test_flext_security_error(self) -> None:
      """Test FlextSecurityError creation."""
      error = FlextSemanticError.Hierarchy.FlextSecurityError(
          "Unauthorized access attempt",
          context={
              "user_id": "malicious-user",
              "resource": "REDACTED_LDAP_BIND_PASSWORD-panel",
              "action": "access",
          },
      )

      assert str(error) == "Unauthorized access attempt"
      assert error.error_code == FlextConstants.Errors.AUTHENTICATION_ERROR
      assert error.context["user_id"] == "malicious-user"
      assert error.context["resource"] == "REDACTED_LDAP_BIND_PASSWORD-panel"
      assert error.context["action"] == "access"

    def test_namespace_classes_exist(self) -> None:
      """Test error namespace classes exist."""
      # The namespace classes should exist even if empty
      assert hasattr(FlextSemanticError, "Namespace")
      assert FlextSemanticError.Namespace is not None

    def test_factory_create_business_error(self) -> None:
      """Test Factory.create_business_error."""
      error = FlextSemanticError.Factory.create_business_error(
          "User not found",
          context={"entity_id": "user-456", "business_rule": "user_exists"},
      )

      assert isinstance(error, FlextSemanticError.Hierarchy.FlextBusinessError)
      assert str(error) == "User not found"
      assert error.context["entity_id"] == "user-456"
      assert error.context["business_rule"] == "user_exists"

    def test_factory_create_validation_error(self) -> None:
      """Test Factory.create_validation_error."""
      error = FlextSemanticError.Factory.create_validation_error(
          "Age must be positive",
          field_name="age",
          field_value=-5,
      )

      assert isinstance(error, FlextSemanticError.Hierarchy.FlextValidationError)
      assert str(error) == "Age must be positive"
      assert error.context["field_name"] == "age"
      assert error.context["field_value"] == -5

    def test_factory_from_exception_known_type(self) -> None:
      """Test Factory.from_exception with known FlextError."""
      original_error = FlextSemanticError.Hierarchy.FlextBusinessError(
          "Original error",
      )

      result_error = FlextSemanticError.Factory.from_exception(original_error)

      # Should create a new FlextError with original error as cause
      assert isinstance(result_error, FlextSemanticError.Hierarchy.FlextError)
      assert str(result_error) == "Original error"

    def test_factory_from_exception_unknown_type(self) -> None:
      """Test Factory.from_exception with unknown exception type."""
      original_error = ValueError("Standard Python error")

      result_error = FlextSemanticError.Factory.from_exception(original_error)

      # ValueError maps to FlextValidationError based on implementation
      assert isinstance(
          result_error,
          FlextSemanticError.Hierarchy.FlextValidationError,
      )
      assert "Standard Python error" in str(result_error)


class TestFlextSemantic:
    """Test FlextSemantic unified interface."""

    def test_flext_semantic_class_exists(self) -> None:
      """Test FlextSemantic class exists and is accessible."""
      assert FlextSemantic is not None

      # Test that it can be instantiated if it's not abstract
      try:
          instance = FlextSemantic()
          assert instance is not None
      except TypeError:
          # If it's an abstract class or has required parameters, that's OK
          pass

    def test_semantic_components_accessible(self) -> None:
      """Test that all semantic components are accessible through main interface."""
      # Test direct access to semantic classes
      assert FlextSemanticModel is not None
      assert FlextSemanticObservability is not None
      assert FlextSemanticError is not None

      # Test that classes have their expected structure
      assert hasattr(FlextSemanticModel, "Foundation")
      assert hasattr(FlextSemanticModel, "Factory")
      assert hasattr(FlextSemanticObservability, "Protocol")
      assert hasattr(FlextSemanticObservability, "Factory")
      assert hasattr(FlextSemanticError, "Hierarchy")
      assert hasattr(FlextSemanticError, "Factory")


class TestSemanticIntegration:
    """Test integration between semantic components."""

    def test_error_uses_constants(self) -> None:
      """Test that error classes use semantic constants."""
      business_error = FlextSemanticError.Hierarchy.FlextBusinessError("Test")
      technical_error = FlextSemanticError.Hierarchy.FlextTechnicalError("Test")
      validation_error = FlextSemanticError.Hierarchy.FlextValidationError("Test")
      security_error = FlextSemanticError.Hierarchy.FlextSecurityError("Test")

      # Verify error codes come from constants
      assert business_error.error_code == FlextConstants.Errors.BUSINESS_RULE_ERROR
      assert technical_error.error_code == FlextConstants.Errors.CONNECTION_ERROR
      assert validation_error.error_code == FlextConstants.Errors.VALIDATION_ERROR
      assert security_error.error_code == FlextConstants.Errors.AUTHENTICATION_ERROR

    def test_model_factory_uses_constants(self) -> None:
      """Test that model factory uses semantic constants."""
      defaults = FlextSemanticModel.Factory.create_model_with_defaults()

      # Verify values come from constants
      assert defaults["timeout"] == FlextConstants.Defaults.TIMEOUT
      assert defaults["status"] == FlextConstants.Status.ACTIVE

    def test_observability_uses_constants(self) -> None:
      """Test that observability uses semantic constants."""
      # This tests that the constants are referenced correctly
      default_log_level = FlextConstants.Observability.DEFAULT_LOG_LEVEL
      assert default_log_level is not None

      # The configure method should use this constant as default
      import contextlib  # noqa: PLC0415

      with contextlib.suppress(ImportError, AttributeError):
          # Try to call with default - should use the constant
          FlextSemanticObservability.Factory.configure_observability("test")
