"""Real coverage boost tests using actual FlextCore APIs.

Focus on uncovered code paths and edge cases in core modules.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextConfig, FlextUtilities, FlextValidations


class TestRealConfigCoverage:
    """Test config paths using real APIs."""

    def test_config_creation_basic(self) -> None:
        """Test basic config creation."""
        config = FlextConfig()
        assert config.name == "flext"
        assert config.environment in {
            "development",
            "production",
            "staging",
            "test",
            "local",
        }

    def test_config_environment_validation(self) -> None:
        """Test environment validation with real values."""
        for env in ["development", "production", "staging", "test", "local"]:
            config = FlextConfig(environment=env)
            assert config.environment == env


class TestRealValidationsCoverage:
    """Test validation paths using real APIs."""

    def test_email_validation_success(self) -> None:
        """Test email validation success path."""
        result = FlextValidations.validate_email("test@example.com")
        assert result.is_success

    def test_email_validation_failure(self) -> None:
        """Test email validation failure path."""
        result = FlextValidations.validate_email("invalid-email")
        assert result.is_failure

    def test_string_field_validation(self) -> None:
        """Test string field validation."""
        result = FlextValidations.validate_string_field("test")
        assert result.is_success

    def test_numeric_field_validation(self) -> None:
        """Test numeric field validation."""
        result = FlextValidations.validate_numeric_field(42)
        assert result.is_success

        # Test validation failure with string
        result = FlextValidations.validate_numeric_field("invalid")
        assert result.is_failure


class TestRealUtilitiesCoverage:
    """Test utilities using real APIs."""

    def test_safe_bool_conversion(self) -> None:
        """Test safe boolean conversion."""
        result = FlextUtilities.safe_bool_conversion("true")
        assert result is True

        result = FlextUtilities.safe_bool_conversion("false")
        assert result is False

        result = FlextUtilities.safe_bool_conversion("invalid", default=False)
        assert result is False

    def test_safe_int_conversion(self) -> None:
        """Test safe integer conversion."""
        result = FlextUtilities.safe_int_conversion("123")
        assert result == 123

        result = FlextUtilities.safe_int_conversion_with_default("invalid", default=0)
        assert result == 0

    def test_json_operations(self) -> None:
        """Test JSON utility operations."""
        data = {"key": "value"}
        json_str = FlextUtilities.safe_json_stringify(data)
        assert '"key"' in json_str

        parsed = FlextUtilities.safe_json_parse(json_str)
        assert parsed is not None
        assert parsed.get("key") == "value"

    def test_id_generation(self) -> None:
        """Test ID generation utilities."""
        entity_id = FlextUtilities.generate_entity_id()
        assert entity_id is not None
        assert len(entity_id) > 0

        uuid_id = FlextUtilities.generate_uuid()
        assert uuid_id is not None
        assert len(uuid_id) > 0

    def test_text_processing(self) -> None:
        """Test text processing utilities."""
        cleaned = FlextUtilities.clean_text("  test  ")
        assert cleaned == "test"

        truncated = FlextUtilities.truncate("very long text", max_length=5)
        assert len(truncated) <= 5

    def test_performance_tracking(self) -> None:
        """Test performance utilities."""
        # Record performance directly since track_performance doesn't support context manager
        FlextUtilities.record_performance("test_operation", 0.001)

        metrics = FlextUtilities.get_performance_metrics()
        assert metrics is not None


class TestRealIntegrationCoverage:
    """Integration tests using real APIs."""

    def test_validation_with_utilities(self) -> None:
        """Test integration between validation and utilities."""
        # Generate test email
        email = "test@example.com"

        # Validate the email
        result = FlextValidations.validate_email(email)
        assert result.is_success

        # Process with utilities
        cleaned_email = FlextUtilities.clean_text(email)
        assert cleaned_email == email

    def test_config_with_validation(self) -> None:
        """Test config creation with validation."""
        # Create test config data
        config_data = {
            "name": "integration-test",
            "environment": "test",
            "debug": True,
        }

        # Validate environment
        if isinstance(config_data["environment"], str):
            # Basic string validation
            assert len(config_data["environment"]) > 0

        # Create config
        config = FlextConfig.create(constants=config_data).unwrap()
        assert config.name == "integration-test"

    def test_batch_processing(self) -> None:
        """Test batch processing capabilities."""
        # Create test items
        items = [f"item_{i}" for i in range(10)]

        # Process items with utilities
        for item in items:
            cleaned = FlextUtilities.clean_text(item)
            assert cleaned == item

            # Validate non-empty
            is_valid = FlextUtilities.is_non_empty_string(cleaned)
            assert is_valid is True


class TestRealEdgeCasesCoverage:
    """Test edge cases using real APIs."""

    def test_empty_string_handling(self) -> None:
        """Test empty string handling."""
        result = FlextValidations.validate_string_field("")
        assert result.is_failure  # Empty string should fail validation

        # Test empty string utility
        is_valid = FlextUtilities.is_non_empty_string("")
        assert is_valid is False

    def test_none_handling(self) -> None:
        """Test None value handling."""
        result = FlextUtilities.safe_bool_conversion(None, default=False)
        assert result is False

        int_result = FlextUtilities.safe_int_conversion_with_default(None, default=0)
        assert int_result == 0

    def test_boundary_values(self) -> None:
        """Test boundary value handling."""
        # Test min/max ports
        min_port = FlextUtilities.MIN_PORT
        max_port = FlextUtilities.MAX_PORT

        assert min_port > 0
        assert max_port > min_port

    def test_invalid_json_handling(self) -> None:
        """Test invalid JSON handling."""
        invalid_json = "{ invalid json }"
        result = FlextUtilities.safe_json_parse(invalid_json)
        # safe_json_parse returns empty dict for invalid JSON, not None
        assert isinstance(result, dict)


class TestRealPerformanceCoverage:
    """Test performance-critical paths."""

    def test_bulk_validation(self) -> None:
        """Test bulk validation operations."""
        emails = [f"test{i}@example.com" for i in range(5)]

        for email in emails:
            result = FlextValidations.validate_email(email)
            assert result.is_success

    def test_bulk_utilities(self) -> None:
        """Test bulk utility operations."""
        texts = [f"  text_{i}  " for i in range(10)]

        cleaned_texts = []
        for text in texts:
            cleaned = FlextUtilities.clean_text(text)
            cleaned_texts.append(cleaned)

        assert len(cleaned_texts) == 10
        assert all("text_" in text for text in cleaned_texts)

    def test_performance_metrics(self) -> None:
        """Test performance metrics collection."""
        # Record some performance metrics
        FlextUtilities.record_performance("test_metric", 0.001)

        # Get performance metrics
        metrics = FlextUtilities.get_performance_metrics()
        assert metrics is not None
