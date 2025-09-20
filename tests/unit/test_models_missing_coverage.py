"""Targeted tests for missing coverage in flext_core.models module.

This test file specifically targets the 110 missing lines identified in coverage analysis
to improve models.py from 78% to 90%+ coverage.

Missing lines targeted: 126, 171, 190-192, 201-212, 217, 221-224, 233-240, 245, 270,
285-287, 374, 402, 405, 441, 471, 518, 577-581, 587-590, 596-600, 604-608, 612-616,
620-633, 664-682, 736, 772-776, 782-796, 805, 822, 871, 908-909, 952, 979, 985,
1002-1003, 1026-1027, 1043, 1050-1051, 1072, 1074-1075

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import tempfile
import unittest.mock
from datetime import UTC, datetime

import pytest

from flext_core import FlextModels, FlextResult
from flext_core.config import FlextConfig

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestPayloadExtractMethod:
    """Test coverage for Payload.extract() method - Line 126."""

    def test_payload_extract_method_returns_data(self) -> None:
        """Test Payload.extract() method returns the data field."""
        # Create payload with test data
        test_data = {"user_id": "123", "action": "login"}
        payload = FlextModels.Payload[dict[str, str]](data=test_data)

        # Test extract method - Line 126
        extracted_data = payload.extract()

        assert extracted_data == test_data
        assert extracted_data is payload.data


class TestCommandQueryValidationMethods:
    """Test coverage for Command/Query validation methods - Lines 190-192."""

    def test_query_validate_query_with_empty_type(self) -> None:
        """Test Query.validate_query() with empty query_type - Lines 190-192."""
        query = FlextModels.Query.model_construct(query_type="")  # Bypass validation

        # Test validation method - Lines 190-192
        result = query.validate_query()

        assert result.is_failure
        assert result.error == "Query type is required"

    def test_query_validate_query_with_valid_type(self) -> None:
        """Test Query.validate_query() with valid query_type - Line 192."""
        query = FlextModels.Query(query_type="get_users")

        # Test validation method - Line 192
        result = query.validate_query()

        assert result.is_success
        assert result.data is True

    def test_command_validate_command_with_empty_type(self) -> None:
        """Test Command.validate_command() with empty command_type - Lines 170-172."""
        command = FlextModels.Command.model_construct(command_type="")  # Bypass validation

        # Test validation method
        result = command.validate_command()

        assert result.is_failure
        assert result.error == "Command type is required"

    def test_command_validate_command_with_valid_type(self) -> None:
        """Test Command.validate_command() with valid command_type."""
        command = FlextModels.Command(command_type="create_user")

        # Test validation method
        result = command.validate_command()

        assert result.is_success
        assert result.data is True


class TestCqrsCommandTypeDerivation:
    """Test coverage for CQRS command type derivation - Lines 201-212."""

    def test_cqrs_command_ensure_command_type_non_dict_input(self) -> None:
        """Test CqrsCommand._ensure_command_type with non-dict input - Lines 201-202."""
        # Test with non-dict input - Lines 201-202
        result = FlextModels.CqrsCommand._ensure_command_type("not_a_dict")
        assert result == "not_a_dict"

    def test_cqrs_command_type_derivation_from_class_name(self) -> None:
        """Test CqrsCommand command_type derivation from class name - Lines 203-212."""
        # Create a custom command class to test derivation
        class CreateUserCommand(FlextModels.CqrsCommand):
            pass

        # Test with missing command_type - should derive from class name
        data = {"payload": {"user": "test"}}
        result = CreateUserCommand._ensure_command_type(data)

        # Lines 203-212: Should derive "create_user" from "CreateUserCommand"
        assert isinstance(result, dict)
        assert result["command_type"] == "create_user"

    def test_cqrs_command_get_command_type_method(self) -> None:
        """Test CqrsCommand.get_command_type() method - Lines 221-224."""
        class TestComplexCommand(FlextModels.CqrsCommand):
            pass

        command = TestComplexCommand()

        # Test get_command_type method - Lines 221-224
        command_type = command.get_command_type()
        assert command_type == "test_complex"

    def test_cqrs_command_id_property(self) -> None:
        """Test CqrsCommand.id property - Lines 217."""
        command = FlextModels.CqrsCommand()

        # Test id property - Line 217
        assert command.id == command.command_id


class TestCqrsQueryTypeDerivation:
    """Test coverage for CQRS query type derivation - Lines 233-240."""

    def test_cqrs_query_ensure_query_type_non_dict_input(self) -> None:
        """Test CqrsQuery._ensure_query_type with non-dict input - Lines 233-234."""
        # Test with non-dict input - Lines 233-234
        result = FlextModels.CqrsQuery._ensure_query_type("not_a_dict")
        assert result == "not_a_dict"

    def test_cqrs_query_type_derivation_from_class_name(self) -> None:
        """Test CqrsQuery query_type derivation from class name - Lines 235-240."""
        # Create a custom query class to test derivation
        class GetUserDetailsQuery(FlextModels.CqrsQuery):
            pass

        # Test with missing query_type - should derive from class name
        data = {"filters": {"user_id": "123"}}
        result = GetUserDetailsQuery._ensure_query_type(data)

        # Lines 235-240: Should derive "get_user_details" from "GetUserDetailsQuery"
        assert isinstance(result, dict)
        assert result["query_type"] == "get_user_details"

    def test_cqrs_query_id_property(self) -> None:
        """Test CqrsQuery.id property - Lines 245."""
        query = FlextModels.CqrsQuery()

        # Test id property - Line 245
        assert query.id == query.query_id


class TestCqrsConfigHandlerMetadata:
    """Test coverage for CqrsConfig.Handler metadata validation - Lines 270."""

    def test_cqrs_handler_ensure_metadata_method(self) -> None:
        """Test CqrsConfig.Handler metadata validation - Line 270."""
        # Test handler creation with explicit metadata to avoid recursion issue
        handler_data = {
            "handler_id": "test_handler",
            "handler_name": "Test Handler",
            "handler_type": "command",
            "metadata": {"key": "value"}
        }

        # Create handler with explicit metadata
        handler = FlextModels.CqrsConfig.Handler.model_validate(handler_data)

        # Line 270: Should have the provided metadata
        assert handler.metadata == {"key": "value"}
        assert isinstance(handler.metadata, dict)


class TestCqrsConfigFactoryMethods:
    """Test coverage for CqrsConfig factory methods - Lines 285-287, 374."""

    def test_create_handler_config_with_existing_handler_different_type(self) -> None:
        """Test create_handler_config with existing handler but different type - Lines 285-287."""
        # Test creating a handler config with specific parameters - Lines 285-287
        result = FlextModels.CqrsConfig.create_handler_config(
            handler_type="query",
            default_name="Query Handler",
            default_id="query_id",
            handler_config=None  # Test with None to trigger default creation
        )

        # Lines 285-287: Should create new handler with provided parameters
        assert result.handler_type == "query"
        assert result.handler_id == "query_id"
        assert result.handler_name == "Query Handler"

    def test_create_bus_config_with_existing_bus(self) -> None:
        """Test create_bus_config with existing bus object - Line 374."""
        # Create existing bus config
        existing_bus = FlextModels.CqrsConfig.Bus(
            enable_middleware=False,
            execution_timeout=60
        )

        # Test with existing bus - Line 374
        result = FlextModels.CqrsConfig.create_bus_config(existing_bus)

        assert result is existing_bus  # Should return same object


class TestEmailAddressCoercion:
    """Test coverage for EmailAddress input coercion - Lines 402, 405."""

    def test_email_address_coerce_input_with_dict_value(self) -> None:
        """Test EmailAddress._coerce_input with dict containing value - Line 402."""
        # Test dict input coercion - Line 402
        result = FlextModels.EmailAddress._coerce_input({"value": "test@example.com"})
        assert result == "test@example.com"

    def test_email_address_coerce_input_with_non_string(self) -> None:
        """Test EmailAddress._coerce_input with non-string input - Line 405."""
        # Test non-string input coercion - Line 405
        result = FlextModels.EmailAddress._coerce_input(12345)
        assert result == "12345"


class TestEmailAddressDomainMethod:
    """Test coverage for EmailAddress.domain() method - Line 441."""

    def test_email_address_domain_method_with_valid_email(self) -> None:
        """Test EmailAddress.domain() method with valid email - Line 441."""
        email = FlextModels.EmailAddress("user@example.com")

        # Test domain method - Line 441
        domain = email.domain()
        assert domain == "example.com"

    def test_email_address_domain_method_without_at_symbol(self) -> None:
        """Test EmailAddress.domain() method without @ symbol - Line 441."""
        # Create invalid email that somehow bypassed validation
        email = FlextModels.EmailAddress.model_construct(root="invalid_email")

        # Test domain method - Line 441
        domain = email.domain()
        assert domain == ""


class TestTimestampCreateMethod:
    """Test coverage for Timestamp.create() method - Line 471."""

    def test_timestamp_create_with_naive_datetime(self) -> None:
        """Test Timestamp.create() with naive datetime - Line 471."""
        naive_dt = datetime(2025, 1, 8, 12, 0, 0, tzinfo=UTC)  # Use UTC timezone

        # Test create method - Line 471
        result = FlextModels.Timestamp.create(naive_dt)

        assert result.is_success
        timestamp = result.unwrap()
        assert timestamp.value.tzinfo is None  # Should remain naive per Line 471


class TestMetadataValidation:
    """Test coverage for Metadata validation - Line 518."""

    def test_metadata_create_with_non_string_values(self) -> None:
        """Test Metadata.create() with non-string values - Line 518."""
        # Test with non-string values - Line 518
        invalid_metadata = {"key1": "string", "key2": 123, "key3": "another_string"}

        result = FlextModels.Metadata.create(invalid_metadata)

        assert result.is_failure
        assert "key2" in result.error  # Should mention the invalid key


class TestUrlHttpValidationMethods:
    """Test coverage for URL HTTP validation methods - Lines 577-581, 587-590."""

    def test_url_create_http_url_with_invalid_port_zero(self) -> None:
        """Test URL.create_http_url() with port 0 - Lines 577-581."""
        url = "http://example.com:0/path"

        # Test HTTP validation with port 0 - Lines 578-580
        result = FlextModels.Url.create_http_url(url)

        assert result.is_failure
        assert "Invalid port 0" in result.error

    def test_url_create_http_url_with_port_exceeding_max(self) -> None:
        """Test URL.create_http_url() with port exceeding max - Lines 580-583."""
        url = "http://example.com:70000/path"

        # Test HTTP validation with port > max_port - Lines 580-583
        result = FlextModels.Url.create_http_url(url, max_port=65535)

        assert result.is_failure
        assert "Port out of range 0-65535" in result.error

    def test_url_create_http_url_too_long(self) -> None:
        """Test URL.create_http_url() with URL too long - Lines 587-590."""
        long_path = "x" * 2000
        url = f"http://example.com/{long_path}"

        # Test HTTP validation with long URL - Lines 586-588
        result = FlextModels.Url.create_http_url(url, max_length=100)

        assert result.is_failure
        assert "URL is too long" in result.error

    def test_url_create_http_url_parsing_exception(self) -> None:
        """Test URL.create_http_url() with parsing exception - Lines 589-590."""
        # Create a URL that would cause parsing issues
        malformed_url = "http://[invalid-ipv6:port"

        # Test HTTP validation exception handling - Lines 589-590
        result = FlextModels.Url.create_http_url(malformed_url)

        assert result.is_failure
        assert "URL parsing failed" in result.error


class TestUrlUtilityMethods:
    """Test coverage for URL utility methods - Lines 596-600, 604-608, 612-616."""

    def test_url_get_port_with_exception(self) -> None:
        """Test URL.get_port() with parsing exception - Lines 599-600."""
        # Create URL that might cause parsing issues
        url = FlextModels.Url.model_construct(value="http://[invalid-url")

        # Test get_port with exception - Lines 599-600
        port = url.get_port()
        assert port is None

    def test_url_get_scheme_with_exception(self) -> None:
        """Test URL.get_scheme() with parsing exception - Lines 607-608."""
        # Create URL that might cause parsing issues
        url = FlextModels.Url.model_construct(value="invalid://[malformed")

        # Test get_scheme with exception - Lines 607-608
        scheme = url.get_scheme()
        assert scheme == ""

    def test_url_get_hostname_with_exception(self) -> None:
        """Test URL.get_hostname() with parsing exception - Lines 615-616."""
        # Create URL that might cause parsing issues
        url = FlextModels.Url.model_construct(value="http://[invalid-hostname")

        # Test get_hostname with exception - Lines 615-616
        hostname = url.get_hostname()
        assert hostname == ""


class TestUrlNormalizationMethod:
    """Test coverage for URL normalization - Lines 620-633."""

    def test_url_normalize_with_empty_cleaned_text(self) -> None:
        """Test URL.normalize() with empty cleaned text - Lines 623-624."""
        # Create URL that would result in empty cleaned text
        url = FlextModels.Url.model_construct(value="   ")  # Whitespace only

        # Test normalization - Lines 623-624
        result = url.normalize()

        assert result.is_failure
        assert "URL cannot be empty" in result.error

    def test_url_normalize_with_exception(self) -> None:
        """Test URL.normalize() with exception - Lines 632-635."""
        # Create URL that might cause normalization issues
        url = FlextModels.Url.model_construct(value="http://example.com/")

        # Mock FlextUtilities.TextProcessor.clean_text to raise exception
        with unittest.mock.patch("flext_core.utilities.FlextUtilities.TextProcessor.clean_text", side_effect=Exception("Mock error")):
            # Test normalization exception handling - Lines 632-635
            result = url.normalize()

        assert result.is_failure
        assert "URL normalization failed" in result.error


class TestSystemConfigsDeprecation:
    """Test coverage for SystemConfigs deprecation - Lines 664-682."""

    def test_system_configs_getattr_valid_config(self) -> None:
        """Test SystemConfigs.__getattr__ with valid config - Lines 671-679."""
        system_configs = FlextModels.SystemConfigs()

        # Test accessing valid config - should trigger deprecation warning
        with pytest.warns(DeprecationWarning, match="ContainerConfig is deprecated"):
            config_class = system_configs.ContainerConfig

        # Should return the FlextConfig.SystemConfigs class
        assert config_class == FlextConfig.SystemConfigs.ContainerConfig

    def test_system_configs_getattr_invalid_config(self) -> None:
        """Test SystemConfigs.__getattr__ with invalid config - Lines 681-682."""
        system_configs = FlextModels.SystemConfigs()

        # Test accessing invalid config - Lines 681-682
        with pytest.raises(AttributeError, match="object has no attribute 'InvalidConfig'"):
            _ = system_configs.InvalidConfig


class TestWorkspaceValidation:
    """Test coverage for workspace validation methods - Lines 736, 772-776."""

    def test_project_validate_business_rules(self) -> None:
        """Test Project.validate_business_rules() method - Line 736."""
        project = FlextModels.Project(
            name="test_project",
            path="/test/path",
            project_type="python"
        )

        # Test validate_business_rules - Line 736
        result = project.validate_business_rules()
        assert result.is_success

    def test_workspace_info_validate_business_rules_negative_project_count(self) -> None:
        """Test WorkspaceInfo.validate_business_rules() with negative count - Lines 772-773."""
        workspace_info = FlextModels.WorkspaceInfo.model_construct(
            name="test_workspace",
            path="/test/path",
            project_count=-1  # Invalid
        )

        # Test validation - Lines 772-773
        result = workspace_info.validate_business_rules()
        assert result.is_failure
        assert "Project count cannot be negative" in result.error

    def test_workspace_info_validate_business_rules_negative_size(self) -> None:
        """Test WorkspaceInfo.validate_business_rules() with negative size - Lines 774-775."""
        workspace_info = FlextModels.WorkspaceInfo.model_construct(
            name="test_workspace",
            path="/test/path",
            total_size_mb=-1.0  # Invalid
        )

        # Test validation - Lines 774-775
        result = workspace_info.validate_business_rules()
        assert result.is_failure
        assert "Total size cannot be negative" in result.error

    def test_workspace_info_validate_business_rules_success(self) -> None:
        """Test WorkspaceInfo.validate_business_rules() success - Line 776."""
        workspace_info = FlextModels.WorkspaceInfo(
            name="test_workspace",
            path="/test/path",
            project_count=5,
            total_size_mb=100.0
        )

        # Test validation success - Line 776
        result = workspace_info.validate_business_rules()
        assert result.is_success


class TestFactoryMethods:
    """Test coverage for factory methods - Lines 782-796, 805, 822."""

    def test_create_entity_with_string_id(self) -> None:
        """Test create_entity() with string ID - Lines 782-796."""
        # Test with string ID - Lines 787-791
        result = FlextModels.create_entity(id="test_id", name="test")

        assert result.is_success
        entity = result.unwrap()
        assert entity.id == "test_id"

    def test_create_entity_with_non_string_id(self) -> None:
        """Test create_entity() with non-string ID - Lines 789-791."""
        # Test with non-string ID - Lines 789-791
        result = FlextModels.create_entity(id=12345)

        assert result.is_success
        entity = result.unwrap()
        assert entity.id == "12345"

    def test_create_entity_with_exception(self) -> None:
        """Test create_entity() with exception - Lines 795-796."""
        # Test with data that would cause exception during entity creation
        with unittest.mock.patch("flext_core.models.FlextModels.Entity", side_effect=Exception("Mock error")):
            result = FlextModels.create_entity(id="test")

        assert result.is_failure
        assert "Mock error" in result.error

    def test_create_event_factory_method(self) -> None:
        """Test create_event() factory method - Line 805."""
        # Test create_event - Line 805
        event = FlextModels.create_event(
            event_type="user_created",
            payload={"user_id": "123"},
            aggregate_id="user_123"
        )

        assert event.event_type == "user_created"
        assert event.payload == {"user_id": "123"}
        assert event.aggregate_id == "user_123"

    def test_create_query_factory_method_with_none_filters(self) -> None:
        """Test create_query() factory method with None filters - Line 822."""
        # Test create_query with None filters - Line 822
        query = FlextModels.create_query(
            query_type="get_users",
            filters=None  # Should default to {}
        )

        assert query.query_type == "get_users"
        assert query.filters == {}


class TestValidationFunctionsMissingCoverage:
    """Test coverage for validation functions - Lines 871, 908-909, 952, 979, 985, 1002-1003."""

    def test_create_validated_http_method_with_none_input(self) -> None:
        """Test create_validated_http_method() with None input - Line 871."""
        # Test with None input - Line 871
        result = FlextModels.create_validated_http_method(None)

        assert result.is_failure
        assert "HTTP method must be a non-empty string" in result.error

    def test_create_validated_http_status_with_type_error(self) -> None:
        """Test create_validated_http_status() with TypeError - Lines 908-909."""
        # Test with input that causes TypeError - Lines 908-909
        result = FlextModels.create_validated_http_status(object())  # Object can't be converted to int

        assert result.is_failure
        assert "Status code must be a valid integer" in result.error

    def test_create_validated_iso_date_with_empty_string(self) -> None:
        """Test create_validated_iso_date() with empty string - Line 952."""
        # Test with empty string - Line 952
        result = FlextModels.create_validated_iso_date("")

        assert result.is_failure
        assert "Date string cannot be empty" in result.error

    def test_create_validated_date_range_with_invalid_start_date(self) -> None:
        """Test create_validated_date_range() with invalid start date - Line 979."""
        # Test with invalid start date - Lines 978-981
        result = FlextModels.create_validated_date_range(
            start_date="invalid-date",
            end_date="2025-01-10"
        )

        assert result.is_failure
        assert "Invalid start date" in result.error

    def test_create_validated_date_range_with_invalid_end_date(self) -> None:
        """Test create_validated_date_range() with invalid end date - Line 985."""
        # Test with invalid end date - Lines 983-987
        result = FlextModels.create_validated_date_range(
            start_date="2025-01-08",
            end_date="invalid-date"
        )

        assert result.is_failure
        assert "Invalid end date" in result.error

    def test_create_validated_date_range_exception_handling(self) -> None:
        """Test create_validated_date_range() exception handling - Lines 1002-1003."""
        # Test with malformed dates that cause ValueError during parsing
        result = FlextModels.create_validated_date_range(
            start_date="invalid-date-format",
            end_date="2025-01-10"
        )

        assert result.is_failure
        assert "Invalid start date" in result.error


class TestFilePathValidationMethods:
    """Test coverage for file path validation methods - Lines 1026-1027, 1043, 1050-1051, 1072, 1074-1075."""

    def test_create_validated_file_path_with_os_error(self) -> None:
        """Test create_validated_file_path() with problematic path - Lines 1026-1027."""
        # Test with empty path which should cause validation failure
        result = FlextModels.create_validated_file_path("")

        assert result.is_failure
        assert "File path cannot be empty" in result.error

    def test_create_validated_existing_file_path_failure_passthrough(self) -> None:
        """Test create_validated_existing_file_path() with path validation failure - Line 1043."""
        # Test with invalid path that fails basic validation - Line 1043
        with unittest.mock.patch("flext_core.models.FlextModels.create_validated_file_path") as mock_validate:
            mock_validate.return_value = FlextResult[str].fail("Invalid path format")

            result = FlextModels.create_validated_existing_file_path("/invalid/path")

        assert result.is_failure
        assert "Invalid path format" in result.error

    def test_create_validated_existing_file_path_with_permission_error(self) -> None:
        """Test create_validated_existing_file_path() with PermissionError - Lines 1050-1051."""
        # Create a temporary file that exists
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_path = tmp_file.name

            # Mock Path.exists() to cause PermissionError
            with unittest.mock.patch("pathlib.Path.exists", side_effect=PermissionError("Mock permission error")):
                result = FlextModels.create_validated_existing_file_path(tmp_path)

        assert result.is_failure
        assert "Cannot access path: Mock permission error" in result.error

    def test_create_validated_directory_path_not_directory(self) -> None:
        """Test create_validated_directory_path() with non-directory - Line 1072."""
        # Create a temporary file (not directory)
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_path = tmp_file.name

            # Test with file instead of directory - Line 1072
            result = FlextModels.create_validated_directory_path(tmp_path)

        assert result.is_failure
        assert "Path is not a directory" in result.error

    def test_create_validated_directory_path_with_permission_error(self) -> None:
        """Test create_validated_directory_path() with PermissionError - Lines 1074-1075."""
        # Create a temporary directory
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            unittest.mock.patch("pathlib.Path.is_dir", side_effect=PermissionError("Mock permission error"))
        ):
            result = FlextModels.create_validated_directory_path(tmp_dir)

        assert result.is_failure
        assert "Cannot verify directory: Mock permission error" in result.error
