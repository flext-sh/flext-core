"""Comprehensive tests for testing utilities module."""

from __future__ import annotations

from flext_core.testing_utilities import (
    FlextTestMocker,
    create_api_test_response,
    create_ldap_test_config,
    create_oud_connection_config,
)


class TestCreateOudConnectionConfig:
    """Test create_oud_connection_config function."""

    def test_returns_dict_with_required_keys(self) -> None:
        """Test that function returns dictionary with all required keys."""
        config = create_oud_connection_config()

        assert isinstance(config, dict)
        required_keys = {
            "host",
            "port",
            "bind_dn",
            "bind_password",
            "base_dn",
            "use_ssl",
            "timeout",
        }
        assert set(config.keys()) == required_keys

    def test_default_values_are_correct(self) -> None:
        """Test that default values match client-a OUD requirements."""
        config = create_oud_connection_config()

        assert config["host"] == "localhost"
        assert config["port"] == "3389"
        assert config["bind_dn"] == "cn=orclREDACTED_LDAP_BIND_PASSWORD"
        assert config["bind_password"] == "Welcome1"
        assert config["base_dn"] == "dc=ctbc,dc=com"
        assert config["use_ssl"] == "false"
        assert config["timeout"] == "30"

    def test_returns_consistent_config(self) -> None:
        """Test that multiple calls return identical configuration."""
        config1 = create_oud_connection_config()
        config2 = create_oud_connection_config()

        assert config1 == config2

    def test_all_values_are_strings(self) -> None:
        """Test that all values are strings for consistency."""
        config = create_oud_connection_config()

        for key, value in config.items():
            assert isinstance(value, str), (
                f"Value for {key} should be string, got {type(value)}"
            )


class TestCreateLdapTestConfig:
    """Test create_ldap_test_config function."""

    def test_returns_dict_with_required_keys(self) -> None:
        """Test that function returns dictionary with all required keys."""
        config = create_ldap_test_config()

        assert isinstance(config, dict)
        required_keys = {
            "host",
            "port",
            "bind_dn",
            "bind_password",
            "base_dn",
            "use_ssl",
            "timeout",
        }
        assert set(config.keys()) == required_keys

    def test_default_values_are_correct(self) -> None:
        """Test that default values are appropriate for testing."""
        config = create_ldap_test_config()

        assert config["host"] == "localhost"
        assert config["port"] == 389
        assert config["bind_dn"] == "cn=REDACTED_LDAP_BIND_PASSWORD,dc=test,dc=com"
        assert config["bind_password"] == "testpass"
        assert config["base_dn"] == "dc=test,dc=com"
        assert config["use_ssl"] is False
        assert config["timeout"] == 30

    def test_returns_consistent_config(self) -> None:
        """Test that multiple calls return identical configuration."""
        config1 = create_ldap_test_config()
        config2 = create_ldap_test_config()

        assert config1 == config2

    def test_port_and_timeout_are_integers(self) -> None:
        """Test that port and timeout values are integers."""
        config = create_ldap_test_config()

        assert isinstance(config["port"], int)
        assert isinstance(config["timeout"], int)

    def test_use_ssl_is_boolean(self) -> None:
        """Test that use_ssl value is boolean."""
        config = create_ldap_test_config()

        assert isinstance(config["use_ssl"], bool)


class TestCreateApiTestResponse:
    """Test create_api_test_response function."""

    def test_default_success_response(self) -> None:
        """Test default success response structure."""
        response = create_api_test_response()

        assert isinstance(response, dict)
        assert response["success"] is True
        assert "data" in response
        assert "timestamp" in response

        # Check default data structure
        data = response["data"]
        assert isinstance(data, dict)
        assert "id" in data
        assert "status" in data

    def test_success_response_with_custom_data(self) -> None:
        """Test success response with custom data."""
        custom_data = {"user_id": "123", "name": "John Doe"}
        response = create_api_test_response(data=custom_data)

        assert response["success"] is True
        assert response["data"] == custom_data
        assert "timestamp" in response

    def test_failure_response(self) -> None:
        """Test failure response structure."""
        response = create_api_test_response(success=False)

        assert isinstance(response, dict)
        assert response["success"] is False
        assert "error" in response
        assert "timestamp" in response

        # Check error structure
        error = response["error"]
        assert isinstance(error, dict)
        assert "code" in error
        assert "message" in error
        assert "details" in error

    def test_failure_response_structure_details(self) -> None:
        """Test detailed failure response structure."""
        from typing import cast

        response = create_api_test_response(success=False)
        error = response["error"]
        error_dict = cast("dict[str, object]", error)

        assert error_dict["code"] == "VALIDATION_ERROR"
        assert error_dict["message"] == "Invalid input data"
        assert isinstance(error_dict["details"], dict)
        details = cast("dict[str, object]", error_dict["details"])
        assert details["field"] == "name"
        assert details["error"] == "required"

    def test_timestamp_format(self) -> None:
        """Test that timestamp is in expected format."""
        response = create_api_test_response()
        timestamp = response["timestamp"]

        assert isinstance(timestamp, str)
        assert timestamp == "2025-01-20T12:00:00Z"

    def test_explicit_success_true(self) -> None:
        """Test explicitly setting success=True."""
        response = create_api_test_response(success=True)

        assert response["success"] is True
        assert "data" in response

    def test_explicit_success_false(self) -> None:
        """Test explicitly setting success=False."""
        response = create_api_test_response(success=False)

        assert response["success"] is False
        assert "error" in response

    def test_none_data_handling(self) -> None:
        """Test handling of None data parameter."""
        response = create_api_test_response(data=None)

        assert response["success"] is True
        # Should use default data when None is provided
        assert response["data"] == {"id": "test_123", "status": "active"}

    def test_custom_data_types(self) -> None:
        """Test various data types as custom data."""
        # String data
        response1 = create_api_test_response(data="test string")
        assert response1["data"] == "test string"

        # List data
        response2 = create_api_test_response(data=["item1", "item2"])
        assert response2["data"] == ["item1", "item2"]

        # Integer data
        response3 = create_api_test_response(data=42)
        assert response3["data"] == 42

    def test_kwargs_only_parameters(self) -> None:
        """Test that parameters are keyword-only."""
        # This should work
        response1 = create_api_test_response(success=True)
        assert response1["success"] is True

        response2 = create_api_test_response(data={"test": "data"})
        assert response2["data"] == {"test": "data"}

        # Both parameters
        response3 = create_api_test_response(success=False, data={"test": "data"})
        assert response3["success"] is False


class TestUtilitiesIntegration:
    """Test integration between different utility functions."""

    def test_oud_vs_ldap_config_differences(self) -> None:
        """Test that OUD and LDAP configs have appropriate differences."""
        oud_config = create_oud_connection_config()
        ldap_config = create_ldap_test_config()

        # Different ports
        assert oud_config["port"] == "3389"  # string
        assert ldap_config["port"] == 389  # int

        # Different base DNs
        assert oud_config["base_dn"] == "dc=ctbc,dc=com"
        assert ldap_config["base_dn"] == "dc=test,dc=com"

        # Different bind DNs
        assert oud_config["bind_dn"] == "cn=orclREDACTED_LDAP_BIND_PASSWORD"
        assert ldap_config["bind_dn"] == "cn=REDACTED_LDAP_BIND_PASSWORD,dc=test,dc=com"

    def test_api_response_success_vs_failure(self) -> None:
        """Test structural differences between success and failure responses."""
        success_response = create_api_test_response(success=True)
        failure_response = create_api_test_response(success=False)

        # Success has data, failure has error
        assert "data" in success_response
        assert "data" not in failure_response
        assert "error" not in success_response
        assert "error" in failure_response

        # Both have success flag and timestamp
        assert "success" in success_response
        assert "success" in failure_response
        assert "timestamp" in success_response
        assert "timestamp" in failure_response

    def test_all_functions_return_dicts(self) -> None:
        """Test that all utility functions return dictionaries."""
        oud_config = create_oud_connection_config()
        ldap_config = create_ldap_test_config()
        api_response = create_api_test_response()

        assert isinstance(oud_config, dict)
        assert isinstance(ldap_config, dict)
        assert isinstance(api_response, dict)

    def test_no_shared_mutable_state(self) -> None:
        """Test that utility functions don't share mutable state."""
        # Modify returned dictionaries and ensure subsequent calls aren't affected
        oud_config1 = create_oud_connection_config()
        oud_config1["host"] = "modified"

        oud_config2 = create_oud_connection_config()
        assert oud_config2["host"] == "localhost"  # Should be unchanged

        # Same for LDAP config
        ldap_config1 = create_ldap_test_config()
        ldap_config1["host"] = "modified"

        ldap_config2 = create_ldap_test_config()
        assert ldap_config2["host"] == "localhost"  # Should be unchanged


class TestFlextTestMockerPatchObject:
    """Test FlextTestMocker.patch_object method."""

    def test_patch_object_with_new_none(self) -> None:
        """Test that patch_object honors new=None and sets attribute to None."""

        class TestTarget:
            attr = "original_value"

        target = TestTarget()
        assert target.attr == "original_value"

        # Test that new=None is properly handled
        with FlextTestMocker.patch_object(target, "attr", new=None):
            assert target.attr is None

        # After context, should be restored
        assert target.attr == "original_value"

    def test_patch_object_with_new_value(self) -> None:
        """Test that patch_object works with new values."""

        class TestTarget:
            attr = "original_value"

        target = TestTarget()

        with FlextTestMocker.patch_object(target, "attr", new="new_value"):
            assert target.attr == "new_value"

        assert target.attr == "original_value"

    def test_patch_object_preserves_kwargs(self) -> None:
        """Test that patch_object preserves all kwargs like create=True."""

        class TestTarget:
            pass  # No attr initially

        target = TestTarget()
        assert not hasattr(target, "attr")

        # Test create=True is honored
        with FlextTestMocker.patch_object(
            target, "attr", new="test_value", create=True,
        ):
            assert target.attr == "test_value"

        # After context, attribute should be removed since it was created
        assert not hasattr(target, "attr")

    def test_patch_object_with_spec(self) -> None:
        """Test that patch_object honors spec parameter."""

        class TestTarget:
            def attr(self) -> str:
                return "original"

        class MockSpec:
            def mock_method(self):
                return "mocked"

        target = TestTarget()

        # Test spec is honored
        with FlextTestMocker.patch_object(target, "attr", spec=MockSpec) as mock:
            assert hasattr(mock, "mock_method")
            # The mock should have spec methods but not others
            assert hasattr(mock, "mock_method")

        # Original should be restored
        assert callable(target.attr)

    def test_patch_object_no_kwargs(self) -> None:
        """Test patch_object with no additional kwargs (creates MagicMock)."""

        class TestTarget:
            attr = "original"

        target = TestTarget()

        with FlextTestMocker.patch_object(target, "attr") as mock:
            # Should be a mock when no new= is provided
            assert mock != "original"
            assert target.attr is mock

        assert target.attr == "original"
