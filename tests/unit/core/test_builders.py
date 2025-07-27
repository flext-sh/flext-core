"""Tests for FlextBuilder patterns and factory utilities.

Comprehensive tests for all builder patterns, factory utilities, and
boilerplate reduction features.
"""

from __future__ import annotations

from flext_core import FlextBuilders


class TestFlextBuilders:
    """Test FlextBuilders functionality."""

    def test_flext_config_builder(self) -> None:
        """Test building configuration."""
        result = FlextBuilders.flext_config(environment="test", debug=True)

        assert result.is_success
        config = result.data
        assert config is not None
        assert config["environment"] == "test"
        assert config["debug"] is True

    def test_flext_api_response(self) -> None:
        """Test API response builder."""
        result = FlextBuilders.flext_api_response(
            data={"message": "success"},
            status=200,
            message="OK",
        )

        assert result.is_success
        response = result.data
        assert response is not None
        assert response["status"] == 200
        assert response["message"] == "OK"
        assert response["data"]["message"] == "success"

    def test_flext_command_builder(self) -> None:
        """Test command builder."""
        result = FlextBuilders.flext_command(
            name="test_command",
            payload={"key": "value"},
        )

        assert result.is_success
        command = result.data
        assert command is not None
        assert command["name"] == "test_command"
        assert command["payload"]["key"] == "value"

    def test_flext_event_builder(self) -> None:
        """Test event builder."""
        result = FlextBuilders.flext_event(
            event_type="user_created",
            data={"user_id": "123"},
        )

        assert result.is_success
        event = result.data
        assert event is not None
        assert event["event_type"] == "user_created"
        assert event["data"]["user_id"] == "123"
