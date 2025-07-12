"""Tests for flext_core.application.handlers module."""

import pytest

from flext_core.application.handlers import CommandHandler
from flext_core.application.handlers import EventHandler
from flext_core.application.handlers import QueryHandler
from flext_core.domain.types import ServiceResult


class MockCommand:
    """Mock command for testing."""

    def __init__(self, data: str) -> None:
        self.data = data


class MockQuery:
    """Mock query for testing."""

    def __init__(self, filter_value: str) -> None:
        self.filter_value = filter_value


class MockEvent:
    """Mock event for testing."""

    def __init__(self, event_type: str) -> None:
        self.event_type = event_type


class MockCommandHandler(CommandHandler[MockCommand, str]):
    """Mock command handler implementation."""

    async def handle(self, command: MockCommand) -> ServiceResult[str]:
        """Handle mock command."""
        if command.data == "error":
            return ServiceResult.failure("Test error")
        return ServiceResult.success(f"Processed: {command.data}")


class MockQueryHandler(QueryHandler[MockQuery, list[str]]):
    """Mock query handler implementation."""

    async def handle(self, query: MockQuery) -> ServiceResult[list[str]]:
        """Handle mock query."""
        if query.filter_value == "empty":
            return ServiceResult.success([])
        return ServiceResult.success([f"Result for: {query.filter_value}"])


class MockEventHandler(EventHandler[MockEvent, None]):
    """Mock event handler implementation."""

    def __init__(self) -> None:
        self.handled_events: list[MockEvent] = []

    async def handle(self, event: MockEvent) -> ServiceResult[None]:
        """Handle mock event."""
        self.handled_events.append(event)
        return ServiceResult.success(None)


class MockCommandHandlerBase:
    """Test CommandHandler base functionality."""

    @pytest.mark.asyncio
    async def test_command_handler_success(self) -> None:
        """Test successful command handling."""
        handler = MockCommandHandler()
        command = MockCommand("test_data")

        result = await handler.handle(command)

        assert result.is_success
        assert result.value == "Processed: test_data"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_command_handler_failure(self) -> None:
        """Test command handling failure."""
        handler = MockCommandHandler()
        command = MockCommand("error")

        result = await handler.handle(command)

        assert not result.is_success
        assert result.error == "Test error"
        assert result.value is None


class MockQueryHandlerBase:
    """Test QueryHandler base functionality."""

    @pytest.mark.asyncio
    async def test_query_handler_success(self) -> None:
        """Test successful query handling."""
        handler = MockQueryHandler()
        query = MockQuery("test_filter")

        result = await handler.handle(query)

        assert result.is_success
        assert result.value == ["Result for: test_filter"]
        assert result.error is None

    @pytest.mark.asyncio
    async def test_query_handler_empty_result(self) -> None:
        """Test query handler with empty result."""
        handler = MockQueryHandler()
        query = MockQuery("empty")

        result = await handler.handle(query)

        assert result.is_success
        assert result.value == []
        assert result.error is None


class MockEventHandlerBase:
    """Test EventHandler base functionality."""

    @pytest.mark.asyncio
    async def test_event_handler_processes_event(self) -> None:
        """Test event handler processes events."""
        handler = MockEventHandler()
        event = MockEvent("test_event")

        await handler.handle(event)

        assert any(e.event_type == "test_event" for e in handler.handled_events)

    @pytest.mark.asyncio
    async def test_event_handler_multiple_events(self) -> None:
        """Test event handler processes multiple events."""
        handler = MockEventHandler()
        events = [
            MockEvent("event1"),
            MockEvent("event2"),
            MockEvent("event3"),
        ]

        for event in events:
            await handler.handle(event)

        assert [e.event_type for e in handler.handled_events] == ["event1", "event2", "event3"]


class TestHandlerImplementations:
    """Test handler implementations functionality."""

    @pytest.mark.asyncio
    async def test_command_handler_concrete_implementation(self) -> None:
        """Test concrete command handler implementation."""
        handler = MockCommandHandler()

        # Test successful command
        success_command = MockCommand("success")
        result = await handler.handle(success_command)
        assert result.is_success
        assert result.value is not None and "Processed: success" in result.value

        # Test error command
        error_command = MockCommand("error")
        result = await handler.handle(error_command)
        assert not result.is_success

    @pytest.mark.asyncio
    async def test_query_handler_concrete_implementation(self) -> None:
        """Test concrete query handler implementation."""
        handler = MockQueryHandler()

        # Test successful query
        query = MockQuery("test")
        result = await handler.handle(query)
        assert result.is_success
        assert isinstance(result.value, list)

    @pytest.mark.asyncio
    async def test_event_handler_concrete_implementation(self) -> None:
        """Test concrete event handler implementation."""
        handler = MockEventHandler()

        # Test event handling
        event = MockEvent("test_event")
        await handler.handle(event)
        assert any(e.event_type == "test_event" for e in handler.handled_events)


class TestHandlerIntegration:
    """Test handler integration scenarios."""

    @pytest.mark.asyncio
    async def test_command_query_event_flow(self) -> None:
        """Test complete command -> event -> query flow."""
        # Setup handlers
        command_handler = MockCommandHandler()
        event_handler = MockEventHandler()
        query_handler = MockQueryHandler()

        # Execute command
        command = MockCommand("integration_test")
        command_result = await command_handler.handle(command)

        # Process resulting event
        event = MockEvent("command_completed")
        await event_handler.handle(event)

        # Query for results
        query = MockQuery("integration")
        query_result = await query_handler.handle(query)

        # Verify flow
        assert command_result.is_success
        assert any(e.event_type == "command_completed" for e in event_handler.handled_events)
        assert query_result.is_success
        assert query_result.value is not None and len(query_result.value) > 0

    @pytest.mark.asyncio
    async def test_error_propagation(self) -> None:
        """Test error propagation through handlers."""
        handler = MockCommandHandler()
        command = MockCommand("error")

        result = await handler.handle(command)

        assert not result.is_success
        assert result.error is not None
        assert result.value is None
