"""Tests for flext_core.application.commands and handlers.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Comprehensive tests for CQRS base classes to boost coverage.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from flext_core.application.commands import Command
from flext_core.application.handlers import (
    CommandHandler,
    EventHandler,
    QueryHandler,
    SimpleQueryHandler,
    VoidCommandHandler,
)
from flext_core.domain.shared_types import ServiceResult


class TestCommand:
    """Test base Command class."""

    def test_command_is_abstract(self) -> None:
        """Test that Command cannot be instantiated directly."""
        # Should not be able to instantiate abstract class directly
        with pytest.raises(TypeError):
            Command()

    def test_concrete_command_creation(self) -> None:
        """Test creating concrete command implementations."""

        class TestCommandImpl(Command):
            def __init__(self, data: str) -> None:
                self.data = data

            def validate_command(self) -> bool:
                return bool(self.data)

        cmd = TestCommandImpl("test data")
        assert cmd.data == "test data"
        assert cmd.validate_command() is True

    def test_command_validation_false(self) -> None:
        """Test command validation returning false."""

        class EmptyCommand(Command):
            def validate_command(self) -> bool:
                return False

        cmd = EmptyCommand()
        assert cmd.validate_command() is False

    def test_command_validation_with_data(self) -> None:
        """Test command validation with data checks."""

        class DataCommand(Command):
            def __init__(self, value: int) -> None:
                self.value = value

            def validate_command(self) -> bool:
                return self.value > 0

        valid_cmd = DataCommand(5)
        assert valid_cmd.validate_command() is True

        invalid_cmd = DataCommand(-1)
        assert invalid_cmd.validate_command() is False

    def test_command_inheritance(self) -> None:
        """Test that Command can be inherited."""

        class BaseCommand(Command):
            def validate_command(self) -> bool:
                return True

        class DerivedCommand(BaseCommand):
            def __init__(self, name: str) -> None:
                self.name = name

            def validate_command(self) -> bool:
                return super().validate_command() and bool(self.name)

        derived = DerivedCommand("test")
        assert derived.name == "test"
        assert derived.validate_command() is True

        empty_derived = DerivedCommand("")
        assert empty_derived.validate_command() is False


class TestCommandHandler:
    """Test CommandHandler base class."""

    def test_command_handler_is_abstract(self) -> None:
        """Test that CommandHandler cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CommandHandler()

    @pytest.mark.asyncio
    async def test_concrete_command_handler(self) -> None:
        """Test creating concrete command handler implementations."""

        class TestCommand:
            def __init__(self, value: str) -> None:
                self.value = value

        class TestCommandHandler(CommandHandler[TestCommand, str]):
            async def handle(self, command: TestCommand) -> ServiceResult[Any]:
                return ServiceResult.ok({"result": f"Handled: {command.value}"}
                )

        handler = TestCommandHandler()
        command = TestCommand("test")
        result = await handler.handle(command)

        assert result.success
        assert cast(dict[str, Any], result.data)["result"] == "Handled: test"

    @pytest.mark.asyncio
    async def test_command_handler_error_handling(self) -> None:
        """Test command handler error handling."""

        class FailCommand:
            pass

        class FailingCommandHandler(CommandHandler[FailCommand, str]):
            async def handle(self, command: FailCommand) -> ServiceResult[Any]:
                return ServiceResult.fail("Command failed")

        handler = FailingCommandHandler()
        command = FailCommand()
        result = await handler.handle(command)

        assert not result.success
        assert result.error == "Command failed"

    @pytest.mark.asyncio
    async def test_command_handler_with_complex_result(self) -> None:
        """Test command handler with complex result types."""

        class ComplexCommand:
            def __init__(self, items: list[str]) -> None:
                self.items = items

        class ComplexCommandHandler(CommandHandler[ComplexCommand, dict[str, Any]]):
            async def handle(self, command: ComplexCommand) -> ServiceResult[Any]:
                result = {
                    "processed_items": len(command.items),
                    "items": command.items,
                    "status": "completed",
                }
                return ServiceResult.ok({"result": result})

        handler = ComplexCommandHandler()
        command = ComplexCommand(["a", "b", "c"])
        result = await handler.handle(command)

        assert result.success
        assert result.data is not None
        assert isinstance(result.data, dict)
        assert cast(dict[str, Any], result.data)["result"]["processed_items"] == 3
        assert cast(dict[str, Any], result.data)["result"]["items"] == ["a", "b", "c"]
        assert cast(dict[str, Any], result.data)["result"]["status"] == "completed"


class TestQueryHandler:
    """Test QueryHandler base class."""

    def test_query_handler_is_abstract(self) -> None:
        """Test that QueryHandler cannot be instantiated directly."""
        with pytest.raises(TypeError):
            QueryHandler()

    @pytest.mark.asyncio
    async def test_concrete_query_handler(self) -> None:
        """Test creating concrete query handler implementations."""

        class GetUserQuery:
            def __init__(self, user_id: str) -> None:
                self.user_id = user_id

        class UserQueryHandler(QueryHandler[GetUserQuery, dict[str, str]]):
            async def handle(self, query: GetUserQuery) -> ServiceResult[Any]:
                user_data = {"id": query.user_id, "name": f"User {query.user_id}"}
                return ServiceResult.ok({"result": user_data})

        handler = UserQueryHandler()
        query = GetUserQuery("123")
        result = await handler.handle(query)

        assert result.success
        assert cast(dict[str, Any], result.data)["result"]["id"] == "123"
        assert cast(dict[str, Any], result.data)["result"]["name"] == "User 123"

    @pytest.mark.asyncio
    async def test_query_handler_not_found(self) -> None:
        """Test query handler returning not found."""

        class SearchQuery:
            def __init__(self, term: str) -> None:
                self.term = term

        class SearchQueryHandler(QueryHandler[SearchQuery, list[str]]):
            async def handle(self, query: SearchQuery) -> ServiceResult[Any]:
                if not query.term:
                    return ServiceResult.fail("Search term required")
                return ServiceResult.ok({"result": []})

        handler = SearchQueryHandler()
        query = SearchQuery("")
        result = await handler.handle(query)

        assert not result.success
        assert result.error == "Search term required"

    @pytest.mark.asyncio
    async def test_query_handler_with_list_result(self) -> None:
        """Test query handler returning list results."""

        class ListItemsQuery:
            def __init__(self, category: str) -> None:
                self.category = category

        class ItemsQueryHandler(QueryHandler[ListItemsQuery, list[dict[str, str]]]):
            async def handle(self, query: ListItemsQuery) -> ServiceResult[Any]:
                items = [
                    {"id": "1", "category": query.category, "name": "Item 1"},
                    {"id": "2", "category": query.category, "name": "Item 2"},
                ]
                return ServiceResult.ok({"result": items})

        handler = ItemsQueryHandler()
        query = ListItemsQuery("electronics")
        result = await handler.handle(query)

        assert result.success
        assert len(cast(dict[str, Any], result.data)["result"]) == 2
        assert (
            cast(dict[str, Any], result.data)["result"][0]["category"] == "electronics"
        )
        assert cast(dict[str, Any], result.data)["result"][1]["name"] == "Item 2"


class TestEventHandler:
    """Test EventHandler base class."""

    def test_event_handler_is_abstract(self) -> None:
        """Test that EventHandler cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EventHandler()

    @pytest.mark.asyncio
    async def test_concrete_event_handler(self) -> None:
        """Test creating concrete event handler implementations."""

        class UserCreatedEvent:
            def __init__(self, user_id: str, email: str) -> None:
                self.user_id = user_id
                self.email = email

        class UserCreatedHandler(EventHandler[UserCreatedEvent, bool]):
            async def handle(self, event: UserCreatedEvent) -> ServiceResult[Any]:
                # Simulate sending welcome email
                if event.email:
                    return ServiceResult.ok({"result": True})
                return ServiceResult.fail("No email provided")

        handler = UserCreatedHandler()
        event = UserCreatedEvent("123", "user@example.com")
        result = await handler.handle(event)

        assert result.success
        assert result.data is True

    @pytest.mark.asyncio
    async def test_event_handler_failure(self) -> None:
        """Test event handler failure scenarios."""

        class PaymentProcessedEvent:
            def __init__(self, amount: float) -> None:
                self.amount = amount

        class PaymentHandler(EventHandler[PaymentProcessedEvent, dict[str, Any]]):
            async def handle(self, event: PaymentProcessedEvent) -> ServiceResult[Any]:
                if event.amount <= 0:
                    return ServiceResult.fail("Invalid payment amount")

                result = {
                    "amount": event.amount,
                    "processed": True,
                    "timestamp": "2025-01-01T00:00:00Z",
                }
                return ServiceResult.ok({"result": result})

        handler = PaymentHandler()

        # Test success case
        success_event = PaymentProcessedEvent(100.0)
        success_result = await handler.handle(success_event)
        assert success_result.success
        assert cast(dict[str, Any], success_result.data)["result"]["amount"] == 100.0

        # Test failure case
        fail_event = PaymentProcessedEvent(-50.0)
        fail_result = await handler.handle(fail_event)
        assert not fail_result.success
        assert fail_result.error == "Invalid payment amount"


class TestVoidCommandHandler:
    """Test VoidCommandHandler specialized class."""

    def test_void_command_handler_is_abstract(self) -> None:
        """Test that VoidCommandHandler cannot be instantiated directly."""
        with pytest.raises(TypeError):
            VoidCommandHandler()

    @pytest.mark.asyncio
    async def test_concrete_void_command_handler(self) -> None:
        """Test creating concrete void command handler implementations."""

        class DeleteCommand:
            def __init__(self, item_id: str) -> None:
                self.item_id = item_id

        class DeleteCommandHandler(VoidCommandHandler[DeleteCommand]):
            async def handle(self, command: DeleteCommand) -> ServiceResult[Any]:
                if not command.item_id:
                    return ServiceResult.fail("Item ID required")
                # Simulate deletion
                return ServiceResult.ok(None)

        handler = DeleteCommandHandler()
        command = DeleteCommand("item-123")
        result = await handler.handle(command)

        assert result.success
        assert result.data is None

    @pytest.mark.asyncio
    async def test_void_command_handler_failure(self) -> None:
        """Test void command handler failure scenarios."""

        class LogoutCommand:
            def __init__(self, user_id: str) -> None:
                self.user_id = user_id

        class LogoutHandler(VoidCommandHandler[LogoutCommand]):
            async def handle(self, command: LogoutCommand) -> ServiceResult[Any]:
                if command.user_id == "invalid":
                    return ServiceResult.fail("Invalid user")
                return ServiceResult.ok({"result": None})

        handler = LogoutHandler()

        # Test failure
        fail_command = LogoutCommand("invalid")
        fail_result = await handler.handle(fail_command)
        assert not fail_result.success
        assert fail_result.error == "Invalid user"


class TestSimpleQueryHandler:
    """Test SimpleQueryHandler specialized class."""

    def test_simple_query_handler_is_abstract(self) -> None:
        """Test that SimpleQueryHandler cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SimpleQueryHandler()

    @pytest.mark.asyncio
    async def test_concrete_simple_query_handler(self) -> None:
        """Test creating concrete simple query handler implementations."""

        class GetConfigQuery:
            def __init__(self, config_key: str) -> None:
                self.config_key = config_key

        class ConfigQueryHandler(SimpleQueryHandler[GetConfigQuery]):
            async def handle(self, query: GetConfigQuery) -> ServiceResult[Any]:
                config_data = {
                    "key": query.config_key,
                    "value": f"config_value_{query.config_key}",
                    "type": "string",
                    "last_modified": "2025-01-01T00:00:00Z",
                }
                return ServiceResult.ok({"result": config_data})

        handler = ConfigQueryHandler()
        query = GetConfigQuery("database_url")
        result = await handler.handle(query)

        assert result.success
        assert cast(dict[str, Any], result.data)["result"]["key"] == "database_url"
        assert (
            cast(dict[str, Any], result.data)["result"]["value"]
            == "config_value_database_url"
        )
        assert cast(dict[str, Any], result.data)["result"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_simple_query_handler_with_nested_data(self) -> None:
        """Test simple query handler with nested dictionary data."""

        class GetStatsQuery:
            def __init__(self, period: str) -> None:
                self.period = period

        class StatsQueryHandler(SimpleQueryHandler[GetStatsQuery]):
            async def handle(self, query: GetStatsQuery) -> ServiceResult[Any]:
                stats_data = {
                    "period": query.period,
                    "metrics": {
                        "total_users": 1500,
                        "active_users": 1200,
                        "conversion_rate": 0.8,
                    },
                    "trends": {"users": "up", "revenue": "stable"},
                }
                return ServiceResult.ok({"result": stats_data})

        handler = StatsQueryHandler()
        query = GetStatsQuery("monthly")
        result = await handler.handle(query)

        assert result.success
        assert cast(dict[str, Any], result.data)["result"]["period"] == "monthly"
        assert (
            cast(dict[str, Any], result.data)["result"]["metrics"]["total_users"]
            == 1500
        )
        assert cast(dict[str, Any], result.data)["result"]["trends"]["users"] == "up"


class TestHandlerIntegration:
    """Test integration scenarios with multiple handlers."""

    @pytest.mark.asyncio
    async def test_command_and_query_workflow(self) -> None:
        """Test workflow combining command and query handlers."""
        # Simulate a data store
        data_store: dict[str, dict[str, Any]] = {}

        class CreateItemCommand:
            def __init__(self, item_id: str, name: str) -> None:
                self.item_id = item_id
                self.name = name

        class GetItemQuery:
            def __init__(self, item_id: str) -> None:
                self.item_id = item_id

        class CreateItemHandler(VoidCommandHandler[CreateItemCommand]):
            async def handle(self, command: CreateItemCommand) -> ServiceResult[Any]:
                data_store[command.item_id] = {"name": command.name}
                return ServiceResult.ok({"result": None})

        class GetItemHandler(SimpleQueryHandler[GetItemQuery]):
            async def handle(self, query: GetItemQuery) -> ServiceResult[Any]:
                if query.item_id not in data_store:
                    return ServiceResult.fail("Item not found")
                return ServiceResult.ok({"result": data_store[query.item_id]}
                )

        # Test the workflow
        create_handler = CreateItemHandler()
        get_handler = GetItemHandler()

        # Create item
        create_command = CreateItemCommand("item-1", "Test Item")
        create_result = await create_handler.handle(create_command)
        assert create_result.success

        # Query item
        get_query = GetItemQuery("item-1")
        get_result = await get_handler.handle(get_query)
        assert get_result.success
        assert cast(dict[str, Any], get_result.data)["result"]["name"] == "Test Item"

        # Query non-existent item
        missing_query = GetItemQuery("item-999")
        missing_result = await get_handler.handle(missing_query)
        assert missing_result.is_failure
        assert missing_result.error == "Item not found"

    @pytest.mark.asyncio
    async def test_event_driven_workflow(self) -> None:
        """Test event-driven workflow with multiple event handlers."""
        # Shared state for event handlers
        events_log: list[str] = []

        class OrderPlacedEvent:
            def __init__(self, order_id: str, amount: float) -> None:
                self.order_id = order_id
                self.amount = amount

        class EmailNotificationHandler(EventHandler[OrderPlacedEvent, bool]):
            async def handle(self, event: OrderPlacedEvent) -> ServiceResult[Any]:
                events_log.append(f"Email sent for order {event.order_id}")
                return ServiceResult.ok({"result": True})

        class InventoryHandler(EventHandler[OrderPlacedEvent, bool]):
            async def handle(self, event: OrderPlacedEvent) -> ServiceResult[Any]:
                events_log.append(f"Inventory updated for order {event.order_id}")
                return ServiceResult.ok({"result": True})

        class PaymentHandler(EventHandler[OrderPlacedEvent, dict[str, Any]]):
            async def handle(self, event: OrderPlacedEvent) -> ServiceResult[Any]:
                if event.amount <= 0:
                    return ServiceResult.fail("Invalid amount")

                events_log.append(f"Payment processed for order {event.order_id}")
                return ServiceResult.ok({
                        "result": {
                            "transaction_id": f"txn_{event.order_id}",
                            "amount": event.amount,
                            "status": "completed",
                        }
                    },
                )

        # Test event handling
        event = OrderPlacedEvent("order-123", 99.99)

        email_handler = EmailNotificationHandler()
        inventory_handler = InventoryHandler()
        payment_handler = PaymentHandler()

        # Process event through all handlers
        email_result = await email_handler.handle(event)
        inventory_result = await inventory_handler.handle(event)
        payment_result = await payment_handler.handle(event)

        assert email_result.success
        assert inventory_result.success
        assert payment_result.success

        # Verify all events were logged
        assert len(events_log) == 3
        assert "Email sent for order order-123" in events_log
        assert "Inventory updated for order order-123" in events_log
        assert "Payment processed for order order-123" in events_log

        # Verify payment result data
        assert (
            cast(dict[str, Any], payment_result.data)["result"]["transaction_id"]
            == "txn_order-123"
        )
        assert cast(dict[str, Any], payment_result.data)["result"]["amount"] == 99.99


class TestAsyncBehavior:
    """Test async behavior and concurrency."""

    @pytest.mark.asyncio
    async def test_concurrent_handlers(self) -> None:
        """Test running multiple handlers concurrently."""
        results: list[str] = []

        class TestEvent:
            def __init__(self, message: str, delay: float) -> None:
                self.message = message
                self.delay = delay

        class SlowHandler(EventHandler[TestEvent, str]):
            async def handle(self, event: TestEvent) -> ServiceResult[Any]:
                await asyncio.sleep(event.delay)
                result = f"Processed: {event.message}"
                results.append(result)
                return ServiceResult.ok({"result": result})

        handler = SlowHandler()
        events = [
            TestEvent("Event 1", 0.1),
            TestEvent("Event 2", 0.05),
            TestEvent("Event 3", 0.02),
        ]

        # Run handlers concurrently
        tasks = [handler.handle(event) for event in events]
        concurrent_results = await asyncio.gather(*tasks)

        # All should succeed
        for result in concurrent_results:
            assert result.success

        # Results should be added in order of completion (shortest delay first)
        assert len(results) == 3
        assert "Event 3" in results[0]  # Shortest delay completes first
        assert all("Processed:" in r for r in results)
