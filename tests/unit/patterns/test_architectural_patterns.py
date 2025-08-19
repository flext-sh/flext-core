"""Advanced architectural pattern tests for FLEXT Core.

This module contains comprehensive tests for architectural patterns including
Clean Architecture, DDD, CQRS, and enterprise design patterns.
"""

from __future__ import annotations

import time
from typing import cast

import pytest
from pydantic import BaseModel

from flext_core import (
    FlextEntity,
    FlextHandlers,
    FlextResult,
    FlextValueObject,
    flext_core,
)


class TestCleanArchitecturePatterns:
    """Test Clean Architecture pattern implementation."""

    @pytest.mark.architecture
    def test_clean_architecture_layers(self) -> None:
        """Test proper Clean Architecture layer separation."""

        # Domain Layer - Entities and Value Objects
        class UserEmail(FlextValueObject):
            """Domain value object."""

            email: str

            def validate_business_rules(self) -> FlextResult[None]:
                """Validate email business rules."""
                if "@" not in self.email:
                    return FlextResult[None].fail("Invalid email format")
                return FlextResult[None].ok(None)

        class User(FlextEntity):
            """Domain entity."""

            name: str
            email_obj: UserEmail

            def validate_domain_rules(self) -> FlextResult[None]:
                """Validate user domain rules."""
                if not self.name.strip():
                    return FlextResult[None].fail("Name cannot be empty")
                return self.email_obj.validate_business_rules()

        # Application Layer - Use Cases (Commands/Handlers)
        class CreateUserCommand(BaseModel):
            """Application command."""

            name: str
            email: str

        class CreateUserHandler(FlextHandlers.CommandHandler):
            """Application command handler."""

            def handle(self, command: object) -> FlextResult[object]:
                """Handle user creation."""
                if not isinstance(command, CreateUserCommand):
                    return FlextResult[None].fail("Invalid command type")

                # Create domain objects
                try:
                    email_obj = UserEmail.model_validate({"email": command.email})
                    email_validation = email_obj.validate_business_rules()
                    if email_validation.is_failure:
                        return FlextResult[None].fail(
                            f"Email validation failed: {email_validation.error}",
                        )
                except Exception as e:
                    return FlextResult[None].fail(f"Email creation failed: {e}")

                # Create entity
                user = User(
                    id="user_123",
                    name=command.name,
                    email_obj=email_obj,
                )
                user_result = user.validate_domain_rules()

                if user_result.is_failure:
                    return FlextResult[None].fail(
                        f"User validation failed: {user_result.error}",
                    )

                return FlextResult[str].ok(FlextResult[None].ok("User created successfully"))

        # Infrastructure Layer - Framework integration
        # Initialize core instance for framework integration
        _ = flext_core()

        # Test the full flow
        command = CreateUserCommand(name="John Doe", email="john@example.com")
        handler = CreateUserHandler()

        result = handler.handle(command)
        assert result.success
        assert result.data == "User created successfully"

    @pytest.mark.architecture
    @pytest.mark.ddd
    def test_ddd_aggregate_pattern(self) -> None:
        """Test Domain-Driven Design aggregate pattern."""
        order_id, money = self._create_ddd_value_objects()
        order = self._create_ddd_aggregate(order_id, money)
        self._test_ddd_validation_and_behavior(order)

    def _create_ddd_value_objects(self) -> tuple[object, object]:
        """Create DDD value objects for testing."""

        # Value Objects
        class OrderId(FlextValueObject):
            """Order identifier value object."""

            value: str

            def validate_business_rules(self) -> FlextResult[None]:
                """Validate order ID format."""
                if not self.value.startswith("ORD-"):
                    return FlextResult[None].fail("Order ID must start with ORD-")
                return FlextResult[None].ok(None)

        class Money(FlextValueObject):
            """Money value object."""

            amount: float
            currency: str = "USD"

            def validate_business_rules(self) -> FlextResult[None]:
                """Validate money rules."""
                if self.amount < 0:
                    return FlextResult[None].fail("Amount cannot be negative")
                return FlextResult[None].ok(None)

        order_id = OrderId.model_validate({"value": "ORD-12345"})
        money = Money.model_validate({"amount": 99.99, "currency": "USD"})
        return order_id, money

    def _create_ddd_aggregate(self, order_id: object, money: object) -> object:
        """Create DDD aggregate for testing."""

        # Aggregate Root
        class Order(FlextEntity):
            """Order aggregate root."""

            order_id: object
            total: object
            status: str = "pending"

            def validate_domain_rules(self) -> FlextResult[None]:
                """Validate order business rules."""
                # Validate value objects with type checking for serialization
                if hasattr(self.order_id, "validate_business_rules"):
                    order_id_validation = self.order_id.validate_business_rules()
                    if (
                        hasattr(order_id_validation, "is_failure")
                        and order_id_validation.is_failure
                    ):
                        return FlextResult[None].fail(str(order_id_validation.error))

                if hasattr(self.total, "validate_business_rules"):
                    total_validation = self.total.validate_business_rules()
                    if (
                        hasattr(total_validation, "is_failure")
                        and total_validation.is_failure
                    ):
                        return FlextResult[None].fail(str(total_validation.error))

                # Business rules
                if self.status not in {"pending", "confirmed", "shipped", "delivered"}:
                    return FlextResult[None].fail("Invalid order status")

                return FlextResult[None].ok(None)

            def confirm_order(self) -> FlextResult[None]:
                """Domain behavior: confirm order."""
                if self.status != "pending":
                    return FlextResult[None].fail("Can only confirm pending orders")

                # Create new instance with updated status (immutable pattern)
                result = self.copy_with(status="confirmed")
                if result.is_failure:
                    return FlextResult[None].fail(f"Failed to confirm order: {result.error}")

                return FlextResult[None].ok(None)

        return Order(
            id="order_123",
            order_id=order_id,
            total=money,
        )

    def _test_ddd_validation_and_behavior(self, order: object) -> None:
        """Test DDD validation and behavior."""
        # Test domain validation
        if hasattr(order, "validate_business_rules"):
            validation_result = order.validate_business_rules()
            assert hasattr(validation_result, "success")
            assert validation_result.success

        # Test domain behavior
        if hasattr(order, "confirm_order"):
            confirm_result = order.confirm_order()
            assert hasattr(confirm_result, "success")
            assert confirm_result.success

    @pytest.mark.architecture
    def test_cqrs_pattern_implementation(self) -> None:
        """Test CQRS (Command Query Responsibility Segregation) pattern."""

        # Commands (Write Operations)
        class UpdateUserCommand(BaseModel):
            """Command to update user information."""

            user_id: str
            name: str

        class UpdateUserHandler(FlextHandlers.CommandHandler):
            """Handler for user update commands."""

            def handle(self, command: object) -> FlextResult[object]:
                """Handle user update command."""
                if not isinstance(command, UpdateUserCommand):
                    return FlextResult[None].fail("Invalid command type")

                # Simulate business logic
                if not command.name.strip():
                    return FlextResult[None].fail("Name cannot be empty")

                return FlextResult[None].ok(f"User {command.user_id} updated")

        # Queries (Read Operations)
        class GetUserQuery(BaseModel):
            """Query to get user information."""

            user_id: str

        class GetUserHandler(
            FlextHandlers.QueryHandler[GetUserQuery, dict[str, object]],
        ):
            """Handler for user queries."""

            def handle(self, query: GetUserQuery) -> FlextResult[dict[str, object]]:
                """Handle user query."""
                # Simulate data retrieval
                user_data: dict[str, object] = {
                    "id": query.user_id,
                    "name": "John Doe",
                    "email": "john@example.com",
                }

                return FlextResult[None].ok(user_data)

        # Test CQRS separation
        command = UpdateUserCommand(user_id="123", name="Jane Doe")
        command_handler = UpdateUserHandler()

        command_result = command_handler.handle(command)
        assert command_result.success
        assert "updated" in str(command_result.data)

        query = GetUserQuery(user_id="123")
        query_handler = GetUserHandler()

        query_result = query_handler.handle(query)
        assert query_result.success
        assert isinstance(query_result.data, dict)


class TestEnterprisePatterns:
    """Test enterprise design patterns."""

    @pytest.mark.architecture
    def test_factory_pattern_implementation(self) -> None:
        """Test Factory pattern implementation."""

        class ServiceFactory:
            """Factory for creating different types of services."""

            @staticmethod
            def create_service(service_type: str) -> FlextResult[object]:
                """Create service based on type."""
                if service_type == "email":
                    return FlextResult[None].ok({"type": "email", "provider": "smtp"})
                if service_type == "sms":
                    return FlextResult[None].ok({"type": "sms", "provider": "twilio"})
                return FlextResult[None].fail(f"Unknown service type: {service_type}")

        # Test factory usage
        email_service = ServiceFactory.create_service("email")
        assert email_service.success
        assert isinstance(email_service.data, dict)
        assert email_service.data["type"] == "email"

        sms_service = ServiceFactory.create_service("sms")
        assert sms_service.success
        assert isinstance(sms_service.data, dict)
        assert sms_service.data["type"] == "sms"

        invalid_service = ServiceFactory.create_service("invalid")
        assert invalid_service.is_failure

    @pytest.mark.architecture
    def test_builder_pattern_implementation(self) -> None:
        """Test Builder pattern implementation."""

        class ConfigurationBuilder:
            """Builder for complex configuration objects."""

            def __init__(self) -> None:
                """Initialize builder."""
                self._config: dict[str, object] = {}

            def with_database(self, host: str, port: int) -> ConfigurationBuilder:
                """Add database configuration."""
                self._config["database"] = {"host": host, "port": port}
                return self

            def with_logging(self, level: str) -> ConfigurationBuilder:
                """Add logging configuration."""
                self._config["logging"] = {"level": level}
                return self

            def with_cache(self, *, enabled: bool) -> ConfigurationBuilder:
                """Add cache configuration."""
                self._config["cache"] = {"enabled": enabled}
                return self

            def build(self) -> FlextResult[dict[str, object]]:
                """Build the configuration."""
                if not self._config:
                    return FlextResult[None].fail("Configuration cannot be empty")

                return FlextResult[None].ok(self._config.copy())

        # Test builder usage
        config_result = (
            ConfigurationBuilder()
            .with_database("localhost", 5432)
            .with_logging("INFO")
            .with_cache(enabled=True)
            .build()
        )

        assert config_result.success
        config = config_result.data
        assert isinstance(config, dict)
        config_dict = config  # Already dict[str, object] from assertion
        database_dict = cast("dict[str, object]", config_dict["database"])
        assert database_dict["host"] == "localhost"
        logging_dict = cast("dict[str, object]", config_dict["logging"])
        assert logging_dict["level"] == "INFO"
        cache_dict = cast("dict[str, object]", config_dict["cache"])
        assert cache_dict["enabled"]

    @pytest.mark.architecture
    @pytest.mark.performance
    def test_repository_pattern_performance(self) -> None:
        """Test Repository pattern with performance considerations."""

        class InMemoryRepository:
            """In-memory repository implementation."""

            def __init__(self) -> None:
                """Initialize repository."""
                self._data: dict[str, object] = {}
                self._query_count = 0

            def save(self, entity_id: str, data: object) -> FlextResult[None]:
                """Save entity to repository."""
                self._data[entity_id] = data
                return FlextResult[None].ok(None)

            def find_by_id(self, entity_id: str) -> FlextResult[object]:
                """Find entity by ID."""
                self._query_count += 1

                if entity_id in self._data:
                    return FlextResult[None].ok(self._data[entity_id])

                return FlextResult[None].fail(f"Entity not found: {entity_id}")

            def get_query_count(self) -> int:
                """Get number of queries executed."""
                return self._query_count

        # Test repository performance
        repo = InMemoryRepository()

        # Save multiple entities
        start_time = time.time()
        for i in range(1000):
            result = repo.save(f"entity_{i}", {"id": i, "name": f"Entity {i}"})
            assert result.success

        save_duration = time.time() - start_time

        # Query entities
        start_time = time.time()
        for i in range(100):
            query_result: FlextResult[object] = repo.find_by_id(f"entity_{i}")
            result = cast("FlextResult[None]", query_result)
            assert result.success
            entity_data = cast("dict[str, object]", result.data)
            assert entity_data["id"] == i

        query_duration = time.time() - start_time

        # Performance assertions
        assert save_duration < 1.0, (
            f"Saving 1000 entities took too long: {save_duration:.3f}s"
        )
        assert query_duration < 0.1, (
            f"Querying 100 entities took too long: {query_duration:.3f}s"
        )
        assert repo.get_query_count() == 100


class TestEventDrivenPatterns:
    """Test event-driven architectural patterns."""

    @pytest.mark.architecture
    @pytest.mark.ddd
    def test_domain_event_pattern(self) -> None:
        """Test Domain Event pattern implementation."""

        # Event classes
        class UserCreatedEvent(BaseModel):
            """Domain event for user creation."""

            user_id: str
            user_name: str
            timestamp: float

        class UserUpdatedEvent(BaseModel):
            """Domain event for user updates."""

            user_id: str
            old_name: str
            new_name: str
            timestamp: float

        # Event handler
        class UserEventHandler:
            """Handler for user domain events."""

            def __init__(self) -> None:
                """Initialize handler."""
                self.processed_events: list[BaseModel] = []

            def handle_user_created(self, event: UserCreatedEvent) -> FlextResult[None]:
                """Handle user created event."""
                self.processed_events.append(event)
                return FlextResult[None].ok(None)

            def handle_user_updated(self, event: UserUpdatedEvent) -> FlextResult[None]:
                """Handle user updated event."""
                self.processed_events.append(event)
                return FlextResult[None].ok(None)

        # Test event processing
        handler = UserEventHandler()

        # Create and process events
        created_event = UserCreatedEvent(
            user_id="123",
            user_name="John Doe",
            timestamp=time.time(),
        )

        updated_event = UserUpdatedEvent(
            user_id="123",
            old_name="John Doe",
            new_name="Jane Doe",
            timestamp=time.time(),
        )

        # Process events
        result1 = handler.handle_user_created(created_event)
        assert result1.success

        result2 = handler.handle_user_updated(updated_event)
        assert result2.success

        # Verify event processing
        assert len(handler.processed_events) == 2
        assert isinstance(handler.processed_events[0], UserCreatedEvent)
        assert isinstance(handler.processed_events[1], UserUpdatedEvent)

    @pytest.mark.architecture
    def test_observer_pattern_implementation(self) -> None:
        """Test Observer pattern implementation."""
        # Simple observer pattern test
        observers: list[dict[str, object]] = []

        def notify_all(state: str) -> None:
            for observer in observers:
                observer["state"] = state

        # Create observers
        obs1: dict[str, object] = {"name": "Observer1", "state": None}
        obs2: dict[str, object] = {"name": "Observer2", "state": None}
        observers.extend([obs1, obs2])

        # Test notifications
        notify_all("new_state")
        assert obs1["state"] == "new_state"
        assert obs2["state"] == "new_state"

        # Test removal
        observers.remove(obs1)
        notify_all("updated_state")
        assert obs1["state"] == "new_state"  # Not updated
        assert obs2["state"] == "updated_state"  # Updated
