"""Advanced architectural pattern tests for FLEXT Core.

This module contains tests for architectural patterns including
Clean Architecture, DDD, CQRS, and enterprise design patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from typing import cast

import pytest

from flext_core import FlextCore, FlextModels


class TestCleanArchitecturePatterns:
    """Test Clean Architecture pattern implementation."""

    @pytest.mark.architecture
    def test_clean_architecture_layers(self) -> None:
        """Test proper Clean Architecture layer separation."""

        # Domain Layer - Entities and Value Objects
        class UserEmail(FlextCore.Models.Value):
            """Domain value object."""

            email: str

            def validate_business_rules(self) -> FlextCore.Result[None]:
                """Validate email business rules.

                Returns:
                    FlextCore.Result[None]: Success or failure result.

                """
                if "@" not in self.email:
                    return FlextCore.Result[None].fail("Invalid email format")
                return FlextCore.Result[None].ok(None)

        class User(
            FlextCore.Models.ArbitraryTypesModel, FlextCore.Models.IdentifiableMixin
        ):
            """Domain entity."""

            name: str
            email_obj: UserEmail

            def validate_business_rules(self) -> FlextCore.Result[None]:
                """Validate business rules for user entity.

                Returns:
                    FlextCore.Result[None]: Success or failure result.

                """
                if not self.name.strip():
                    return FlextCore.Result[None].fail("Name cannot be empty")
                return self.email_obj.validate_business_rules()

            def validate_domain_rules(self) -> FlextCore.Result[None]:
                """Validate user domain rules.

                Returns:
                    FlextCore.Result[None]: Success or failure result.

                """
                return self.validate_business_rules()

        # Application Layer - Use Cases (Commands/Handlers)
        class CreateUserCommand(FlextCore.Models.Command):
            """Application command using FlextCore.Models foundation."""

            name: str
            email: str

        class CreateUserHandler(FlextCore.Processors.Implementation.BasicHandler):
            """Application command handler."""

            @property
            def handler_name(self) -> str:
                """Get handler name."""
                return "CreateUserHandler"

            def can_handle(self, message_type: type) -> bool:
                """Check if handler can handle the message type.

                Returns:
                    bool: True if handler can handle the message type.

                """
                return message_type == CreateUserCommand or issubclass(
                    message_type,
                    CreateUserCommand,
                )

            def handle(self, request: object) -> FlextCore.Result[str]:
                """Handle user creation.

                Returns:
                    FlextCore.Result[str]: Success or failure result.

                """
                if not isinstance(request, CreateUserCommand):
                    return FlextCore.Result[str].fail("Invalid command type")

                command = request

                # Create domain objects
                try:
                    email_obj = UserEmail.model_validate({"email": command.email})
                    email_validation = email_obj.validate_business_rules()
                    if email_validation.is_failure:
                        return FlextCore.Result[str].fail(
                            f"Email validation failed: {email_validation.error}",
                        )
                except Exception as e:
                    return FlextCore.Result[str].fail(f"Email creation failed: {e}")

                # Create entity
                user = User(
                    id="user_123",
                    name=command.name,
                    email_obj=email_obj,
                )
                user_result = user.validate_domain_rules()

                if user_result.is_failure:
                    return FlextCore.Result[str].fail(
                        f"User validation failed: {user_result.error}",
                    )

                return FlextCore.Result[str].ok("User created successfully")

        # Infrastructure Layer - Framework integration
        # Use FlextCore.Result pattern for framework integration
        FlextCore.Result[str].ok("Framework initialized")

        # Test the full flow
        command = CreateUserCommand(name="John Doe", email="john@example.com")
        handler = CreateUserHandler("CreateUserHandler")

        result = handler.handle(command)
        assert result.is_success
        assert result.value == "User created successfully"

    @pytest.mark.architecture
    @pytest.mark.ddd
    def test_ddd_aggregate_pattern(self) -> None:
        """Test Domain-Driven Design aggregate pattern."""
        order_id, money = self._create_ddd_value_objects()
        order = self._create_ddd_aggregate(order_id, money)
        self._test_ddd_validation_and_behavior(order)

    def _create_ddd_value_objects(self) -> tuple[object, object]:
        """Create DDD value objects for testing.

        Returns:
            tuple[object, object]: Tuple of created value objects.

        """

        # Value Objects
        class OrderId(FlextCore.Models.Value):
            """Order identifier value object."""

            value: str

            def validate_business_rules(self) -> FlextCore.Result[None]:
                """Validate order ID format."""
                if not self.value.startswith("ORD-"):
                    return FlextCore.Result[None].fail("Order ID must start with ORD-")
                return FlextCore.Result[None].ok(None)

        class Money(FlextCore.Models.Value):
            """Money value object."""

            amount: float
            currency: str = "USD"

            def validate_business_rules(self) -> FlextCore.Result[None]:
                """Validate money rules."""
                if self.amount < 0:
                    return FlextCore.Result[None].fail("Amount cannot be negative")
                return FlextCore.Result[None].ok(None)

        order_id = OrderId.model_validate({"value": "ORD-12345"})
        money = Money.model_validate({"amount": 99.99, "currency": "USD"})
        return order_id, money

    def _create_ddd_aggregate(self, order_id: object, money: object) -> object:
        """Create DDD aggregate for testing."""

        # Aggregate Root
        class Order(FlextModels.AggregateRoot):
            """Order aggregate root."""

            order_id: str
            total: object
            status: str = "pending"

            def validate_business_rules(self) -> FlextCore.Result[None]:
                """Validate business rules for order entity."""
                return self.validate_domain_rules()

            def validate_domain_rules(self) -> FlextCore.Result[None]:
                """Validate order business rules."""
                # Validate value objects with type checking for serialization
                if hasattr(self.order_id, "validate_business_rules"):
                    validate_method = getattr(
                        self.order_id,
                        "validate_business_rules",
                        None,
                    )
                    order_id_validation = (
                        validate_method()
                        if validate_method
                        else FlextCore.Result[None].ok(None)
                    )
                    if (
                        hasattr(order_id_validation, "is_failure")
                        and order_id_validation.is_failure
                    ):
                        return FlextCore.Result[None].fail(
                            str(order_id_validation.error)
                        )

                if hasattr(self.total, "validate_business_rules"):
                    validate_method = getattr(
                        self.total,
                        "validate_business_rules",
                        None,
                    )
                    total_validation = (
                        validate_method()
                        if validate_method
                        else FlextCore.Result[None].ok(None)
                    )
                    if (
                        hasattr(total_validation, "is_failure")
                        and total_validation.is_failure
                    ):
                        return FlextCore.Result[None].fail(str(total_validation.error))

                # Business rules
                if self.status not in {"pending", "confirmed", "shipped", "delivered"}:
                    return FlextCore.Result[None].fail("Invalid order status")

                return FlextCore.Result[None].ok(None)

            def confirm_order(self) -> FlextCore.Result[None]:
                """Domain behavior: confirm order."""
                if self.status != "pending":
                    return FlextCore.Result[None].fail(
                        "Can only confirm pending orders"
                    )

                # Create new instance with updated status (immutable pattern)
                # Use object type to avoid forward reference issues with locally defined Order class
                result = FlextCore.Result[object].ok(
                    self.model_copy(update={"status": "confirmed"}),
                )
                if result.is_failure:
                    return FlextCore.Result[None].fail(
                        f"Failed to confirm order: {result.error}",
                    )

                return FlextCore.Result[None].ok(None)

        return Order(
            id="order_123",
            order_id=getattr(order_id, "value", str(order_id)),
            total=money,
        )

    def _test_ddd_validation_and_behavior(self, order: object) -> None:
        """Test DDD validation and behavior."""
        # Test domain validation
        if hasattr(order, "validate_business_rules"):
            validate_method = getattr(order, "validate_business_rules", None)
            validation_result = (
                validate_method()
                if validate_method
                else FlextCore.Result[None].ok(None)
            )
            assert hasattr(validation_result, "is_success")
            assert validation_result.is_success

        # Test domain behavior
        if hasattr(order, "confirm_order"):
            confirm_method = getattr(order, "confirm_order", None)
            confirm_result = (
                confirm_method() if confirm_method else FlextCore.Result[None].ok(None)
            )
            assert hasattr(confirm_result, "is_success")
            assert confirm_result.is_success

    @pytest.mark.architecture
    def test_cqrs_pattern_implementation(self) -> None:
        """Test CQRS (Command Query Responsibility Segregation) pattern."""

        # Commands (Write Operations)
        class UpdateUserCommand(FlextCore.Models.Command):
            """Command to update user information using FlextCore.Models foundation."""

            user_id: str
            name: str

        class UpdateUserHandler(FlextCore.Processors.Handler):
            """Handler for user update commands."""

            def handle(self, request: object) -> FlextCore.Result[object]:
                """Handle user update command."""
                if not isinstance(request, UpdateUserCommand):
                    return FlextCore.Result[object].fail("Invalid command type")

                command = request

                # Simulate business logic
                if not command.name.strip():
                    return FlextCore.Result[object].fail("Name cannot be empty")

                return FlextCore.Result[object].ok(f"User {command.user_id} updated")

        # Queries (Read Operations)
        class GetUserQuery(FlextCore.Models.Query):
            """Query to get user information using FlextCore.Models foundation."""

            user_id: str

        class GetUserHandler(FlextCore.Processors.Handler):
            """Handler for user queries."""

            def handle(self, request: object) -> FlextCore.Result[object]:
                """Handle user query."""
                if not isinstance(request, GetUserQuery):
                    return FlextCore.Result[object].fail("Invalid query type")

                query = request

                # Simulate data retrieval
                user_data: FlextCore.Types.Dict = {
                    "id": query.user_id,
                    "name": "John Doe",
                    "email": "john@example.com",
                }

                return FlextCore.Result[object].ok(user_data)

        # Test CQRS separation
        command = UpdateUserCommand(user_id="123", name="Jane Doe")
        command_handler = UpdateUserHandler()

        command_result = command_handler.handle(command)
        assert command_result.is_success
        assert "updated" in str(command_result.value)

        query = GetUserQuery(user_id="123")
        query_handler = GetUserHandler()

        query_result = query_handler.handle(query)
        assert query_result.is_success
        assert isinstance(query_result.value, dict)


class TestEnterprisePatterns:
    """Test enterprise design patterns."""

    @pytest.mark.architecture
    def test_factory_pattern_implementation(self) -> None:
        """Test Factory pattern implementation."""

        class ServiceFactory:
            """Factory for creating different types of services."""

            @staticmethod
            def create_service(
                service_type: str,
            ) -> FlextCore.Result[FlextCore.Types.StringDict]:
                """Create service based on type."""
                if service_type == "email":
                    return FlextCore.Result[FlextCore.Types.StringDict].ok(
                        {
                            "type": "email",
                            "provider": "smtp",
                        },
                    )
                if service_type == "sms":
                    return FlextCore.Result[FlextCore.Types.StringDict].ok(
                        {
                            "type": "sms",
                            "provider": "twilio",
                        },
                    )
                return FlextCore.Result[FlextCore.Types.StringDict].fail(
                    f"Unknown service type: {service_type}",
                )

        # Test factory usage
        email_service = ServiceFactory.create_service("email")
        assert email_service.is_success
        assert isinstance(email_service.value, dict)
        assert email_service.value["type"] == "email"

        sms_service = ServiceFactory.create_service("sms")
        assert sms_service.is_success
        assert isinstance(sms_service.value, dict)
        assert sms_service.value["type"] == "sms"

        invalid_service = ServiceFactory.create_service("invalid")
        assert invalid_service.is_failure

    @pytest.mark.architecture
    def test_builder_pattern_implementation(self) -> None:
        """Test Builder pattern implementation."""

        class ConfigurationBuilder:
            """Builder for complex configuration objects."""

            def __init__(self) -> None:
                """Initialize builder."""
                super().__init__()
                self._config: FlextCore.Types.Dict = {}

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

            def build(self) -> FlextCore.Result[FlextCore.Types.Dict]:
                """Build the configuration."""
                if not self._config:
                    return FlextCore.Result[FlextCore.Types.Dict].fail(
                        "Configuration cannot be empty",
                    )

                return FlextCore.Result[FlextCore.Types.Dict].ok(self._config.copy())

        # Test builder usage
        config_result = (
            ConfigurationBuilder()
            .with_database("localhost", 5432)
            .with_logging("INFO")
            .with_cache(enabled=True)
            .build()
        )

        assert config_result.is_success
        config = config_result.value
        assert isinstance(config, dict)
        config_dict = config  # Already FlextCore.Types.Dict from assertion
        database_dict = cast("FlextCore.Types.Dict", config_dict["database"])
        assert database_dict["host"] == "localhost"
        logging_dict = cast("FlextCore.Types.Dict", config_dict["logging"])
        assert logging_dict["level"] == "INFO"
        cache_dict = cast("FlextCore.Types.Dict", config_dict["cache"])
        assert cache_dict["enabled"]

    @pytest.mark.architecture
    @pytest.mark.performance
    def test_repository_pattern_performance(self) -> None:
        """Test Repository pattern with performance considerations."""

        class InMemoryRepository:
            """In-memory repository implementation."""

            def __init__(self) -> None:
                """Initialize repository."""
                super().__init__()
                self._data: FlextCore.Types.Dict = {}
                self._query_count = 0

            def save(self, entity_id: str, data: object) -> FlextCore.Result[None]:
                """Save entity to repository."""
                self._data[entity_id] = data
                return FlextCore.Result[None].ok(None)

            def find_by_id(self, entity_id: str) -> FlextCore.Result[object]:
                """Find entity by ID."""
                self._query_count += 1

                if entity_id in self._data:
                    return FlextCore.Result[object].ok(self._data[entity_id])

                return FlextCore.Result[object].fail(f"Entity not found: {entity_id}")

            def get_query_count(self) -> int:
                """Get number of queries executed."""
                return self._query_count

        # Test repository performance
        repo = InMemoryRepository()

        # Save multiple entities
        start_time = time.time()
        for i in range(1000):
            result = repo.save(f"entity_{i}", {"id": i, "name": f"Entity {i}"})
            assert result.is_success

        save_duration = time.time() - start_time

        # Query entities
        start_time = time.time()
        for i in range(100):
            query_result: FlextCore.Result[object] = repo.find_by_id(f"entity_{i}")
            assert query_result.is_success
            entity_data = cast("FlextCore.Types.Dict", query_result.value)
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
        class UserCreatedEvent(FlextCore.Models.DomainEvent):
            """Domain event for user creation using FlextCore.Models foundation."""

            user_id: str
            user_name: str
            timestamp: float

        class UserUpdatedEvent(FlextCore.Models.DomainEvent):
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
                super().__init__()
                self.processed_events: list[FlextCore.Models.DomainEvent] = []

            def handle_user_created(
                self, event: UserCreatedEvent
            ) -> FlextCore.Result[None]:
                """Handle user created event."""
                self.processed_events.append(event)
                return FlextCore.Result[None].ok(None)

            def handle_user_updated(
                self, event: UserUpdatedEvent
            ) -> FlextCore.Result[None]:
                """Handle user updated event."""
                self.processed_events.append(event)
                return FlextCore.Result[None].ok(None)

        # Test event processing
        handler = UserEventHandler()

        # Create and process events with required fields
        created_event = UserCreatedEvent(
            event_type="UserCreated",
            aggregate_id="user_123",
            user_id="123",
            user_name="John Doe",
            timestamp=time.time(),
        )

        updated_event = UserUpdatedEvent(
            event_type="UserUpdated",
            aggregate_id="user_123",
            user_id="123",
            old_name="John Doe",
            new_name="Jane Doe",
            timestamp=time.time(),
        )

        # Process events
        result1 = handler.handle_user_created(created_event)
        assert result1.is_success

        result2 = handler.handle_user_updated(updated_event)
        assert result2.is_success

        # Verify event processing
        assert len(handler.processed_events) == 2
        assert isinstance(handler.processed_events[0], UserCreatedEvent)
        assert isinstance(handler.processed_events[1], UserUpdatedEvent)

    @pytest.mark.architecture
    def test_observer_pattern_implementation(self) -> None:
        """Test Observer pattern implementation."""
        observers: list[FlextCore.Types.Dict] = []

        def notify_all(state: str) -> None:
            for observer in observers:
                observer["state"] = state

        # Create observers
        obs1: FlextCore.Types.Dict = {"name": "Observer1", "state": None}
        obs2: FlextCore.Types.Dict = {"name": "Observer2", "state": None}
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
