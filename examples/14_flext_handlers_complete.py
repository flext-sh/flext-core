#!/usr/bin/env python3
"""14 - FlextHandlers: Complete CQRS Handler Implementation.

This example demonstrates the COMPLETE FlextHandlers API for implementing
CQRS handlers with configuration-based architecture, type safety, and pipeline execution.

Key Concepts Demonstrated:
- Handler Configuration: Config-based handler creation
- CQRS Pattern: Command and Query handler separation
- Type Safety: Generic message and result types
- Pipeline Execution: Validation → Processing → Completion
- Error Handling: Explicit FlextResult error patterns
- Factory Methods: from_callable handler creation
- Companion Modules: Configuration, validation, metrics, context
- Integration: Real-world usage patterns

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from decimal import Decimal

# ========== MESSAGE DEFINITIONS ==========
from uuid import uuid4

from flext_core import (
    FlextContainer,
    FlextHandlers,
    FlextLogger,
    FlextModels,
    FlextResult,
    FlextService,
)


@dataclass
class CreateUserCommand:
    """Command to create a new user."""

    user_id: str
    name: str
    email: str
    age: int


@dataclass
class UpdateUserCommand:
    """Command to update user information."""

    user_id: str
    name: str | None = None
    email: str | None = None


@dataclass
class GetUserQuery:
    """Query to retrieve user information."""

    user_id: str


@dataclass
class ListUsersQuery:
    """Query to list users with pagination."""

    limit: int = 10
    offset: int = 0


@dataclass
class UserCreatedEvent:
    """Event emitted when user is created."""

    user_id: str
    name: str
    email: str
    timestamp: float


# ========== DOMAIN MODELS ==========


@dataclass
class User:
    """User domain model."""

    id: str
    name: str
    email: str
    age: int
    created_at: float
    updated_at: float | None = None


# ========== HANDLERS SERVICE ==========


class FlextHandlersService(FlextService[dict[str, str | bool]]):
    """Service demonstrating ALL FlextHandlers patterns."""

    def __init__(self) -> None:
        """Initialize with dependencies."""
        super().__init__()
        self._logger = FlextLogger(__name__)
        self._container = FlextContainer.get_global()
        # In-memory stores for demo
        self._users: dict[str, User] = {}
        self._events: list[UserCreatedEvent] = []

    def execute(self) -> FlextResult[dict[str, str | bool]]:
        """Execute method required by FlextService."""
        self._logger.info("Executing FlextHandlers demo")
        return FlextResult[dict[str, str | bool]].ok({
            "status": "processed",
            "handlers_executed": True,
        })

    # ========== COMMAND HANDLERS ==========

    def demonstrate_command_handlers(self) -> None:
        """Show command handler patterns with new FlextHandlers API."""
        print("\n=== Command Handlers with FlextHandlers ===")

        # Create User Command Handler
        class CreateUserHandler(FlextHandlers[CreateUserCommand, str]):
            """Handler for creating new users."""

            def __init__(
                self, user_store: dict[str, User], event_store: list[UserCreatedEvent]
            ) -> None:
                # Create handler configuration
                config = FlextModels.CqrsConfig.Handler(
                    handler_id="create_user_handler",
                    handler_name="Create User Handler",
                    handler_type="command",
                    handler_mode="command",
                    command_timeout=5000,
                    max_command_retries=3,
                    metadata={"description": "Handles user creation commands"},
                )
                super().__init__(config=config)
                self._users = user_store
                self._events = event_store
                self._logger = FlextLogger(__name__)

            def handle(self, message: CreateUserCommand) -> FlextResult[str]:
                """Handle user creation command."""
                self._logger.info(f"Creating user {message.user_id}")

                # Validation
                if message.user_id in self._users:
                    return FlextResult[str].fail(
                        f"User {message.user_id} already exists",
                        error_code="USER_EXISTS",
                    )

                if message.age < 0 or message.age > 150:
                    return FlextResult[str].fail(
                        "Invalid age provided", error_code="VALIDATION_ERROR"
                    )

                # Create user
                user = User(
                    id=message.user_id,
                    name=message.name,
                    email=message.email,
                    age=message.age,
                    created_at=time.time(),
                )
                self._users[message.user_id] = user

                # Emit event
                event = UserCreatedEvent(
                    user_id=message.user_id,
                    name=message.name,
                    email=message.email,
                    timestamp=time.time(),
                )
                self._events.append(event)

                return FlextResult[str].ok(message.user_id)

        # Update User Command Handler
        class UpdateUserHandler(FlextHandlers[UpdateUserCommand, None]):
            """Handler for updating existing users."""

            def __init__(self, user_store: dict[str, User]) -> None:
                config = FlextModels.CqrsConfig.Handler(
                    handler_id="update_user_handler",
                    handler_name="Update User Handler",
                    handler_type="command",
                    handler_mode="command",
                    command_timeout=3000,
                    max_command_retries=2,
                    metadata={"description": "Handles user update commands"},
                )
                super().__init__(config=config)
                self._users = user_store
                self._logger = FlextLogger(__name__)

            def handle(self, message: UpdateUserCommand) -> FlextResult[None]:
                """Handle user update command."""
                self._logger.info(f"Updating user {message.user_id}")

                if message.user_id not in self._users:
                    return FlextResult[None].fail(
                        f"User {message.user_id} not found", error_code="USER_NOT_FOUND"
                    )

                user = self._users[message.user_id]

                # Apply updates
                if message.name is not None:
                    user.name = message.name
                if message.email is not None:
                    user.email = message.email

                user.updated_at = time.time()

                return FlextResult[None].ok(None)

        # Use command handlers
        print("\n1. Creating Command Handlers:")
        create_handler = CreateUserHandler(self._users, self._events)
        update_handler = UpdateUserHandler(self._users)

        print("✅ Handlers created with configuration-based initialization")

        print("\n2. Executing Commands:")

        # Create user command
        create_cmd = CreateUserCommand(
            user_id="USER-001", name="Alice Smith", email="alice@example.com", age=28
        )

        create_result = create_handler.handle(create_cmd)
        if create_result.is_success:
            print(f"  ✅ User created: {create_result.unwrap()}")
        else:
            print(f"  ❌ Creation failed: {create_result.error}")

        # Update user command
        update_cmd = UpdateUserCommand(
            user_id="USER-001", name="Alice Johnson", email="alice.johnson@example.com"
        )

        update_result = update_handler.handle(update_cmd)
        if update_result.is_success:
            print("  ✅ User updated successfully")
        else:
            print(f"  ❌ Update failed: {update_result.error}")

        print("\n3. Results:")
        print(f"  Users in store: {len(self._users)}")
        print(f"  Events emitted: {len(self._events)}")

        if self._users:
            user = next(iter(self._users.values()))
            print(f"  Sample user: {user.name} ({user.email})")

    # ========== QUERY HANDLERS ==========

    def demonstrate_query_handlers(self) -> None:
        """Show query handler patterns with new FlextHandlers API."""
        print("\n=== Query Handlers with FlextHandlers ===")

        # Get User Query Handler
        class GetUserHandler(FlextHandlers[GetUserQuery, User]):
            """Handler for retrieving individual users."""

            def __init__(self, user_store: dict[str, User]) -> None:
                config = FlextModels.CqrsConfig.Handler(
                    handler_id="get_user_handler",
                    handler_name="Get User Handler",
                    handler_type="query",
                    handler_mode="query",
                    command_timeout=1000,
                    max_command_retries=1,
                    metadata={"description": "Handles single user queries"},
                )
                super().__init__(config=config)
                self._users = user_store
                self._logger = FlextLogger(__name__)

            def handle(self, message: GetUserQuery) -> FlextResult[User]:
                """Handle get user query."""
                self._logger.info(f"Getting user {message.user_id}")

                user = self._users.get(message.user_id)
                if not user:
                    return FlextResult[User].fail(
                        f"User {message.user_id} not found", error_code="USER_NOT_FOUND"
                    )

                return FlextResult[User].ok(user)

        # List Users Query Handler
        class ListUsersHandler(FlextHandlers[ListUsersQuery, list[User]]):
            """Handler for listing users with pagination."""

            def __init__(self, user_store: dict[str, User]) -> None:
                config = FlextModels.CqrsConfig.Handler(
                    handler_id="list_users_handler",
                    handler_name="List Users Handler",
                    handler_type="query",
                    handler_mode="query",
                    command_timeout=2000,
                    max_command_retries=1,
                    metadata={
                        "description": "Handles user list queries with pagination"
                    },
                )
                super().__init__(config=config)
                self._users = user_store
                self._logger = FlextLogger(__name__)

            def handle(self, message: ListUsersQuery) -> FlextResult[list[User]]:
                """Handle list users query."""
                self._logger.info(
                    f"Listing users (limit: {message.limit}, offset: {message.offset})"
                )

                users = list(self._users.values())
                # Apply pagination
                paginated_users = users[message.offset : message.offset + message.limit]

                return FlextResult[list[User]].ok(paginated_users)

        # Use query handlers
        print("\n1. Creating Query Handlers:")
        get_handler = GetUserHandler(self._users)
        list_handler = ListUsersHandler(self._users)

        print("✅ Query handlers created with type-safe generics")

        print("\n2. Executing Queries:")

        # Get user query
        get_query = GetUserQuery(user_id="USER-001")
        get_result = get_handler.handle(get_query)

        if get_result.is_success:
            user = get_result.unwrap()
            print(f"  ✅ User retrieved: {user.name} (age: {user.age})")
        else:
            print(f"  ❌ Get failed: {get_result.error}")

        # List users query
        list_query = ListUsersQuery(limit=5, offset=0)
        list_result = list_handler.handle(list_query)

        if list_result.is_success:
            users = list_result.unwrap()
            print(f"  ✅ Users listed: {len(users)} users retrieved")
            for user in users:
                print(f"    - {user.name} ({user.email})")
        else:
            print(f"  ❌ List failed: {list_result.error}")

    # ========== FACTORY METHODS ==========

    def demonstrate_factory_methods(self) -> None:
        """Show handler creation using factory methods."""
        print("\n=== Handler Factory Methods ===")

        # Create handler from callable function
        def validate_email(email: str) -> FlextResult[bool]:
            """Simple email validation function."""
            if "@" in email and "." in email:
                return FlextResult[bool].ok(True)
            return FlextResult[bool].fail("Invalid email format")

        # Create wrapper function that handles object -> str conversion
        def validate_email_wrapper(data: object) -> FlextResult[bool]:
            """Wrapper function that converts object to str for email validation."""
            if not isinstance(data, str):
                return FlextResult[bool].fail("Email must be a string")
            return validate_email(data)

        print("\n1. Creating Handler from Callable:")

        # Create handler config for the callable
        handler_config = FlextModels.CqrsConfig.Handler(
            handler_id="email_validator",
            handler_name="Email Validator",
            handler_type="command",
            handler_mode="command",
            command_timeout=1000,
            max_command_retries=1,
            metadata={"description": "Validates email addresses"},
        )

        # Use from_callable factory method with wrapper
        email_handler = FlextHandlers.from_callable(
            handler_func=validate_email_wrapper,
            mode="command",
            handler_config=handler_config,
        )

        print("  ✅ Handler created from callable function")

        # Test the handler
        test_result = email_handler.handle("alice@example.com")
        if test_result.is_success:
            print(f"    ✅ Email validation: {test_result.unwrap()}")
        else:
            print(f"    ❌ Validation failed: {test_result.error}")

        # Test with invalid email
        invalid_result = email_handler.handle("invalid-email")
        if invalid_result.is_failure:
            print(f"    ✅ Invalid email detected: {invalid_result.error}")

        # Lambda function handler
        print("\n2. Creating Handler from Lambda:")

        lambda_config = FlextModels.CqrsConfig.Handler(
            handler_id="string_processor",
            handler_name="String Processor",
            handler_type="command",
            handler_mode="command",
            command_timeout=500,
            max_command_retries=1,
            metadata={"description": "Processes strings"},
        )

        # Create handler from lambda with proper type handling
        def string_processor_wrapper(data: object) -> FlextResult[str]:
            """Wrapper function that handles object -> str conversion for string processing."""
            if not isinstance(data, str):
                return FlextResult[str].fail("Input must be a string")
            return FlextResult[str].ok(data.upper() if data else "")

        string_handler = FlextHandlers.from_callable(
            handler_func=string_processor_wrapper,
            mode="command",
            handler_config=lambda_config,
        )

        print("  ✅ Lambda handler created successfully")

        # Test the lambda handler
        test_string_result = string_handler.handle("hello world")
        if test_string_result.is_success:
            print(f"    ✅ String processed: '{test_string_result.unwrap()}'")
        else:
            print(f"    ❌ String processing failed: {test_string_result.error}")

    # ========== ERROR HANDLING ==========

    def demonstrate_error_handling(self) -> None:
        """Show comprehensive error handling patterns."""
        print("\n=== Error Handling Patterns ===")

        class ValidationHandler(
            FlextHandlers[dict[str, str | int | float], dict[str, str | int | float]]
        ):
            """Handler demonstrating various error scenarios."""

            def __init__(self) -> None:
                config = FlextModels.CqrsConfig.Handler(
                    handler_id="validation_handler",
                    handler_name="Validation Handler",
                    handler_type="command",
                    handler_mode="command",
                    command_timeout=2000,
                    max_command_retries=2,
                    metadata={"description": "Demonstrates error handling patterns"},
                )
                super().__init__(config=config)
                self._logger = FlextLogger(__name__)

            def handle(
                self, message: dict[str, str | int | float]
            ) -> FlextResult[dict[str, str | int | float]]:
                """Handle validation with comprehensive error handling."""
                self._logger.info("Processing validation request")

                # Required field validation
                if not message.get("id"):
                    return FlextResult[dict[str, str | int | float]].fail(
                        "Missing required field: id",
                        error_code="MISSING_FIELD",
                        error_data={
                            "field": "id",
                            "provided_fields": list(message.keys()),
                        },
                    )

                # Type validation
                if not isinstance(message.get("amount"), (int, float)):
                    return FlextResult[dict[str, str | int | float]].fail(
                        "Invalid type for amount field",
                        error_code="TYPE_ERROR",
                        error_data={
                            "field": "amount",
                            "expected": "number",
                            "actual": type(message.get("amount")).__name__,
                        },
                    )

                # Business rule validation
                amount_value = message.get("amount", 0)
                if not isinstance(amount_value, (int, float)):
                    return FlextResult[dict[str, str | int | float]].fail(
                        "Amount must be a number",
                        error_code="TYPE_ERROR",
                    )

                amount: int | float = amount_value
                if amount < 0:
                    return FlextResult[dict[str, str | int | float]].fail(
                        "Amount cannot be negative",
                        error_code="BUSINESS_RULE_VIOLATION",
                        error_data={"rule": "positive_amount", "value": amount},
                    )

                # Success case
                validated_data = {
                    "id": message["id"],
                    "amount": amount,
                    "validated_at": time.time(),
                    "status": "validated",
                }

                return FlextResult[dict[str, str | int | float]].ok(validated_data)

        # Test error handling
        print("\n1. Error Handling Scenarios:")
        validator = ValidationHandler()

        # Test missing field error
        missing_field_result = validator.handle({"amount": 100})
        if missing_field_result.is_failure:
            print(f"  ❌ Missing field: {missing_field_result.error}")
            print(f"    Error code: {missing_field_result.error_code}")
            print(f"    Error data: {missing_field_result.error_data}")

        # Test type error
        type_error_result = validator.handle({"id": "TEST", "amount": "not_a_number"})
        if type_error_result.is_failure:
            print(f"  ❌ Type error: {type_error_result.error}")
            print(f"    Error code: {type_error_result.error_code}")

        # Test business rule violation
        business_error_result = validator.handle({"id": "TEST", "amount": -50})
        if business_error_result.is_failure:
            print(f"  ❌ Business rule violation: {business_error_result.error}")
            print(f"    Error code: {business_error_result.error_code}")

        # Test success case
        success_result = validator.handle({"id": "TEST", "amount": 100})
        if success_result.is_success:
            print(f"  ✅ Validation successful: {success_result.unwrap()}")

    # ========== INTEGRATION PATTERNS ==========

    def demonstrate_integration_patterns(self) -> None:
        """Show real-world integration patterns."""
        print("\n=== Integration Patterns ===")

        # Order processing pipeline using multiple handlers
        @dataclass
        class ProcessOrderCommand:
            order_id: str
            customer_id: str
            items: list[dict[str, str | int | float]]
            total_amount: Decimal

        class OrderValidationHandler(
            FlextHandlers[ProcessOrderCommand, ProcessOrderCommand]
        ):
            """First stage: Order validation."""

            def __init__(self) -> None:
                config = FlextModels.CqrsConfig.Handler(
                    handler_id="order_validation",
                    handler_name="Order Validation Handler",
                    handler_type="command",
                    handler_mode="command",
                    command_timeout=3000,
                    max_command_retries=1,
                    metadata={"stage": "validation", "pipeline_order": 1},
                )
                super().__init__(config=config)

            def handle(
                self, message: ProcessOrderCommand
            ) -> FlextResult[ProcessOrderCommand]:
                """Validate order before processing."""
                if not message.items:
                    return FlextResult[ProcessOrderCommand].fail(
                        "Order must contain items"
                    )

                if message.total_amount <= 0:
                    return FlextResult[ProcessOrderCommand].fail(
                        "Total amount must be positive"
                    )

                return FlextResult[ProcessOrderCommand].ok(message)

        class OrderEnrichmentHandler(
            FlextHandlers[ProcessOrderCommand, ProcessOrderCommand]
        ):
            """Second stage: Order enrichment."""

            def __init__(self) -> None:
                config = FlextModels.CqrsConfig.Handler(
                    handler_id="order_enrichment",
                    handler_name="Order Enrichment Handler",
                    handler_type="command",
                    handler_mode="command",
                    command_timeout=5000,
                    max_command_retries=2,
                    metadata={"stage": "enrichment", "pipeline_order": 2},
                )
                super().__init__(config=config)

            def handle(
                self, message: ProcessOrderCommand
            ) -> FlextResult[ProcessOrderCommand]:
                """Enrich order with additional data."""
                # Simulate enrichment by adding timestamps and IDs
                enriched_items: list[dict[str, str | int | float]] = []
                for item in message.items:
                    enriched_item = item.copy()
                    enriched_item["item_id"] = f"ITEM-{uuid4().hex[:8]}"
                    enriched_item["processed_at"] = time.time()
                    enriched_items.append(enriched_item)

                # Create enriched command
                enriched_command = ProcessOrderCommand(
                    order_id=message.order_id,
                    customer_id=message.customer_id,
                    items=enriched_items,
                    total_amount=message.total_amount,
                )

                return FlextResult[ProcessOrderCommand].ok(enriched_command)

        class OrderPersistenceHandler(FlextHandlers[ProcessOrderCommand, str]):
            """Final stage: Order persistence."""

            def __init__(self) -> None:
                config = FlextModels.CqrsConfig.Handler(
                    handler_id="order_persistence",
                    handler_name="Order Persistence Handler",
                    handler_type="command",
                    handler_mode="command",
                    command_timeout=10000,
                    max_command_retries=3,
                    metadata={"stage": "persistence", "pipeline_order": 3},
                )
                super().__init__(config=config)

            def handle(self, message: ProcessOrderCommand) -> FlextResult[str]:
                """Persist order to storage."""
                # Simulate persistence
                confirmation_id = f"CONF-{uuid4().hex[:8]}"
                print(
                    f"    💾 Order {message.order_id} persisted with confirmation {confirmation_id}"
                )
                return FlextResult[str].ok(confirmation_id)

        # Execute integration pipeline
        print("\n1. Order Processing Pipeline:")

        # Create handlers
        validator = OrderValidationHandler()
        enricher = OrderEnrichmentHandler()
        persister = OrderPersistenceHandler()

        # Create test order
        order = ProcessOrderCommand(
            order_id=f"ORDER-{uuid4().hex[:8]}",
            customer_id="CUST-001",
            items=[
                {"product": "Widget", "quantity": 2, "price": 29.99},
                {"product": "Gadget", "quantity": 1, "price": 49.99},
            ],
            total_amount=Decimal("109.97"),
        )

        print(f"  📦 Processing order: {order.order_id}")

        # Execute pipeline using railway pattern
        pipeline_result = (
            FlextResult[ProcessOrderCommand]
            .ok(order)
            .flat_map(validator.handle)
            .flat_map(enricher.handle)
            .flat_map(persister.handle)
        )

        if pipeline_result.is_success:
            confirmation = pipeline_result.unwrap()
            print(f"  ✅ Pipeline completed successfully: {confirmation}")
        else:
            print(f"  ❌ Pipeline failed: {pipeline_result.error}")

    # ========== DEPRECATED PATTERNS ==========

    def demonstrate_deprecated_patterns(self) -> None:
        """Show deprecated handler patterns to avoid."""
        print("\n=== ⚠️ DEPRECATED PATTERNS ===")

        # OLD: Direct instantiation without config (DEPRECATED)
        warnings.warn(
            "Direct FlextHandlers instantiation is DEPRECATED! Use config-based initialization.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("❌ OLD WAY (direct instantiation):")
        print("class OldHandler(FlextHandlers):")
        print("    def __init__(self):")
        print("        super().__init__()  # No config provided")

        print("\n✅ CORRECT WAY (config-based):")
        print("class NewHandler(FlextHandlers[MessageT, ResultT]):")
        print("    def __init__(self):")
        print("        config = FlextModels.CqrsConfig.Handler(...)")
        print("        super().__init__(config=config)")

        # OLD: Exception-based error handling (DEPRECATED)
        warnings.warn(
            "Exception-based error handling is DEPRECATED! Use FlextResult patterns.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (exceptions):")
        print("def handle(self, message):")
        print("    if invalid:")
        print("        raise ValueError('Invalid input')")
        print("    return result")

        print("\n✅ CORRECT WAY (FlextResult):")
        print("def handle(self, message) -> FlextResult[T]:")
        print("    if invalid:")
        print("        return FlextResult[T].fail('Invalid input')")
        print("    return FlextResult[T].ok(result)")

        # OLD: Mixed command/query handlers (DEPRECATED)
        warnings.warn(
            "Mixed command/query handlers are DEPRECATED! Separate by responsibility.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (mixed responsibilities):")
        print("class UserHandler:")
        print("    def create_and_get_user(self): ...")  # Both write and read

        print("\n✅ CORRECT WAY (separated):")
        print("class CreateUserHandler(FlextHandlers[CreateCommand, str]): ...")
        print("class GetUserHandler(FlextHandlers[GetQuery, User]): ...")


def main() -> None:
    """Main entry point demonstrating all FlextHandlers capabilities."""
    service = FlextHandlersService()

    print("=" * 70)
    print("FLEXTHANDLERS COMPLETE API DEMONSTRATION")
    print("Configuration-based CQRS Handler Implementation")
    print("=" * 70)

    # Core handler patterns
    service.demonstrate_command_handlers()
    service.demonstrate_query_handlers()

    # Advanced patterns
    service.demonstrate_factory_methods()
    service.demonstrate_error_handling()
    service.demonstrate_integration_patterns()

    # Deprecation warnings
    service.demonstrate_deprecated_patterns()

    print("\n" + "=" * 70)
    print("✅ ALL FlextHandlers patterns demonstrated!")
    print("🎯 Key Features:")
    print("   • Configuration-based initialization")
    print("   • Type-safe generic handlers")
    print("   • Railway-oriented error handling")
    print("   • CQRS command/query separation")
    print("   • Factory method support")
    print("   • Pipeline composition")
    print("=" * 70)


if __name__ == "__main__":
    main()
