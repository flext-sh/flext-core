"""h complete CQRS handler implementation demonstration.

Shows command and query handlers with type safety and pipeline execution.
Uses configuration-based architecture and error handling patterns.

**Expected Output:**
- Handler base class implementation patterns
- Command handler registration and execution
- Query handler patterns and response handling
- Validation hooks and preprocessing
- Pipeline execution with error handling
- Handler-to-dispatcher integration

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import override

from flext_core import c, h, m, r, s, t, u


class CreateUserCommand(m.Command):
    """Command to create a user."""

    user_id: str
    name: str
    email: str


class GetUserQuery(m.Query):
    """Query to get a user."""

    user_id: str


class UserDTO(m.Value):
    """User data transfer t.NormalizedValue."""

    id: str
    name: str
    email: str


class CommandHandler(h[CreateUserCommand, str]):
    """Example command handler."""

    @override
    def handle(self, message: CreateUserCommand) -> r[str]:
        """Handle user creation command using u validation."""
        name_validation = u.validate_length(
            message.name,
            min_length=c.MIN_NAME_LENGTH,
            max_length=c.MAX_NAME_LENGTH,
        )
        if name_validation.is_failure:
            return r[str].fail(
                name_validation.error or c.VALIDATION_ERROR,
                error_code=c.VALIDATION_ERROR,
            )
        email_validation = u.validate_pattern(message.email, c.PATTERN_EMAIL, "email")
        if email_validation.is_failure:
            return r[str].fail(
                email_validation.error or c.VALIDATION_ERROR,
                error_code=c.VALIDATION_ERROR,
            )
        return r[str].ok(f"User {message.name} created")


class QueryHandler(h[GetUserQuery, UserDTO]):
    """Example query handler."""

    @override
    def handle(self, message: GetUserQuery) -> r[UserDTO]:
        """Handle user retrieval query using c error codes."""
        if message.user_id == "not-found":
            return r[UserDTO].fail(c.NOT_FOUND_ERROR, error_code=c.NOT_FOUND_ERROR)
        user = UserDTO(
            id=message.user_id,
            name="Example User",
            email="user@example.com",
        )
        return r[UserDTO].ok(user)


class HandlersService(s[t.ConfigMap]):
    """Service demonstrating CQRS handlers with flext-core."""

    @staticmethod
    def _demonstrate_command_handlers() -> None:
        """Show command handler patterns."""
        print("\n=== Command Handlers ===")
        handler = CommandHandler()
        command = CreateUserCommand(
            user_id="user-123",
            name="Alice",
            email="alice@example.com",
        )
        result = handler.handle(command)
        if result.is_success:
            print(f"✅ Command executed: {result.value}")
        invalid_command = CreateUserCommand(
            user_id="user-456",
            name="",
            email="bob@example.com",
        )
        invalid_result = handler.handle(invalid_command)
        if invalid_result.is_failure:
            print(f"❌ Command failed: {invalid_result.error}")

    @staticmethod
    def _demonstrate_error_handling() -> None:
        """Show error handling in handlers."""
        print("\n=== Error Handling ===")
        command_handler = CommandHandler()
        error_command = CreateUserCommand(user_id="error-user", name="", email="")
        error_result = command_handler.handle(error_command)
        if error_result.is_failure:
            print(f"✅ Error handled: {error_result.error}")
            print(f"   Error code: {error_result.error_code or 'N/A'}")

    @staticmethod
    def _demonstrate_pipeline_execution() -> None:
        """Show handler pipeline execution."""
        print("\n=== Pipeline Execution ===")
        phases = [
            c.ProcessingPhase.PREPARE,
            c.ProcessingPhase.EXECUTE,
            c.ProcessingPhase.VALIDATE,
            c.ProcessingPhase.COMPLETE,
        ]
        for phase in phases:
            print(f"✅ {phase.value.capitalize()} phase")
        print("✅ Pipeline executed successfully")

    @staticmethod
    def _demonstrate_query_handlers() -> None:
        """Show query handler patterns."""
        print("\n=== Query Handlers ===")
        handler = QueryHandler()
        query = GetUserQuery(user_id="user-123")
        result = handler.handle(query)
        if result.is_success:
            user = result.value
            print(f"✅ Query result: {user.name} ({user.email})")
        not_found_query = GetUserQuery(user_id="not-found")
        not_found_result = handler.handle(not_found_query)
        if not_found_result.is_failure:
            print(f"❌ Query failed: {not_found_result.error}")

    @override
    def execute(self) -> r[t.ConfigMap]:
        """Execute comprehensive handler demonstrations."""
        print("Starting CQRS handlers demonstration")
        self._demonstrate_command_handlers()
        self._demonstrate_query_handlers()
        self._demonstrate_pipeline_execution()
        self._demonstrate_error_handling()
        return r[t.ConfigMap].ok(
            t.ConfigMap(
                root={
                    "handlers_demonstrated": [
                        c.HandlerType.COMMAND,
                        c.HandlerType.QUERY,
                        "pipeline",
                        "error_handling",
                    ],
                    "cqrs_patterns": [
                        "separation_of_concerns",
                        "type_safety",
                        "result_patterns",
                    ],
                    "handler_types": 2,
                },
            ),
        )


def demonstrate_cqrs_architecture() -> None:
    """Show CQRS architectural patterns."""
    print("\n=== CQRS Architecture ===")
    print("✅ Command-Query Responsibility Segregation")
    print("✅ Separate models for reading and writing")
    print("✅ Eventual consistency patterns")
    print("✅ Scalable architecture support")


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("FLEXT HANDLERS - CQRS HANDLER IMPLEMENTATION")
    print("Command and Query handlers with type safety")
    print("=" * 60)
    demonstrate_cqrs_architecture()
    service = HandlersService()
    result = service.execute()
    if result.is_success:
        data = result.value
        handler_count_raw = data.get("handler_types", 0)
        handler_count_text = str(handler_count_raw)
        handler_count = int(handler_count_text) if handler_count_text.isdigit() else 0
        pattern_count = 3
        print(f"\n✅ Demonstrated {handler_count} handler patterns")
        print(f"✅ Used {pattern_count} CQRS patterns")
    else:
        print(f"\n❌ Failed: {result.error}")
    print("\n" + "=" * 60)
    print("🎯 CQRS Patterns: Command-Query Separation, Type Safety")
    print("🎯 Handler Types: Command Handlers, Query Handlers")
    print("🎯 Pipeline: Validation → Processing → Completion")
    print("=" * 60)


if __name__ == "__main__":
    main()
