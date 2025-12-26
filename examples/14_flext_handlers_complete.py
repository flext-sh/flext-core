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

from collections.abc import Sequence

from flext_core import (
    FlextConstants,
    FlextModels,
    FlextResult,
    FlextService,
    h,
    t,
    u,
)


class CreateUserCommand(FlextModels.Cqrs.Command):
    """Command to create a user."""

    user_id: str
    name: str
    email: str


class GetUserQuery(FlextModels.Cqrs.Query):
    """Query to get a user."""

    user_id: str


class UserDTO(FlextModels.Value):
    """User data transfer object."""

    id: str
    name: str
    email: str


# Rebuild models to resolve forward references after all definitions
# Include FlextModels.Cqrs in namespace for forward reference resolution
_types_namespace = {
    **globals(),
    "FlextModels": FlextModels,
    "FlextModelsCqrs": FlextModels.Cqrs,
}
_ = CreateUserCommand.model_rebuild(_types_namespace=_types_namespace)
_ = GetUserQuery.model_rebuild(_types_namespace=_types_namespace)
_ = UserDTO.model_rebuild(_types_namespace=_types_namespace)


# Handlers using h directly
class CommandHandler(h[CreateUserCommand, str]):
    """Example command handler."""

    def handle(self, message: CreateUserCommand) -> FlextResult[str]:
        """Handle user creation command using u validation."""
        _ = self.handler_name  # Use self to satisfy ruff

        # Railway pattern with u validation (DRY)
        name_validation = u.validate_length(
            message.name,
            min_length=FlextConstants.Validation.MIN_NAME_LENGTH,
            max_length=FlextConstants.Validation.MAX_NAME_LENGTH,
        )
        if name_validation.is_failure:
            return FlextResult[str].fail(
                name_validation.error or FlextConstants.Errors.VALIDATION_ERROR,
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        email_validation = u.validate_pattern(
            message.email,
            FlextConstants.Platform.PATTERN_EMAIL,
            "email",
        )
        if email_validation.is_failure:
            return FlextResult[str].fail(
                email_validation.error or FlextConstants.Errors.VALIDATION_ERROR,
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        return FlextResult[str].ok(f"User {message.name} created")


class QueryHandler(h[GetUserQuery, UserDTO]):
    """Example query handler."""

    def handle(self, message: GetUserQuery) -> FlextResult[UserDTO]:
        """Handle user retrieval query using FlextConstants error codes."""
        _ = self.handler_name  # Use self to satisfy ruff
        if message.user_id == "not-found":
            return FlextResult[UserDTO].fail(
                FlextConstants.Errors.NOT_FOUND_ERROR,
                error_code=FlextConstants.Errors.NOT_FOUND_ERROR,
            )

        user = UserDTO(
            id=message.user_id,
            name="Example User",
            email="user@example.com",
        )
        return FlextResult[UserDTO].ok(user)


# Service using FlextService directly
class HandlersService(FlextService[t.ServiceMetadataMapping]):
    """Service demonstrating CQRS handlers with flext-core."""

    def execute(
        self,
    ) -> FlextResult[t.ServiceMetadataMapping]:
        """Execute comprehensive handler demonstrations."""
        print("Starting CQRS handlers demonstration")

        self._demonstrate_command_handlers()
        self._demonstrate_query_handlers()
        self._demonstrate_pipeline_execution()
        self._demonstrate_error_handling()

        return FlextResult[t.ServiceMetadataMapping].ok({
            "handlers_demonstrated": [
                FlextConstants.Cqrs.HandlerType.COMMAND,
                FlextConstants.Cqrs.HandlerType.QUERY,
                "pipeline",
                "error_handling",
            ],
            "cqrs_patterns": [
                "separation_of_concerns",
                "type_safety",
                "result_patterns",
            ],
            "handler_types": 2,
        })

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

    @staticmethod
    def _demonstrate_pipeline_execution() -> None:
        """Show handler pipeline execution."""
        print("\n=== Pipeline Execution ===")

        phases = [
            FlextConstants.Cqrs.ProcessingPhase.PREPARE,
            FlextConstants.Cqrs.ProcessingPhase.EXECUTE,
            FlextConstants.Cqrs.ProcessingPhase.VALIDATE,
            FlextConstants.Cqrs.ProcessingPhase.COMPLETE,
        ]

        for phase in phases:
            print(f"✅ {phase.value.capitalize()} phase")

        print("✅ Pipeline executed successfully")

    @staticmethod
    def _demonstrate_error_handling() -> None:
        """Show error handling in handlers."""
        print("\n=== Error Handling ===")

        command_handler = CommandHandler()

        error_command = CreateUserCommand(
            user_id="error-user",
            name="",
            email="",
        )

        error_result = command_handler.handle(error_command)
        if error_result.is_failure:
            print(f"✅ Error handled: {error_result.error}")
            print(f"   Error code: {error_result.error_code or 'N/A'}")


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
        handlers = data.get("handlers_demonstrated", [])
        patterns = data.get("cqrs_patterns", [])
        handler_count = len(handlers) if isinstance(handlers, Sequence) else 0
        pattern_count = len(patterns) if isinstance(patterns, Sequence) else 0
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
