"""Example: Log Configuration Once Pattern.

Demonstrates how to log configuration ONCE without it repeating
in all subsequent log messages.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from flext_core import FlextResult, FlextService


class DatabaseService(FlextService[dict[str, object]]):
    """Example service showing config log-once pattern."""

    def __init__(self, config: dict[str, object]) -> None:
        """Initialize service with configuration.

        Args:
            config: Database configuration

        """
        super().__init__()

        # ✅ CORRECT: Log config ONCE, doesn't appear in all subsequent logs
        self._log_config_once(config, message="Database configuration loaded")

        # ❌ WRONG: DO NOT pass config to _with_operation_context
        # self._with_operation_context("init", config=config)  # ← This binds config to ALL logs!

    def execute(self) -> FlextResult[dict[str, object]]:
        """Execute database operations.

        Returns:
            FlextResult[dict]: Operation results

        """
        # Set operation context WITHOUT config
        self._with_operation_context(
            "database_query",
            operation_type="select",
            table="users",
        )

        # This log will NOT include config - only operation context
        self.logger.info("Executing database query")

        # Simulate query
        results: dict[str, object] = {"users": [{"id": 1, "name": "Alice"}]}

        return FlextResult[dict[str, object]].ok(results)


class MigrationService(FlextService[dict[str, object]]):
    """Example migration service with config log-once pattern."""

    def __init__(self, input_dir: str, output_dir: str, sync: bool) -> None:
        """Initialize migration service.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            sync: Enable synchronization

        """
        super().__init__()

        # Build config dict - use dict[str, object] for type compatibility
        config: dict[str, object] = {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "sync": sync,
            "batch_size": 100,
            "max_workers": 4,
        }

        # ✅ CORRECT: Log config ONCE at initialization
        self._log_config_once(config, message="Migration configuration loaded")

    def execute(self) -> FlextResult[dict[str, object]]:
        """Execute migration.

        Returns:
            FlextResult[dict[str, object]]: Migration results

        """
        # Set operation context with business data (NOT config)
        self._with_operation_context(
            "migration_process",
            total_entries=1000,
            batch_count=10,
        )

        # Config is NOT in this log or any subsequent logs
        self.logger.info("Starting migration process")

        # Simulate migration
        self.logger.info("Processing batch 1 of 10")
        self.logger.info("Processing batch 2 of 10")
        # Config does NOT repeat in these logs!

        return FlextResult[dict[str, object]].ok({"migrated": 1000, "failed": 0})


def main() -> None:
    """Demonstrate config log-once pattern."""
    print("=== Example 1: Database Service ===")
    db_config: dict[str, object] = {
        "host": "localhost",
        "port": 5432,
        "database": "mydb",
        "pool_size": 10,
    }

    db_service = DatabaseService(db_config)
    result = db_service.execute()

    if result.is_success:
        print(f"✅ Database query successful: {result.unwrap()}")

    print("\n=== Example 2: Migration Service ===")
    migration_service = MigrationService(
        input_dir="/data/input",
        output_dir="/data/output",
        sync=True,
    )

    result = migration_service.execute()

    if result.is_success:
        print(f"✅ Migration successful: {result.unwrap()}")

    print("\n=== Key Observations ===")
    print("1. Config logged ONCE when service initialized")
    print("2. Config does NOT appear in subsequent logs")
    print("3. Operation context (total_entries, batch_count) appears in all logs")
    print("4. Clean, relevant logs without config repetition!")


if __name__ == "__main__":
    main()
