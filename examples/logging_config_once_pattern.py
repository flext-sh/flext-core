"""Example: Log Configuration Once Pattern.

Demonstrates how to log configuration ONCE without it repeating
in all subsequent log messages.

**Expected Output:**
- Initial configuration logging demonstration
- Subsequent log messages without repeated configuration
- Idempotent logging configuration pattern
- Context-aware logging setup

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import override

from flext_core import c, m, r, s, t


class DatabaseService(s[m.ConfigMap]):
    """Example service showing config log-once pattern."""

    db_config: m.ConfigMap

    @override
    def execute(self) -> r[m.ConfigMap]:
        """Execute database operations.

        Returns:
            r[dict]: Operation results

        """
        self._with_operation_context(
            "database_query", operation_type="select", table="users"
        )
        self.logger.info("Executing database query")
        results = m.ConfigMap(root={"users": [{"id": 1, "name": "Alice"}]})
        return r[m.ConfigMap].ok(results)

    @override
    def model_post_init(self, /, __context: t.Container | None) -> None:
        """Post-initialization hook.

        Args:
            __context: Pydantic context (unused)

        """
        super().model_post_init(__context)
        self._log_config_once(self.db_config, message="Database configuration loaded")


class MigrationService(s[m.ConfigMap]):
    """Example migration service with config log-once pattern."""

    def __init__(self, input_dir: str, output_dir: str, sync: bool) -> None:
        """Initialize migration service.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            sync: Enable synchronization

        """
        super().__init__()
        config = m.ConfigMap(
            root={
                "input_dir": input_dir,
                "output_dir": output_dir,
                "sync": sync,
                "batch_size": 100,
                "max_workers": 4,
            }
        )
        self._log_config_once(config, message="Migration configuration loaded")

    @override
    def execute(self) -> r[m.ConfigMap]:
        """Execute migration.

        Returns:
            r[dict]: Migration results

        """
        self._with_operation_context(
            "migration_process", total_entries=1000, batch_count=10
        )
        self.logger.info("Starting migration process")
        self.logger.info("Processing batch 1 of 10")
        self.logger.info("Processing batch 2 of 10")
        return r[m.ConfigMap].ok(m.ConfigMap(root={"migrated": 1000, "failed": 0}))


def main() -> None:
    """Demonstrate config log-once pattern."""
    print("=== Example 1: Database Service ===")
    db_config = m.ConfigMap(
        root={
            "host": c.Network.LOCALHOST,
            "port": 5432,
            "database": "mydb",
            "pool_size": 10,
        }
    )
    db_service = DatabaseService.model_construct()
    setattr(db_service, "db_config", db_config)
    result = db_service.execute()
    if result.is_success:
        print(f"✅ Database query successful: {result.value}")
    print("\n=== Example 2: Migration Service ===")
    migration_service = MigrationService(
        input_dir="/data/input", output_dir="/data/output", sync=True
    )
    result = migration_service.execute()
    if result.is_success:
        print(f"✅ Migration successful: {result.value}")
    print("\n=== Key Observations ===")
    print("1. Config logged ONCE when service initialized")
    print("2. Config does NOT appear in subsequent logs")
    print("3. Operation context (total_entries, batch_count) appears in all logs")
    print("4. Clean, relevant logs without config repetition!")


if __name__ == "__main__":
    main()
