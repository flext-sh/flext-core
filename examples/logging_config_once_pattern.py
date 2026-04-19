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

from examples import c, d, m, p, r, s, t, u


class DatabaseService(s):
    """Example service showing settings log-once pattern."""

    db_config: m.ConfigMap

    @override
    @d.log_operation("database_query")
    def execute(self) -> p.Result[m.ConfigMap]:
        """Execute database operations.

        Returns:
            r[dict]: Operation results

        """
        self.logger.info(
            "Executing database query",
            operation_type="select",
            table="users",
        )
        results = m.ConfigMap(root={"users": [{"id": 1, "name": "Alice"}]})
        return r[m.ConfigMap].ok(results)

    @override
    def model_post_init(self, /, __context: t.ScalarMapping | None) -> None:
        """Post-initialization hook.

        Args:
            __context: Pydantic context (unused)

        """
        super().model_post_init(__context)
        normalized_db_config: t.FlatContainerMapping = {
            str(key): u.to_plain_container(u.normalize_to_container(value))
            for key, value in self.db_config.root.items()
        }
        self.logger.info(
            "Database configuration loaded",
            **normalized_db_config,
        )


class MigrationService(s):
    """Example migration service with settings log-once pattern."""

    input_dir: str
    output_dir: str
    sync: bool

    @override
    def model_post_init(self, /, __context: t.ScalarMapping | None) -> None:
        """Post-initialization hook.

        Args:
            __context: Pydantic context (unused)

        """
        super().model_post_init(__context)
        settings = m.ConfigMap(
            root={
                "input_dir": self.input_dir,
                "output_dir": self.output_dir,
                "sync": self.sync,
                "batch_size": 100,
                "max_workers": 4,
            },
        )
        normalized_settings: t.FlatContainerMapping = {
            str(key): u.to_plain_container(u.normalize_to_container(value))
            for key, value in settings.root.items()
        }
        self.logger.info(
            "Migration configuration loaded",
            **normalized_settings,
        )

    @override
    @d.log_operation("migration_process")
    def execute(self) -> p.Result[m.ConfigMap]:
        """Execute migration.

        Returns:
            r[dict]: Migration results

        """
        self.logger.info(
            "Starting migration process",
            total_entries=1000,
            batch_count=10,
        )
        self.logger.info("Processing batch 1 of 10")
        self.logger.info("Processing batch 2 of 10")
        return r[m.ConfigMap].ok(m.ConfigMap(root={"migrated": 1000, "failed": 0}))


def main() -> None:
    """Demonstrate settings log-once pattern."""
    print("=== Example 1: Database Service ===")
    db_config = m.ConfigMap(
        root={
            "host": c.LOCALHOST,
            "port": 5432,
            "database": "mydb",
            "pool_size": 10,
        },
    )
    db_service = DatabaseService.model_construct(db_config=db_config)
    result = db_service.execute()
    if result.success:
        print(f"✅ Database query successful: {result.value}")
    print("\n=== Example 2: Migration Service ===")
    migration_service = MigrationService(
        input_dir="/data/input",
        output_dir="/data/output",
        sync=True,
    )
    result = migration_service.execute()
    if result.success:
        print(f"✅ Migration successful: {result.value}")
    print("\n=== Key Observations ===")
    print("1. Config logged ONCE when service initialized")
    print("2. Config does NOT appear in subsequent logs")
    print("3. Operation context (total_entries, batch_count) appears in all logs")
    print("4. Clean, relevant logs without settings repetition!")


if __name__ == "__main__":
    main()
