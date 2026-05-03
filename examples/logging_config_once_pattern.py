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

from typing import Annotated, override

from examples import c, d, m, p, r, s, t, u


class ExamplesFlextDatabaseService(s[m.ConfigMap]):
    """Example service showing settings log-once pattern."""

    db_config: Annotated[
        m.ConfigMap,
        m.Field(description="Database connection settings."),
    ]

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
        normalized_db_config: t.JsonMapping = {
            key: u.normalize_to_metadata(value)
            for key, value in self.db_config.root.items()
        }
        self.logger.info(
            "Database configuration loaded",
            **normalized_db_config,
        )




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
    db_service = ExamplesFlextDatabaseService.model_construct(db_config=db_config)
    result = db_service.execute()
    if result.success:
        print(f"✅ Database query successful: {result.value}")
    print("\n=== Example 2: Migration Service ===")
    migration_service = ExamplesFlextMigrationService(
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
