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

from examples.constants import c
from examples.models import m
from examples.protocols import p
from examples.typings import p, t
from examples.utilities import u
from flext_core import d, r, s


class ExamplesFlextDatabaseService(s[p.ConfigMap]):
    """Example service showing settings log-once pattern."""

    db_config: Annotated[
        m.ConfigMap, m.Field(description="Database connection settings.")
    ]

    @d.log_operation("database_query")
    @override
    def execute(self) -> p.Result[p.ConfigMap]:
        """Execute database operations.

        Returns:
            r[dict]: Operation results

        """
        self.logger.info(
            "Executing database query", operation_type="select", table="users"
        )
        results = m.ConfigMap(root={"users": [{"id": 1, "name": "Alice"}]})
        return r[p.ConfigMap].ok(results)

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
        self.logger.info("Database configuration loaded", **normalized_db_config)


class ExamplesFlextMigrationService(s[p.ConfigMap]):
    """Example migration service with settings log-once pattern."""

    input_dir: Annotated[str, m.Field(description="Source migration directory.")]
    output_dir: Annotated[str, m.Field(description="Target migration directory.")]
    sync: Annotated[
        bool, m.Field(description="Whether to perform synchronous migration.")
    ]

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
            }
        )
        normalized_settings = {
            key: u.normalize_to_metadata(value) for key, value in settings.root.items()
        }
        self.logger.info("Migration configuration loaded", **normalized_settings)

    @d.log_operation("migration_process")
    @override
    def execute(self) -> p.Result[p.ConfigMap]:
        """Execute migration.

        Returns:
            r[dict]: Migration results

        """
        self.logger.info(
            "Starting migration process", total_entries=1000, batch_count=10
        )
        self.logger.info("Processing batch 1 of 10")
        self.logger.info("Processing batch 2 of 10")
        return r[p.ConfigMap].ok(m.ConfigMap(root={"migrated": 1000, "failed": 0}))


def main() -> None:
    """Demonstrate settings log-once pattern."""
    db_config = m.ConfigMap(
        root={"host": c.LOCALHOST, "port": 5432, "database": "mydb", "pool_size": 10}
    )
    db_service = ExamplesFlextDatabaseService.model_construct(db_config=db_config)
    db_service.logger.info("Example started", example="database_service")
    result = db_service.execute()
    if result.success:
        db_service.logger.info(
            "Database query successful", result=u.normalize_to_metadata(result.value)
        )
    migration_service = ExamplesFlextMigrationService(
        input_dir="/data/input", output_dir="/data/output", sync=True
    )
    migration_service.logger.info("Example started", example="migration_service")
    result = migration_service.execute()
    if result.success:
        migration_service.logger.info(
            "Migration successful", result=u.normalize_to_metadata(result.value)
        )
    observations = (
        "Configuration is logged once when the service is initialized",
        "Configuration does not appear in subsequent logs",
        "Operation context appears in every operation log",
        "Logs remain focused without repeating settings",
    )
    for position, observation in enumerate(observations, start=1):
        migration_service.logger.info(
            "Key observation", position=position, observation=observation
        )


if __name__ == "__main__":
    main()
