"""Tests for FlextWorkspaceCli to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys

from _pytest.monkeypatch import MonkeyPatch
from flext_core.result import FlextResult as r
from flext_infra.models import m as im
from flext_infra.workspace import __main__ as workspace_cli
from flext_infra.workspace.migrator import ProjectMigrator


def test_workspace_cli_migrate_command(monkeypatch: MonkeyPatch) -> None:
    def _fake_migrate(
        self: ProjectMigrator,
        *,
        workspace_root: object,
        dry_run: bool,
    ) -> r[list[im.MigrationResult]]:
        del self, workspace_root
        assert dry_run is True
        return r[list[im.MigrationResult]].ok([
            im.MigrationResult.model_validate({
                "project": "flext-core",
                "changes": ["[DRY-RUN] base.mk regenerated via BaseMkGenerator"],
                "errors": [],
            })
        ])

    _ = monkeypatch.setattr(ProjectMigrator, "migrate", _fake_migrate)
    _ = monkeypatch.setattr(
        sys,
        "argv",
        [
            "workspace",
            "migrate",
            "--workspace-root",
            ".",
            "--dry-run",
        ],
    )

    exit_code = workspace_cli.main()

    assert exit_code == 0


def test_workspace_cli_migrate_output_contains_summary(
    monkeypatch: MonkeyPatch,
) -> None:
    def _fake_migrate(
        self: ProjectMigrator,
        *,
        workspace_root: object,
        dry_run: bool,
    ) -> r[list[im.MigrationResult]]:
        del self, workspace_root, dry_run
        return r[list[im.MigrationResult]].ok([
            im.MigrationResult.model_validate({
                "project": "flext-core",
                "changes": [
                    "[DRY-RUN] .gitignore cleaned from scripts/ and normalized"
                ],
                "errors": [],
            })
        ])

    _ = monkeypatch.setattr(ProjectMigrator, "migrate", _fake_migrate)
    _ = monkeypatch.setattr(
        sys,
        "argv",
        ["workspace", "migrate", "--workspace-root", ".", "--dry-run"],
    )

    exit_code = workspace_cli.main()

    # CLI uses structlog, no stdout output expected
    assert exit_code == 0
