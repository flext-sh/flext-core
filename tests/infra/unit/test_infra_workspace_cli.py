"""Tests for FlextWorkspaceCli to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch

from flext_core import r
from flext_infra import __main__ as workspace_cli, m
from flext_infra.workspace.migrator import FlextInfraProjectMigrator


def test_workspace_cli_migrate_command(monkeypatch: MonkeyPatch) -> None:

    def _fake_migrate(
        self: FlextInfraProjectMigrator,
        *,
        workspace_root: Path,
        dry_run: bool,
    ) -> r[list[m.Infra.Workspace.MigrationResult]]:
        del self, workspace_root
        assert dry_run is True
        return r[list[m.Infra.Workspace.MigrationResult]].ok([
            m.Infra.Workspace.MigrationResult.model_validate(
                obj={
                    "project": "flext-core",
                    "changes": ["[DRY-RUN] base.mk regenerated via BaseMkGenerator"],
                    "errors": [],
                }
            ),
        ])

    _ = monkeypatch.setattr(FlextInfraProjectMigrator, "migrate", _fake_migrate)
    _ = monkeypatch.setattr(
        sys,
        "argv",
        ["flext-infra", "workspace", "--workspace", ".", "--dry-run", "migrate"],
    )
    exit_code = workspace_cli.main()
    assert exit_code == 0


def test_workspace_cli_migrate_output_contains_summary(
    monkeypatch: MonkeyPatch,
) -> None:

    def _fake_migrate(
        self: FlextInfraProjectMigrator,
        *,
        workspace_root: Path,
        dry_run: bool,
    ) -> r[list[m.Infra.Workspace.MigrationResult]]:
        del self, workspace_root, dry_run
        return r[list[m.Infra.Workspace.MigrationResult]].ok([
            m.Infra.Workspace.MigrationResult.model_validate(
                obj={
                    "project": "flext-core",
                    "changes": [
                        "[DRY-RUN] .gitignore cleaned from scripts/ and normalized",
                    ],
                    "errors": [],
                }
            ),
        ])

    _ = monkeypatch.setattr(FlextInfraProjectMigrator, "migrate", _fake_migrate)
    _ = monkeypatch.setattr(
        sys,
        "argv",
        ["flext-infra", "workspace", "--workspace", ".", "--dry-run", "migrate"],
    )
    exit_code = workspace_cli.main()
    assert exit_code == 0
