"""Workflow sync service for canonical GitHub workflow distribution.

Wraps workflow sync operations with FlextResult error handling,
replacing scripts/github/sync_workflows.py with a service class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path

from flext_core.result import FlextResult, r
from flext_core.typings import t

from flext_infra.constants import ic
from flext_infra.json_io import JsonService
from flext_infra.selection import ProjectSelector
from flext_infra.templates import TemplateEngine


@dataclass(frozen=True)
class SyncOperation:
    """Describe one workflow sync operation."""

    project: str
    path: str
    action: str
    reason: str


MANAGED_FILES: frozenset[str] = frozenset({"ci.yml"})


class WorkflowSyncer:
    """Infrastructure service for syncing canonical workflow files.

    Distributes a source workflow template to all workspace projects,
    optionally pruning non-canonical workflow files.
    """

    def __init__(
        self,
        selector: ProjectSelector | None = None,
        json_io: JsonService | None = None,
        templates: TemplateEngine | None = None,
    ) -> None:
        """Initialize the workflow syncer."""
        self._selector = selector or ProjectSelector()
        self._json = json_io or JsonService()
        self._templates = templates or TemplateEngine()

    def resolve_source_workflow(
        self,
        workspace_root: Path,
        source_workflow: Path | None = None,
    ) -> FlextResult[Path]:
        """Resolve the source workflow file path.

        Args:
            workspace_root: The workspace root directory.
            source_workflow: Optional explicit source path.

        Returns:
            FlextResult with the resolved source path.

        """
        if source_workflow is not None:
            candidate = (
                source_workflow
                if source_workflow.is_absolute()
                else (workspace_root / source_workflow)
            ).resolve()
            if candidate.exists():
                return r[Path].ok(candidate)
            return r[Path].fail(f"missing source workflow: {candidate}")

        default_source = (workspace_root / ".github" / "workflows" / "ci.yml").resolve()
        if default_source.exists():
            return r[Path].ok(default_source)
        return r[Path].fail(f"missing source workflow: {default_source}")

    def render_template(self, template_path: Path) -> FlextResult[str]:
        """Read and render a workflow template with generated header.

        Args:
            template_path: Path to the template file.

        Returns:
            FlextResult with the rendered template content.

        """
        try:
            body = template_path.read_text(encoding=ic.Encoding.DEFAULT)
        except OSError as exc:
            return r[str].fail(f"failed to read template: {exc}")

        header = self._templates.GENERATED_SHELL_HEADER.format(
            source="flext_infra.github.workflows",
        )
        if body.startswith(header):
            return r[str].ok(body)
        return r[str].ok(header + body)

    def sync_project(
        self,
        *,
        project_name: str,
        project_root: Path,
        rendered_template: str,
        apply: bool = False,
        prune: bool = False,
    ) -> FlextResult[list[SyncOperation]]:
        """Sync workflow to a single project.

        Args:
            project_name: Project identifier.
            project_root: Project root directory.
            rendered_template: Rendered workflow content.
            apply: If True, actually write files.
            prune: If True, remove non-canonical workflows.

        Returns:
            FlextResult with list of sync operations.

        """
        operations: list[SyncOperation] = []
        workflows_dir = project_root / ".github" / "workflows"
        destination = workflows_dir / "ci.yml"

        try:
            if destination.exists():
                current = destination.read_text(encoding=ic.Encoding.DEFAULT)
                if current != rendered_template:
                    if apply:
                        _ = destination.write_text(
                            rendered_template,
                            encoding=ic.Encoding.DEFAULT,
                        )
                    operations.append(
                        SyncOperation(
                            project=project_name,
                            path=str(destination.relative_to(project_root)),
                            action="update",
                            reason="force overwrite ci.yml",
                        )
                    )
                else:
                    operations.append(
                        SyncOperation(
                            project=project_name,
                            path=str(destination.relative_to(project_root)),
                            action="noop",
                            reason="already synced",
                        )
                    )
            else:
                if apply:
                    workflows_dir.mkdir(parents=True, exist_ok=True)
                    _ = destination.write_text(
                        rendered_template,
                        encoding=ic.Encoding.DEFAULT,
                    )
                operations.append(
                    SyncOperation(
                        project=project_name,
                        path=str(destination.relative_to(project_root)),
                        action="create",
                        reason="missing ci.yml",
                    )
                )

            if prune and workflows_dir.exists():
                candidates = sorted(workflows_dir.glob("*.yml")) + sorted(
                    workflows_dir.glob("*.yaml"),
                )
                for path in candidates:
                    if path.name in MANAGED_FILES:
                        continue
                    if apply:
                        path.unlink()
                    operations.append(
                        SyncOperation(
                            project=project_name,
                            path=str(path.relative_to(project_root)),
                            action="prune",
                            reason="remove non-canonical workflow",
                        )
                    )
        except OSError as exc:
            return r[list[SyncOperation]].fail(f"sync error: {exc}")

        return r[list[SyncOperation]].ok(operations)

    def sync_workspace(
        self,
        workspace_root: Path,
        *,
        source_workflow: Path | None = None,
        report_path: Path | None = None,
        apply: bool = False,
        prune: bool = False,
    ) -> FlextResult[list[SyncOperation]]:
        """Sync workflows across all workspace projects.

        Args:
            workspace_root: Workspace root directory.
            source_workflow: Optional explicit source path.
            report_path: Optional report output path.
            apply: If True, actually write files.
            prune: If True, remove non-canonical workflows.

        Returns:
            FlextResult with all sync operations.

        """
        source_result = self.resolve_source_workflow(workspace_root, source_workflow)
        if source_result.is_failure:
            return r[list[SyncOperation]].fail(
                source_result.error or "source resolution failed",
            )

        template_result = self.render_template(source_result.value)
        if template_result.is_failure:
            return r[list[SyncOperation]].fail(
                template_result.error or "template render failed",
            )

        projects_result = self._selector.resolve_projects(workspace_root, [])
        if projects_result.is_failure:
            return r[list[SyncOperation]].fail(
                projects_result.error or "project discovery failed",
            )

        all_operations: list[SyncOperation] = []
        for project in projects_result.value:
            ops_result = self.sync_project(
                project_name=project.name,
                project_root=project.path,
                rendered_template=template_result.value,
                apply=apply,
                prune=prune,
            )
            if ops_result.is_success:
                all_operations.extend(ops_result.value)

        if report_path is not None:
            self._write_report(report_path, apply, all_operations)

        return r[list[SyncOperation]].ok(all_operations)

    def _write_report(
        self,
        report_path: Path,
        apply: bool,
        operations: list[SyncOperation],
    ) -> None:
        """Write a JSON report of sync operations."""
        by_action: MutableMapping[str, int] = {}
        for op in operations:
            by_action[op.action] = by_action.get(op.action, 0) + 1

        payload: MutableMapping[str, t.ConfigMapValue] = {
            "mode": "apply" if apply else "dry-run",
            "summary": by_action,
            "operations": [
                {
                    "project": op.project,
                    "path": op.path,
                    "action": op.action,
                    "reason": op.reason,
                }
                for op in operations
            ],
        }
        self._json.write(report_path, payload, sort_keys=True)


__all__ = ["SyncOperation", "WorkflowSyncer"]
