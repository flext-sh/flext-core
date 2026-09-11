"""GitHub workflow synchronization and lint helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from flext_cli import u
from flext_core import r
from flext_infra._utilities._github_sync import FlextInfraUtilitiesGithubSyncMixin
from flext_infra._utilities.docs_scope import FlextInfraUtilitiesDocsScope
from flext_infra.constants import c
from flext_infra.models import m

if TYPE_CHECKING:
    from flext_infra.protocols import p
    from flext_infra.typings import t


class FlextInfraUtilitiesGithub(FlextInfraUtilitiesGithubSyncMixin):
    """Utilities for GitHub workflow synchronization and linting."""

    @classmethod
    def lint_github_workflows(
        cls, request: m.Infra.GithubWorkflowLintRequest
    ) -> p.Result[m.Infra.GithubWorkflowLintOutcome]:
        """Run actionlint on the repository and return results."""
        actionlint = shutil.which("actionlint")
        repository_root = request.workspace_path
        if actionlint is None:
            payload_skipped = m.Infra.GithubWorkflowLintOutcome(
                status=c.Infra.WorkflowLintStatus.SKIPPED.value,
                reason="actionlint not installed",
            )
            if request.report_path is not None:
                _ = u.Cli.json_write(
                    request.report_path,
                    payload_skipped,
                    m.Cli.JsonWriteOptions(sort_keys=True),
                )
            return r[m.Infra.GithubWorkflowLintOutcome].ok(payload_skipped)
        result = u.Cli.run_raw([actionlint], cwd=repository_root)
        if result.success:
            output = result.value
            payload = m.Infra.GithubWorkflowLintOutcome(
                status=c.Infra.WorkflowLintStatus.OK.value,
                exit_code=output.exit_code,
                stdout=output.stdout,
                stderr=output.stderr,
            )
        else:
            payload = m.Infra.GithubWorkflowLintOutcome(
                status=c.Infra.WorkflowLintStatus.FAIL.value,
                exit_code=1,
                detail=result.error or "",
            )
        if request.report_path is not None:
            _ = u.Cli.json_write(
                request.report_path, payload, m.Cli.JsonWriteOptions(sort_keys=True)
            )
        if payload.status == c.Infra.WorkflowLintStatus.FAIL.value and request.strict:
            return r[m.Infra.GithubWorkflowLintOutcome].fail(
                result.error or "actionlint found issues"
            )
        return r[m.Infra.GithubWorkflowLintOutcome].ok(payload)

    @classmethod
    def sync_github_workflows(
        cls, request: m.Infra.GithubWorkflowSyncRequest
    ) -> p.Result[m.Infra.GithubWorkflowSyncReport]:
        """Sync workflows across all workspace projects."""
        repository_root = request.workspace_path
        source_result = cls._github_resolve_source_workflow(repository_root, None)
        if source_result.failure:
            return r[m.Infra.GithubWorkflowSyncReport].fail(
                source_result.error or "source resolution failed"
            )
        template_result = cls._github_render_template(source_result.value)
        if template_result.failure:
            return r[m.Infra.GithubWorkflowSyncReport].fail(
                template_result.error or "template render failed"
            )
        projects_result = FlextInfraUtilitiesDocsScope.resolve_projects(
            repository_root, list(request.projects or [])
        )
        if projects_result.failure:
            return r[m.Infra.GithubWorkflowSyncReport].fail(
                projects_result.error or "project discovery failed"
            )
        all_operations: t.MutableSequenceOf[m.Infra.GithubWorkflowSyncOperation] = []
        for project in projects_result.value:
            ctx = m.Infra.GithubWorkflowSyncContext(
                project_name=project.name,
                project_root=project.path,
                rendered_template=cls._github_render_project_template(
                    template_result.value
                ),
                request=request,
            )
            ops_result = cls._github_sync_project(ctx)
            if ops_result.success:
                all_operations.extend(ops_result.value)
        report = m.Infra.GithubWorkflowSyncReport.from_operations(
            apply=request.apply, operations=all_operations
        )
        if request.report_path is not None:
            _ = u.Cli.json_write(
                request.report_path, report, m.Cli.JsonWriteOptions(sort_keys=True)
            )
        return r[m.Infra.GithubWorkflowSyncReport].ok(report)


__all__: list[str] = ["FlextInfraUtilitiesGithub"]
