"""GitHub workflow sync helpers: template render + per-project ci.yml."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flext_core import r
from flext_infra.constants import c
from flext_infra.models import m
from flext_infra.typings import t

if TYPE_CHECKING:
    from flext_infra.protocols import p


class FlextInfraUtilitiesGithubSyncMixin:
    """Render the canonical workflow template and sync/prune it per project.

    Composed into FlextInfraUtilitiesGithub via inheritance; the public
    ``sync_github_workflows`` resolves these helpers through ``cls`` FLEXT.
    """

    @classmethod
    def _github_render_template(cls, template_path: Path) -> p.Result[str]:
        """Github render template."""
        try:
            body = template_path.read_text(encoding=c.Cli.ENCODING_DEFAULT)
        except OSError as exc:
            return r[str].fail(f"failed to read template: {exc}")
        header_template = c.Infra.GENERATED_SHELL_HEADER
        header = header_template.format(source="flext_infra.github.workflows")
        if body.startswith(header):
            return r[str].ok(body)
        return r[str].ok(header + body)

    @classmethod
    def _github_render_project_template(cls, rendered_template: str) -> str:
        """Adapt workspace workflow commands to standalone project bootstrap semantics."""
        return rendered_template.replace(
            "- name: Boot (blocking)", "- name: Setup (blocking)"
        ).replace("run: make boot", "run: make setup")

    @classmethod
    def _github_resolve_source_workflow(
        cls, repository_root: Path, source_workflow: Path | None = None
    ) -> p.Result[Path]:
        """Github resolve source workflow."""
        if source_workflow is not None:
            candidate = (
                source_workflow
                if source_workflow.is_absolute()
                else repository_root / source_workflow
            ).resolve()
            if candidate.exists():
                return r[Path].ok(candidate)
            return r[Path].fail(f"missing source workflow: {candidate}")
        default_source = (
            repository_root / ".github" / "workflows" / "ci.yml"
        ).resolve()
        if default_source.exists():
            return r[Path].ok(default_source)
        return r[Path].fail(f"missing source workflow: {default_source}")

    @classmethod
    def _github_sync_project(
        cls, ctx: m.Infra.GithubWorkflowSyncContext
    ) -> p.Result[t.SequenceOf[m.Infra.GithubWorkflowSyncOperation]]:
        """Github sync project."""
        operations: t.MutableSequenceOf[m.Infra.GithubWorkflowSyncOperation] = []
        try:
            cls._github_sync_ci_yml(ctx, operations)
            if ctx.prune and ctx.workflows_dir.exists():
                cls._github_prune_workflows(ctx, operations)
        except OSError as exc:
            return r[t.SequenceOf[m.Infra.GithubWorkflowSyncOperation]].fail(
                f"sync error: {exc}"
            )
        return r[t.SequenceOf[m.Infra.GithubWorkflowSyncOperation]].ok(operations)

    @classmethod
    def _github_sync_ci_yml(
        cls,
        ctx: m.Infra.GithubWorkflowSyncContext,
        operations: t.MutableSequenceOf[m.Infra.GithubWorkflowSyncOperation],
    ) -> None:
        """Sync a single ci.yml file for a project."""
        destination = ctx.ci_destination
        rel_path = str(destination.relative_to(ctx.project_root))
        if destination.exists():
            current = destination.read_text(encoding=c.Cli.ENCODING_DEFAULT)
            if current != ctx.rendered_template:
                if ctx.apply:
                    _ = destination.write_text(
                        ctx.rendered_template, encoding=c.Cli.ENCODING_DEFAULT
                    )
                operations.append(
                    m.Infra.GithubWorkflowSyncOperation(
                        project=ctx.project_name,
                        path=rel_path,
                        action="update",
                        reason="force overwrite ci.yml",
                    )
                )
            else:
                operations.append(
                    m.Infra.GithubWorkflowSyncOperation(
                        project=ctx.project_name,
                        path=rel_path,
                        action="noop",
                        reason="already synced",
                    )
                )
        else:
            if ctx.apply:
                ctx.workflows_dir.mkdir(parents=True, exist_ok=True)
                _ = destination.write_text(
                    ctx.rendered_template, encoding=c.Cli.ENCODING_DEFAULT
                )
            operations.append(
                m.Infra.GithubWorkflowSyncOperation(
                    project=ctx.project_name,
                    path=rel_path,
                    action="create",
                    reason="missing ci.yml",
                )
            )

    @staticmethod
    def _github_prune_workflows(
        ctx: m.Infra.GithubWorkflowSyncContext,
        operations: t.MutableSequenceOf[m.Infra.GithubWorkflowSyncOperation],
    ) -> None:
        """Remove non-canonical workflow files from a project."""
        candidates = sorted(ctx.workflows_dir.glob("*.yml")) + sorted(
            ctx.workflows_dir.glob("*.yaml")
        )
        for path in candidates:
            if path.name in c.Infra.MANAGED_FILES:
                continue
            if ctx.apply:
                path.unlink()
            operations.append(
                m.Infra.GithubWorkflowSyncOperation(
                    project=ctx.project_name,
                    path=str(path.relative_to(ctx.project_root)),
                    action="prune",
                    reason="remove non-canonical workflow",
                )
            )


__all__: list[str] = ["FlextInfraUtilitiesGithubSyncMixin"]
