"""Multi-project orchestration service."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Self, override

from flext_core import r
from flext_infra import c, m, t, u
from flext_infra.base_selection import FlextInfraProjectSelectionServiceBase
from flext_infra.validate.pytest_selector import FlextInfraPytestSelectorValidator
from flext_infra.workspace._orchestrator_discovery import (
    FlextInfraWorkspaceOrchestratorDiscoveryMixin,
)
from flext_infra.workspace._orchestrator_execution import (
    FlextInfraWorkspaceOrchestratorExecutionMixin,
)

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraOrchestratorService(
    FlextInfraProjectSelectionServiceBase[bool],
    FlextInfraWorkspaceOrchestratorDiscoveryMixin,
    FlextInfraWorkspaceOrchestratorExecutionMixin,
):
    """Infrastructure service for multi-project make orchestration."""

    verb: Annotated[str, m.Field(description="Make verb to execute")]
    make_arg: Annotated[
        t.StrSequence,
        m.Field(
            default_factory=tuple,
            description="Additional arguments passed to each make invocation.",
        ),
    ] = m.Field(default_factory=tuple)
    file: Annotated[
        str | None,
        m.Field(
            default=None,
            min_length=1,
            description="Exact repository-relative pytest path or nodeid.",
        ),
    ] = None
    match: Annotated[
        str | None,
        m.Field(
            default=None,
            min_length=1,
            description="Exact pytest -k expression forwarded as one argument.",
        ),
    ] = None
    what: Annotated[
        str | None,
        m.Field(
            default=None,
            min_length=1,
            description="Exact test submode forwarded as one Make assignment.",
        ),
    ] = None

    @u.model_validator(mode="after")
    def _validate_test_selectors(self) -> Self:
        """Keep pytest selectors typed and out of generic Make argument strings."""
        if self.verb != "test" and any(
            value is not None for value in (self.file, self.match, self.what)
        ):
            msg = "file, match, and what selectors are only valid for the test verb"
            raise ValueError(msg)
        if self.verb == "test" and self.make_arg:
            msg = "generic make-arg is forbidden for the test verb"
            raise ValueError(msg)
        if msg := FlextInfraPytestSelectorValidator.syntax_violation(
            file=self.file, match=self.match, what=self.what
        ):
            raise ValueError(msg)
        return self

    def _make_args(self, *, file: str | None = None) -> t.StrSequence:
        """Build exact Make argv elements from already validated fields."""
        normalized = u.Infra.normalize_make_args(self.make_arg)
        if self.verb != "test":
            return normalized
        selectors = (
            *((f"FILE={file}",) if file is not None else ()),
            *((f"MATCH={self.match}",) if self.match is not None else ()),
            *((f"WHAT={self.what}",) if self.what is not None else ()),
        )
        return (*normalized, *selectors)

    @property
    def make_args(self) -> t.StrSequence:
        """Exact Make argv for direct, non-workspace consumers."""
        return self._make_args(file=self.file)

    def _select_file_owner(
        self, projects: t.SequenceOf[m.Infra.ProjectInfo], *, repository_root: Path
    ) -> p.Result[t.Pair[m.Infra.ProjectInfo, str]]:
        """Resolve one workspace-relative FILE to its most specific project."""
        if self.file is None:
            return r.fail("file owner selection requires a file")
        path_prefix, separator, node_suffix = self.file.partition("::")
        resolved = FlextInfraPytestSelectorValidator.resolve_file(
            repository_root, self.file
        )
        if resolved.failure:
            return r.fail(resolved.error or "pytest FILE resolution failed")
        resolved_file = resolved.value
        candidates = tuple(
            project
            for project in projects
            if resolved_file.is_relative_to(project.path.resolve())
        )
        if not candidates:
            return r.fail(f"no selected project owns file: {path_prefix}")

        def project_path_depth(project: m.Infra.ProjectInfo) -> int:
            return len(project.path.resolve().parts)

        owner = max(candidates, key=project_path_depth)
        relative_prefix = resolved_file.relative_to(owner.path.resolve()).as_posix()
        relative_file = (
            f"{relative_prefix}::{node_suffix}" if separator else relative_prefix
        )
        return r.ok((owner, relative_file))

    @override
    def execute(self) -> p.Result[bool]:
        """Execute the workspace-orchestrate CLI flow."""
        allowed_verbs = c.Infra.ORCHESTRATED_PROJECT_VERBS
        if self.verb not in allowed_verbs:
            allowed = ", ".join(allowed_verbs)
            return r[bool].fail(
                f"unsupported orchestrate verb '{self.verb}' (allowed: {allowed})"
            )

        resolved_projects = self._resolved_projects()
        if resolved_projects.failure:
            return r[bool].fail(resolved_projects.error or "project resolution failed")

        projects = resolved_projects.value
        if not projects:
            return r[bool].fail("no projects discovered")

        repository_root: Path = self.root
        effective_file = self.file
        if self.file is not None:
            owner_result = self._select_file_owner(
                projects, repository_root=repository_root
            )
            if owner_result.failure:
                return r[bool].fail(
                    owner_result.error or "file owner resolution failed"
                )
            owner, effective_file = owner_result.value
            projects = (owner,)
        orchestrate_result = self.orchestrate(
            projects=[
                self._project_target(project, repository_root=repository_root)
                for project in projects
            ],
            verb=self.verb,
            fail_fast=self.fail_fast,
            make_args=self._make_args(file=effective_file),
        )
        if orchestrate_result.failure:
            return r[bool].fail(
                orchestrate_result.error or "orchestration completed with failures"
            )
        return r[bool].ok(True)


__all__: list[str] = ["FlextInfraOrchestratorService"]
