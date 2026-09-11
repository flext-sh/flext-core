"""Git identity report models — nested container for FLEXT composition."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar

from flext_cli import m
from flext_infra import t


class FlextInfraModelsGitIdentity:
    """Git identity report, composed into m.Infra via FLEXT."""

    class GitIdentityReport(m.ContractModel):
        """Consolidated Git identity snapshot for one repository path."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        repo_root: Annotated[Path, m.Field(description="Repository worktree root")]
        head_oid: Annotated[t.NonEmptyStr, m.Field(description="HEAD commit hex SHA")]
        porcelain: Annotated[
            str, m.Field(description="Raw git status --porcelain output")
        ]
        dirty: Annotated[bool, m.Field(description="Whether the worktree is dirty")]
        git_dir: Annotated[Path, m.Field(description="Absolute .git directory")]
        common_dir: Annotated[
            Path,
            m.Field(
                description="Git common directory (shared across linked worktrees)"
            ),
        ]
        branch: Annotated[
            t.NonEmptyStr | None,
            m.Field(
                default=None, description="Active branch name, None if detached HEAD"
            ),
        ]
        origin_remote: Annotated[
            t.NonEmptyStr | None, m.Field(default=None, description="Origin remote URL")
        ]
        superproject_root: Annotated[
            Path | None,
            m.Field(
                default=None,
                description="Superproject root if nested, None if standalone",
            ),
        ]
        requested_path: Annotated[
            Path | None,
            m.Field(
                default=None,
                description="Filesystem path submitted to the identity probe, if any",
            ),
        ]
        is_worktree: Annotated[
            bool,
            m.Field(
                default=False,
                description="Whether the checkout uses a linked (non-primary) Git dir",
            ),
        ]
        is_submodule: Annotated[
            bool,
            m.Field(
                default=False,
                description="Whether the checkout is a nested Git submodule",
            ),
        ]
        has_submodules: Annotated[
            bool, m.Field(description="Whether the repository declares any submodules")
        ] = False
        is_inside_work_tree: Annotated[
            bool, m.Field(description="Whether the path is inside a Git work tree")
        ] = False


__all__: list[str] = ["FlextInfraModelsGitIdentity"]
