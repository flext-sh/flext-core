"""Typed contracts for isolated worktree command transactions."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar

from flext_cli import m
from flext_infra import t


class FlextInfraModelsWorktree:
    """Declaration-only models for transactional fix and codegen execution."""

    class LintSnapshot(m.ContractModel):
        """Captured diagnostics from one lint tool invocation."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        tool: Annotated[t.NonEmptyStr, m.Field(description="Canonical tool name")]
        exit_code: Annotated[int, m.Field(description="Tool process exit code")]
        errors: Annotated[
            t.NonNegativeInt, m.Field(description="Detected error count")
        ] = 0
        warnings: Annotated[
            t.NonNegativeInt, m.Field(description="Detected warning count")
        ] = 0
        output: Annotated[str, m.Field(description="Combined captured tool output")] = (
            ""
        )

    class RepositoryDelta(m.ContractModel):
        """Operation-only patch for one repository in a workspace transaction."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        relative_path: Annotated[
            t.NonEmptyStr,
            m.Field(description="Repository path relative to the repository root"),
        ]
        source_root: Annotated[
            Path, m.Field(description="Original repository worktree root")
        ]
        worktree_root: Annotated[
            Path, m.Field(description="Temporary repository worktree root")
        ]
        checkpoint_sha: Annotated[
            t.NonEmptyStr, m.Field(description="Synthetic dirty-state checkpoint SHA")
        ]
        changed_files: Annotated[
            t.StrSequence, m.Field(description="Files changed by the isolated command")
        ] = ()
        patch: Annotated[
            bytes,
            m.Field(b"", description="Binary Git patch relative to the checkpoint"),
        ] = b""

    class RepositoryWorktree(m.ContractModel):
        """One source repository paired with its isolated worktree checkpoint."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        relative_path: Annotated[
            t.NonEmptyStr,
            m.Field(description="Repository path relative to the repository root"),
        ]
        source_root: Annotated[
            Path, m.Field(description="Original repository worktree root")
        ]
        worktree_root: Annotated[
            Path, m.Field(description="Temporary detached repository worktree root")
        ]
        checkpoint_sha: Annotated[
            t.NonEmptyStr, m.Field(description="Current isolated checkpoint SHA")
        ]


__all__: list[str] = ["FlextInfraModelsWorktree"]
