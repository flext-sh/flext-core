"""Typed Git request/report contracts for flext-infra public Git API."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar

from flext_cli import m
from flext_infra import t
from flext_infra._models._git.identity import FlextInfraModelsGitIdentity


class FlextInfraModelsGit(FlextInfraModelsGitIdentity):
    """Declaration-only models for Git facade and FlextInfraGitService.

    Composed via FLEXT with FlextInfraModelsGitIdentity (GitIdentityReport).
    """

    class GitRepoRequest(m.ContractModel):
        """One repository path for a monomorphic Git query."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        repo_root: Annotated[Path, m.Field(description="Repository worktree root")]

    class GitStatusRequest(m.ContractModel):
        """Request porcelain status for one repository."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        repo_root: Annotated[Path, m.Field(description="Repository worktree root")]

    class GitStatusReport(m.ContractModel):
        """Porcelain status snapshot for one repository."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        repo_root: Annotated[Path, m.Field(description="Repository worktree root")]
        porcelain: Annotated[
            str, m.Field(description="Raw git status --porcelain output")
        ]
        dirty: Annotated[bool, m.Field(description="Whether the worktree is dirty")]

    class GitPrimaryRootReport(m.ContractModel):
        """Resolved primary worktree root."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        primary_root: Annotated[
            Path, m.Field(description="Canonical primary worktree root")
        ]

    class GitRootReport(m.ContractModel):
        """Resolved workspace or superproject root."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        repository_root: Annotated[
            Path, m.Field(description="Workspace or superproject root")
        ]

    class GitOidReport(m.ContractModel):
        """Resolved Git object id."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        oid: Annotated[t.NonEmptyStr, m.Field(description="Git object id (hex)")]

    class GitTextReport(m.ContractModel):
        """Captured Git stdout text for one semantic operation."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        text: Annotated[str, m.Field(description="Captured stdout text")]

    class GitBytesReport(m.ContractModel):
        """Captured Git stdout bytes for one semantic operation."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        payload: Annotated[bytes, m.Field(description="Captured stdout bytes")]

    class GitBoolReport(m.ContractModel):
        """Boolean outcome for one semantic Git predicate."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        value: Annotated[bool, m.Field(description="Predicate result")]

    class GitBranchRequest(m.ContractModel):
        """Repository plus branch name for branch-scoped operations."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        repo_root: Annotated[Path, m.Field(description="Repository worktree root")]
        branch: Annotated[t.NonEmptyStr, m.Field(description="Branch name")]

    class GitRefRequest(m.ContractModel):
        """Repository plus exact Git ref name."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        repo_root: Annotated[Path, m.Field(description="Repository worktree root")]
        reference: Annotated[t.NonEmptyStr, m.Field(description="Exact Git ref")]

    class GitSubmoduleConfigRequest(m.ContractModel):
        """One ``.gitmodules`` section key of a declared submodule."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        repo_root: Annotated[Path, m.Field(description="Superproject worktree root")]
        section: Annotated[
            t.NonEmptyStr, m.Field(description="Section, e.g. submodule.flext-core")
        ]
        key: Annotated[t.NonEmptyStr, m.Field(description="Key inside the section")]

    class GitCommitishRequest(m.ContractModel):
        """Repository plus commit-ish for resolve/ancestor/merge ops."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        repo_root: Annotated[Path, m.Field(description="Repository worktree root")]
        commitish: Annotated[t.NonEmptyStr, m.Field(description="Commit-ish")]

    class GitPathPairRequest(m.ContractModel):
        """Repository plus source/target relative paths."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        repo_root: Annotated[Path, m.Field(description="Repository worktree root")]
        source: Annotated[t.NonEmptyStr, m.Field(description="Source path relative")]
        target: Annotated[t.NonEmptyStr, m.Field(description="Target path relative")]

    class GitRelativePathRequest(m.ContractModel):
        """Repository plus one relative path."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        repo_root: Annotated[Path, m.Field(description="Repository worktree root")]
        relative_path: Annotated[
            t.NonEmptyStr, m.Field(description="Path relative to repo root")
        ]

    class GitDeleteRefRequest(m.ContractModel):
        """Delete a ref when it still points at an expected oid."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        repo_root: Annotated[Path, m.Field(description="Repository worktree root")]
        reference: Annotated[t.NonEmptyStr, m.Field(description="Exact Git ref")]
        expected_oid: Annotated[
            t.NonEmptyStr, m.Field(description="Expected tip oid for CAS delete")
        ]

    class GitPushRequest(m.ContractModel):
        """Push HEAD to a remote branch ref."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        repo_root: Annotated[Path, m.Field(description="Repository worktree root")]
        remote: Annotated[t.NonEmptyStr, m.Field(description="Remote name")] = "origin"
        branch: Annotated[t.NonEmptyStr, m.Field(description="Branch to publish")]

    class GitWorktreeAddRequest(m.ContractModel):
        """Add a development worktree lane."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        repo_root: Annotated[Path, m.Field(description="Primary repository root")]
        lane: Annotated[Path, m.Field(description="Absolute lane worktree path")]
        branch: Annotated[t.NonEmptyStr, m.Field(description="Lane branch name")]
        base: Annotated[
            t.NonEmptyStr, m.Field(description="Base commit-ish when creating branch")
        ]
        track_remote: Annotated[
            bool, m.Field(description="Track origin/<branch> when creating")
        ] = False
        local_branch_exists: Annotated[
            bool, m.Field(description="Whether refs/heads/<branch> already exists")
        ] = False

    class GitNumstatReport(m.ContractModel):
        """HEAD commit subject plus parent..HEAD numstat text."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        subject: Annotated[str, m.Field(description="HEAD commit subject")]
        numstat: Annotated[str, m.Field(description="git diff --numstat HEAD~1 HEAD")]

    class GitFingerprintInputsReport(m.ContractModel):
        """Byte-exact inputs for workspace fingerprinting."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        paths_z: Annotated[
            bytes, m.Field(description="NUL-delimited ls-files path list")
        ]
        index_z: Annotated[
            bytes, m.Field(description="NUL-delimited ls-files --stage list")
        ]
        head: Annotated[bytes, m.Field(description="HEAD oid bytes or UNBORN")]

    class GitUpdateIndexGitlinkRequest(m.ContractModel):
        """Stage one gitlink (mode 160000) into the index."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        repo_root: Annotated[Path, m.Field(description="Repository worktree root")]
        oid: Annotated[t.NonEmptyStr, m.Field(description="Gitlink commit oid")]
        relative_path: Annotated[
            t.NonEmptyStr, m.Field(description="Gitlink path relative to repo")
        ]

    class GitPathsRequest(m.ContractModel):
        """Repository plus multiple relative paths for add/restore operations."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        repo_root: Annotated[Path, m.Field(description="Repository worktree root")]
        paths: Annotated[
            t.SequenceOf[t.NonEmptyStr],
            m.Field(description="Relative paths to operate on"),
        ]

    class GitCommitRequest(m.ContractModel):
        """Repository plus commit message for staging a commit."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        repo_root: Annotated[Path, m.Field(description="Repository worktree root")]
        message: Annotated[t.NonEmptyStr, m.Field(description="Commit message")]

    class GitRemoteUrlRequest(m.ContractModel):
        """Repository plus optional remote name for URL resolution."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        repo_root: Annotated[Path, m.Field(description="Repository worktree root")]
        remote: Annotated[t.NonEmptyStr, m.Field(description="Remote name")] = "origin"

    class GitSubmoduleContractRequest(m.ContractModel):
        """Superproject plus one declared submodule path for contract reads."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        repo_root: Annotated[Path, m.Field(description="Superproject worktree root")]
        member_path: Annotated[
            t.NonEmptyStr,
            m.Field(description="Submodule path relative to the superproject root"),
        ]

    class GitSubmoduleContractReport(m.ContractModel):
        """Declared ``.gitmodules`` URL and branch for one submodule path."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        url: Annotated[t.NonEmptyStr, m.Field(description="Declared submodule URL")]
        branch: Annotated[
            t.NonEmptyStr, m.Field(description="Declared submodule branch")
        ]

    class GitCheckoutPathsRequest(m.ContractModel):
        """Repository plus optional paths for checkout/restore operations."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)

        repo_root: Annotated[Path, m.Field(description="Repository worktree root")]
        paths: Annotated[
            t.SequenceOf[t.NonEmptyStr],
            m.Field(description="Relative paths to restore; empty restores all"),
        ] = ()


__all__: list[str] = ["FlextInfraModelsGit"]
