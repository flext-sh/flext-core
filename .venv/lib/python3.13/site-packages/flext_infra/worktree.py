"""Repository-local development worktree lifecycle service."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, override

from flext_core import r
from flext_infra import c, m, u
from flext_infra._utilities.worktree_lifecycle import FlextInfraWorktreeLifecycle
from flext_infra._utilities.worktree_provisioning import FlextInfraWorktreeProvisioning
from flext_infra.base import s

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraWorktreeService(s[str]):
    """List, add, update, and remove development lanes under the repository."""

    operation: Annotated[
        c.Infra.WorktreeOperation, m.Field(description="Worktree lifecycle operation")
    ]
    branch: Annotated[
        str | None, m.Field(description="Git branch identifying the development lane")
    ] = None
    base: Annotated[
        str | None,
        m.Field(description="Commit-ish used to create or fast-forward a branch"),
    ] = None
    epic_lane: Annotated[
        Path | None,
        m.Field(description="Registered epic lane owning this nested child lane"),
    ] = None

    def _primary_root(self) -> p.Result[Path]:
        """Resolve the primary worktree from Git's canonical registry."""
        primary = u.Infra.git_primary_worktree_root(
            m.Infra.GitRepoRequest(repo_root=self.repository_root)
        )
        if primary.failure:
            return r[Path].fail(primary.error or "failed to resolve primary worktree")
        return r[Path].ok(primary.value.primary_root)

    def _validated_branch(self) -> p.Result[str]:
        """Validate and return the branch required by mutating operations."""
        branch = (self.branch or "").strip()
        if not branch:
            return r.fail(f"worktree {self.operation} requires --branch")
        checked = u.Infra.git_check_branch_format(
            m.Infra.GitBranchRequest(repo_root=self.repository_root, branch=branch)
        )
        if checked.failure or not checked.value.value:
            return r.fail(checked.error or f"invalid branch name: {branch}")
        return r.ok(branch)

    @staticmethod
    def _lanes_root(
        primary_root: Path, epic_lane: Path | None = None
    ) -> p.Result[Path]:
        """Place new lanes outside every ancestor project discovery boundary.

        A child lane is namespaced by its epic instead of by the repository:
        the epic owns the container, so Git's own registry proves the
        parent/child topology and no epic can be retired while one of its
        children is still registered.
        """
        if epic_lane is not None:
            return r.ok((epic_lane.resolve() / c.Infra.WORKTREES_DIRNAME).resolve())
        resolved_primary = primary_root.resolve()
        outermost_project = resolved_primary
        for candidate in resolved_primary.parents:
            if (candidate / c.Infra.PYPROJECT_FILENAME).is_file():
                outermost_project = candidate
            if (candidate / ".git").exists():
                break
        namespace_digest = u.Cli.sha256_content(str(resolved_primary))[
            : c.Infra.WORKTREE_NAMESPACE_DIGEST_LENGTH
        ]
        namespace = f"{resolved_primary.name}-{namespace_digest}"
        return r.ok(
            (outermost_project.parent / c.Infra.WORKTREES_DIRNAME / namespace).resolve()
        )

    @classmethod
    def _lane_path(
        cls, primary_root: Path, branch: str, epic_lane: Path | None = None
    ) -> p.Result[Path]:
        """Derive an isolated lane path and reject branch traversal."""
        root_result = cls._lanes_root(primary_root, epic_lane)
        if root_result.failure:
            return r.fail(root_result.error or "failed to resolve worktree lanes root")
        lanes_root = root_result.value
        lane_name = (
            branch.rsplit("/", maxsplit=1)[-1] if epic_lane is not None else branch
        )
        lane_path = (lanes_root / lane_name).resolve()
        if not lane_path.is_relative_to(lanes_root):
            return r.fail(f"branch resolves outside {c.Infra.WORKTREES_DIRNAME}")
        return r.ok(lane_path)

    @classmethod
    def canonical_lane_path(
        cls, primary_root: Path, branch: str, epic_lane: Path | None = None
    ) -> p.Result[Path]:
        """Return the canonical path reserved by one branch topology."""
        return cls._lane_path(primary_root, branch, epic_lane)

    @staticmethod
    def _registered_worktrees(
        primary_root: Path,
    ) -> p.Result[tuple[tuple[Path, str], ...]]:
        """Pair every registered worktree root with the branch it checks out."""
        listed = u.Infra.git_list_worktrees(
            m.Infra.GitRepoRequest(repo_root=primary_root)
        )
        if listed.failure:
            return r.fail(listed.error or "failed to list Git worktrees")
        entries: list[tuple[Path, str]] = []
        current: Path | None = None
        branch = ""
        for line in (*listed.value.text.splitlines(), ""):
            if line.startswith("worktree "):
                current = Path(line.removeprefix("worktree ").strip()).resolve()
                branch = ""
            elif line.startswith("branch refs/heads/"):
                branch = line.removeprefix("branch refs/heads/").strip()
            elif not line and current is not None:
                entries.append((current, branch))
                current = None
                branch = ""
        return r.ok(tuple(entries))

    @classmethod
    def registered_lane(cls, primary_root: Path, branch: str) -> p.Result[Path]:
        """Resolve an existing branch lane from Git's canonical registry."""
        entries = cls._registered_worktrees(primary_root)
        if entries.failure:
            return r.fail(entries.error or "failed to list Git worktrees")
        for root, registered_branch in entries.value:
            if registered_branch == branch:
                return r.ok(root)
        return r.fail(f"worktree branch is not registered: {branch}")

    @classmethod
    def registered_children(
        cls, primary_root: Path, epic_lane: Path
    ) -> p.Result[tuple[Path, ...]]:
        """Return every registered lane nested under one epic lane container."""
        entries = cls._registered_worktrees(primary_root)
        if entries.failure:
            return r.fail(entries.error or "failed to list Git worktrees")
        container = (epic_lane.resolve() / c.Infra.WORKTREES_DIRNAME).resolve()
        return r.ok(
            tuple(
                sorted(
                    root
                    for root, _ in entries.value
                    if root != container and root.is_relative_to(container)
                )
            )
        )

    def _ref_exists(self, reference: str) -> p.Result[bool]:
        """Return whether an exact Git ref exists, preserving command failures."""
        checked = u.Infra.git_ref_exists(
            m.Infra.GitRefRequest(repo_root=self.repository_root, reference=reference)
        )
        if checked.failure:
            return r.fail(checked.error or f"failed to inspect Git ref: {reference}")
        return r.ok(checked.value.value)

    @classmethod
    def setup_lane(cls, lane: Path) -> p.Result[bool]:
        """Provision an isolated environment inside one lane."""
        return FlextInfraWorktreeProvisioning.setup_lane(lane)

    @staticmethod
    def _rollback_new_lane(
        primary_root: Path,
        lane: Path,
        branch: str,
        created_branch_oid: str | None,
        setup_error: str,
    ) -> p.Result[str]:
        """Roll back only a clean lane created by the current add operation."""
        return FlextInfraWorktreeLifecycle.rollback_new_lane(
            primary_root, lane, branch, created_branch_oid, setup_error
        )

    def _add(self, primary_root: Path, branch: str, base: str) -> p.Result[str]:
        """Create one branch worktree without provisioning it."""
        if not self.apply_changes:
            return r.fail("worktree add requires --apply")
        if base.startswith("-"):
            return r.fail(f"invalid base commitish: {base}")
        resolved = u.Infra.git_resolve_commit(
            m.Infra.GitCommitishRequest(repo_root=primary_root, commitish=base)
        )
        if resolved.failure:
            return r.fail(
                f"cannot resolve worktree base {base}: "
                f"{resolved.error or 'unknown commitish'}"
            )
        base_oid = resolved.value.oid
        if self.epic_lane is not None:
            if self.epic_lane.is_symlink():
                return r.fail(f"epic lane worktree is a symlink: {self.epic_lane}")
            if not self.epic_lane.is_dir():
                return r.fail(f"epic lane worktree does not exist: {self.epic_lane}")
            registered = self._registered_worktrees(primary_root)
            if registered.failure:
                return r.fail(
                    registered.error or "failed to inspect registered epic lane"
                )
            if self.epic_lane.resolve() not in {root for root, _ in registered.value}:
                return r.fail(f"registered epic lane is required: {self.epic_lane}")
            container = self.epic_lane / c.Infra.WORKTREES_DIRNAME
            if container.is_symlink():
                return r.fail(f"epic worktree container is a symlink: {container}")
        existing = self.registered_lane(primary_root, branch)
        if existing.success:
            return r.fail(f"worktree branch is already registered: {branch}")
        lane_result = self._lane_path(primary_root, branch, self.epic_lane)
        if lane_result.failure:
            return r.fail(lane_result.error or "invalid worktree lane path")
        lane = lane_result.value
        if lane.exists():
            return r.fail(f"worktree lane already exists: {lane}")
        ensured = u.Cli.ensure_dir(lane.parent)
        if ensured.failure:
            return r.fail(ensured.error or f"failed to create {lane.parent}")
        local = self._ref_exists(f"refs/heads/{branch}")
        if local.failure:
            return r.fail(local.error or "failed to inspect local branch")
        remote = self._ref_exists(f"refs/remotes/origin/{branch}")
        if remote.failure:
            return r.fail(remote.error or "failed to inspect remote branch")
        added = u.Infra.git_add_lane_worktree(
            m.Infra.GitWorktreeAddRequest(
                repo_root=self.repository_root,
                lane=lane,
                branch=branch,
                base=base_oid,
                local_branch_exists=local.value,
                track_remote=not local.value and remote.value,
            )
        )
        if added.failure:
            return r.fail(added.error or f"failed to add worktree for {branch}")
        created_branch_oid: str | None = None
        if not local.value:
            created_oid = u.Infra.git_repository_head(
                m.Infra.GitRepoRequest(repo_root=lane)
            )
            if created_oid.failure:
                return self._rollback_new_lane(
                    primary_root,
                    lane,
                    branch,
                    None,
                    created_oid.error or "failed to retain created branch identity",
                )
            created_branch_oid = created_oid.value.oid
        pyproject = lane / c.Infra.PYPROJECT_FILENAME
        if pyproject.is_file():
            metadata = u.read_project_metadata(lane)
            if metadata.failure:
                return self._rollback_new_lane(
                    primary_root,
                    lane,
                    branch,
                    created_branch_oid,
                    metadata.error or "invalid lane project metadata",
                )
        return r.ok(str(lane))

    def _remove(self, primary_root: Path, branch: str) -> p.Result[str]:
        """Remove one clean canonical lane without deleting its branch."""
        if not self.apply_changes:
            return r.fail("worktree remove requires --apply")
        lane_result = self.registered_lane(primary_root, branch)
        if lane_result.failure:
            return r.fail(lane_result.error or "invalid worktree lane path")
        lane = lane_result.value
        # Why: removing an epic lane deletes the directory that physically holds
        # its children, so Git would keep registering worktrees whose checkout
        # no longer exists. The registry is the authority on that topology.
        children = self.registered_children(primary_root, lane)
        if children.failure:
            return r.fail(children.error or "failed to inspect nested child lanes")
        if children.value:
            nested = ", ".join(str(child) for child in children.value)
            return r.fail(
                f"worktree remove refuses lane {branch} while children are "
                f"registered: {nested}"
            )
        removed = u.Infra.git_remove_clean_worktree(primary_root, lane)
        if removed.failure:
            return r.fail(removed.error or f"failed to remove worktree for {branch}")
        return r.ok(str(lane))

    def _update(self, primary_root: Path, branch: str, base: str) -> p.Result[str]:
        """Merge-forward one clean canonical lane to the requested base."""
        if not self.apply_changes:
            return r.fail("worktree update requires --apply")
        lane_result = self.registered_lane(primary_root, branch)
        if lane_result.failure:
            return r.fail(lane_result.error or "invalid worktree lane path")
        lane = lane_result.value
        return FlextInfraWorktreeLifecycle.update_lane(lane, branch, base)

    @override
    def execute(self) -> p.Result[str]:
        """Execute the selected worktree operation."""
        primary = self._primary_root()
        if primary.failure:
            return r.fail(primary.error or "failed to resolve primary worktree")
        if self.operation == c.Infra.WorktreeOperation.LIST:
            listed = u.Infra.git_list_worktrees(
                m.Infra.GitRepoRequest(repo_root=primary.value)
            )
            if listed.failure:
                return r.fail(listed.error or "failed to list Git worktrees")
            return r.ok(listed.value.text)
        branch = self._validated_branch()
        if branch.failure:
            return r.fail(branch.error or "invalid worktree branch")
        base = (self.base or "").strip()
        if (
            self.operation
            in {c.Infra.WorktreeOperation.ADD, c.Infra.WorktreeOperation.UPDATE}
            and not base
        ):
            return r.fail(f"worktree {self.operation} requires --base")
        if self.operation == c.Infra.WorktreeOperation.ADD:
            return self._add(primary.value, branch.value, base)
        if self.operation == c.Infra.WorktreeOperation.UPDATE:
            return self._update(primary.value, branch.value, base)
        return self._remove(primary.value, branch.value)


__all__: list[str] = ["FlextInfraWorktreeService"]
