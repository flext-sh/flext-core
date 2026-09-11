"""Deterministic cleanup planning and guarded apply for workspace residues."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from flext_cli import u
from flext_tests import m, p, r
from flext_tests._utilities.workspace_cleanup_inspect import (
    FlextTestsWorkspaceCleanupInspectUtilitiesMixin,
)


class FlextTestsWorkspaceCleanupPlanUtilitiesMixin(
    FlextTestsWorkspaceCleanupInspectUtilitiesMixin
):
    """Build and apply exact ignored-residue plans with stale-drift protection."""

    @staticmethod
    def _candidate_sort_key(candidate: m.Tests.WorkspaceCleanupCandidate) -> str:
        """Return the deterministic path key for cleanup plan ordering."""
        return candidate.relative_path.as_posix()

    @classmethod
    def _candidate(
        cls, root: Path, relative_path: Path
    ) -> p.Result[m.Tests.WorkspaceCleanupCandidate]:
        """Validate and describe one existing cleanup candidate."""
        lexical_result = cls._lexical_path(root, relative_path)
        if lexical_result.failure:
            return r[m.Tests.WorkspaceCleanupCandidate].fail(lexical_result.error)
        path = lexical_result.value
        protected_result = cls._reject_protected(root, relative_path)
        if protected_result.failure:
            return r[m.Tests.WorkspaceCleanupCandidate].fail(protected_result.error)
        ancestor_result = cls._reject_symlink_ancestor(root, relative_path)
        if ancestor_result.failure:
            return r[m.Tests.WorkspaceCleanupCandidate].fail(ancestor_result.error)
        node_result = cls._reject_unsafe_node(path, relative_path)
        if node_result.failure:
            return r[m.Tests.WorkspaceCleanupCandidate].fail(node_result.error)
        ignored_result = cls._ignored(root, relative_path)
        if ignored_result.failure:
            return r[m.Tests.WorkspaceCleanupCandidate].fail(ignored_result.error)
        clean_result = cls._untracked_and_clean(root, relative_path)
        if clean_result.failure:
            return r[m.Tests.WorkspaceCleanupCandidate].fail(clean_result.error)
        fingerprint_result = cls._path_fingerprint(path)
        if fingerprint_result.failure:
            return r[m.Tests.WorkspaceCleanupCandidate].fail(fingerprint_result.error)
        kind: Literal["file", "directory", "symlink"] = (
            "symlink" if path.is_symlink() else "directory" if path.is_dir() else "file"
        )
        candidate = m.Tests.WorkspaceCleanupCandidate(
            relative_path=relative_path,
            path=path,
            kind=kind,
            fingerprint=fingerprint_result.value,
        )
        return r[m.Tests.WorkspaceCleanupCandidate].ok(candidate)

    @staticmethod
    def _reject_nested(
        candidates: tuple[m.Tests.WorkspaceCleanupCandidate, ...],
    ) -> p.Result[bool]:
        """Reject overlapping parent and child cleanup targets."""
        for index, parent in enumerate(candidates):
            for child in candidates[index + 1 :]:
                if (
                    parent.path in child.path.parents
                    or child.path in parent.path.parents
                ):
                    return r[bool].fail(
                        "cleanup residues overlap: "
                        f"{parent.relative_path} and {child.relative_path}"
                    )
        return r[bool].ok(True)

    @classmethod
    def workspace_cleanup_plan(
        cls, request: p.Tests.WorkspaceCleanupRequest
    ) -> p.Result[p.Tests.WorkspaceCleanupPlan]:
        """Build a deterministic read-only plan for exact ignored residues."""
        if not isinstance(request, m.Tests.WorkspaceCleanupRequest):
            return r[p.Tests.WorkspaceCleanupPlan].fail(
                "cleanup request must be the canonical WorkspaceCleanupRequest model"
            )
        root_result = cls._workspace_root(request)
        if root_result.failure:
            return r[p.Tests.WorkspaceCleanupPlan].fail(root_result.error)
        root = root_result.value
        relative_paths: set[Path] = set()
        candidates: list[m.Tests.WorkspaceCleanupCandidate] = []
        for declared in request.policy.residues:
            relative_result = cls._relative_path(declared)
            if relative_result.failure:
                return r[p.Tests.WorkspaceCleanupPlan].fail(relative_result.error)
            relative_path = relative_result.value
            if relative_path in relative_paths:
                return r[p.Tests.WorkspaceCleanupPlan].fail(
                    f"cleanup residue is declared more than once: {relative_path}"
                )
            relative_paths.add(relative_path)
            lexical_result = cls._lexical_path(root, relative_path)
            if lexical_result.failure:
                return r[p.Tests.WorkspaceCleanupPlan].fail(lexical_result.error)
            lexical = lexical_result.value
            if not lexical.exists() and not lexical.is_symlink():
                continue
            candidate_result = cls._candidate(root, relative_path)
            if candidate_result.failure:
                return r[p.Tests.WorkspaceCleanupPlan].fail(candidate_result.error)
            candidates.append(candidate_result.value)
        ordered = tuple(sorted(candidates, key=cls._candidate_sort_key))
        nested_result = cls._reject_nested(ordered)
        if nested_result.failure:
            return r[p.Tests.WorkspaceCleanupPlan].fail(nested_result.error)
        plan = m.Tests.WorkspaceCleanupPlan(request=request, candidates=ordered)
        return r[p.Tests.WorkspaceCleanupPlan].ok(plan)

    @classmethod
    def workspace_cleanup_apply(
        cls, plan: p.Tests.WorkspaceCleanupPlan
    ) -> p.Result[p.Tests.WorkspaceCleanupReport]:
        """Apply exactly one fresh canonical dry-run plan and fail loudly."""
        if not isinstance(plan, m.Tests.WorkspaceCleanupPlan):
            return r[p.Tests.WorkspaceCleanupReport].fail(
                "cleanup plan must be the canonical WorkspaceCleanupPlan model"
            )
        replanned_result = cls.workspace_cleanup_plan(plan.request)
        if replanned_result.failure:
            return r[p.Tests.WorkspaceCleanupReport].fail(
                f"cleanup plan is stale: {replanned_result.error}"
            )
        replanned = replanned_result.value
        if replanned.candidates != plan.candidates:
            return r[p.Tests.WorkspaceCleanupReport].fail(
                "cleanup plan is stale: candidates changed since dry-run"
            )
        root_result = cls._workspace_root(plan.request)
        if root_result.failure:
            return r[p.Tests.WorkspaceCleanupReport].fail(
                f"cleanup plan is stale: {root_result.error}"
            )
        root = root_result.value
        removed: list[Path] = []
        for candidate in plan.candidates:
            fresh_result = cls._candidate(root, candidate.relative_path)
            if fresh_result.failure:
                return r[p.Tests.WorkspaceCleanupReport].fail(
                    f"cleanup plan is stale: {fresh_result.error}"
                )
            if fresh_result.value != candidate:
                return r[p.Tests.WorkspaceCleanupReport].fail(
                    f"cleanup plan is stale for {candidate.relative_path}: "
                    "filesystem state changed since dry-run"
                )
            delete_result = u.Cli.files_delete(candidate.path)
            if delete_result.failure:
                completed = ", ".join(path.as_posix() for path in removed)
                return r[p.Tests.WorkspaceCleanupReport].fail(
                    f"cleanup deletion failed for {candidate.relative_path}: "
                    f"{delete_result.error}; already removed=[{completed}]"
                )
            if candidate.path.exists() or candidate.path.is_symlink():
                return r[p.Tests.WorkspaceCleanupReport].fail(
                    f"cleanup deletion reported success but path remains: "
                    f"{candidate.relative_path}"
                )
            removed.append(candidate.path)
        report = m.Tests.WorkspaceCleanupReport(plan=plan, removed=tuple(removed))
        return r[p.Tests.WorkspaceCleanupReport].ok(report)


__all__: tuple[str, ...] = ("FlextTestsWorkspaceCleanupPlanUtilitiesMixin",)
