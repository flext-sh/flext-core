"""Cohesive doc-sync + transitive-dependency mixin for FlextInfraExtraPathsManager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import c, u

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from flext_infra import p, t


class FlextInfraExtraPathsSyncMixin:
    """Mixin holding TOML doc-sync plus transitive-dependency path resolution."""

    if TYPE_CHECKING:
        # Provided by the concrete FlextInfraExtraPathsManager / its base; declared
        # for static resolution only so they don't shadow the runtime implementations.
        root: Path
        _workspace_project_names: t.Infra.StrSet
        pyright_extra_paths: Callable[..., t.StrSequence]
        pyrefly_search_paths: Callable[..., t.StrSequence]
        mypy_search_paths: Callable[..., t.StrSequence]

    def resolve_transitive_dependency_names(
        self, direct_names: t.StrSequence
    ) -> t.StrSequence:
        """Return the transitive workspace path-dependency closure of direct_names."""
        return self._resolve_transitive_deps(direct_names)

    def _resolve_transitive_deps(
        self, direct_names: t.StrSequence, *, visited: t.Infra.StrSet | None = None
    ) -> t.StrSequence:
        """Recursively resolve transitive workspace path dependencies."""
        resolved_visited: set[str] = visited if visited is not None else set()
        all_paths: set[str] = set(direct_names)
        for name in direct_names:
            if name in resolved_visited:
                continue
            resolved_visited.add(name)
            dep_pyproject = self.root / name / c.Infra.PYPROJECT_FILENAME
            if not dep_pyproject.exists():
                continue
            dep_payload = u.Infra.pyproject_payload(dep_pyproject)
            transitive = u.Infra.local_dependency_names_from_payload(
                dep_payload,
                workspace_project_names=tuple(self._workspace_project_names),
            )
            if not transitive:
                continue
            all_paths.update(transitive)
            all_paths.update(
                self._resolve_transitive_deps(transitive, visited=resolved_visited)
            )
        return sorted(all_paths)

    def sync_doc(
        self, doc: t.Cli.TomlDocument, *, project_dir: Path, is_root: bool
    ) -> t.StrSequence:
        """Apply computed extra paths to an in-memory TOMLDocument."""
        # Path producers and toml_as_string_list both yield immutable
        # sequences, so equal content compares equal (no list/tuple churn).
        expected = self.pyright_extra_paths(project_dir=project_dir, is_root=is_root)
        tool_table = u.Cli.toml_table_child(doc, c.Infra.TOOL)
        if tool_table is None:
            return list[str]()
        pyright_table = u.Cli.toml_table_child(tool_table, c.Infra.PYRIGHT)
        if pyright_table is None:
            return list[str]()
        mypy_table = u.Cli.toml_table_child(tool_table, c.Infra.MYPY)
        changes: t.MutableSequenceOf[str] = []
        pyright_extra_paths = u.Cli.toml_item_child(pyright_table, "extraPaths")
        current_pyright = u.Cli.toml_as_string_list(
            pyright_extra_paths if pyright_extra_paths is not None else []
        )
        if current_pyright != expected:
            pyright_table["extraPaths"] = expected
            changes.append("synchronized pyright extraPaths")
        if mypy_table is not None:
            # NOT the pyrefly search path any more (cosmos-45hiv, 2026-08-31).
            # mypy enumerates every search-path root as a package root, so a
            # root that re-spells an already-rooted module makes it report the
            # same file twice ("Source file found twice under different module
            # names") and abort repo-wide. pyrefly resolves first-match and
            # tolerates what mypy cannot. mypy gets source + shared config
            # paths only; the project root stays a pyrefly-only resolution aid.
            expected_mypy = self.mypy_search_paths(
                project_dir=project_dir, is_root=is_root
            )
            mypy_path_item = u.Cli.toml_item_child(mypy_table, "mypy_path")
            current_mypy = u.Cli.toml_as_string_list(
                mypy_path_item if mypy_path_item is not None else []
            )
            if current_mypy != expected_mypy:
                mypy_table["mypy_path"] = expected_mypy
                tool_table[c.Infra.MYPY] = mypy_table
                changes.append("synchronized mypy mypy_path")
        if changes:
            tool_table[c.Infra.PYRIGHT] = pyright_table
            doc["tool"] = tool_table
        return changes

    def sync_payload(
        self, payload: t.MutableJsonMapping, *, project_dir: Path, is_root: bool
    ) -> t.StrSequence:
        """Apply computed extra paths to one normalized TOML payload."""
        expected = self.pyright_extra_paths(project_dir=project_dir, is_root=is_root)
        tool_table = u.Cli.toml_mapping_path(payload, (c.Infra.TOOL,))
        if tool_table is None:
            return list[str]()
        pyright_table = u.Cli.toml_mapping_path(tool_table, (c.Infra.PYRIGHT,))
        if pyright_table is None:
            return list[str]()
        mypy_table = u.Cli.toml_mapping_path(tool_table, (c.Infra.MYPY,))
        changes: t.MutableSequenceOf[str] = []
        if u.Cli.toml_mapping_sync_string_list(
            u.Cli.toml_mapping_ensure_path(payload, (c.Infra.TOOL, c.Infra.PYRIGHT)),
            "extraPaths",
            expected,
        ):
            changes.append("synchronized pyright extraPaths")
        # Mypy resolves the same import graph as Pyrefly, so it needs the same
        # roots. pyright_extra_paths omits path dependencies, which left sibling
        # packages unresolvable and degraded every symbol they export to Any.
        # It does NOT take pyrefly_search_paths any more (cosmos-45hiv): mypy
        # enumerates each root as a package root and aborts repo-wide when two
        # roots re-spell one file (source-file-found-twice). First-match
        # resolution is pyrefly-only.
        if mypy_table is not None and u.Cli.toml_mapping_sync_string_list(
            u.Cli.toml_mapping_ensure_path(payload, (c.Infra.TOOL, c.Infra.MYPY)),
            "mypy_path",
            self.mypy_search_paths(project_dir=project_dir, is_root=is_root),
        ):
            changes.append("synchronized mypy mypy_path")
        return changes

    def sync_one(
        self, pyproject_path: Path, *, dry_run: bool = False, is_root: bool = False
    ) -> p.Result[bool]:
        """Synchronize pyright and mypy paths for one pyproject.toml."""
        if not pyproject_path.exists():
            return r[bool].fail(f"pyproject not found: {pyproject_path}")
        doc_result = u.Cli.toml_read_document(pyproject_path)
        if doc_result.failure:
            return r[bool].fail(doc_result.error or f"failed to read {pyproject_path}")
        changes = self.sync_doc(
            doc_result.value, project_dir=pyproject_path.parent, is_root=is_root
        )
        if changes and (not dry_run):
            write_result = u.Cli.toml_write_document(pyproject_path, doc_result.value)
            if write_result.failure:
                return r[bool].fail(
                    write_result.error or f"failed to write {pyproject_path}"
                )
        return r[bool].ok(bool(changes))

    def sync_extra_paths(
        self, *, dry_run: bool = False, project_dirs: t.SequenceOf[Path] | None = None
    ) -> p.Result[int]:
        """Synchronize extraPaths and mypy_path across projects."""
        if project_dirs:
            updated_selected = 0
            for project_dir in project_dirs:
                pyproject = project_dir / c.Infra.PYPROJECT_FILENAME
                # A governed worktree transaction materializes only the scoped
                # submodule's source tree, so a selected member legitimately has
                # no pyproject there. Failing closed made every scoped apply
                # report breakage and silently skip writing the derived paths.
                if not pyproject.exists():
                    continue
                sync_result = self.sync_one(
                    pyproject, dry_run=dry_run, is_root=project_dir == self.root
                )
                if sync_result.failure:
                    return r[int].fail(
                        sync_result.error or f"sync failed for {pyproject}"
                    )
                if sync_result.value and (not dry_run):
                    updated_selected += 1
                    u.Cli.info(f"Updated {pyproject}")
            return r[int].ok(updated_selected)
        # WHAT=all is the canonical default: an unselected run owns the root AND
        # every managed member. Syncing only the root left each member's
        # mypy_path/extraPaths frozen at whatever was last written by hand, so
        # sibling path dependencies never reached the member analyzers and every
        # symbol imported from them degraded to Any.
        targets: t.MutableSequenceOf[Path] = [self.root]
        targets.extend(
            self.root / member for member in u.Infra.workspace_project_paths(self.root)
        )
        updated = 0
        for target in targets:
            pyproject = target / c.Infra.PYPROJECT_FILENAME
            if not pyproject.exists():
                if target == self.root:
                    return r[int].fail(f"Missing {pyproject}")
                continue
            sync_result = self.sync_one(
                pyproject, dry_run=dry_run, is_root=target == self.root
            )
            if sync_result.failure:
                return r[int].fail(sync_result.error or f"sync failed for {pyproject}")
            if sync_result.value and (not dry_run):
                updated += 1
                u.Cli.info(f"Updated {pyproject}")
        return r[int].ok(updated)


__all__: list[str] = ["FlextInfraExtraPathsSyncMixin"]
