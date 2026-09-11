"""Child-package merging for the lazy-init planner."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from flext_infra import c

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import m, p, t


class FlextInfraCodegenLazyInitPlannerChildrenMixin:
    if TYPE_CHECKING:
        rope_workspace: p.Infra.RopeWorkspaceDsl

        def _package_entry(
            self, pkg_dir: Path
        ) -> m.Infra.RopePackageIndexEntry | None: ...

        def context(self, pkg_dir: Path) -> m.Infra.LazyInitPackageContext: ...

        _source_plan_cache: dict[str, m.Infra.LazyInitPlan]

        def _add(
            self, index: t.MutableLazyAliasMap, name: str, target: t.StrPair
        ) -> None: ...

        @staticmethod
        def _publish(name: str, *, allow_main: bool) -> bool: ...

    def _has_live_package_content(
        self, package_entry: m.Infra.RopePackageIndexEntry
    ) -> bool:
        """Return whether a package owns a module or a manual initializer."""
        candidates = (package_entry.package_dir, *package_entry.descendant_child_dirs)
        for candidate in candidates:
            entry = self._package_entry(candidate)
            if entry is None:
                continue
            if any(not module.is_package_init for module in entry.modules):
                return True
            if entry.init_path.is_file():
                source = entry.init_path.read_text(encoding=c.Cli.ENCODING_DEFAULT)
                if source.strip() and not source.startswith(c.Infra.AUTOGEN_HEADER):
                    return True
        return False

    def _merge_children(
        self,
        pkg_dir: Path,
        lazy_map: t.MutableLazyAliasMap,
        dir_exports: t.MappingKV[str, t.LazyAliasMap],
    ) -> t.StrSequence:
        """Merge direct child packages into the parent lazy map."""
        package_entry = self._package_entry(pkg_dir)
        if package_entry is None:
            return ()
        resolved_pkg_dir = pkg_dir.resolve()
        parent_pkg = self.context(pkg_dir).current_pkg
        direct: list[str] = []
        for child_dir in package_entry.descendant_child_dirs:
            # flext-pulj (codex): do not merge retired root registries into the
            # inline map that replaces them.
            if child_dir.name in c.Infra.OBSOLETE_ROOT_SUPPORT_NAMES:
                continue
            resolved_child_dir = child_dir.resolve()
            # flext-mh7g4: children are planned before their parent (depth
            # descending), so the parent inventory follows the child's plan in
            # the same pass — a planned WRITE counts as a package even before
            # its initializer exists on disk, and a planned REMOVE/SKIP (for
            # example a stdlib-shadowing name) never does. Without a plan the
            # on-disk initializer decides. This keeps check and apply at a
            # fixed point after one run.
            child_plan = self._source_plan_cache.get(str(resolved_child_dir))
            planned_action = child_plan.action if child_plan is not None else None
            if planned_action is c.Infra.LazyInitAction.REMOVE:
                continue
            planned_write = planned_action is c.Infra.LazyInitAction.WRITE
            child_init = child_dir / c.Infra.INIT_PY
            if not planned_write and not child_init.is_file():
                continue
            child_entry = self._package_entry(child_dir)
            is_fixture_child = self._is_fixture_package(child_dir)
            child_exports = dir_exports.get(str(resolved_child_dir), {})
            child_pkg_name = (
                child_entry.package_name
                if child_entry is not None and child_entry.package_name
                else (f"{parent_pkg}.{child_dir.name}" if planned_write else "")
            )
            if not child_pkg_name:
                continue
            if (
                not child_exports
                and not planned_write
                and child_entry is not None
                and not self._has_live_package_content(child_entry)
            ):
                continue
            if resolved_child_dir.parent != resolved_pkg_dir:
                continue
            # flext-pulj (codex): private fixture modules are pytest-owned plugin
            # boundaries and never bubble into their production package root.
            if is_fixture_child:
                continue
            direct.append(child_pkg_name)
            self._add(
                lazy_map,
                child_pkg_name.rsplit(".", maxsplit=1)[-1],
                (child_pkg_name, ""),
            )
            for name, (module_name, attr) in child_exports.items():
                source_module_name = module_name.rsplit(".", maxsplit=1)[-1]
                test_only_source_module = (
                    c.Infra.TEST_ONLY_SOURCE_MODULE_RE.fullmatch(
                        f"{source_module_name}.py"
                    )
                    is not None
                )
                if (
                    attr
                    and not test_only_source_module
                    and self._publish(name, allow_main=True)
                ):
                    self._add(lazy_map, name, (module_name, attr))
        return tuple(sorted(direct))

    def _shadows_stdlib_module(self, pkg_dir: Path) -> bool:
        """Return True when the package's importable name shadows a stdlib module.

        Ruff (``stdlib-module-shadowing``, non-strict) flags a module whose
        path relative to a configured source root (``src``, ``tests``,
        ``examples``, ``scripts``) is exactly a stdlib module name. So
        ``tests/typing/`` shadows (importable as ``typing`` from the ``tests``
        root) while ``tests/unit/io/`` and ``pkg/services/http/`` do not.
        The generator's own Ruff gate rejects the shadowing render, and no
        generated content can repair the package name.
        """
        current_pkg = self.context(pkg_dir).current_pkg
        parts = (
            current_pkg.split(".")
            if current_pkg
            else [pkg_dir.parent.name, pkg_dir.name]
        )
        relative_to_root = (
            parts[1:] if parts[0] in c.Infra.NON_PUBLIC_LAZY_ROOTS else parts
        )
        return (
            len(relative_to_root) == 1
            and relative_to_root[0] in sys.stdlib_module_names
        )

    @staticmethod
    def _is_fixture_package(pkg_dir: Path) -> bool:
        """Return True when the directory is the ``_fixtures`` convention package."""
        return pkg_dir.name == "_fixtures"

    @classmethod
    def _is_private_test_fixture_package(cls, pkg_dir: Path, surface: str) -> bool:
        """Return True when the package is a private fixture under the tests surface."""
        return surface == "tests" and cls._is_fixture_package(pkg_dir)
