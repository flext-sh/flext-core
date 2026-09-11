"""Package/module entry and name-resolution caches for the lazy-init planner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import c, m, u

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p, t


class FlextInfraCodegenLazyInitPlannerCacheMixin:
    if TYPE_CHECKING:
        rope_workspace: p.Infra.RopeWorkspaceDsl
        _package_exports_cache: dict[str, frozenset[str]]
        _source_exports_cache: dict[str, frozenset[str]]
        _source_plan_cache: dict[str, m.Infra.LazyInitPlan]
        _source_exports_visiting: set[str]
        _module_file_by_name: dict[str, Path]

        def build_plan(
            self, pkg_dir: Path, *, dir_exports: t.MappingKV[str, t.LazyAliasMap]
        ) -> m.Infra.LazyInitPlan: ...

    def _export_names_for_package(self, package_name: str) -> frozenset[str]:
        """Return all export names for a package (init + source plans)."""
        cached = self._package_exports_cache.get(package_name)
        if cached is not None:
            return cached
        exports = frozenset((
            *self._package_init_exports(package_name),
            *self._source_export_names_for_package(package_name),
        ))
        self._package_exports_cache[package_name] = exports
        return exports

    def _package_init_exports(self, package_name: str) -> frozenset[str]:
        """Return names exported from the package __init__.py."""
        package_dir = self.rope_workspace.workspace_index.package_dir_by_name.get(
            package_name
        )
        if package_dir is None:
            return u.Infra.installed_package_exports(package_name)
        init_path = package_dir / c.Infra.INIT_PY
        if self.rope_workspace.resource(init_path) is None:
            return frozenset()
        if init_path.read_text(encoding=c.Cli.ENCODING_DEFAULT).startswith(
            c.Infra.AUTOGEN_HEADERS
        ):
            return frozenset()
        return frozenset(
            self.rope_workspace.exports(
                init_path, export_options=m.Infra.ExportOptions(allow_assignments=True)
            )
        )

    def _source_export_names_for_package(self, package_name: str) -> frozenset[str]:
        """Return names exported by a full source build_plan run (cycle-safe)."""
        cached = self._source_exports_cache.get(package_name)
        if cached is not None:
            return cached
        if package_name in self._source_exports_visiting:
            return frozenset()
        package_dir = self.rope_workspace.workspace_index.package_dir_by_name.get(
            package_name
        )
        if package_dir is None:
            return frozenset()
        self._source_exports_visiting.add(package_name)
        try:
            cache_key = str(package_dir.resolve())
            plan = self._source_plan_cache.get(cache_key)
            if plan is None:
                plan = self.build_plan(package_dir, dir_exports={})
                self._source_plan_cache[cache_key] = plan
            exports = frozenset(plan.exports)
        finally:
            self._source_exports_visiting.remove(package_name)
        self._source_exports_cache[package_name] = exports
        return exports

    def _source_package_name(self, pkg_dir: Path, inherited_key: str) -> str:
        """Return the project-root package name for the given directory."""
        if inherited_key == "src":
            return ""
        package_entry = self._package_entry(pkg_dir)
        if package_entry is None or package_entry.project_root is None:
            return ""
        project_pkg: str = (
            self.rope_workspace.workspace_index.project_package_by_root.get(
                str(package_entry.project_root), ""
            )
        )
        return project_pkg

    def _package_entry(self, pkg_dir: Path) -> m.Infra.RopePackageIndexEntry | None:
        """Return the workspace index entry for a package directory."""
        return self.rope_workspace.package(pkg_dir)

    def _module_file(self, module_path: str) -> Path | None:
        """Resolve a dotted module path to its source file (lazy index build)."""
        resolved = self._module_file_by_name.get(module_path)
        if resolved is not None:
            return resolved
        if not self._module_file_by_name:
            index = self.rope_workspace.workspace_index
            self._module_file_by_name = {
                entry.module_name: entry.file_path
                for entry in index.modules_by_path.values()
                if entry.module_name
            }
            resolved = self._module_file_by_name.get(module_path)
        return resolved
