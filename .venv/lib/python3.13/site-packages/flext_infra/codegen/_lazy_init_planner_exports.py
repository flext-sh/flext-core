"""Per-package and per-module export resolution for the lazy-init planner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import c, m, u

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p, t


class FlextInfraCodegenLazyInitPlannerExportsMixin:
    if TYPE_CHECKING:
        rope_workspace: p.Infra.RopeWorkspaceDsl
        lazy_init: m.Infra.LazyInitConfig
        _module_exports_cache: dict[
            tuple[str, bool, bool, bool, bool, bool], t.LazyAliasMap
        ]
        _version_module_name: str

        @classmethod
        def _is_private_test_fixture_package(
            cls, pkg_dir: Path, surface: str
        ) -> bool: ...

        def _package_entry(
            self, pkg_dir: Path
        ) -> m.Infra.RopePackageIndexEntry | None: ...

        def _add(
            self, index: t.MutableLazyAliasMap, name: str, target: t.StrPair
        ) -> None: ...

        @staticmethod
        def _publish(name: str, *, allow_main: bool) -> bool: ...

    def _package_exports(
        self, context: m.Infra.LazyInitPackageContext
    ) -> t.MutableLazyAliasMap:
        """Return the lazy export map for a package (excluding child packages)."""
        if self._is_private_test_fixture_package(context.pkg_dir, context.surface):
            return {}
        package_entry = self._package_entry(context.pkg_dir)
        if package_entry is None:
            return {}
        index: t.MutableLazyAliasMap = {}
        # flext-i6nq.10: Generated support modules are output, never public input.
        skip_names = {
            c.Infra.INIT_PY,
            "__main__.py",
            self._version_module_name,
            *c.Infra.OBSOLETE_GENERATED_INIT_FILES,
        }
        for module_entry in package_entry.modules:
            py_file = module_entry.file_path
            child_dir = py_file.parent / py_file.stem
            child_entry = self._package_entry(child_dir)
            # flext-pulj: test artifacts never enter an installable package ABI.
            test_only_source_module = (
                context.surface != c.Infra.DIR_TESTS
                or context.current_pkg == c.Infra.DIR_TESTS
            ) and (
                c.Infra.TEST_ONLY_SOURCE_MODULE_RE.fullmatch(py_file.name) is not None
            )
            # flext-6int (claude-ulw): extract predicate to satisfy PLR0916
            # (>5 boolean expressions); retired/generated/test modules are
            # never semantic input for the lazy export map.
            is_generated_or_test = (
                py_file.name in skip_names
                or c.Infra.GENERATED_EXPORT_SIDECAR_RE.match(py_file.name)
                or py_file.stem in c.Infra.OBSOLETE_ROOT_SUPPORT_NAMES
                or test_only_source_module
            )
            is_child_package = child_entry is not None and child_entry.package_name
            if is_generated_or_test or is_child_package:
                continue
            convention = self.rope_workspace.convention(
                py_file, rel_path=py_file.relative_to(context.pkg_dir)
            )
            policy = convention.module_policy
            root_private_contract = (
                py_file.parent == context.pkg_dir
                and py_file.stem in {"_config", "_settings"}
                and bool(
                    self._module_exports(
                        py_file,
                        convention.module_name,
                        export_options=m.Infra.ExportOptions(
                            allow_main=True,
                            allow_assignments=True,
                            allow_functions=True,
                            require_explicit_all=True,
                        ),
                    )
                )
            )
            if (
                not policy.include_in_lazy_init and not root_private_contract
            ) or not module_entry.module_name:
                continue
            # In public src packages, public submodules (without expected_alias) derive
            # from their explicit __all__; non-public/private subpackages auto-discover.
            require_explicit_all = (
                context.surface not in c.Infra.NON_PUBLIC_LAZY_ROOTS
                and not any(part.startswith("_") for part in context.pkg_dir.parts)
                and not py_file.stem.startswith("_")
                and (
                    u.Infra.matches_root_namespace_file(py_file.name)
                    or policy.expected_alias is not None
                    or "." in context.current_pkg
                )
            )
            targets = self._module_exports(
                py_file,
                convention.module_name,
                export_options=m.Infra.ExportOptions(
                    allow_main=True,
                    allow_assignments=True,
                    allow_functions=True,
                    require_explicit_all=require_explicit_all,
                ),
            )
            if (
                policy.expected_alias
                and u.Infra.matches_project_namespace_package(context.current_pkg)
                and u.Infra.matches_root_namespace_file(py_file.name)
            ):
                targets.setdefault(
                    policy.expected_alias,
                    (module_entry.module_name, policy.expected_alias),
                )
            for name, target in targets.items():
                self._add(index, name, target)
        return index

    def _module_exports(
        self,
        py_file: Path,
        module_path: str,
        *,
        export_options: m.Infra.ExportOptions | None = None,
    ) -> t.MutableLazyAliasMap:
        """Return the lazy export map for one Python module (cache-backed)."""
        resolved_export_options = export_options or m.Infra.ExportOptions()
        cache_key = (
            str(py_file.resolve()),
            resolved_export_options.include_dunder,
            resolved_export_options.allow_main,
            resolved_export_options.allow_assignments,
            resolved_export_options.allow_functions,
            resolved_export_options.require_explicit_all,
        )
        cached = self._module_exports_cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        if self.rope_workspace.resource(py_file) is None:
            return {}
        names = self.rope_workspace.exports(
            py_file,
            export_options=resolved_export_options.model_copy(
                update={
                    "require_explicit_all": (
                        resolved_export_options.require_explicit_all
                        and not resolved_export_options.include_dunder
                    )
                }
            ),
        )
        exports = {
            name: (module_path, name)
            for name in names
            if resolved_export_options.include_dunder
            or self._publish(name, allow_main=resolved_export_options.allow_main)
        }
        self._module_exports_cache[cache_key] = exports
        return dict(exports)
