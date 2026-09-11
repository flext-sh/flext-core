"""Root public-export decisions for the lazy-init planner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import c, m, t, u

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraCodegenLazyInitPlannerPublicRootMixin:
    """Public root-facade export filtering helpers."""

    if TYPE_CHECKING:
        lazy_init: m.Infra.LazyInitConfig
        rope_workspace: p.Infra.RopeWorkspaceDsl

    def _filter_public_root_exports(
        self,
        *,
        context: m.Infra.LazyInitPackageContext,
        export_names: set[str],
        lazy_map: t.MutableLazyAliasMap,
        eager_names: frozenset[str],
    ) -> tuple[set[str], t.MutableLazyAliasMap]:
        declared_contract = (
            self._declared_root_contract(context)
            if context.current_pkg.startswith("flext_")
            else None
        )
        governed_lazy_map = {
            name: target
            for name, target in lazy_map.items()
            if self._is_declared_root_export(
                name,
                target,
                root_pkg=context.current_pkg,
                declared_contract=declared_contract,
            )
        }
        lazy_map.clear()
        lazy_map.update(governed_lazy_map)
        public_export_names = {
            name
            for name in export_names
            if name in eager_names
            or (name in governed_lazy_map and name not in c.Infra.PUBLISHED_ALL_EXCLUDE)
        }
        filtered_lazy_map = {
            name: target
            for name, target in lazy_map.items()
            if name in public_export_names
        }
        return public_export_names, filtered_lazy_map

    def _declared_root_contract(
        self, context: m.Infra.LazyInitPackageContext
    ) -> frozenset[str] | None:
        if context.generated_init or not context.init_path.is_file():
            return None
        # If the project declares subpackages (e.g. services/), root aggregates from sources;
        # only single-directory/flat projects can declare an ABI filter via manual __init__.py.
        entry = self.rope_workspace.package(context.pkg_dir)
        if entry is not None and entry.descendant_child_dirs:
            return None
        constants_path = context.pkg_dir / c.Infra.CONSTANTS_PY
        if self.rope_workspace.resource(constants_path) is not None:
            imports = self.rope_workspace.semantic(constants_path).declared_imports
            if any(
                name != "annotations" and not target.startswith("__future__")
                for name, target in imports.items()
            ):
                return None
        contract = frozenset(
            self.rope_workspace.exports(
                context.init_path,
                export_options=m.Infra.ExportOptions(allow_assignments=True),
            )
        )
        return contract or None

    @staticmethod
    def _is_declared_root_export(
        name: str,
        target: t.StrPair,
        *,
        root_pkg: str,
        declared_contract: frozenset[str] | None,
    ) -> bool:
        # Private names never widen the public root ABI.
        if name.startswith("_"):
            return False
        if declared_contract is not None and name not in declared_contract:
            return False
        module_path, _attr_name = target
        runtime_module = f"{module_path.rsplit('.', maxsplit=1)[-1]}.py"
        if u.Infra.runtime_singleton_export(runtime_module) == name:
            return True
        if (module_path == f"{root_pkg}._config" and name.endswith("Config")) or (
            module_path == f"{root_pkg}._settings" and name.endswith("Settings")
        ):
            return True
        if module_path == root_pkg:
            return True
        if module_path.startswith(f"{root_pkg}."):
            # Any underscore-prefixed source segment
            # marks the owner as private; the symbol stays behind its facade.
            tail = module_path[len(root_pkg) + 1 :].split(".")
            return not any(
                part.startswith("_") and not part.startswith("__") for part in tail
            )
        return True


__all__: list[str] = ["FlextInfraCodegenLazyInitPlannerPublicRootMixin"]
