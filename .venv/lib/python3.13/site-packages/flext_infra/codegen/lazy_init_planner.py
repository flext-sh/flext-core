"""Lazy-init planning over generic Rope workspace indexes."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, override

from flext_infra import c, m, p, t, u
from flext_infra.codegen._lazy_init_planner_aliases import (
    FlextInfraCodegenLazyInitPlannerAliasesMixin,
)
from flext_infra.codegen._lazy_init_planner_cache import (
    FlextInfraCodegenLazyInitPlannerCacheMixin,
)
from flext_infra.codegen._lazy_init_planner_children import (
    FlextInfraCodegenLazyInitPlannerChildrenMixin,
)
from flext_infra.codegen._lazy_init_planner_collision import (
    FlextInfraCodegenLazyInitPlannerCollisionMixin,
)
from flext_infra.codegen._lazy_init_planner_exports import (
    FlextInfraCodegenLazyInitPlannerExportsMixin,
)
from flext_infra.codegen._lazy_init_planner_parents import (
    FlextInfraCodegenLazyInitPlannerParentsMixin,
)
from flext_infra.codegen._lazy_init_planner_public_root import (
    FlextInfraCodegenLazyInitPlannerPublicRootMixin,
)


class FlextInfraCodegenLazyInitPlannerBase(m.ArbitraryTypesModel):
    """Pydantic state base for lazy-init planning."""

    rope_workspace: Annotated[
        p.Infra.RopeWorkspaceDsl,
        m.Field(description="Shared Rope workspace DSL reused by the planner"),
    ]
    lazy_init: m.Infra.LazyInitConfig = m.Field(
        description="Validated lazy-init policy document"
    )

    _module_exports_cache: dict[
        tuple[str, bool, bool, bool, bool, bool], t.LazyAliasMap
    ] = u.PrivateAttr(default_factory=dict)
    _package_exports_cache: dict[str, frozenset[str]] = u.PrivateAttr(
        default_factory=dict
    )
    _source_exports_cache: dict[str, frozenset[str]] = u.PrivateAttr(
        default_factory=dict
    )
    _source_plan_cache: dict[str, m.Infra.LazyInitPlan] = u.PrivateAttr(
        default_factory=dict
    )
    _source_exports_visiting: set[str] = u.PrivateAttr(default_factory=set)
    _parent_package_cache: dict[str, t.StrSequence] = u.PrivateAttr(
        default_factory=dict
    )
    _module_file_by_name: dict[str, Path] = u.PrivateAttr(default_factory=dict)
    _version_module_name: str = u.PrivateAttr(
        default_factory=lambda: f"{c.Infra.DUNDER_VERSION}.py"
    )
    _collision_count: int = u.PrivateAttr(default_factory=int)

    @property
    def collision_count(self) -> int:
        """Number of export collisions resolved so far."""
        return self._collision_count


class FlextInfraCodegenLazyInitPlanner(
    FlextInfraCodegenLazyInitPlannerBase,
    FlextInfraCodegenLazyInitPlannerAliasesMixin,
    FlextInfraCodegenLazyInitPlannerExportsMixin,
    FlextInfraCodegenLazyInitPlannerChildrenMixin,
    FlextInfraCodegenLazyInitPlannerCollisionMixin,
    FlextInfraCodegenLazyInitPlannerParentsMixin,
    FlextInfraCodegenLazyInitPlannerCacheMixin,
    FlextInfraCodegenLazyInitPlannerPublicRootMixin,
):
    """Resolve lazy-init plans using one shared Rope workspace index."""

    @override
    def build_plan(
        self, pkg_dir: Path, *, dir_exports: t.MappingKV[str, t.LazyAliasMap]
    ) -> m.Infra.LazyInitPlan:
        """Build the lazy-init render plan for one package directory."""
        context = self.context(pkg_dir)
        if self._shadows_stdlib_module(pkg_dir):
            # flext-mh7g4: no generated content can repair a package name that
            # shadows a stdlib module, so the plan removes generator-owned
            # residue and otherwise skips the directory. The ALL_SCAN_PATTERNS
            # contract is unchanged: every surface is still scanned; only
            # directories that cannot legally be packages are not rendered,
            # and _merge_children applies the same predicate to the parent.
            residue_action = (
                c.Infra.LazyInitAction.REMOVE
                if context.generated_init
                else c.Infra.LazyInitAction.SKIP
            )
            return self._publish_plan(
                m.Infra.LazyInitPlan(context=context, action=residue_action)
            )
        is_test_child_package = (
            context.surface == c.Infra.DIR_TESTS
            and context.current_pkg != c.Infra.DIR_TESTS
        )
        empty_action: c.Infra.LazyInitAction = (
            c.Infra.LazyInitAction.WRITE
            if is_test_child_package
            else (
                c.Infra.LazyInitAction.REMOVE
                if context.generated_init
                else c.Infra.LazyInitAction.SKIP
            )
        )
        if not context.importable:
            return self._publish_plan(
                m.Infra.LazyInitPlan(context=context, action=empty_action)
            )
        lazy_map = self._package_exports(context)
        version_map = self._module_exports(
            context.pkg_dir / self._version_module_name,
            f"{context.current_pkg}.{c.Infra.DUNDER_VERSION}",
            export_options=m.Infra.ExportOptions(include_dunder=True),
        )
        child_lazy = self._merge_children(context.pkg_dir, lazy_map, dir_exports)
        # Version-submodule dunders are emitted as eager imports rather than
        # lazy. The submodule shares its name (``__version__``) with the dunder
        # string it exports; lazy resolution would let Python's import
        # machinery shadow the dunder with the submodule object on first
        # access. Eager binding at __init__.py load time pins the canonical
        # strings in the package dict permanently.
        eager_dunders = dict(version_map)
        for name in eager_dunders:
            lazy_map.pop(name, None)
        self._resolve_aliases(
            lazy_map,
            current_pkg=context.current_pkg,
            pkg_dir=context.pkg_dir,
            surface=context.surface,
        )
        for name in c.Infra.INFRA_ONLY_EXPORTS:
            lazy_map.pop(name, None)
            eager_dunders.pop(name, None)
        if not lazy_map and not eager_dunders:
            return m.Infra.LazyInitPlan(context=context, action=empty_action)
        excluded_lazy_names: t.StrSequence = ()
        is_public_project_root = (
            context.pkg_dir.parent.name == c.Infra.DEFAULT_SRC_DIR
            and context.current_pkg
            and "." not in context.current_pkg
            # Why (flext-27a9e.1, multi-agent): governed consumers such as ai_hub
            # are first-class project roots; package prefixes are not architecture.
            and u.Infra.matches_project_namespace_package(context.current_pkg)
        )
        is_test_facade_root = (
            context.current_pkg == c.Infra.DIR_TESTS
            and context.pkg_dir.name == c.Infra.DIR_TESTS
            and context.surface == c.Infra.DIR_TESTS
        )
        is_facade_root = is_public_project_root or is_test_facade_root
        export_names = {*lazy_map, *eager_dunders}
        if not is_facade_root:
            # flext-udpm5: a nested package's own modules commonly consume
            # the project root's already-published facade aliases directly
            # (``from <root> import c, m, p, ...``) without defining any
            # local class of their own under that alias. _resolve_aliases
            # then inherits the ROOT's own alias entry (package_name ==
            # this package's own project root) into lazy_map so internal
            # code can still resolve it, but republishing it as part of
            # THIS package's own __all__/TYPE_CHECKING contract is a pure
            # upstream re-export, not a local owner: it is redundant with
            # the root's own contract and, being an absolute self-import of
            # the project root, fails the relative-owner validation in
            # generate_type_checking. Exclude only that exact self-pointing
            # case (an alias whose target is literally the project root
            # under its own name) here, at the one place that decides the
            # publishable contract; every alias with a genuine local or
            # foreign-parent owner (module path does not equal the bare
            # root package) stays published exactly as before.
            root_pkg_name = context.current_pkg.split(".", maxsplit=1)[0]
            if root_pkg_name and root_pkg_name != context.current_pkg:
                export_names = {
                    name
                    for name in export_names
                    if name not in c.Infra.ALIAS_NAMES
                    or lazy_map.get(name) != (root_pkg_name, name)
                }
        if is_public_project_root:
            package_alias = u.Infra.package_alias(package_name=context.current_pkg)
            if (
                package_alias not in export_names
                and (context.pkg_dir / c.Infra.API_PY).is_file()
            ):
                export_names.add(package_alias)
        if is_facade_root:
            # flext-pulj (codex) + ulw follow-up: __all__ is the one public
            # contract (dir()/star-import/docs already respect it). Do NOT
            # narrow lazy_map/_LAZY_MODULES to match -- internal fragments across
            # the package rely on lazy __getattr__ resolving the root facade
            # (`from flext_core import c`) during their own eager import chain;
            # pruning the lazy map to the public subset breaks that resolution
            # with a circular ImportError the moment any pruned name is touched
            # before __init__ finishes executing. Only export_names (-> __all__)
            # is filtered; lazy_map stays the full discovered set.
            export_names, filtered_lazy_map = self._filter_public_root_exports(
                context=context,
                export_names=export_names,
                lazy_map=lazy_map,
                eager_names=frozenset(eager_dunders),
            )
            lazy_map = filtered_lazy_map
            child_lazy = ()
            excluded_lazy_names = ()
        preserve_manual_init = (
            not is_facade_root
            and context.init_path.is_file()
            and not context.generated_init
            and bool(
                context.init_path.read_text(encoding=c.Cli.ENCODING_DEFAULT).strip()
            )
        )
        type_checking_map = dict(lazy_map)
        all_export_names = tuple(sorted(export_names))
        plan = m.Infra.LazyInitPlan(
            context=context,
            action=(
                c.Infra.LazyInitAction.SKIP
                if preserve_manual_init
                else c.Infra.LazyInitAction.WRITE
            ),
            exports=u.Infra.ordered_namespace_exports(
                package_dir=context.pkg_dir,
                package_name=context.current_pkg,
                export_names=all_export_names,
            ),
            lazy_map=dict(lazy_map),
            type_checking_map=type_checking_map,
            eager_dunders=eager_dunders,
            wildcard_runtime_modules=(),
            child_packages_for_lazy=child_lazy,
            excluded_lazy_names=excluded_lazy_names,
        )
        self._source_exports_cache[context.current_pkg] = frozenset(plan.exports)
        return self._publish_plan(plan)

    def _publish_plan(self, plan: m.Infra.LazyInitPlan) -> m.Infra.LazyInitPlan:
        """Publish one bottom-up plan so parents follow it in the same pass.

        flext-pulj (codex): later alias resolution never rebuilds a package
        without its children. flext-mh7g4: every plan, including REMOVE and
        SKIP decided before rendering, is published so the parent inventory in
        ``_merge_children`` sees the child's action instead of the on-disk
        initializer.
        """
        self._source_plan_cache[str(plan.context.pkg_dir.resolve())] = plan
        return plan

    @override
    def context(self, pkg_dir: Path) -> m.Infra.LazyInitPackageContext:
        """Return the lazy-init package context for the requested package directory."""
        return self.rope_workspace.package_context(pkg_dir)


__all__: list[str] = ["FlextInfraCodegenLazyInitPlanner"]
