"""Census parent-package facade-alias inventory + collision detection."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from flext_infra import p, u

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import m, t


class FlextInfraRefactorCensusInventoryMixin:
    """Governed-parent facade-alias inventory + workspace collision cross-ref.

    Composed into FlextInfraRefactorCensus via inheritance; ``_is_flext_owned``
    is provided by the sibling objects mixin through FLEXT.
    """

    if TYPE_CHECKING:

        @staticmethod
        def _is_flext_owned(value: p.ModuleOwned) -> bool: ...

    @classmethod
    def _build_parent_inventory(
        cls, repository_root: Path
    ) -> t.MappingKV[str, t.StrSequence]:
        """Inventory governed-package alias top-level facade names.

        Discovers governed projects via ``u.Infra.projects(repository_root)``
        (canonical workspace project discovery — SSOT). For each project
        whose hyphenated name converts to a Python package, imports the
        package and walks dynamic facade aliases at depth 1
        (top-level facade attributes such as ``flext_core.c.Result``).

        Returns ``{symbol_name: (parent_path, ...)}`` so a consumer-defined
        symbol with the same name can be cross-referenced against every
        parent that declares it.

        Only ``type`` instances (classes) are inventoried; method names
        inherited from ABCs (``clear``, ``get``, …) are skipped — every
        mapping class shares them, so they are not collision candidates.

        Filters out attributes whose values' ``__module__`` is not in the
        flext package tree.

        Read-only runtime introspection — NO Rope, NO source-tree walking,
        NO subprocess. Skips packages that fail to import (sub-repo
        environments may not have every flext-* installed).
        """
        projects_result = u.Infra.projects(repository_root)
        if projects_result.failure:
            return {}
        inventory: dict[str, list[str]] = defaultdict(list)
        for project in projects_result.unwrap():
            pkg_name = project.name.replace("-", "_")
            try:
                module = __import__(pkg_name)
            except ImportError:
                continue
            import_name = pkg_name.replace("-", "_")
            for alias_name, module_name, _ in u.lazy_alias_suffixes(import_name):
                if module_name.split(".", 1)[0] != import_name:
                    continue
                alias = getattr(module, alias_name, None)
                if alias is None:
                    continue
                for attr in dir(alias):
                    if attr.startswith("_"):
                        continue
                    nested = getattr(alias, attr, None)
                    if (
                        nested is None
                        or not isinstance(nested, p.ModuleOwned)
                        or not cls._is_flext_owned(nested)
                    ):
                        continue
                    if not isinstance(nested, type):
                        continue
                    inventory[attr].append(f"{pkg_name}.{alias_name}.{attr}")
        return {name: tuple(paths) for name, paths in inventory.items()}

    @classmethod
    def parent_alias_collisions(
        cls, report: m.Infra.Census.WorkspaceReport, *, repository_root: Path
    ) -> tuple[tuple[m.Infra.Census.Object, t.StrSequence], ...]:
        """Cross-reference workspace objects against upstream parent inventory.

        Returns ``(symbol, parent_paths)`` pairs where the consumer's
        public symbol name appears on at least one governed parent
        package's dynamically derived facade aliases. Sorted descending by the number of
        matching parent paths (broadest collision surface first).

        Self-references are filtered: a symbol in project ``flext-core``
        whose name matches a symbol on ``flext_core.<alias>.<name>`` is
        the canonical owner, not a duplicate.

        Args:
            report: A ``WorkspaceReport`` produced by ``execute()`` or
                ``_collect_report(...)``. Reusing the existing report
                avoids a second Rope-walk; the inventory is the only new
                I/O.
            repository_root: Repository root used to discover governed
                projects (parent packages).

        Returns:
            Tuple of ``(symbol, parent_paths)`` pairs. Empty tuple if
            no collisions are found.

        """
        inventory = cls._build_parent_inventory(repository_root)
        collisions: list[tuple[m.Infra.Census.Object, t.StrSequence]] = []

        def collision_breadth(
            entry: tuple[m.Infra.Census.Object, t.StrSequence],
        ) -> int:
            return -len(entry[1])

        for project_report in report.projects:
            self_pkg_prefix = f"{project_report.project.replace('-', '_')}."
            for obj in project_report.objects:
                if obj.name.startswith("_"):
                    continue
                parent_paths = inventory.get(obj.name)
                if not parent_paths:
                    continue
                foreign_paths = tuple(
                    path
                    for path in parent_paths
                    if not path.startswith(self_pkg_prefix)
                )
                if not foreign_paths:
                    continue
                collisions.append((obj, foreign_paths))
        collisions.sort(key=collision_breadth)
        return tuple(collisions)


__all__: list[str] = ["FlextInfraRefactorCensusInventoryMixin"]
