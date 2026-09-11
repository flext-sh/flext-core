"""Guard 1 — ROPE-backed import cycle detector.

Builds a module import graph using rope's semantic import resolution
(which correctly handles ``if TYPE_CHECKING:`` blocks, ``__init__.py``
re-exports, and relative imports) and runs Tarjan's SCC algorithm.
Any strongly-connected component with more than one module is a
runtime-import cycle and blocks the validate gate.

Mandate: This is a flext-infra detector and is 100% ROPE-based per the
ROPE/beartype mandate; raw ``ast`` / ``libcst`` source analysis is
forbidden in flext-infra detectors.

Architecture: flext-infra validate layer — consumes
``u.Infra`` Rope boundary helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

from flext_core import r
from flext_infra import c, m, u
from flext_infra.base import s

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from flext_infra import p, t


class FlextInfraValidateImportCycles(s[bool]):
    """Detects runtime import cycles using rope's semantic import resolver.

    Guard 1 of the circular-import defense-in-depth suite. Unlike a raw
    AST walker, rope correctly excludes ``if TYPE_CHECKING:`` imports
    from the runtime graph, so this validator only flags imports that
    actually execute at module load time.
    """

    _MIN_CYCLE_SIZE: ClassVar[int] = 2

    def build_report(self, repository_root: Path) -> p.Result[m.Infra.ValidationReport]:
        """Scan ``repository_root`` for runtime import cycles via rope.

        Detection is scoped per project root: each governed project is an
        independent import unit with its own ``sys.path`` at test/runtime, so
        same-named top-level packages (for example ``tests``) from different
        projects are never merged into a single graph. When no governed project
        roots are discoverable (for example a bare synthetic tree), the whole
        root is scanned as one unit.

        Args:
            repository_root: Path under which to discover and scan projects.

        Returns:
            r with ValidationReport listing each cyclic SCC as a violation.

        """
        try:
            graphs = self._build_graphs(repository_root)
        except OSError as exc:
            return r[m.Infra.ValidationReport].fail_op("import-cycles scan", exc)
        total_modules = 0
        cycles: list[tuple[str, t.StrSequence]] = []
        for label, graph in graphs:
            total_modules += len(graph)
            cycles.extend(
                (label, tuple(sorted(scc)))
                for scc in u.Infra.strongly_connected_components(graph)
                if len(scc) >= self._MIN_CYCLE_SIZE
            )
        violations: t.MutableSequenceOf[str] = [
            (f"[{label}] cycle: " + " -> ".join(scc))
            if label
            else "cycle: " + " -> ".join(scc)
            for label, scc in cycles
        ]
        passed = not violations
        summary = (
            f"no runtime import cycles (scanned {total_modules} modules)"
            if passed
            else f"found {len(cycles)} runtime import cycle(s) across {total_modules} modules"
        )
        return r[m.Infra.ValidationReport].ok(
            m.Infra.ValidationReport(
                passed=passed, violations=violations, summary=summary
            )
        )

    def _build_graphs(
        self, repository_root: Path
    ) -> list[tuple[str, MutableMapping[str, set[str]]]]:
        """Build one import graph per governed project root (one import unit).

        Falls back to a single graph over ``repository_root`` when no governed
        project roots are discoverable, preserving behaviour for bare synthetic
        trees used by unit tests.
        """
        roots = u.Infra.discover_project_roots(repository_root)
        if not roots:
            return [("", self._build_graph(repository_root))]
        return [(root.name, self._build_graph(root)) for root in roots]

    def _build_graph(self, repository_root: Path) -> MutableMapping[str, set[str]]:
        """Build ``{module_name: {imported_modules}}`` via rope."""
        graph: MutableMapping[str, set[str]] = {}
        with u.Infra.open_project(repository_root) as project:
            for resource in u.Infra.python_resources(project):
                module_name_result = self._module_name_for(project, resource)
                if module_name_result.failure:
                    continue
                module_name = module_name_result.value
                graph.setdefault(module_name, set())
                module_imports = u.Infra.get_module_imports(project, resource)
                if module_imports is None:
                    continue
                for imported_name in self._iter_imported_modules(module_imports):
                    graph[module_name].add(imported_name)
        return graph

    def _module_name_for(
        self, project: t.Infra.RopeProject, resource: t.Infra.RopeResource
    ) -> p.Result[str]:
        """Resolve a rope resource to its fully-qualified module name.

        ``r.ok(name)`` when rope identifies the module or its on-disk
        path can be projected onto a dotted name. ``r.fail(reason)`` for
        rope runtime/type errors or paths that do not yield a name.
        """
        try:
            pymodule = u.Infra.get_pymodule(project, resource)
        except u.Infra.rope_runtime_errors() as exc:
            return r[str].fail(f"get_pymodule rope error: {exc!s}")
        except TypeError as exc:
            return r[str].fail(f"get_pymodule type error: {exc!s}")
        try:
            name = pymodule.get_name()
        except c.EXC_ATTR_TYPE:
            name = None
        if isinstance(name, str) and name:
            return r[str].ok(name)
        resource_path = resource.path
        parts = Path(resource_path).with_suffix("").parts
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts:
            return r[str].fail(f"could not derive module name for {resource_path}")
        return r[str].ok(".".join(parts))

    def _iter_imported_modules(
        self, module_imports: t.Infra.RopeModuleImports
    ) -> t.StrSequence:
        """Extract imported module names from the boundary import-info collection.

        Rope's parser already excludes ``if TYPE_CHECKING:`` blocks and
        resolves relative imports to absolute module names.
        """
        return u.Infra.imported_module_paths(module_imports)

    @override
    def execute(self) -> p.Result[bool]:
        """Execute the cycle-detection CLI flow using ``self.repository_root``."""
        report_result = self.build_report(self.repository_root)
        if report_result.failure:
            return r[bool].fail(
                report_result.error or "import-cycles validation failed"
            )
        report = report_result.unwrap()
        return r[bool].ok(True) if report.passed else r[bool].fail(report.summary)


__all__: t.StrSequence = ("FlextInfraValidateImportCycles",)
