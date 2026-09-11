"""Detect cyclic import dependencies in projects via rope.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flext_infra import c, m, u

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraCyclicImportDetector:
    """Detect cyclic imports at project level via rope semantic import resolution."""

    @staticmethod
    def scan_project(
        *,
        project_root: Path,
        rope_project: t.Infra.RopeProject,
        proposed_sources: t.MappingKV[Path, str] | None = None,
        _parse_failures: t.SequenceOf[m.Infra.ParseFailureViolation] | None = None,
    ) -> t.SequenceOf[m.Infra.CyclicImportViolation]:
        """Build the current or prospective import graph and detect cycles."""
        del _parse_failures
        source_updates = {
            path.resolve(): source for path, source in (proposed_sources or {}).items()
        }
        scan_dirs = [
            (project_root / d).resolve()
            for d in u.Infra.namespace_scan_dirs(project_root)
            if (project_root / d).is_dir()
        ]
        if not scan_dirs:
            return []

        module_resources: list[tuple[str, str, t.Infra.RopeResource]] = []
        for resource in rope_project.get_python_files():
            real_path = Path(resource.real_path).resolve()
            if not any(real_path.is_relative_to(scan_dir) for scan_dir in scan_dirs):
                continue
            try:
                module_name = u.Infra.get_pymodule(rope_project, resource).get_name()
            except (
                *u.Infra.rope_runtime_errors(),
                *u.Infra.rope_syntax_errors(),
                TypeError,
            ):
                continue
            if module_name:
                module_resources.append((module_name, str(real_path), resource))

        file_map: t.MutableStrMapping = {
            module_name: file_path for module_name, file_path, _ in module_resources
        }
        graph: dict[str, t.Infra.StrSet] = {module: set() for module in file_map}
        for module_name, file_path, resource in module_resources:
            resolved_file = Path(file_path).resolve()
            semantic_targets = (
                FlextInfraCyclicImportDetector._prospective_import_targets(
                    rope_project, resource, source_updates[resolved_file]
                )
                if resolved_file in source_updates
                else tuple(
                    u.Infra.get_semantic_module_imports(rope_project, resource).values()
                )
            )
            for semantic_target in semantic_targets:
                target_module = semantic_target
                while target_module not in file_map and "." in target_module:
                    target_module = target_module.rsplit(".", maxsplit=1)[0]
                if target_module in file_map:
                    graph[module_name].add(target_module)

        return tuple(
            m.Infra.CyclicImportViolation(
                cycle=tuple(sorted(component)),
                files=tuple(file_map.get(node, node) for node in sorted(component)),
            )
            for component in u.Infra.strongly_connected_components(graph)
            if len(component) > 1
            or (len(component) == 1 and component[0] in graph[component[0]])
        )

    @staticmethod
    def _prospective_import_targets(
        rope_project: t.Infra.RopeProject, resource: t.Infra.RopeResource, source: str
    ) -> t.StrSequence:
        """Return Rope-resolved import targets for one proposed source."""
        pymodule = u.Infra.get_string_module(rope_project, source, resource=resource)
        module_imports = u.Infra.module_imports_for_pymodule(rope_project, pymodule)
        module_name = pymodule.get_name()
        current_package = (
            module_name
            if resource.path == c.Infra.INIT_PY
            or resource.path.endswith(f"/{c.Infra.INIT_PY}")
            else module_name.rsplit(".", maxsplit=1)[0]
            if "." in module_name
            else ""
        )
        return u.Infra.imported_module_paths(
            module_imports, current_package=current_package
        )


__all__: list[str] = ["FlextInfraCyclicImportDetector"]
