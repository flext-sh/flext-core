"""Direct constants consolidation command service."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, override

from flext_core import r
from flext_infra import c, m, p, t, u
from flext_infra.base import s
from flext_infra.codegen._consolidator_steps import (
    FlextInfraCodegenConsolidatorStepsMixin,
)
from flext_infra.workspace.rope import FlextInfraRopeWorkspace


class FlextInfraCodegenConsolidator(s[str], FlextInfraCodegenConsolidatorStepsMixin):
    """Consolidate inline constants into canonical ``c.*`` references."""

    project_name: Annotated[
        str | None,
        m.Field(alias="project", description="Single project to consolidate"),
    ] = None

    @override
    def execute(self) -> p.Result[str]:
        """Execute constants consolidation with normalized command context."""
        output_lines: t.MutableSequenceOf[str] = (
            ["[DRY-RUN] Scanning...\n"] if self.dry_run else []
        )
        found = applied = failed = 0
        file_results: t.MutableSequenceOf[m.Infra.ConsolidatorFileResult] = []

        with FlextInfraRopeWorkspace.open_workspace(self.repository_root) as rope:
            projects_result = self._selected_projects(rope)
            if projects_result.failure:
                return r[str].fail("Failed to discover projects")
            selected_projects = projects_result.unwrap()

            for project in selected_projects:
                project_layout = u.Infra.layout(project.path)
                if project_layout is None or not project_layout.init_path.is_file():
                    continue

                constants_file = project_layout.package_dir / c.Infra.CONSTANTS_PY
                value_map_result = self._build_value_map_from_constants_file(
                    constants_file
                )
                if value_map_result.failure:
                    return r[str].fail(
                        value_map_result.error or "constants file read failed"
                    )
                value_map = value_map_result.value
                if not value_map:
                    continue

                project_files = self._project_python_files(rope, project.path)
                if project_files.failure:
                    return r[str].fail(
                        project_files.error or "project python file discovery failed"
                    )
                for python_file in project_files.value:
                    scanned = self._scan_file(rope.rope_project, python_file, value_map)
                    if scanned is None:
                        continue
                    resource, source, matches = scanned
                    found += len(matches)
                    rel_path = python_file.relative_to(self.repository_root)
                    if self.dry_run:
                        output_lines.extend(
                            (
                                f"  {rel_path}:{symbol.line}  {symbol.name} = "
                                f"{value} -> {ref}"
                            )
                            for symbol, ref, value in matches
                        )
                        continue
                    ok, changes, lines = self._apply_and_validate(
                        rope.rope_project,
                        resource,
                        python_file,
                        self.repository_root,
                        project_layout.package_name,
                        source,
                        matches,
                    )
                    output_lines.extend(lines)
                    file_results.append(
                        m.Infra.ConsolidatorFileResult(
                            file=str(rel_path),
                            status="applied" if ok else "reverted",
                            changes=tuple(changes),
                        )
                    )
                    if ok:
                        applied += len(changes)
                    else:
                        failed += 1

        summary = (
            f"Found {found} canonical matches across {len(selected_projects)} projects"
            if self.dry_run
            else f"Applied {applied} replacements, {failed} files reverted"
        )
        output_lines.extend(("", summary))
        if self.output_format == c.Cli.OutputFormats.JSON:
            report = m.Infra.ConsolidatorReport(
                total_found=found,
                total_applied=applied,
                total_failed=failed,
                files=tuple(file_results),
            )
            return r[str].ok(report.model_dump_json())
        return r[str].ok("\n".join(output_lines))

    def _project_python_files(
        self, rope_workspace: p.Infra.RopeWorkspaceDsl, project_root: Path
    ) -> p.Result[t.SequenceOf[Path]]:
        """Return indexed Python wrapper files for one consolidation pass."""
        resolved_root = project_root.resolve()
        constants_directory = c.Infra.FAMILY_DIRECTORIES["c"]
        indexed_files: t.MutableSequenceOf[Path] = []
        for module in rope_workspace.modules():
            if (
                module.project_root is None
                or module.project_root.resolve() != resolved_root
            ):
                continue
            file_path = module.file_path.resolve()
            if not file_path.is_relative_to(resolved_root):
                continue
            relative_path = file_path.relative_to(resolved_root)
            if (
                file_path.suffix != c.Infra.EXT_PYTHON
                or not relative_path.parts
                or relative_path.parts[0] not in c.Infra.ROOT_WRAPPER_SEGMENTS
                or constants_directory in relative_path.parts
            ):
                continue
            indexed_files.append(file_path)
        return r[t.SequenceOf[Path]].ok(tuple(sorted(indexed_files)))

    def _selected_projects(
        self, rope_workspace: p.Infra.RopeWorkspaceDsl
    ) -> p.Result[t.SequenceOf[p.Infra.ProjectInfo]]:
        """Return the selected projects."""
        _ = rope_workspace
        discovered = u.Infra.projects(self.repository_root)
        if discovered.failure:
            return r[t.SequenceOf[p.Infra.ProjectInfo]].fail(
                discovered.error or "project discovery failed"
            )
        selected = tuple(
            project
            for project in discovered.unwrap()
            if self.project_name is None or project.name == self.project_name
        )
        return r[t.SequenceOf[p.Infra.ProjectInfo]].ok(selected)


__all__: list[str] = ["FlextInfraCodegenConsolidator"]
