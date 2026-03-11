"""Phase: Consolidate optional-dependencies and Poetry groups into single dev group."""

from __future__ import annotations

import tomlkit
from tomlkit.container import Container
from tomlkit.items import Item, Table

from flext_infra import c, u


class ConsolidateGroupsPhase:
    """Consolidate optional-dependencies and Poetry groups into single dev group."""

    def apply(self, doc: tomlkit.TOMLDocument, canonical_dev: list[str]) -> list[str]:
        changes: list[str] = []
        project: Item | Container | None = None
        if c.Infra.Toml.PROJECT in doc:
            project = doc[c.Infra.Toml.PROJECT]
        if not isinstance(project, Table):
            project = tomlkit.table()
            doc[c.Infra.Toml.PROJECT] = project

        optional: Item | Container | None = None
        if c.Infra.Toml.OPTIONAL_DEPENDENCIES in project:
            optional = project[c.Infra.Toml.OPTIONAL_DEPENDENCIES]
        if not isinstance(optional, Table):
            optional = tomlkit.table()
            project[c.Infra.Toml.OPTIONAL_DEPENDENCIES] = optional
        existing = u.Infra.project_dev_groups(doc)
        merged_dev = u.Infra.dedupe_specs([
            *canonical_dev,
            *existing.get(c.Infra.Toml.DEV, []),
            *existing.get(c.Infra.Directories.DOCS, []),
            *existing.get(c.Infra.Gates.SECURITY, []),
            *existing.get(c.Infra.Toml.TEST, []),
            *existing.get(c.Infra.Directories.TYPINGS, []),
        ])
        current_dev = u.Infra.as_string_list(u.Infra.get(optional, c.Infra.Toml.DEV))
        if sorted(current_dev) != sorted(merged_dev):
            optional[c.Infra.Toml.DEV] = u.Infra.array(sorted(merged_dev))
            changes.append("project.optional-dependencies.dev consolidated")
        for old_key in (
            c.Infra.Toml.DOCS,
            c.Infra.Toml.SECURITY,
            c.Infra.Toml.TEST,
            c.Infra.Directories.TYPINGS,
        ):
            if old_key in optional:
                del optional[old_key]
                changes.append(f"project.optional-dependencies.{old_key} removed")
        tool: Item | Container | None = None
        if c.Infra.Toml.TOOL in doc:
            tool = doc[c.Infra.Toml.TOOL]
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool
        poetry = u.Infra.ensure_table(tool, c.Infra.Toml.POETRY)
        poetry_group_raw: Item | Container | None = None
        if c.Infra.Toml.GROUP in poetry:
            poetry_group_raw = poetry[c.Infra.Toml.GROUP]
        poetry_group = poetry_group_raw if isinstance(poetry_group_raw, Table) else None
        poetry_dev_table: Table | None = None
        for old_group in (
            c.Infra.Toml.DOCS,
            c.Infra.Toml.SECURITY,
            c.Infra.Toml.TEST,
            c.Infra.Directories.TYPINGS,
        ):
            if poetry_group is None:
                continue
            old_group_table: Item | Container | None = None
            if old_group in poetry_group:
                old_group_table = poetry_group[old_group]
            if not isinstance(old_group_table, Table):
                continue
            old_deps: Item | Container | None = None
            if c.Infra.Toml.DEPENDENCIES in old_group_table:
                old_deps = old_group_table[c.Infra.Toml.DEPENDENCIES]
            if isinstance(old_deps, Table):
                if poetry_dev_table is None:
                    poetry_dev_table = u.Infra.ensure_table(
                        u.Infra.ensure_table(poetry_group, c.Infra.Toml.DEV),
                        c.Infra.Toml.DEPENDENCIES,
                    )
                for dep_name_raw in old_deps:
                    dep_name = dep_name_raw
                    dep_value = old_deps[dep_name_raw]
                    if dep_name not in poetry_dev_table:
                        poetry_dev_table[dep_name] = dep_value
            del poetry_group[old_group]
            changes.append(f"tool.poetry.group.{old_group} removed")
        deptry = u.Infra.ensure_table(tool, c.Infra.Toml.DEPTRY)
        current_groups = u.Infra.as_string_list(
            u.Infra.get(deptry, "pep621_dev_dependency_groups"),
        )
        if current_groups != [c.Infra.Toml.DEV]:
            deptry["pep621_dev_dependency_groups"] = u.Infra.array([c.Infra.Toml.DEV])
            changes.append("tool.deptry.pep621_dev_dependency_groups set to ['dev']")
        return changes
