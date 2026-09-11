"""Phase: Consolidate optional-dependencies and Poetry groups into single dev group."""

from __future__ import annotations

from flext_infra import c, t, u


class FlextInfraConsolidateGroupsPhase:
    """Consolidate optional-dependencies and Poetry groups into single dev group."""

    def apply(
        self, doc: t.Cli.TomlDocument, canonical_dev: t.StrSequence
    ) -> t.StrSequence:
        """Merge all legacy optional groups into canonical ``project.optional-dependencies.dev``."""
        changes: t.MutableSequenceOf[str] = []
        project = u.Cli.toml_ensure_table(doc, c.Infra.PROJECT)
        optional = u.Cli.toml_ensure_table(project, c.Infra.OPTIONAL_DEPENDENCIES)
        existing = u.Infra.project_dev_groups(doc)
        merged_dev = u.Infra.dedupe_specs([
            *canonical_dev,
            *[
                requirement
                for group in c.Infra.CANONICAL_DEV_DEPENDENCY_GROUPS
                for requirement in existing.get(str(group), ())
            ],
        ])
        current_dev = [
            str(item)
            for item in u.Cli.json_as_sequence(u.Cli.toml_value(optional, c.Infra.DEV))
        ]
        if sorted(current_dev) != sorted(merged_dev):
            optional[c.Infra.DEV] = u.Cli.toml_array(sorted(merged_dev))
            changes.append("project.optional-dependencies.dev consolidated")
        for old_key in c.Infra.LEGACY_DEV_DEPENDENCY_GROUPS:
            if old_key in optional:
                del optional[old_key]
                changes.append(f"project.optional-dependencies.{old_key} removed")
        tool = u.Cli.toml_ensure_table(doc, c.Infra.TOOL)
        poetry = u.Cli.toml_ensure_table(tool, c.Infra.POETRY)
        poetry_group = u.Cli.toml_ensure_table(poetry, c.Infra.GROUP)
        poetry_dev_table: t.Cli.TomlTable | None = None
        for old_group in c.Infra.LEGACY_DEV_DEPENDENCY_GROUPS:
            if old_group not in poetry_group:
                continue
            old_group_table = poetry_group[old_group]
            if not u.Cli.toml_is_table(old_group_table):
                continue
            old_deps = old_group_table.get(c.Infra.DEPENDENCIES)
            if u.Cli.toml_is_table(old_deps):
                if poetry_dev_table is None:
                    poetry_dev_table = u.Cli.toml_ensure_table(
                        u.Cli.toml_ensure_table(poetry_group, c.Infra.DEV),
                        c.Infra.DEPENDENCIES,
                    )
                current_dev_table: t.Cli.TomlTable = poetry_dev_table
                for dep_name_raw in old_deps:
                    dep_name = dep_name_raw
                    dep_value = old_deps[dep_name_raw]
                    if dep_name not in current_dev_table:
                        current_dev_table[dep_name] = dep_value
            del poetry_group[old_group]
            changes.append(f"tool.poetry.group.{old_group} removed")
        deptry = u.Cli.toml_ensure_table(tool, c.Infra.DEPTRY)
        current_groups = [
            str(item)
            for item in u.Cli.json_as_sequence(
                u.Cli.toml_value(deptry, "pep621_dev_dependency_groups")
            )
        ]
        if current_groups != [c.Infra.DEV]:
            deptry["pep621_dev_dependency_groups"] = u.Cli.toml_array([c.Infra.DEV])
            changes.append("tool.deptry.pep621_dev_dependency_groups set to ['dev']")
        return changes

    def apply_payload(
        self, payload: t.MutableJsonMapping, canonical_dev: t.StrSequence
    ) -> t.StrSequence:
        """Merge legacy groups into one canonical dev group in one plain payload."""
        changes: t.MutableSequenceOf[str] = []
        project = u.Cli.toml_mapping_ensure_table(payload, c.Infra.PROJECT)
        optional = u.Cli.toml_mapping_ensure_table(
            project, c.Infra.OPTIONAL_DEPENDENCIES
        )
        existing = u.Infra.project_dev_groups_from_payload(payload)
        merged_dev = u.Infra.dedupe_specs([
            *canonical_dev,
            *[
                requirement
                for group in c.Infra.CANONICAL_DEV_DEPENDENCY_GROUPS
                for requirement in existing.get(str(group), ())
            ],
        ])
        if u.Cli.toml_mapping_sync_string_list(
            optional, c.Infra.DEV, sorted(merged_dev)
        ):
            changes.append("project.optional-dependencies.dev consolidated")
        for old_key in c.Infra.LEGACY_DEV_DEPENDENCY_GROUPS:
            if u.Cli.toml_mapping_remove_key_if_present(optional, old_key):
                changes.append(f"project.optional-dependencies.{old_key} removed")
        poetry_group = u.Cli.toml_mapping_path(
            payload, (c.Infra.TOOL, c.Infra.POETRY, c.Infra.GROUP)
        )
        poetry_dev_table: t.MutableJsonMapping | None = None
        for old_group in c.Infra.LEGACY_DEV_DEPENDENCY_GROUPS:
            if poetry_group is None:
                continue
            old_group_table = u.Cli.toml_mapping_child(poetry_group, old_group)
            old_deps = (
                u.Cli.toml_mapping_child(old_group_table, c.Infra.DEPENDENCIES)
                if old_group_table is not None
                else None
            )
            if old_deps is not None:
                if poetry_dev_table is None:
                    current_poetry_dev_table = u.Cli.toml_mapping_ensure_path(
                        payload,
                        (
                            c.Infra.TOOL,
                            c.Infra.POETRY,
                            c.Infra.GROUP,
                            c.Infra.DEV,
                            c.Infra.DEPENDENCIES,
                        ),
                    )
                    poetry_dev_table = current_poetry_dev_table
                else:
                    current_poetry_dev_table = poetry_dev_table
                for dep_name, dep_value in old_deps.items():
                    if dep_name not in current_poetry_dev_table:
                        current_poetry_dev_table[dep_name] = dep_value
            if u.Cli.toml_mapping_remove_key_if_present(poetry_group, old_group):
                changes.append(f"tool.poetry.group.{old_group} removed")
        deptry = u.Cli.toml_mapping_ensure_path(payload, (c.Infra.TOOL, c.Infra.DEPTRY))
        if u.Cli.toml_mapping_sync_string_list(
            deptry, "pep621_dev_dependency_groups", [c.Infra.DEV]
        ):
            changes.append("tool.deptry.pep621_dev_dependency_groups set to ['dev']")
        return changes


__all__: list[str] = ["FlextInfraConsolidateGroupsPhase"]
