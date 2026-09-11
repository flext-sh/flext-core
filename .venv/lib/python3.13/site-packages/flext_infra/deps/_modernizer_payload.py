"""Pyproject payload normalization + TOML key reordering — extracted concern."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import c, u

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from flext_infra import t


class FlextInfraPyprojectModernizerPayloadMixin:
    """Build-system/poetry-group normalization + deterministic TOML key ordering.

    Composed into FlextInfraPyprojectModernizer via inheritance; self-contained
    (``cls``/``self``-internal recursion only, operates on the passed payload/doc).
    """

    def _ensure_build_system_payload(
        self, payload: t.MutableJsonMapping
    ) -> t.StrSequence:
        """Ensure canonical build-system backend/requirements in one plain payload."""
        changes: t.MutableSequenceOf[str] = []
        build_system_existing = payload.get("build-system", None)
        build_system = u.Cli.toml_mapping_ensure_table(payload, "build-system")
        if not isinstance(build_system_existing, dict):
            changes.append("created [build-system]")
        expected_backend = "hatchling.build"
        current_backend = u.norm_str(str(build_system.get("build-backend", "")))
        if current_backend != expected_backend:
            build_system["build-backend"] = expected_backend
            changes.append("build-system.build-backend set to hatchling.build")
        expected_requires = ["hatchling"]
        requires_value = build_system.get("requires", None)
        current_requires = sorted(
            str(item) for item in u.Cli.json_as_sequence(requires_value)
        )
        if current_requires != expected_requires:
            requires_list: t.JsonValueList = list(expected_requires)
            build_system["requires"] = requires_list
            changes.append("build-system.requires set to ['hatchling']")
        metadata_table = u.Cli.toml_mapping_ensure_path(
            payload, (c.Infra.TOOL, "hatch", "metadata")
        )
        current_allow = metadata_table.get("allow-direct-references", None) is True
        if not current_allow:
            metadata_table["allow-direct-references"] = True
            changes.append("tool.hatch.metadata.allow-direct-references set to true")
        return changes

    def _remove_empty_poetry_groups_payload(
        self, payload: t.MutableJsonMapping
    ) -> t.StrSequence:
        """Remove empty Poetry group tables from one normalized payload."""
        poetry_groups = u.Cli.toml_mapping_path(
            payload, (c.Infra.TOOL, c.Infra.POETRY, c.Infra.GROUP)
        )
        if poetry_groups is None:
            return []
        empty_groups: t.MutableSequenceOf[str] = []
        for name, group_value in poetry_groups.items():
            group_table = (
                group_value
                if isinstance(group_value, dict)
                else u.Cli.toml_mapping_child(poetry_groups, name)
            )
            if group_table is None:
                continue
            deps_item = u.Cli.toml_mapping_child(group_table, c.Infra.DEPENDENCIES)
            if deps_item is not None and not deps_item:
                empty_groups.append(name)
        changes: t.MutableSequenceOf[str] = []
        for name in empty_groups:
            del poetry_groups[name]
            changes.append(f"removed empty poetry group '{name}'")
        if poetry_groups:
            return changes
        poetry_table = u.Cli.toml_mapping_path(payload, (c.Infra.TOOL, c.Infra.POETRY))
        if poetry_table is None or c.Infra.GROUP not in poetry_table:
            return changes
        del poetry_table[c.Infra.GROUP]
        changes.append("removed empty poetry group container")
        return changes

    def _ordered_keys(
        self, keys: t.StrSequence, *, preferred_first: t.StrSequence | None = None
    ) -> t.StrSequence:
        """Return keys with optional preferred-first order then alphabetical."""
        preferred = list(preferred_first or [])
        key_set = set(keys)
        ordered: t.MutableSequenceOf[str] = [key for key in preferred if key in key_set]
        remaining = sorted(key for key in keys if key not in set(ordered))
        ordered.extend(remaining)
        return ordered

    def _recurse_into_item(
        self, item: t.Cli.TomlContainer | t.Cli.TomlItem, table_key: str
    ) -> None:
        """Reorder children of one TOML node; Table/AoT only, no-op otherwise."""
        if u.Cli.toml_is_table(item):
            self._reorder_table_inplace(item, table_key=table_key)
        elif u.Cli.toml_is_aot(item):
            for entry in item.body:
                self._reorder_table_inplace(entry, table_key=table_key)

    def _reorder_table_inplace(
        self,
        table: t.Cli.TomlTable,
        *,
        preferred_first: t.StrSequence | None = None,
        table_key: str | None = None,
    ) -> None:
        """Reorder table keys in-place recursively (tables/AoT items)."""
        if table_key == "per-file-ignores":
            return
        original_keys = [str(key) for key in table]
        ordered_keys = self._ordered_keys(
            original_keys, preferred_first=preferred_first
        )
        if ordered_keys == original_keys:
            for key in ordered_keys:
                self._recurse_into_item(table[key], key)
            return
        items: MutableMapping[str, t.Cli.TomlItem] = {
            key: table[key] for key in original_keys
        }
        for key in original_keys:
            del table[key]
        for key in ordered_keys:
            item_value: t.Cli.TomlItem = items[key]
            self._recurse_into_item(item_value, key)
            table[key] = item_value

    def _reorder_document_inplace(
        self, doc: t.Cli.TomlDocument, *, preferred_first: t.StrSequence | None = None
    ) -> None:
        """Apply deterministic ordering for top-level groups and nested tables."""
        root_keys = [str(key) for key in doc]
        ordered_root = self._ordered_keys(root_keys, preferred_first=preferred_first)
        if ordered_root != root_keys:
            root_items: dict[str, t.Cli.TomlItem | t.Cli.TomlContainer] = {
                key: doc[key] for key in root_keys
            }
            for key in root_keys:
                del doc[key]
            for key in ordered_root:
                doc[key] = root_items[key]
        tool_child = u.Cli.toml_table_child(doc, "tool")
        if tool_child is not None:
            self._reorder_table_inplace(tool_child, table_key="tool")
        for key in ordered_root:
            if key == "tool":
                continue
            self._recurse_into_item(doc[key], key)


__all__: list[str] = ["FlextInfraPyprojectModernizerPayloadMixin"]
