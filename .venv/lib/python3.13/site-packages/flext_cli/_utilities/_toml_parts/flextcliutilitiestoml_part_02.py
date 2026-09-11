"""Generic TOML helpers shared through ``u.Cli.toml_*``."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeIs

import tomlkit
from tomlkit.container import OutOfOrderTableProxy
from tomlkit.items import AoT, Item, Table
from tomlkit.toml_document import TOMLDocument

if TYPE_CHECKING:
    from flext_cli import t


class FlextCliUtilitiesToml:
    """Implementation part for FlextCliUtilitiesToml."""

    @staticmethod
    def toml_item_from_json_value(value: t.JsonValue) -> Item | t.JsonValue:
        """Convert one JSON-compatible value into one TOML runtime value."""
        if value is None:
            msg = "TOML does not support null values"
            raise TypeError(msg)
        if isinstance(value, bool | int | float | str):
            return tomlkit.item(value)
        if isinstance(value, list):
            return tomlkit.item(value)
        return tomlkit.item(value)

    @staticmethod
    def toml_is_document(value: t.Cli.TomlRuntimeSource) -> TypeIs[TOMLDocument]:
        """Return True when the value is a TOML document."""
        return isinstance(value, TOMLDocument)

    @staticmethod
    def toml_is_table(value: t.Cli.TomlRuntimeSource) -> TypeIs[Table]:
        """Return True when the value is an explicit TOML table."""
        return isinstance(value, Table)

    @staticmethod
    def _toml_consolidate_proxy(proxy: OutOfOrderTableProxy) -> Table:
        """Materialize a fragmented out-of-order table as one explicit table.

        A ``[section]`` split across the document by an intervening top-level
        table is valid TOML; tomlkit exposes it as an ``OutOfOrderTableProxy``.
        Copying its entries into a single ``Table`` gives callers a normal,
        fully readable table without altering the source document.
        """
        table = tomlkit.table()
        for entry_key in list(proxy):
            table[entry_key] = proxy[entry_key]
        return table

    @staticmethod
    def toml_is_item(value: t.Cli.TomlRuntimeSource) -> TypeIs[Item]:
        """Return True when the value is a TOML item."""
        return isinstance(value, Item)

    @staticmethod
    def toml_is_aot(value: t.Cli.TomlRuntimeSource) -> TypeIs[AoT]:
        """Return True when the value is a TOML array-of-tables."""
        return isinstance(value, AoT)

    @staticmethod
    def toml_table_child(container: TOMLDocument | Table, key: str) -> Table | None:
        """Return a table child from a TOML container.

        A fragmented (out-of-order) child is consolidated into one explicit
        table so callers always receive a normal ``Table``, regardless of the
        physical section order in the source document.
        """
        if key not in container:
            return None
        value = container[key]
        if isinstance(value, OutOfOrderTableProxy):
            return FlextCliUtilitiesToml._toml_consolidate_proxy(value)
        return value if isinstance(value, Table) else None

    @staticmethod
    def toml_item_child(container: TOMLDocument | Table, key: str) -> Item | None:
        """Return a raw TOML item from a container."""
        if key not in container:
            return None
        value = container[key]
        return value if FlextCliUtilitiesToml.toml_is_item(value) else None

    @staticmethod
    def toml_ensure_table(parent: TOMLDocument | Table, key: str) -> Table:
        """Return an explicit table child, promoting implicit super-tables when needed."""
        existing: t.Cli.TomlRuntimeSource | None = None
        if key in parent:
            existing = parent[key]
        if isinstance(existing, OutOfOrderTableProxy):
            # A fragmented (out-of-order) table carries real entries spread
            # across the document. Consolidate them into one explicit table so
            # subsequent mutation targets a single contiguous section instead of
            # silently overwriting the fragments with an empty table.
            table = tomlkit.table()
            for entry_key in list(existing):
                table[entry_key] = existing[entry_key]
            del parent[key]
            parent[key] = table
            return table
        if isinstance(existing, Table):
            table = existing
            if not table.is_super_table():
                return table
            del parent[key]
            table = tomlkit.table()
            for entry_key in list(existing):
                table[entry_key] = existing[entry_key]
            parent[key] = table
            return table
        table = tomlkit.table()
        parent[key] = table
        return table

    @staticmethod
    def toml_ensure_path(parent: TOMLDocument | Table, path: t.StrSequence) -> Table:
        """Return a nested table path, creating intermediate tables as needed."""
        current: TOMLDocument | Table = parent
        for segment in path:
            current = FlextCliUtilitiesToml.toml_ensure_table(current, segment)
        if FlextCliUtilitiesToml.toml_is_table(current):
            return current
        msg = "toml_ensure_path must return a TOML table"
        raise TypeError(msg)


__all__: list[str] = ["FlextCliUtilitiesToml"]
