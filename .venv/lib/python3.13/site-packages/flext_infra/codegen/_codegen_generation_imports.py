"""Import rendering helpers for lazy-init generation."""

from __future__ import annotations

import operator
from collections import defaultdict
from typing import TYPE_CHECKING

from flext_infra.codegen._codegen_generation_paths import (
    FlextInfraCodegenGenerationPathsMixin,
)

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraCodegenGenerationImportsMixin(FlextInfraCodegenGenerationPathsMixin):
    """Import grouping and rendering helper methods."""

    @staticmethod
    def _format_import_part(imported_name: str, export_name: str) -> str:
        """Format one imported symbol, preserving aliases only when names differ."""
        if imported_name == export_name:
            return imported_name
        return f"{imported_name} as {export_name}"

    @staticmethod
    def _format_import(indent: str, mod: str, parts: t.StrSequence) -> t.StrSequence:
        """Emit one valid import statement for canonical normalization."""
        return (f"{indent}from {mod} import {', '.join(parts)}",)

    @staticmethod
    def _format_module_alias_import(indent: str, mod: str, export_name: str) -> str:
        """Format a module alias import."""
        if mod.startswith(".") and mod != ".":
            parent_mod, _, child_name = mod.rpartition(".")
            return (
                f"{indent}from {parent_mod or '.'} import {child_name} as {export_name}"
            )
        return f"{indent}import {mod} as {export_name}"

    @staticmethod
    def _format_type_checking_module_alias_import(
        indent: str, mod: str, export_name: str
    ) -> t.StrSequence:
        """Format one explicit static reexport."""
        return (
            FlextInfraCodegenGenerationImportsMixin._format_module_alias_import(
                indent, mod, export_name
            ),
        )

    @staticmethod
    def _group_imports(
        import_map: t.LazyAliasMap,
    ) -> t.MappingKV[str, t.MutableSequenceOf[t.StrPair]]:
        """Group import map entries by module."""
        groups: dict[str, list[t.StrPair]] = defaultdict(list)
        for export_name in sorted(import_map):
            mod, attr = import_map[export_name]
            groups[mod].append((export_name, attr))
        return groups

    @staticmethod
    def _import_item_sort_key(item: t.StrPair) -> tuple[str, bool]:
        """Order an imported symbol by source name, then alias status."""
        export_name, imported_name = item
        return imported_name or export_name, export_name != imported_name

    @staticmethod
    def _generate_import_lines(
        groups: t.MappingKV[str, t.StrPairSequence], *, indent: str = ""
    ) -> t.StrSequence:
        """Generate import lines grouped by module path."""
        if not groups:
            return ()
        lines: t.MutableSequenceOf[str] = []

        def _emit_module(mod: str) -> None:
            items = groups[mod]
            alias_items = sorted(
                (item for item in items if not item[1]), key=operator.itemgetter(0)
            )
            sorted_items = sorted(
                (item for item in items if item[1]),
                key=FlextInfraCodegenGenerationImportsMixin._import_item_sort_key,
            )
            for export_name, _ in alias_items:
                lines.append(
                    FlextInfraCodegenGenerationImportsMixin._format_module_alias_import(
                        indent, mod, export_name
                    )
                )
            if not sorted_items:
                return
            parts: t.StrSequence = [
                FlextInfraCodegenGenerationImportsMixin._format_import_part(
                    attr_name, export_name
                )
                for export_name, attr_name in sorted_items
            ]
            lines.extend(
                FlextInfraCodegenGenerationImportsMixin._format_import(
                    indent, mod, parts
                )
            )

        for mod in sorted(groups, key=str.lower):
            _emit_module(mod)
        return lines


__all__: list[str] = ["FlextInfraCodegenGenerationImportsMixin"]
