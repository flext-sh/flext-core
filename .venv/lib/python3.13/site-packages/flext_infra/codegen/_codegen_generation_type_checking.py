"""TYPE_CHECKING render helpers for lazy-init generation."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from flext_infra import c
from flext_infra.codegen._codegen_generation_imports import (
    FlextInfraCodegenGenerationImportsMixin,
)

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraCodegenGenerationTypeCheckingMixin(
    FlextInfraCodegenGenerationImportsMixin
):
    """TYPE_CHECKING block generation helper methods."""

    @staticmethod
    def _collapse_to_children(
        groups: t.MappingKV[str, t.StrPairSequence],
        child_packages: t.StrSequence | None,
    ) -> t.MappingKV[str, t.MutableSequenceOf[t.StrPair]]:
        """Collapse child module imports into configured child packages."""
        sorted_children: list[str] = sorted(
            set(child_packages or []), key=len, reverse=True
        )
        collapsed: dict[str, list[t.StrPair]] = defaultdict(list)
        for mod, items in groups.items():
            target = mod
            for child_package in sorted_children:
                if mod.startswith(child_package + ".") or mod == child_package:
                    target = child_package
                    break
            collapsed[target].extend(items)
        return collapsed

    @staticmethod
    def _has_flext_types(collapsed: t.MappingKV[str, t.StrPairSequence]) -> bool:
        """Return whether a collapsed import map already imports FlextTypes."""
        return any(
            export_name == "FlextTypes"
            for items in collapsed.values()
            for export_name, _ in items
        )

    @staticmethod
    def _type_checking_sort_key(mod: str) -> t.StrPair:
        """Order absolute imports before relative imports, then by module path."""
        return ("1" if mod.startswith(".") else "0", mod.lower())

    @staticmethod
    def _type_checking_sort_owner(mod: str, items: t.StrPairSequence) -> str:
        """Return the module that owns the emitted import for sorting."""
        module_basename = mod.rsplit(".", maxsplit=1)[-1]
        if (
            "." in mod
            and items
            and all(
                not attr_name and export_name == module_basename
                for export_name, attr_name in items
            )
        ):
            # flext-i6nq.10: Module aliases emit from their parent package.
            return mod.rsplit(".", maxsplit=1)[0]
        return mod

    @staticmethod
    def _should_skip_type_checking_module_export(
        mod: str, export_name: str, attr_name: str, root_name: str
    ) -> bool:
        """Return whether a symbol import is a redundant root self-import."""
        if export_name in c.Infra.ALIAS_NAMES or not export_name:
            return False
        if export_name in {"cli", "main", "infra"}:
            return False
        if export_name != export_name.lower():
            return False
        if not attr_name:
            # flext-i6nq.10: Literal __all__ requires every module alias binding.
            return False
        # A lowercase ``from mod import name`` (package-name alias like ``grpc``
        # or a module-level function like ``smell_fixer_for``) is a real symbol
        # that must stay statically visible; only skip a redundant self-import
        # from the root package itself.
        return mod == root_name and attr_name == export_name

    @staticmethod
    def _emit_type_checking_module(
        mod: str,
        items: t.StrPairSequence,
        root_name: str,
        lines: t.MutableSequenceOf[str],
    ) -> None:
        """Emit one TYPE_CHECKING module import group."""
        alias_exports: t.MutableSequenceOf[str] = []
        parts: t.MutableSequenceOf[str] = []
        module_basename = mod.rsplit(".", maxsplit=1)[-1]
        selected_items = tuple(
            item
            for item in sorted(
                items,
                key=FlextInfraCodegenGenerationTypeCheckingMixin._import_item_sort_key,
            )
            if not FlextInfraCodegenGenerationTypeCheckingMixin._should_skip_type_checking_module_export(
                mod, item[0], item[1], root_name
            )
        )
        for export_name, attr_name in selected_items:
            if not attr_name:
                if export_name == module_basename:
                    alias_exports.append(export_name)
                else:
                    parts.append(
                        FlextInfraCodegenGenerationTypeCheckingMixin._format_import_part(
                            export_name, export_name
                        )
                    )
                continue
            parts.append(
                FlextInfraCodegenGenerationTypeCheckingMixin._format_import_part(
                    attr_name, export_name
                )
            )
        for export_name in tuple(dict.fromkeys(alias_exports)):
            lines.extend(
                FlextInfraCodegenGenerationTypeCheckingMixin._format_type_checking_module_alias_import(
                    "    ", mod, export_name
                )
            )
        deduped_parts = tuple(dict.fromkeys(parts))
        if deduped_parts:
            lines.extend(
                FlextInfraCodegenGenerationTypeCheckingMixin._format_import(
                    "    ", mod, deduped_parts
                )
            )

    @staticmethod
    def generate_type_checking(
        groups: t.MappingKV[str, t.StrPairSequence],
        *,
        include_flext_types: bool = True,
        child_packages: t.StrSequence | None = None,
        local_package_root: str | None = None,
    ) -> t.StrSequence:
        """Generate a TYPE_CHECKING import block."""
        if not groups and not include_flext_types:
            return ()
        if not groups:
            return ("if TYPE_CHECKING:", "    from flext_core import FlextTypes")
        normalized_groups: dict[str, t.StrPairSequence] = {}
        for mod, items in groups.items():
            resolved = FlextInfraCodegenGenerationTypeCheckingMixin._normalize_type_checking_module_path(
                mod, local_package_root
            )
            FlextInfraCodegenGenerationTypeCheckingMixin._reject_noncanonical_type_checking_import(
                resolved, local_package_root, items
            )
            normalized_groups[resolved] = (*normalized_groups.get(resolved, ()), *items)
        collapsed = FlextInfraCodegenGenerationTypeCheckingMixin._collapse_to_children(
            normalized_groups, child_packages
        )
        root_name = "" if not local_package_root else local_package_root.split(".")[0]
        lines: t.MutableSequenceOf[str] = ["if TYPE_CHECKING:"]
        if include_flext_types and (
            not FlextInfraCodegenGenerationTypeCheckingMixin._has_flext_types(collapsed)
        ):
            lines.append("    from flext_core import FlextTypes")

        def type_checking_module_key(mod: str) -> t.StrPair:
            owner = (
                FlextInfraCodegenGenerationTypeCheckingMixin._type_checking_sort_owner(
                    mod, collapsed[mod]
                )
            )
            return FlextInfraCodegenGenerationTypeCheckingMixin._type_checking_sort_key(
                owner
            )

        sorted_mods = sorted(collapsed, key=type_checking_module_key)
        previous_is_relative: bool | None = False if include_flext_types else None
        for mod in sorted_mods:
            is_relative = mod.startswith(".")
            if previous_is_relative is False and is_relative:
                lines.append("")
            FlextInfraCodegenGenerationTypeCheckingMixin._emit_type_checking_module(
                mod, collapsed[mod], root_name, lines
            )
            previous_is_relative = is_relative
        return () if len(lines) == 1 else lines


__all__: list[str] = ["FlextInfraCodegenGenerationTypeCheckingMixin"]
