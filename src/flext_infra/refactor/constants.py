"""Constants namespace for flext_infra.refactor."""

from __future__ import annotations

from enum import Enum, auto
from typing import ClassVar


class FlextInfraRefactorConstants:
    """Shared constants for refactor engine modules."""

    RUNTIME_ALIAS_NAMES: ClassVar[frozenset[str]] = frozenset(
        {
            "c",
            "m",
            "r",
            "t",
            "u",
            "p",
            "d",
            "e",
            "h",
            "s",
            "x",
        }
    )

    LEGACY_FIX_ACTIONS: ClassVar[frozenset[str]] = frozenset(
        {
            "remove",
            "inline_and_remove",
            "remove_and_update_refs",
            "keep_try_only",
        }
    )

    IMPORT_FIX_ACTIONS: ClassVar[frozenset[str]] = frozenset(
        {
            "replace_with_alias",
            "hoist_to_module_top",
        }
    )

    CLASS_FIX_ACTIONS: ClassVar[frozenset[str]] = frozenset(
        {
            "reorder_methods",
        }
    )

    MRO_FIX_ACTIONS: ClassVar[frozenset[str]] = frozenset(
        {
            "remove_inheritance_keep_class",
            "fix_mro_redeclaration",
        }
    )

    PROPAGATION_FIX_ACTIONS: ClassVar[frozenset[str]] = frozenset(
        {
            "propagate_symbol_renames",
            "rename_imported_symbols",
            "propagate_signature_migrations",
        }
    )

    PATTERN_FIX_ACTIONS: ClassVar[frozenset[str]] = frozenset(
        {
            "convert_dict_to_mapping_annotations",
            "remove_redundant_casts",
        }
    )

    FUTURE_FIX_ACTIONS: ClassVar[frozenset[str]] = frozenset(
        {
            "ensure_future_annotations",
        }
    )

    FUTURE_CHECKS: ClassVar[frozenset[str]] = frozenset(
        {
            "missing_future_import",
        }
    )

    class MethodCategory(Enum):
        """Categorias de metodos para ordenacao."""

        MAGIC = auto()
        PROPERTY = auto()
        STATIC = auto()
        CLASS = auto()
        PUBLIC = auto()
        PROTECTED = auto()
        PRIVATE = auto()


__all__ = ["FlextInfraRefactorConstants"]
