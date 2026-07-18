"""Type and module introspection helpers — annotation inspection + bytecode analysis."""

from __future__ import annotations

import functools
import importlib
import sys
from typing import Any, TypeAliasType, get_args, get_origin

import typing_extensions

# pydantic ships JsonValue as a typing_extensions.TypeAliasType, which is a
# distinct class from typing.TypeAliasType; unwrap must recognize both variants
# so a nested alias (e.g. flext JsonValue -> pydantic JsonValue) fully resolves.
_TYPE_ALIAS_TYPES: tuple[type, ...] = (TypeAliasType, typing_extensions.TypeAliasType)

from flext_core._typings.base import FlextTypingBase as t


class FlextUtilitiesBeartypeHelpers:
    """Annotation + bytecode inspection helpers."""

    @staticmethod
    def _lazy_suffix(module_path: str) -> str:
        name = module_path.rsplit(".", 1)[-1]
        if name == "typings":
            return "Types"
        return "".join(part[:1].upper() + part[1:] for part in name.split("_") if part)

    @staticmethod
    @functools.cache
    def lazy_alias_suffixes(package_name: str) -> tuple[tuple[str, str, str], ...]:
        """Return ``(alias, module_path, suffix)`` rows from package ``_LAZY_IMPORTS``."""
        package = sys.modules.get(package_name)
        if package is None:
            try:
                package = importlib.import_module(package_name)
            except (ImportError, ModuleNotFoundError):
                return ()
        if not hasattr(package, "_LAZY_IMPORTS"):
            return ()
        lazy_module = importlib.import_module("flext_core.lazy")
        lazy_imports = lazy_module.normalize_lazy_imports(
            package.__name__, getattr(package, "_LAZY_IMPORTS")
        )
        return tuple(
            (
                alias,
                module_path,
                FlextUtilitiesBeartypeHelpers._lazy_suffix(module_path),
            )
            for alias, entry in lazy_imports.items()
            if len(alias) == 1 and alias.islower()
            for module_path in (entry if isinstance(entry, str) else entry[0],)
        )

    @staticmethod
    def runtime_alias_names(package_name: str) -> frozenset[str]:
        """Return runtime alias names derived from generated lazy exports."""
        return frozenset(
            alias
            for alias, _, _ in FlextUtilitiesBeartypeHelpers.lazy_alias_suffixes(
                package_name
            )
        )

    @staticmethod
    def facade_module_names(package_name: str) -> frozenset[str]:
        """Return local facade module names derived from generated lazy exports."""
        return frozenset(
            module_path.rsplit(".", 1)[-1]
            for _, module_path, suffix in FlextUtilitiesBeartypeHelpers.lazy_alias_suffixes(
                package_name
            )
            if module_path.split(".", 1)[0] == package_name
            and suffix in {"Constants", "Models", "Protocols", "Types", "Utilities"}
        )

    @staticmethod
    def unwrap_type_alias(
        hint: t.TypeHintSpecifier | None,
    ) -> t.TypeHintSpecifier | None:
        current = hint
        seen: set[int] = set()
        while isinstance(current, _TYPE_ALIAS_TYPES):
            current_id = id(current)
            if current_id in seen:
                return current
            seen.add(current_id)
            current = current.__value__
        return current

    @staticmethod
    def contains_any_recursive(
        hint: t.TypeHintSpecifier | None, *, seen: set[int]
    ) -> bool:
        h = FlextUtilitiesBeartypeHelpers
        hint = h.unwrap_type_alias(hint)
        if hint is None:
            return False
        hint_id = id(hint)
        if hint_id in seen:
            return False
        seen.add(hint_id)
        if hint is Any or hint is object:
            return True
        if hint is type or get_origin(hint) is type:
            return False
        return any(
            h.contains_any_recursive(child, seen=seen) for child in get_args(hint)
        )

    @staticmethod
    def has_forbidden_collection_origin(
        hint: t.TypeHintSpecifier | None, forbidden: frozenset[str]
    ) -> tuple[bool, str]:
        h = FlextUtilitiesBeartypeHelpers
        hint = h.unwrap_type_alias(hint)
        if hint is None:
            return False, ""
        origin = get_origin(hint)
        if origin is None or not hasattr(origin, "__name__"):
            return False, ""
        name: str = origin.__name__
        return (True, name) if name in forbidden else (False, "")

    @staticmethod
    def has_runtime_protocol_marker(value: type) -> bool:
        return bool(getattr(value, "_is_protocol", False))

    @staticmethod
    def has_abstract_contract(value: type) -> bool:
        return bool(getattr(value, "__abstractmethods__", None)) or any(
            getattr(base, "__name__", "") == "ABC" for base in value.__mro__
        )

    @staticmethod
    def has_nested_namespace(value: type) -> bool:
        for base in value.__mro__:
            if base is not object and any(
                isinstance(member, type) and not name.startswith("_")
                for name, member in vars(base).items()
            ):
                return True
        return False


__all__: list[str] = ["FlextUtilitiesBeartypeHelpers"]
