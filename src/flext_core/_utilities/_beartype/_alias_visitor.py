"""Alias rebind / compatibility / foreign-canonical alias visitors."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flext_core._constants.enforcement import FlextConstantsEnforcement as c

from .helpers import FlextUtilitiesBeartypeHelpers as _ubh

if TYPE_CHECKING:
    from flext_core._models.enforcement import FlextModelsEnforcement as me
    from flext_core._typings.base import FlextTypingBase as t

_NO_VIOLATION: t.StrMapping | None = None


class FlextUtilitiesBeartypeAliasVisitor:
    """ALIAS_REBIND / COMPATIBILITY_ALIAS / FOREIGN_CANONICAL_ALIAS_IMPORT visitors."""

    @staticmethod
    def v_foreign_canonical_alias_import(
        params: me.ForeignCanonicalAliasImportParams,
        target: type,
    ) -> t.StrMapping | None:
        """FOREIGN_CANONICAL_ALIAS_IMPORT — alias owned locally must not be imported from upstream.

        Flags ``from flext_core import c`` (etc.) inside a project that re-exports
        the same canonical alias locally. The local facade is the only legal source
        for c/m/p/t/u when the project owns that slot.
        """
        if not params.project_alias_owners:
            return _NO_VIOLATION
        module = _ubh.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        module_name = module.__name__
        package = module_name.split(".")[0]
        if package not in params.project_alias_owners:
            return _NO_VIOLATION
        local_aliases = frozenset(params.project_alias_owners[package])
        for name, value in vars(module).items():
            if name not in local_aliases:
                continue
            origin = _ubh.object_module_name_for(value)
            if origin is None:
                continue
            origin_package = origin.split(".")[0]
            if origin_package == package:
                continue
            if origin_package.startswith("flext_"):
                return {
                    "alias": name,
                    "origin": origin_package,
                    "local": package,
                }
        return _NO_VIOLATION

    @staticmethod
    def v_alias_rebind(
        params: me.AliasRebindParams,
        target: type,
    ) -> t.StrMapping | None:
        """ALIAS_REBIND — canonical alias rebind / sibling-import discipline."""
        module = _ubh.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        src_file = _ubh.module_filename_for(module) or ""
        filename = Path(src_file).name
        module_name = module.__name__
        package = module_name.split(".")[0]
        variant = params.expected_form
        violation = _NO_VIOLATION
        match variant:
            case "rebound_at_module_end" if filename in c.ENFORCEMENT_CANONICAL_FILES:
                target_name = target.__name__
                alias_char: str | None = next(
                    (
                        alias_name
                        for alias_name, _, suffix in _ubh.lazy_alias_suffixes(package)
                        if suffix in target_name
                    ),
                    None,
                )
                if alias_char and getattr(module, alias_char, None) is not target:
                    violation = {"alias": alias_char, "class": target_name}
            case "no_self_root_import_in_core_files" if (
                filename in c.ENFORCEMENT_CANONICAL_FILES
            ):
                violation = next(
                    (
                        {"package": package, "alias": alias_char}
                        for alias_char in _ubh.runtime_alias_names(package)
                        if (alias_value := getattr(module, alias_char, None))
                        is not None
                        and (
                            (_ubh.object_module_name_for(alias_value) or "").split(
                                ".",
                                1,
                            )[0]
                        )
                        == package
                    ),
                    _NO_VIOLATION,
                )
            case "sibling_models_type_checking":
                if "_models" in src_file:
                    violation = _NO_VIOLATION
            case _:
                pass
        return violation

    @staticmethod
    def v_compatibility_alias(
        params: me.CompatibilityAliasParams,
        target: type,
    ) -> t.StrMapping | None:
        """COMPATIBILITY_ALIAS — long facade class name must use canonical alias."""
        if not params.alias_renames:
            return _NO_VIOLATION
        module = _ubh.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        src_file = _ubh.module_filename_for(module) or ""
        filename = Path(src_file).name
        if filename in c.ENFORCEMENT_CANONICAL_FILES:
            return _NO_VIOLATION
        alias_renames = dict(params.alias_renames)
        for name, value in vars(module).items():
            alias = alias_renames.get(name)
            if alias is None:
                continue
            origin = _ubh.object_module_name_for(value)
            if origin is None:
                continue
            origin_package = origin.split(".")[0]
            current_package = module.__name__.split(".")[0]
            if origin_package == current_package:
                # Same-package definitions are not compatibility imports.
                continue
            return {
                "file": filename,
                "name": name,
                "alias": alias,
                "module": origin_package,
            }
        return _NO_VIOLATION


__all__: list[str] = ["FlextUtilitiesBeartypeAliasVisitor"]
