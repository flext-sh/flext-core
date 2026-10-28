"""Alias rebind and compatibility alias visitors.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flext_core import c, p, t
from flext_core._utilities._beartype import FlextUtilitiesBeartypeHelpers
from flext_core._utilities.family_surface import FlextUtilitiesFamilySurface

if TYPE_CHECKING:
    from flext_core import m

_NO_VIOLATION: t.StrMapping | None = None


class FlextUtilitiesBeartypeAliasVisitor:
    """ALIAS_REBIND / COMPATIBILITY_ALIAS visitors."""

    @staticmethod
    def _self_root_alias_import(
        module: p.AttributeProbe,
        alias_char: str,
        package: str,
        module_name: str,
    ) -> bool:
        """Return True when one canonical alias rebinds a self-root family import.

        Returns:
            True when the alias binds a same-package family member from outside
            the canonical root files.

        """
        alias_value = getattr(module, alias_char, None)
        if alias_value is None:
            return False
        origin = FlextUtilitiesBeartypeHelpers.object_module_name_for(alias_value) or ""
        if not origin:
            return False
        canonical_stems = frozenset(
            name.removesuffix(".py") for name in c.ENFORCEMENT_CANONICAL_FILES
        )
        return (
            origin.split(".", 1)[0] == package
            and origin != module_name
            and origin not in {f"{package}.{stem}" for stem in canonical_stems}
        )

    @staticmethod
    def v_alias_rebind(
        params: m.AliasRebindParams,
        target: type,
    ) -> t.StrMapping | None:
        """ALIAS_REBIND — canonical alias rebind / sibling-import discipline.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        module = FlextUtilitiesBeartypeHelpers.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        src_file = FlextUtilitiesBeartypeHelpers.module_filename_for(module) or ""
        filename = Path(src_file).name
        module_name = module.__name__
        package = module_name.split(".")[0]
        variant = params.expected_form
        violation = _NO_VIOLATION
        surface = FlextUtilitiesFamilySurface
        match variant:
            case "rebound_at_module_end" if filename in c.ENFORCEMENT_CANONICAL_FILES:
                target_name = target.__name__
                alias_char: str | None = next(
                    (
                        alias_name
                        for alias_name, _, suffix in surface.lazy_alias_suffixes(
                            package,
                        )
                        if suffix in target_name
                    ),
                    None,
                )
                if alias_char and getattr(module, alias_char, None) is not target:
                    violation = {
                        "alias": alias_char,
                        "class": target_name,
                        "rebind_form": f"{alias_char} = {target_name}",
                    }
            case "no_self_root_import_in_core_files" if (
                filename in c.ENFORCEMENT_CANONICAL_FILES
            ):
                violation = next(
                    (
                        {"package": package, "alias": alias_char}
                        for alias_char in surface.runtime_alias_names(
                            package,
                        )
                        if FlextUtilitiesBeartypeAliasVisitor._self_root_alias_import(
                            module,
                            alias_char,
                            package,
                            module_name,
                        )
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
        params: m.CompatibilityAliasParams,
        target: type,
    ) -> t.StrMapping | None:
        """COMPATIBILITY_ALIAS — long facade class name must use canonical alias.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        if not params.alias_renames:
            return _NO_VIOLATION
        module = FlextUtilitiesBeartypeHelpers.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        src_file = FlextUtilitiesBeartypeHelpers.module_filename_for(module) or ""
        filename = Path(src_file).name
        if filename in c.ENFORCEMENT_CANONICAL_FILES:
            return _NO_VIOLATION
        alias_renames = dict(params.alias_renames)
        for name, value in vars(module).items():
            alias = alias_renames.get(name)
            if alias is None:
                continue
            origin = FlextUtilitiesBeartypeHelpers.object_module_name_for(value)
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
