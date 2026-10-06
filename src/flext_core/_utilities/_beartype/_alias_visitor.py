"""Alias rebind and compatibility alias visitors.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_core._constants.enforcement import FlextConstantsEnforcement
from flext_core._models.enforcement import FlextModelsEnforcement
from flext_core._typings.base import FlextTypingBase
from flext_core._utilities._beartype.helpers import FlextUtilitiesBeartypeHelpers

_NO_VIOLATION: FlextTypingBase.StrMapping | None = None


class FlextUtilitiesBeartypeAliasVisitor:
    """ALIAS_REBIND / COMPATIBILITY_ALIAS visitors."""

    @staticmethod
    def v_alias_rebind(
        params: FlextModelsEnforcement.AliasRebindParams,
        target: type,
    ) -> FlextTypingBase.StrMapping | None:
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
        match variant:
            case "rebound_at_module_end" if (
                filename in FlextConstantsEnforcement.ENFORCEMENT_CANONICAL_FILES
            ):
                target_name = target.__name__
                alias_char: str | None = next(
                    (
                        alias_name
                        for alias_name, _, suffix in FlextUtilitiesBeartypeHelpers.lazy_alias_suffixes(
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
                filename in FlextConstantsEnforcement.ENFORCEMENT_CANONICAL_FILES
            ):
                canonical_stems = frozenset(
                    name.removesuffix(".py")
                    for name in FlextConstantsEnforcement.ENFORCEMENT_CANONICAL_FILES
                )
                violation = next(
                    (
                        {"package": package, "alias": alias_char}
                        for alias_char in FlextUtilitiesBeartypeHelpers.runtime_alias_names(
                            package,
                        )
                        if (alias_value := getattr(module, alias_char, None))
                        is not None
                        and (
                            origin
                            := FlextUtilitiesBeartypeHelpers.object_module_name_for(
                                alias_value,
                            )
                            or ""
                        )
                        and origin.split(".", 1)[0] == package
                        and origin != module_name
                        and origin
                        not in {f"{package}.{stem}" for stem in canonical_stems}
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
        params: FlextModelsEnforcement.CompatibilityAliasParams,
        target: type,
    ) -> FlextTypingBase.StrMapping | None:
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
        if filename in FlextConstantsEnforcement.ENFORCEMENT_CANONICAL_FILES:
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
