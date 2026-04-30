"""Import discipline enforcement — blacklist + alias rebind + library owners."""

from __future__ import annotations

from pathlib import Path

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._typings.base import FlextTypingBase as t
from flext_core._utilities._beartype.helpers import (
    FlextUtilitiesBeartypeHelpers as _ubh,
)

_NO_VIOLATION: t.StrMapping | None = None


class FlextUtilitiesBeartypeImportVisitor:
    """IMPORT_BLACKLIST + ALIAS_REBIND + LIBRARY_IMPORT visitors."""

    @staticmethod
    def v_import_blacklist(
        params: me.ImportBlacklistParams,
        target: type,
    ) -> t.StrMapping | None:
        """IMPORT_BLACKLIST — concrete-class / pydantic consumer-import discipline."""
        module = _ubh.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        src_file = _ubh.module_filename_for(module) or ""
        filename = Path(src_file).name
        module_name = module.__name__
        violation = _NO_VIOLATION
        if filename in c.ENFORCEMENT_CANONICAL_FILES and not params.forbidden_symbols:
            tier_prefixes = tuple(
                value.__name__
                for value in vars(module).values()
                if isinstance(value, type)
            )
            violation = next(
                (
                    {"file": filename, "import": name}
                    for name, value in vars(module).items()
                    if isinstance(value, type)
                    and name.startswith(tier_prefixes)
                    and (origin := _ubh.object_module_name_for(value) or "").startswith(
                        "flext_"
                    )
                    and origin != module_name
                ),
                _NO_VIOLATION,
            )
        elif params.forbidden_symbols:
            package = module_name.split(".")[0]
            if not (
                package.startswith("flext_") and module_name.startswith(f"{package}._")
            ):
                forbidden = frozenset(params.forbidden_symbols)
                allowed_roots = frozenset(params.forbidden_modules) or frozenset({
                    "pydantic"
                })
                violation = next(
                    (
                        {"import": name, "package": package}
                        for name, value in vars(module).items()
                        if name in forbidden
                        and ((_ubh.object_module_name_for(value) or "").split(".")[0])
                        in allowed_roots
                    ),
                    _NO_VIOLATION,
                )
        return violation

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
                                ".", 1
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
    def v_library_import(
        params: me.LibraryImportParams,
        target: type,
    ) -> t.StrMapping | None:
        """LIBRARY_IMPORT — §2.7 library abstraction owner enforcement (Phase 3 hook)."""
        if not params.library_owners:
            return _NO_VIOLATION
        module = _ubh.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        module_name = getattr(target, "__module__", "") or ""
        package = module_name.split(".")[0].replace("_", "-")
        for value in vars(module).values():
            origin = _ubh.object_module_name_for(value)
            if origin is None:
                continue
            origin_root = origin.split(".")[0]
            owner = params.library_owners.get(origin_root)
            if owner is None or package == owner:
                continue
            return {"lib": origin_root, "owner": owner, "package": package}
        return _NO_VIOLATION
