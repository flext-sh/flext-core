"""Import discipline enforcement — blacklist + alias rebind + library owners."""

from __future__ import annotations

from pathlib import Path

from beartype._util.module.utilmodget import (
    get_module_filename_or_none,
    get_object_module_name_or_none,
)

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._constants.project_metadata import FlextConstantsProjectMetadata as cp
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._typings.base import FlextTypingBase as t

_NO_VIOLATION: t.StrMapping | None = None


class FlextUtilitiesBeartypeImportVisitor:
    """IMPORT_BLACKLIST + ALIAS_REBIND + LIBRARY_IMPORT visitors."""

    @staticmethod
    def _v_import_blacklist(
        params: me.ImportBlacklistParams,
        target: type,
    ) -> t.StrMapping | None:
        """IMPORT_BLACKLIST — concrete-class / pydantic consumer-import discipline."""
        from flext_core._utilities.beartype_engine import ube

        module = ube.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        src_file = get_module_filename_or_none(module) or ""
        filename = Path(src_file).name
        module_name = module.__name__
        if filename in c.ENFORCEMENT_CANONICAL_FILES and not params.forbidden_symbols:
            canonical_import = next(
                (
                    {"file": filename, "import": name}
                    for name, value in vars(module).items()
                    if isinstance(value, type)
                    and name.startswith("Flext")
                    and (
                        origin := get_object_module_name_or_none(value) or ""
                    ).startswith("flext_")
                    and origin != module_name
                ),
                None,
            )
            return canonical_import or _NO_VIOLATION
        if not params.forbidden_symbols:
            return _NO_VIOLATION
        package = module_name.split(".")[0]
        if package.startswith("flext_") and module_name.startswith(f"{package}._"):
            return _NO_VIOLATION
        forbidden = frozenset(params.forbidden_symbols)
        allowed_roots = frozenset(params.forbidden_modules) or frozenset({"pydantic"})
        banned_import = next(
            (
                {"import": name, "package": package}
                for name, value in vars(module).items()
                if name in forbidden
                and ((get_object_module_name_or_none(value) or "").split(".")[0])
                in allowed_roots
            ),
            None,
        )
        if banned_import is not None:
            return banned_import
        return _NO_VIOLATION

    @staticmethod
    def _v_alias_rebind(
        params: me.AliasRebindParams,
        target: type,
    ) -> t.StrMapping | None:
        """ALIAS_REBIND — canonical alias rebind / sibling-import discipline."""
        from flext_core._utilities.beartype_engine import ube

        module = ube.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        src_file = get_module_filename_or_none(module) or ""
        filename = Path(src_file).name
        module_name = module.__name__
        package = module_name.split(".")[0]
        variant = params.expected_form
        if (
            variant == "rebound_at_module_end"
            and filename in c.ENFORCEMENT_CANONICAL_FILES
        ):
            target_name = target.__name__
            alias_char: str | None = next(
                (
                    alias_name
                    for alias_name, suffix in cp.ALIAS_TO_SUFFIX.items()
                    if alias_name in cp.FACADE_ALIAS_NAMES and suffix in target_name
                ),
                None,
            )
            if alias_char and getattr(module, alias_char, None) is not target:
                return {"alias": alias_char, "class": target_name}
            return _NO_VIOLATION
        if (
            variant == "no_self_root_import_in_core_files"
            and filename in c.ENFORCEMENT_CANONICAL_FILES
        ):
            for alias_char in cp.RUNTIME_ALIAS_NAMES:
                alias_value = getattr(module, alias_char, None)
                if alias_value is None:
                    continue
                origin = get_object_module_name_or_none(alias_value) or ""
                if origin.split(".", 1)[0] == package:
                    return {"package": package, "alias": alias_char}
            return _NO_VIOLATION
        if variant == "sibling_models_type_checking" and "_models" in src_file:
            return _NO_VIOLATION
        return _NO_VIOLATION

    @staticmethod
    def _v_library_import(
        params: me.LibraryImportParams,
        target: type,
    ) -> t.StrMapping | None:
        """LIBRARY_IMPORT — §2.7 library abstraction owner enforcement (Phase 3 hook)."""
        from flext_core._utilities.beartype_engine import ube

        if not params.library_owners:
            return _NO_VIOLATION
        module = ube.runtime_module_for(target)
        if module is None:
            return _NO_VIOLATION
        module_name = getattr(target, "__module__", "") or ""
        package = module_name.split(".")[0].replace("_", "-")
        for value in vars(module).values():
            origin = get_object_module_name_or_none(value)
            if origin is None:
                continue
            origin_root = origin.split(".")[0]
            owner = params.library_owners.get(origin_root)
            if owner is None or package == owner:
                continue
            return {"lib": origin_root, "owner": owner, "package": package}
        return _NO_VIOLATION
