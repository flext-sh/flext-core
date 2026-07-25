"""Import discipline enforcement — blacklist + alias rebind + library owners."""

from __future__ import annotations

from pathlib import Path

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._typings.base import FlextTypingBase as t

from ._alias_visitor import FlextUtilitiesBeartypeAliasVisitor
from ._library_visitor import FlextUtilitiesBeartypeLibraryVisitor
from .helpers import FlextUtilitiesBeartypeHelpers as _ubh


class FlextUtilitiesBeartypeImportVisitor:
    """IMPORT_BLACKLIST + ALIAS_REBIND + LIBRARY_IMPORT visitor facade."""

    @staticmethod
    def v_import_blacklist(
        params: me.ImportBlacklistParams,
        target: type,
    ) -> t.StrMapping | None:
        """IMPORT_BLACKLIST — concrete-class / pydantic consumer-import discipline."""
        return _ImportBlacklistVisitor.v_import_blacklist(params, target)

    @staticmethod
    def v_foreign_canonical_alias_import(
        params: me.ForeignCanonicalAliasImportParams,
        target: type,
    ) -> t.StrMapping | None:
        """FOREIGN_CANONICAL_ALIAS_IMPORT."""
        return FlextUtilitiesBeartypeAliasVisitor.v_foreign_canonical_alias_import(
            params,
            target,
        )

    @staticmethod
    def v_alias_rebind(
        params: me.AliasRebindParams,
        target: type,
    ) -> t.StrMapping | None:
        """ALIAS_REBIND."""
        return FlextUtilitiesBeartypeAliasVisitor.v_alias_rebind(params, target)

    @staticmethod
    def v_compatibility_alias(
        params: me.CompatibilityAliasParams,
        target: type,
    ) -> t.StrMapping | None:
        """COMPATIBILITY_ALIAS."""
        return FlextUtilitiesBeartypeAliasVisitor.v_compatibility_alias(params, target)

    @staticmethod
    def v_library_import(
        params: me.LibraryImportParams,
        target: type,
    ) -> t.StrMapping | None:
        """LIBRARY_IMPORT."""
        return FlextUtilitiesBeartypeLibraryVisitor.v_library_import(params, target)


class _ImportBlacklistVisitor:
    """IMPORT_BLACKLIST implementation extracted for LOC cap."""

    @staticmethod
    def v_import_blacklist(
        params: me.ImportBlacklistParams,
        target: type,
    ) -> t.StrMapping | None:
        """IMPORT_BLACKLIST — concrete-class / pydantic consumer-import discipline."""
        no_violation: t.StrMapping | None = None
        module = _ubh.runtime_module_for(target)
        if module is None:
            return no_violation
        src_file = _ubh.module_filename_for(module) or ""
        filename = Path(src_file).name
        module_name = module.__name__
        violation = no_violation
        if (
            filename in c.ENFORCEMENT_CANONICAL_FILES
            and not params.forbidden_symbols
            and not params.private_package_only
        ):
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
                        "flext_",
                    )
                    and origin != module_name
                ),
                no_violation,
            )
        elif params.private_package_only:
            package = module_name.split(".")[0]
            subpath = module_name.split(".")[1:]
            families = frozenset(
                f"_{name.removesuffix('.py')}" for name in c.ENFORCEMENT_CANONICAL_FILES
            )
            consumer_exempt = (
                filename in c.ENFORCEMENT_CANONICAL_FILES
                or any(part.startswith("_") for part in subpath)
                or len(subpath) <= 1
            )
            if package.startswith("flext_"):
                violation = next(
                    (
                        {"import": name, "origin": origin, "file": filename}
                        for name, value in vars(module).items()
                        if _ImportBlacklistVisitor._is_private_family_import(
                            value,
                            origin := _ubh.object_module_name_for(value) or "",
                            package,
                            families,
                            consumer_exempt=consumer_exempt,
                        )
                    ),
                    no_violation,
                )
        elif params.forbidden_symbols:
            package = module_name.split(".")[0]
            if not (
                package.startswith("flext_") and module_name.startswith(f"{package}._")
            ):
                forbidden = frozenset(params.forbidden_symbols)
                allowed_roots = frozenset(params.forbidden_modules) or frozenset({
                    "pydantic",
                })
                violation = next(
                    (
                        {"import": name, "package": package}
                        for name, value in vars(module).items()
                        if name in forbidden
                        and ((_ubh.object_module_name_for(value) or "").split(".")[0])
                        in allowed_roots
                    ),
                    no_violation,
                )
        return violation

    @staticmethod
    def _is_private_family_import(
        value: object,
        origin: str,
        package: str,
        families: frozenset[str],
        *,
        consumer_exempt: bool,
    ) -> bool:
        """Return True when a module re-exports a private-family flext symbol."""
        if not isinstance(value, type):
            return False
        if not origin.startswith("flext_"):
            return False
        if not families.intersection(origin.split(".")):
            return False
        return not origin.startswith(f"{package}.") or not consumer_exempt


__all__: list[str] = ["FlextUtilitiesBeartypeImportVisitor"]
