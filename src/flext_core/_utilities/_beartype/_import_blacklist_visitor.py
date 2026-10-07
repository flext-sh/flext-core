"""IMPORT_BLACKLIST implementation extracted for LOC cap.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._typings.base import FlextTypingBase as t
from flext_core._utilities._beartype.helpers import (
    FlextUtilitiesBeartypeHelpers as _ubh,
)

_MIN_FAMILY_MODULE_PARTS = 2


class _ImportBlacklistVisitor:
    """IMPORT_BLACKLIST implementation extracted for LOC cap."""

    @staticmethod
    def implementation(
        params: me.ImportBlacklistParams,
        target: type,
    ) -> t.StrMapping | None:
        """Run the IMPORT_BLACKLIST checks behind the visitor facade.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
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
                        c.NAMESPACE_FAMILY_PREFIX,
                    )
                    and origin != module_name
                    and not _ImportBlacklistVisitor._is_local_family_import(
                        origin,
                        module_name,
                    )
                    and not (
                        (
                            value in target.__bases__
                            and _ubh.family_facade(target)
                            and _ubh.family_facade(value)
                        )
                        or _ImportBlacklistVisitor._is_lazy_published_member(
                            module,
                            value,
                        )
                    )
                ),
                no_violation,
            )
        elif params.private_package_only:
            package = module_name.split(".")[0]
            subpath = module_name.split(".")[1:]
            families = c.ENFORCEMENT_PRIVATE_FAMILY_PACKAGES
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
    def _is_lazy_published_member(module: object, value: type) -> bool:
        """Return True when the module publishes the lazy export contract.

        *value* must be one of its lazy-published family members. The catalog
        exempts "a consumer facade whose package publishes the lazy export
        contract and requires a family member": ``install_lazy_exports`` binds
        family members into the consumer facade's namespace through
        ``__getattr__``, so the census observes them in ``vars(module)``
        (after first access) although the module never imports them eagerly.
        The published ``_LAZY_IMPORTS`` map names every member's owning
        module.

        Returns:
            True when *value* is a lazy-published family member of *module*.

        """
        lazy_imports = getattr(module, "_LAZY_IMPORTS", None)
        values = getattr(lazy_imports, "values", None)
        if values is None:
            return False
        module_name = getattr(module, "__name__", "")
        origin = _ubh.object_module_name_for(value) or ""
        owners = {
            rel
            if rel.startswith(".")
            else f"{module_name}.{rel}"
            if not rel.startswith(module_name)
            else rel
            for rel in values()
            if isinstance(rel, str)
        }
        return origin in owners

    @staticmethod
    def _is_local_family_import(origin: str, module_name: str) -> bool:
        """Return True when *origin* is a same-package private-family import.

        Facade classes legitimately compose their private family sub-classes
        (``FlextApi[Tipo]*``) through MRO (R1, R3).  When the runtime origin
        of an imported class lives under the same package's private sub-package
        (e.g. ``flext_api._constants.api`` imported by ``flext_api.constants``),
        the ``no_concrete_namespace_import`` rule (ENFORCE-046) must not flag it.

        The allowed family suffixes are driven by the
        :attr:`FlextConstantsEnforcementTargets.ENFORCEMENT_PRIVATE_FAMILY_PACKAGES`
        constant (derived from ``ENFORCEMENT_CANONICAL_FILES``); evaluation
        iterates that set rather than hardcoding path fragments.

        Returns:
            True when *origin* is a same-package private-family import.

        """
        if "." not in origin or "." not in module_name:
            return False
        origin_parts = origin.split(".")
        module_parts = module_name.split(".")
        if (
            len(origin_parts) < _MIN_FAMILY_MODULE_PARTS
            or len(module_parts) < _MIN_FAMILY_MODULE_PARTS
        ):
            return False
        return (
            origin_parts[0] == module_parts[0]
            and origin_parts[1] in c.ENFORCEMENT_PRIVATE_FAMILY_PACKAGES
        )

    @staticmethod
    def _is_private_family_import(
        value: object,
        origin: str,
        package: str,
        families: frozenset[str],
        *,
        consumer_exempt: bool,
    ) -> bool:
        """Return True when a module re-exports a private-family flext symbol.

        Returns:
            True when a module re-exports a private-family flext symbol.

        """
        if not isinstance(value, type):
            return False
        if not origin.startswith("flext_"):
            return False
        if not families.intersection(origin.split(".")):
            return False
        return not origin.startswith(f"{package}.") or not consumer_exempt


__all__: list[str] = ["_ImportBlacklistVisitor"]
