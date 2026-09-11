"""Detect private-module imports that should route through canonical facades.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

from flext_infra import c, m, u
from flext_infra.detectors.internal_import_detector import (
    FlextInfraInternalImportDetector,
)

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraPrivateImportBypassDetector:
    """Detect imports of private family modules that should use a facade.

    Mirrors ENFORCE-068 semantics: same-package facade assembly is exempt,
    cross-package private imports are always flagged, and the fix points to
    the canonical facade module derived from the family directory name.
    """

    _FAMILY_TO_FACADE: ClassVar[t.StrMapping] = MappingProxyType({
        f"_{layer_lower}": layer_name[0].lower()
        for layer_name, layer_lower in c.ENFORCEMENT_NAMESPACE_LAYER_MAP
    })

    @classmethod
    def detect_file(
        cls, ctx: m.Infra.DetectorContext
    ) -> t.SequenceOf[m.Infra.PrivateImportBypassViolation]:
        """Return private-import bypass violations for one file."""
        res = u.Infra.fetch_python_resource(
            ctx.rope_project, ctx.file_path, skip_init_py=True
        )
        if res is None:
            return ()
        current_module = u.Infra.package_name(ctx.file_path)
        declared_imports = u.Infra.get_declared_module_imports(ctx.rope_project, res)
        semantic_imports = u.Infra.get_semantic_module_imports(ctx.rope_project, res)
        violations: list[m.Infra.PrivateImportBypassViolation] = []
        for local_name, fqn in semantic_imports.items():
            if "._" not in fqn:
                continue
            if FlextInfraInternalImportDetector.facade_assembly_exempt(
                current_module, fqn
            ):
                continue
            private_module = declared_imports.get(local_name, fqn)
            family = cls._family_for_module(private_module)
            if family is None:
                continue
            package = cls._package_for_module(private_module, family)
            if package is None:
                continue
            suggested_facade = f"{package}.{cls._FAMILY_TO_FACADE[family]}"
            symbol_exported = cls._symbol_exported(
                ctx.rope_project, suggested_facade, local_name
            )
            violations.append(
                m.Infra.PrivateImportBypassViolation(
                    file=str(ctx.file_path),
                    line=1,
                    current_import=f"from {private_module} import {local_name}",
                    detail=(
                        f"Private import '{private_module}.{local_name}' "
                        f"should use facade '{suggested_facade}'"
                    ),
                    private_module=private_module,
                    imported_symbol=local_name,
                    suggested_facade=suggested_facade,
                    symbol_exported=symbol_exported,
                )
            )
        return tuple(violations)

    @classmethod
    def _family_for_module(cls, module_name: str) -> str | None:
        """Return the private family segment of a module path, if any."""
        for part in module_name.split("."):
            if part in cls._FAMILY_TO_FACADE:
                return part
        return None

    @classmethod
    def _package_for_module(cls, module_name: str, family: str) -> str | None:
        """Return the owning package prefix before the private family."""
        parts = module_name.split(".")
        try:
            family_index = parts.index(family)
        except ValueError:
            return None
        return ".".join(parts[:family_index])

    @classmethod
    def _symbol_exported(
        cls, rope_project: t.Infra.RopeProject, facade_module: str, symbol: str
    ) -> bool:
        """Check whether ``symbol`` is reachable from ``facade_module``."""
        resource = rope_project.find_module(facade_module)
        if resource is None:
            return False
        try:
            pymodule = u.Infra.get_pymodule(rope_project, resource)
        except c.EXC_BROAD_IO_TYPE:
            return False
        return symbol in pymodule.get_attributes()


__all__: list[str] = ["FlextInfraPrivateImportBypassDetector"]
