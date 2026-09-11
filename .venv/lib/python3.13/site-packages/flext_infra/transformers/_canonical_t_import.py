"""Thin mixin that delegates canonical-alias import injection to ``_header``.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from flext_infra import c, u
from flext_infra.transformers import _header

if TYPE_CHECKING:
    from pathlib import Path


class FlextInfraEnsureCanonicalTImportMixin:
    """Inject ``from <pkg> import t`` only when the alias is actually used."""

    _DEFAULT_ALIAS_MODULE: ClassVar[str] = c.Infra.PKG_CORE_UNDERSCORE
    _IMPORT_ALIAS_PARTS: ClassVar[int] = 2

    def _ensure_t_import(self, source: str, module_name: str) -> tuple[str, bool]:
        """Inject ``from <module_name> import t`` if needed."""
        target_module = module_name or self._DEFAULT_ALIAS_MODULE
        updated = _header.ensure_alias_import(source, target_module, "t")
        return updated, updated != source

    def _ensure_alias_import(
        self, *, source: str, module_name: str, alias: str
    ) -> tuple[str, bool]:
        """Inject ``from <module_name> import <alias>`` when the alias is used."""
        target_module = module_name or self._DEFAULT_ALIAS_MODULE
        updated = _header.ensure_alias_import(source, target_module, alias)
        return updated, updated != source

    @classmethod
    def _t_alias_used(cls, source: str) -> bool:
        """Return whether the source contains a real use of the ``t`` alias."""
        return _header.alias_used(source, "t")

    @staticmethod
    def _has_t_import(source: str) -> bool:
        """Return whether the source already imports ``t``."""
        return _header.has_alias_import(source, "t")

    @classmethod
    def _alias_used(cls, source: str, alias: str) -> bool:
        """Return whether ``alias`` is used as a standalone dotted facade name."""
        return _header.alias_used(source, alias)

    @staticmethod
    def _has_alias_import(source: str, alias: str) -> bool:
        """Return whether source already binds ``alias`` through a ``from`` import."""
        return _header.has_alias_import(source, alias)

    @staticmethod
    def _canonical_import_module(file_path: Path | None) -> str:
        """Return the root package name for a transformed file."""
        default_module: str = c.Infra.PKG_CORE_UNDERSCORE
        if file_path is None or not file_path.exists():
            return default_module
        package_name: str = u.Infra.package_name(file_path)
        if not package_name:
            return default_module
        root_package: str = package_name.split(".", maxsplit=1)[0]
        return root_package


__all__: list[str] = ["FlextInfraEnsureCanonicalTImportMixin"]
