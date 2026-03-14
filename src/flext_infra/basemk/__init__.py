# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Base.mk template engine service.

Provides services for managing, validating, and rendering base.mk templates
for workspace build orchestration.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes
    from flext_infra.basemk.engine import FlextInfraBaseMkTemplateEngine
    from flext_infra.basemk.generator import FlextInfraBaseMkGenerator

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FlextInfraBaseMkGenerator": ("flext_infra.basemk.generator", "FlextInfraBaseMkGenerator"),
    "FlextInfraBaseMkTemplateEngine": ("flext_infra.basemk.engine", "FlextInfraBaseMkTemplateEngine"),
}

__all__ = [
    "FlextInfraBaseMkGenerator",
    "FlextInfraBaseMkTemplateEngine",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
