"""Base.mk template engine service.

Provides services for managing, validating, and rendering base.mk templates
for workspace build orchestration.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_infra.basemk.__main__ import main
    from flext_infra.basemk.engine import TemplateEngine
    from flext_infra.basemk.generator import BaseMkGenerator

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BaseMkGenerator": ("flext_infra.basemk.generator", "BaseMkGenerator"),
    "TemplateEngine": ("flext_infra.basemk.engine", "TemplateEngine"),
    "main": ("flext_infra.basemk.__main__", "main"),
}

__all__ = [
    "BaseMkGenerator",
    "TemplateEngine",
    "main",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
