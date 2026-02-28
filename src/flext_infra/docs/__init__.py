"""Documentation services.

Provides services for documentation generation, validation, and maintenance
across the workspace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_infra.docs.auditor import DocAuditor
    from flext_infra.docs.builder import DocBuilder
    from flext_infra.docs.fixer import DocFixer
    from flext_infra.docs.generator import DocGenerator
    from flext_infra.docs.validator import DocValidator

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "DocAuditor": ("flext_infra.docs.auditor", "DocAuditor"),
    "DocBuilder": ("flext_infra.docs.builder", "DocBuilder"),
    "DocFixer": ("flext_infra.docs.fixer", "DocFixer"),
    "DocGenerator": ("flext_infra.docs.generator", "DocGenerator"),
    "DocValidator": ("flext_infra.docs.validator", "DocValidator"),
}

__all__ = [
    "DocAuditor",
    "DocBuilder",
    "DocFixer",
    "DocGenerator",
    "DocValidator",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
