# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Documentation services.

Provides services for documentation generation, validation, and maintenance
across the workspace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes
    from flext_infra.docs.auditor import FlextInfraDocAuditor
    from flext_infra.docs.builder import FlextInfraDocBuilder
    from flext_infra.docs.fixer import FlextInfraDocFixer
    from flext_infra.docs.generator import FlextInfraDocGenerator
    from flext_infra.docs.shared import FlextInfraDocsShared
    from flext_infra.docs.validator import FlextInfraDocValidator

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FlextInfraDocAuditor": ("flext_infra.docs.auditor", "FlextInfraDocAuditor"),
    "FlextInfraDocBuilder": ("flext_infra.docs.builder", "FlextInfraDocBuilder"),
    "FlextInfraDocFixer": ("flext_infra.docs.fixer", "FlextInfraDocFixer"),
    "FlextInfraDocGenerator": ("flext_infra.docs.generator", "FlextInfraDocGenerator"),
    "FlextInfraDocValidator": ("flext_infra.docs.validator", "FlextInfraDocValidator"),
    "FlextInfraDocsShared": ("flext_infra.docs.shared", "FlextInfraDocsShared"),
}

__all__ = [
    "FlextInfraDocAuditor",
    "FlextInfraDocBuilder",
    "FlextInfraDocFixer",
    "FlextInfraDocGenerator",
    "FlextInfraDocValidator",
    "FlextInfraDocsShared",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
