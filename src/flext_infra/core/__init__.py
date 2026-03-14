# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Core infrastructure services.

Provides foundational services for inventory management, validation rules,
base.mk sync checking, pytest diagnostics, pattern scanning, skill validation,
and stub supply chain management.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes
    from flext_infra.core.basemk_validator import FlextInfraBaseMkValidator
    from flext_infra.core.inventory import (
        FlextInfraInventoryService,
        FlextInfraInventoryService as s,
    )
    from flext_infra.core.namespace_validator import FlextInfraNamespaceValidator
    from flext_infra.core.pytest_diag import FlextInfraPytestDiagExtractor
    from flext_infra.core.scanner import FlextInfraTextPatternScanner
    from flext_infra.core.skill_validator import FlextInfraSkillValidator
    from flext_infra.core.stub_chain import FlextInfraStubSupplyChain

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FlextInfraBaseMkValidator": (
        "flext_infra.core.basemk_validator",
        "FlextInfraBaseMkValidator",
    ),
    "FlextInfraInventoryService": (
        "flext_infra.core.inventory",
        "FlextInfraInventoryService",
    ),
    "FlextInfraNamespaceValidator": (
        "flext_infra.core.namespace_validator",
        "FlextInfraNamespaceValidator",
    ),
    "FlextInfraPytestDiagExtractor": (
        "flext_infra.core.pytest_diag",
        "FlextInfraPytestDiagExtractor",
    ),
    "FlextInfraSkillValidator": (
        "flext_infra.core.skill_validator",
        "FlextInfraSkillValidator",
    ),
    "FlextInfraStubSupplyChain": (
        "flext_infra.core.stub_chain",
        "FlextInfraStubSupplyChain",
    ),
    "FlextInfraTextPatternScanner": (
        "flext_infra.core.scanner",
        "FlextInfraTextPatternScanner",
    ),
    "s": ("flext_infra.core.inventory", "FlextInfraInventoryService"),
}

__all__ = [
    "FlextInfraBaseMkValidator",
    "FlextInfraInventoryService",
    "FlextInfraNamespaceValidator",
    "FlextInfraPytestDiagExtractor",
    "FlextInfraSkillValidator",
    "FlextInfraStubSupplyChain",
    "FlextInfraTextPatternScanner",
    "s",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
