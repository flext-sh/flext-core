"""Core infrastructure services.

Provides foundational services for inventory management, validation rules,
base.mk sync checking, pytest diagnostics, pattern scanning, skill validation,
and stub supply chain management.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_infra.core.__main__ import main
    from flext_infra.core.basemk_validator import FlextInfraBaseMkValidator
    from flext_infra.core.inventory import FlextInfraInventoryService
    from flext_infra.core.pytest_diag import FlextInfraPytestDiagExtractor
    from flext_infra.core.scanner import FlextInfraTextPatternScanner
    from flext_infra.core.skill_validator import FlextInfraSkillValidator
    from flext_infra.core.stub_chain import FlextInfraStubSupplyChain

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FlextInfraBaseMkValidator": ("flext_infra.core.basemk_validator", "FlextInfraBaseMkValidator"),
    "FlextInfraInventoryService": ("flext_infra.core.inventory", "FlextInfraInventoryService"),
    "FlextInfraPytestDiagExtractor": ("flext_infra.core.pytest_diag", "FlextInfraPytestDiagExtractor"),
    "FlextInfraSkillValidator": ("flext_infra.core.skill_validator", "FlextInfraSkillValidator"),
    "FlextInfraStubSupplyChain": ("flext_infra.core.stub_chain", "FlextInfraStubSupplyChain"),
    "FlextInfraTextPatternScanner": ("flext_infra.core.scanner", "FlextInfraTextPatternScanner"),
    "main": ("flext_infra.core.__main__", "main"),
}

__all__ = [
    "FlextInfraBaseMkValidator",
    "FlextInfraInventoryService",
    "FlextInfraPytestDiagExtractor",
    "FlextInfraSkillValidator",
    "FlextInfraStubSupplyChain",
    "FlextInfraTextPatternScanner",
    "main",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
