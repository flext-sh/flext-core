# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Infra.workspace package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .detector import FlextInfraWorkspaceDetector
    from .environment_contracts import envrc_contract_violations
    from .environment_provenance import FlextInfraWorkspaceEnvironmentProvenance
    from .flext_binding import FlextInfraFlextBindingService
    from .orchestrator import FlextInfraOrchestratorService
    from .rope import FlextInfraRopeWorkspace
    from .sandbox_orchestrator import FlextInfraSandboxOrchestrator
__all__: tuple[str, ...] = (
    "FlextInfraFlextBindingService",
    "FlextInfraOrchestratorService",
    "FlextInfraRopeWorkspace",
    "FlextInfraSandboxOrchestrator",
    "FlextInfraWorkspaceDetector",
    "FlextInfraWorkspaceEnvironmentProvenance",
    "envrc_contract_violations",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".detector": ("FlextInfraWorkspaceDetector",),
            ".environment_contracts": ("envrc_contract_violations",),
            ".environment_provenance": ("FlextInfraWorkspaceEnvironmentProvenance",),
            ".flext_binding": ("FlextInfraFlextBindingService",),
            ".orchestrator": ("FlextInfraOrchestratorService",),
            ".rope": ("FlextInfraRopeWorkspace",),
            ".sandbox_orchestrator": ("FlextInfraSandboxOrchestrator",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
