# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Infra.refactor package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .accessor_migration import FlextInfraAccessorMigrationOrchestrator
    from .census import FlextInfraRefactorCensus
    from .class_nesting_analyzer import FlextInfraRefactorClassNestingAnalyzer
    from .classvar_constant_autofix import FlextInfraRefactorClassvarConstantAutofix
    from .file_executor import (
        FlextInfraClassNestingPostCheckGate,
        FlextInfraRefactorFileExecutor,
    )
    from .legacy_text_ops import FlextInfraRefactorLegacyTextOps
    from .loader import FlextInfraRefactorRuleLoader
    from .modernize_orchestrator import FlextInfraModernizeOrchestrator
    from .namespace_enforcer import FlextInfraNamespaceEnforcer
    from .namespace_enforcer_phases import FlextInfraNamespaceEnforcerPhasesMixin
    from .orchestrator import FlextInfraRefactorOrchestrator
    from .project_classifier import FlextInfraProjectClassifier
    from .safety import FlextInfraRefactorSafetyManager
    from .scanner import FlextInfraRefactorLooseClassScanner
    from .service import FlextInfraRefactorService
    from .text_executor import FlextInfraRefactorTextExecutor
    from .violation_analyzer import FlextInfraRefactorViolationAnalyzer
    from .wrapper_root_namespace import FlextInfraWrapperRootNamespaceRefactor
__all__: tuple[str, ...] = (
    "FlextInfraAccessorMigrationOrchestrator",
    "FlextInfraClassNestingPostCheckGate",
    "FlextInfraModernizeOrchestrator",
    "FlextInfraNamespaceEnforcer",
    "FlextInfraNamespaceEnforcerPhasesMixin",
    "FlextInfraProjectClassifier",
    "FlextInfraRefactorCensus",
    "FlextInfraRefactorClassNestingAnalyzer",
    "FlextInfraRefactorClassvarConstantAutofix",
    "FlextInfraRefactorFileExecutor",
    "FlextInfraRefactorLegacyTextOps",
    "FlextInfraRefactorLooseClassScanner",
    "FlextInfraRefactorOrchestrator",
    "FlextInfraRefactorRuleLoader",
    "FlextInfraRefactorSafetyManager",
    "FlextInfraRefactorService",
    "FlextInfraRefactorTextExecutor",
    "FlextInfraRefactorViolationAnalyzer",
    "FlextInfraWrapperRootNamespaceRefactor",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".accessor_migration": ("FlextInfraAccessorMigrationOrchestrator",),
            ".census": ("FlextInfraRefactorCensus",),
            ".class_nesting_analyzer": ("FlextInfraRefactorClassNestingAnalyzer",),
            ".classvar_constant_autofix": (
                "FlextInfraRefactorClassvarConstantAutofix",
            ),
            ".file_executor": (
                "FlextInfraClassNestingPostCheckGate",
                "FlextInfraRefactorFileExecutor",
            ),
            ".legacy_text_ops": ("FlextInfraRefactorLegacyTextOps",),
            ".loader": ("FlextInfraRefactorRuleLoader",),
            ".modernize_orchestrator": ("FlextInfraModernizeOrchestrator",),
            ".namespace_enforcer": ("FlextInfraNamespaceEnforcer",),
            ".namespace_enforcer_phases": ("FlextInfraNamespaceEnforcerPhasesMixin",),
            ".orchestrator": ("FlextInfraRefactorOrchestrator",),
            ".project_classifier": ("FlextInfraProjectClassifier",),
            ".safety": ("FlextInfraRefactorSafetyManager",),
            ".scanner": ("FlextInfraRefactorLooseClassScanner",),
            ".service": ("FlextInfraRefactorService",),
            ".text_executor": ("FlextInfraRefactorTextExecutor",),
            ".violation_analyzer": ("FlextInfraRefactorViolationAnalyzer",),
            ".wrapper_root_namespace": ("FlextInfraWrapperRootNamespaceRefactor",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
