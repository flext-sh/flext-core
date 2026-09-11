# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Infra.deps package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import phases as phases
    from .detection import FlextInfraDependencyDetectionService
    from .detection_analysis import FlextInfraDependencyDetectionAnalysis
    from .detector import FlextInfraRuntimeDevDependencyDetector
    from .detector_runtime import FlextInfraDependencyDetectorRuntime
    from .extra_paths import FlextInfraExtraPathsManager
    from .fix_pyrefly_config import FlextInfraConfigFixer
    from .modernizer import FlextInfraPyprojectModernizer
    from .phases.consolidate_groups import FlextInfraConsolidateGroupsPhase
    from .phases.ensure_coverage import FlextInfraEnsureCoverageConfigPhase
    from .phases.ensure_formatting import FlextInfraEnsureFormattingToolingPhase
    from .phases.ensure_mypy import FlextInfraEnsureMypyConfigPhase
    from .phases.ensure_namespace import FlextInfraEnsureNamespaceToolingPhase
    from .phases.ensure_packaging import FlextInfraEnsurePackagingPhase
    from .phases.ensure_pydantic_mypy import FlextInfraEnsurePydanticMypyConfigPhase
    from .phases.ensure_pyrefly import FlextInfraEnsurePyreflyConfigPhase
    from .phases.ensure_pyright import FlextInfraEnsurePyrightConfigPhase
    from .phases.ensure_pytest import FlextInfraEnsurePytestConfigPhase
    from .phases.ensure_ruff import FlextInfraEnsureRuffConfigPhase
    from .phases.ensure_vulture import FlextInfraEnsureVultureConfigPhase
    from .phases.inject_comments import FlextInfraInjectCommentsPhase
    from .toml_phase import FlextInfraTomlPhaseService
__all__: tuple[str, ...] = (
    "FlextInfraConfigFixer",
    "FlextInfraConsolidateGroupsPhase",
    "FlextInfraDependencyDetectionAnalysis",
    "FlextInfraDependencyDetectionService",
    "FlextInfraDependencyDetectorRuntime",
    "FlextInfraEnsureCoverageConfigPhase",
    "FlextInfraEnsureFormattingToolingPhase",
    "FlextInfraEnsureMypyConfigPhase",
    "FlextInfraEnsureNamespaceToolingPhase",
    "FlextInfraEnsurePackagingPhase",
    "FlextInfraEnsurePydanticMypyConfigPhase",
    "FlextInfraEnsurePyreflyConfigPhase",
    "FlextInfraEnsurePyrightConfigPhase",
    "FlextInfraEnsurePytestConfigPhase",
    "FlextInfraEnsureRuffConfigPhase",
    "FlextInfraEnsureVultureConfigPhase",
    "FlextInfraExtraPathsManager",
    "FlextInfraInjectCommentsPhase",
    "FlextInfraPyprojectModernizer",
    "FlextInfraRuntimeDevDependencyDetector",
    "FlextInfraTomlPhaseService",
    "phases",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".detection": ("FlextInfraDependencyDetectionService",),
            ".detection_analysis": ("FlextInfraDependencyDetectionAnalysis",),
            ".detector": ("FlextInfraRuntimeDevDependencyDetector",),
            ".detector_runtime": ("FlextInfraDependencyDetectorRuntime",),
            ".extra_paths": ("FlextInfraExtraPathsManager",),
            ".fix_pyrefly_config": ("FlextInfraConfigFixer",),
            ".modernizer": ("FlextInfraPyprojectModernizer",),
            ".phases": ("phases",),
            ".phases.consolidate_groups": ("FlextInfraConsolidateGroupsPhase",),
            ".phases.ensure_coverage": ("FlextInfraEnsureCoverageConfigPhase",),
            ".phases.ensure_formatting": ("FlextInfraEnsureFormattingToolingPhase",),
            ".phases.ensure_mypy": ("FlextInfraEnsureMypyConfigPhase",),
            ".phases.ensure_namespace": ("FlextInfraEnsureNamespaceToolingPhase",),
            ".phases.ensure_packaging": ("FlextInfraEnsurePackagingPhase",),
            ".phases.ensure_pydantic_mypy": (
                "FlextInfraEnsurePydanticMypyConfigPhase",
            ),
            ".phases.ensure_pyrefly": ("FlextInfraEnsurePyreflyConfigPhase",),
            ".phases.ensure_pyright": ("FlextInfraEnsurePyrightConfigPhase",),
            ".phases.ensure_pytest": ("FlextInfraEnsurePytestConfigPhase",),
            ".phases.ensure_ruff": ("FlextInfraEnsureRuffConfigPhase",),
            ".phases.ensure_vulture": ("FlextInfraEnsureVultureConfigPhase",),
            ".phases.inject_comments": ("FlextInfraInjectCommentsPhase",),
            ".toml_phase": ("FlextInfraTomlPhaseService",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
