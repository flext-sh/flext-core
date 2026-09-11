# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Infra.codegen package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .census import FlextInfraCodegenCensus
    from .codegen_generation import FlextInfraCodegenGeneration
    from .conform import FlextInfraCodegenConform
    from .consolidator import FlextInfraCodegenConsolidator
    from .constants_quality_gate import FlextInfraCodegenQualityGate
    from .fixer import FlextInfraCodegenFixer
    from .layout import FlextInfraCodegenLayout
    from .lazy_init import FlextInfraCodegenLazyInit
    from .lazy_init_planner import FlextInfraCodegenLazyInitPlanner
    from .make_bootstrap import FlextInfraCodegenMakeBootstrap
    from .managed_conflicts import FlextInfraCodegenManagedConflicts
    from .managed_conflicts_bootstrap import (
        ManagedConflictBootstrapError,
        prepare_managed_conflicts,
    )
    from .managed_conflicts_core import ManagedConflictError, recover_managed_toml
    from .mise_artifacts import FlextInfraCodegenMiseArtifacts
    from .pipeline import FlextInfraCodegenPipeline
    from .project_new import FlextInfraCodegenProjectNew
    from .py_typed import FlextInfraCodegenPyTyped
    from .scaffolder import FlextInfraCodegenScaffolder
    from .version_file import FlextInfraCodegenVersionFile
__all__: tuple[str, ...] = (
    "FlextInfraCodegenCensus",
    "FlextInfraCodegenConform",
    "FlextInfraCodegenConsolidator",
    "FlextInfraCodegenFixer",
    "FlextInfraCodegenGeneration",
    "FlextInfraCodegenLayout",
    "FlextInfraCodegenLazyInit",
    "FlextInfraCodegenLazyInitPlanner",
    "FlextInfraCodegenMakeBootstrap",
    "FlextInfraCodegenManagedConflicts",
    "FlextInfraCodegenMiseArtifacts",
    "FlextInfraCodegenPipeline",
    "FlextInfraCodegenProjectNew",
    "FlextInfraCodegenPyTyped",
    "FlextInfraCodegenQualityGate",
    "FlextInfraCodegenScaffolder",
    "FlextInfraCodegenVersionFile",
    "ManagedConflictBootstrapError",
    "ManagedConflictError",
    "prepare_managed_conflicts",
    "recover_managed_toml",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".census": ("FlextInfraCodegenCensus",),
            ".codegen_generation": ("FlextInfraCodegenGeneration",),
            ".conform": ("FlextInfraCodegenConform",),
            ".consolidator": ("FlextInfraCodegenConsolidator",),
            ".constants_quality_gate": ("FlextInfraCodegenQualityGate",),
            ".fixer": ("FlextInfraCodegenFixer",),
            ".layout": ("FlextInfraCodegenLayout",),
            ".lazy_init": ("FlextInfraCodegenLazyInit",),
            ".lazy_init_planner": ("FlextInfraCodegenLazyInitPlanner",),
            ".make_bootstrap": ("FlextInfraCodegenMakeBootstrap",),
            ".managed_conflicts": ("FlextInfraCodegenManagedConflicts",),
            ".managed_conflicts_bootstrap": (
                "ManagedConflictBootstrapError",
                "prepare_managed_conflicts",
            ),
            ".managed_conflicts_core": ("ManagedConflictError", "recover_managed_toml"),
            ".mise_artifacts": ("FlextInfraCodegenMiseArtifacts",),
            ".pipeline": ("FlextInfraCodegenPipeline",),
            ".project_new": ("FlextInfraCodegenProjectNew",),
            ".py_typed": ("FlextInfraCodegenPyTyped",),
            ".scaffolder": ("FlextInfraCodegenScaffolder",),
            ".version_file": ("FlextInfraCodegenVersionFile",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
