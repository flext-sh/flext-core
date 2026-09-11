# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Infra. Constants package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .adapters import FlextInfraConstantsAdapters
    from .base import FlextInfraConstantsBase
    from .census import FlextInfraConstantsCensus
    from .check import FlextInfraConstantsCheck
    from .cli import FlextInfraConstantsCli
    from .codegen import FlextInfraConstantsCodegen
    from .codegen_detection import FlextInfraConstantsCodegenDetection
    from .codegen_lazy import FlextInfraConstantsCodegenLazy
    from .codegen_project import FlextInfraConstantsCodegenProject
    from .codegen_render_names import FlextInfraConstantsCodegenRenderNames
    from .deps import FlextInfraConstantsDeps
    from .detectors import FlextInfraConstantsDetectors
    from .docs import FlextInfraConstantsDocs
    from .git import FlextInfraConstantsGit
    from .github import FlextInfraConstantsGithub
    from .make import FlextInfraConstantsMake
    from .namespace import FlextInfraConstantsNamespace
    from .refactor import FlextInfraConstantsRefactor
    from .release import FlextInfraConstantsRelease
    from .rope import FlextInfraConstantsRope
    from .source_code import FlextInfraConstantsSourceCode
    from .validate import FlextInfraConstantsSharedInfra
    from .workspace import FlextInfraConstantsWorkspace
__all__: tuple[str, ...] = (
    "FlextInfraConstantsAdapters",
    "FlextInfraConstantsBase",
    "FlextInfraConstantsCensus",
    "FlextInfraConstantsCheck",
    "FlextInfraConstantsCli",
    "FlextInfraConstantsCodegen",
    "FlextInfraConstantsCodegenDetection",
    "FlextInfraConstantsCodegenLazy",
    "FlextInfraConstantsCodegenProject",
    "FlextInfraConstantsCodegenRenderNames",
    "FlextInfraConstantsDeps",
    "FlextInfraConstantsDetectors",
    "FlextInfraConstantsDocs",
    "FlextInfraConstantsGit",
    "FlextInfraConstantsGithub",
    "FlextInfraConstantsMake",
    "FlextInfraConstantsNamespace",
    "FlextInfraConstantsRefactor",
    "FlextInfraConstantsRelease",
    "FlextInfraConstantsRope",
    "FlextInfraConstantsSharedInfra",
    "FlextInfraConstantsSourceCode",
    "FlextInfraConstantsWorkspace",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".adapters": ("FlextInfraConstantsAdapters",),
            ".base": ("FlextInfraConstantsBase",),
            ".census": ("FlextInfraConstantsCensus",),
            ".check": ("FlextInfraConstantsCheck",),
            ".cli": ("FlextInfraConstantsCli",),
            ".codegen": ("FlextInfraConstantsCodegen",),
            ".codegen_detection": ("FlextInfraConstantsCodegenDetection",),
            ".codegen_lazy": ("FlextInfraConstantsCodegenLazy",),
            ".codegen_project": ("FlextInfraConstantsCodegenProject",),
            ".codegen_render_names": ("FlextInfraConstantsCodegenRenderNames",),
            ".deps": ("FlextInfraConstantsDeps",),
            ".detectors": ("FlextInfraConstantsDetectors",),
            ".docs": ("FlextInfraConstantsDocs",),
            ".git": ("FlextInfraConstantsGit",),
            ".github": ("FlextInfraConstantsGithub",),
            ".make": ("FlextInfraConstantsMake",),
            ".namespace": ("FlextInfraConstantsNamespace",),
            ".refactor": ("FlextInfraConstantsRefactor",),
            ".release": ("FlextInfraConstantsRelease",),
            ".rope": ("FlextInfraConstantsRope",),
            ".source_code": ("FlextInfraConstantsSourceCode",),
            ".validate": ("FlextInfraConstantsSharedInfra",),
            ".workspace": ("FlextInfraConstantsWorkspace",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
