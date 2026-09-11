# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Tests. Models package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from .base import FlextTestsBaseModelsMixin
    from .batch import FlextTestsBatchModelsMixin
    from .docker import FlextTestsDockerModelsMixin
    from .domains import FlextTestsDomainModelsMixin
    from .filesystem import FlextTestsFilesystemModelsMixin
    from .make import FlextTestsMakeModelsMixin
    from .matchers import FlextTestsMatchersModelsMixin
    from .validator import FlextTestsValidatorModelsMixin
    from .workspace_cleanup import FlextTestsWorkspaceCleanupModelsMixin
__all__: tuple[str, ...] = (
    "FlextTestsBaseModelsMixin",
    "FlextTestsBatchModelsMixin",
    "FlextTestsDockerModelsMixin",
    "FlextTestsDomainModelsMixin",
    "FlextTestsFilesystemModelsMixin",
    "FlextTestsMakeModelsMixin",
    "FlextTestsMatchersModelsMixin",
    "FlextTestsValidatorModelsMixin",
    "FlextTestsWorkspaceCleanupModelsMixin",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".base": ("FlextTestsBaseModelsMixin",),
            ".batch": ("FlextTestsBatchModelsMixin",),
            ".docker": ("FlextTestsDockerModelsMixin",),
            ".domains": ("FlextTestsDomainModelsMixin",),
            ".filesystem": ("FlextTestsFilesystemModelsMixin",),
            ".make": ("FlextTestsMakeModelsMixin",),
            ".matchers": ("FlextTestsMatchersModelsMixin",),
            ".validator": ("FlextTestsValidatorModelsMixin",),
            ".workspace_cleanup": ("FlextTestsWorkspaceCleanupModelsMixin",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
