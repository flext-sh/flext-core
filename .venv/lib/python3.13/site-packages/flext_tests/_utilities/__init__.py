# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Tests. Utilities package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import _files as _files
    from . import _matchers as _matchers
    from ._files._assertions import FlextTestsFilesAssertionsMixin
    from ._files._batch import FlextTestsFilesBatchMixin
    from ._files._comparison import FlextTestsFilesComparisonMixin
    from ._files._contexts import FlextTestsFilesContextsMixin
    from ._files._creation import FlextTestsFilesCreationMixin
    from ._files._info import FlextTestsFilesInfoMixin
    from ._files._lifecycle import FlextTestsFilesLifecycleMixin
    from ._files._reading import FlextTestsFilesReadingMixin
    from ._matchers._assertions import FlextTestsMatchersAssertionsMixin
    from ._matchers._containment import FlextTestsMatchersContainmentMixin
    from ._matchers._immutability import FlextTestsMatchersImmutabilityMixin
    from ._matchers._result import FlextTestsMatchersResultMixin
    from ._matchers._scope import FlextTestsMatchersScopeMixin
    from ._matchers._that import FlextTestsMatchersThatMixin
    from ._matchers._typeguards import FlextTestsMatchersTypeGuardsMixin
    from .container import FlextTestsContainerHelpersUtilitiesMixin
    from .files import FlextTestsFilesUtilitiesMixin
    from .fixtures_dsl import FlextTestsFixturesDSLMixin
    from .generic import FlextTestsGenericHelpersUtilitiesMixin
    from .handler import FlextTestsHandlerHelpersUtilitiesMixin
    from .make import FlextTestsMakeUtilitiesMixin
    from .make_contract import FlextTestsMakeContractUtilitiesMixin
    from .make_parsing import FlextTestsMakeParsingUtilitiesMixin
    from .make_registry import FlextTestsMakeRegistryUtilitiesMixin
    from .make_rendering import FlextTestsMakeRenderingUtilitiesMixin
    from .matchers import FlextTestsMatchersUtilities, tm
    from .payload import FlextTestsPayloadUtilities
    from .result import FlextTestsResultUtilitiesMixin
    from .settings import FlextTestsConfigHelpersUtilitiesMixin
    from .testcontext import FlextTestsTestContextUtilitiesMixin
    from .validator import FlextTestsValidatorUtilitiesMixin
    from .workspace_cleanup import FlextTestsWorkspaceCleanupUtilitiesMixin
    from .workspace_cleanup_git import FlextTestsWorkspaceCleanupGitUtilitiesMixin
    from .workspace_cleanup_inspect import (
        FlextTestsWorkspaceCleanupInspectUtilitiesMixin,
    )
    from .workspace_cleanup_paths import FlextTestsWorkspaceCleanupPathsUtilitiesMixin
    from .workspace_cleanup_plan import FlextTestsWorkspaceCleanupPlanUtilitiesMixin
__all__: tuple[str, ...] = (
    "FlextTestsConfigHelpersUtilitiesMixin",
    "FlextTestsContainerHelpersUtilitiesMixin",
    "FlextTestsFilesAssertionsMixin",
    "FlextTestsFilesBatchMixin",
    "FlextTestsFilesComparisonMixin",
    "FlextTestsFilesContextsMixin",
    "FlextTestsFilesCreationMixin",
    "FlextTestsFilesInfoMixin",
    "FlextTestsFilesLifecycleMixin",
    "FlextTestsFilesReadingMixin",
    "FlextTestsFilesUtilitiesMixin",
    "FlextTestsFixturesDSLMixin",
    "FlextTestsGenericHelpersUtilitiesMixin",
    "FlextTestsHandlerHelpersUtilitiesMixin",
    "FlextTestsMakeContractUtilitiesMixin",
    "FlextTestsMakeParsingUtilitiesMixin",
    "FlextTestsMakeRegistryUtilitiesMixin",
    "FlextTestsMakeRenderingUtilitiesMixin",
    "FlextTestsMakeUtilitiesMixin",
    "FlextTestsMatchersAssertionsMixin",
    "FlextTestsMatchersContainmentMixin",
    "FlextTestsMatchersImmutabilityMixin",
    "FlextTestsMatchersResultMixin",
    "FlextTestsMatchersScopeMixin",
    "FlextTestsMatchersThatMixin",
    "FlextTestsMatchersTypeGuardsMixin",
    "FlextTestsMatchersUtilities",
    "FlextTestsPayloadUtilities",
    "FlextTestsResultUtilitiesMixin",
    "FlextTestsTestContextUtilitiesMixin",
    "FlextTestsValidatorUtilitiesMixin",
    "FlextTestsWorkspaceCleanupGitUtilitiesMixin",
    "FlextTestsWorkspaceCleanupInspectUtilitiesMixin",
    "FlextTestsWorkspaceCleanupPathsUtilitiesMixin",
    "FlextTestsWorkspaceCleanupPlanUtilitiesMixin",
    "FlextTestsWorkspaceCleanupUtilitiesMixin",
    "_files",
    "_matchers",
    "tm",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._files": ("_files",),
            "._files._assertions": ("FlextTestsFilesAssertionsMixin",),
            "._files._batch": ("FlextTestsFilesBatchMixin",),
            "._files._comparison": ("FlextTestsFilesComparisonMixin",),
            "._files._contexts": ("FlextTestsFilesContextsMixin",),
            "._files._creation": ("FlextTestsFilesCreationMixin",),
            "._files._info": ("FlextTestsFilesInfoMixin",),
            "._files._lifecycle": ("FlextTestsFilesLifecycleMixin",),
            "._files._reading": ("FlextTestsFilesReadingMixin",),
            "._matchers": ("_matchers",),
            "._matchers._assertions": ("FlextTestsMatchersAssertionsMixin",),
            "._matchers._containment": ("FlextTestsMatchersContainmentMixin",),
            "._matchers._immutability": ("FlextTestsMatchersImmutabilityMixin",),
            "._matchers._result": ("FlextTestsMatchersResultMixin",),
            "._matchers._scope": ("FlextTestsMatchersScopeMixin",),
            "._matchers._that": ("FlextTestsMatchersThatMixin",),
            "._matchers._typeguards": ("FlextTestsMatchersTypeGuardsMixin",),
            ".container": ("FlextTestsContainerHelpersUtilitiesMixin",),
            ".files": ("FlextTestsFilesUtilitiesMixin",),
            ".fixtures_dsl": ("FlextTestsFixturesDSLMixin",),
            ".generic": ("FlextTestsGenericHelpersUtilitiesMixin",),
            ".handler": ("FlextTestsHandlerHelpersUtilitiesMixin",),
            ".make": ("FlextTestsMakeUtilitiesMixin",),
            ".make_contract": ("FlextTestsMakeContractUtilitiesMixin",),
            ".make_parsing": ("FlextTestsMakeParsingUtilitiesMixin",),
            ".make_registry": ("FlextTestsMakeRegistryUtilitiesMixin",),
            ".make_rendering": ("FlextTestsMakeRenderingUtilitiesMixin",),
            ".matchers": ("FlextTestsMatchersUtilities", "tm"),
            ".payload": ("FlextTestsPayloadUtilities",),
            ".result": ("FlextTestsResultUtilitiesMixin",),
            ".settings": ("FlextTestsConfigHelpersUtilitiesMixin",),
            ".testcontext": ("FlextTestsTestContextUtilitiesMixin",),
            ".validator": ("FlextTestsValidatorUtilitiesMixin",),
            ".workspace_cleanup": ("FlextTestsWorkspaceCleanupUtilitiesMixin",),
            ".workspace_cleanup_git": ("FlextTestsWorkspaceCleanupGitUtilitiesMixin",),
            ".workspace_cleanup_inspect": (
                "FlextTestsWorkspaceCleanupInspectUtilitiesMixin",
            ),
            ".workspace_cleanup_paths": (
                "FlextTestsWorkspaceCleanupPathsUtilitiesMixin",
            ),
            ".workspace_cleanup_plan": (
                "FlextTestsWorkspaceCleanupPlanUtilitiesMixin",
            ),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
