# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Infra. Utilities package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import _git as _git
    from . import _rope as _rope
    from ._docs_audit_detectors import FlextInfraUtilitiesDocsAuditDetectorsMixin
    from ._docs_github_links import FlextInfraUtilitiesDocsGithubLinks
    from ._docs_scope_build import FlextInfraUtilitiesDocsScopeBuildMixin
    from ._docs_scope_selection import FlextInfraUtilitiesDocsScopeSelectionMixin
    from ._git.attestation import FlextInfraUtilitiesGitAttestationMixin
    from ._git.remote import canonical_origin_remote, redact_origin_remote
    from ._git.repo import FlextInfraUtilitiesGitRepo
    from ._git.scope import FlextInfraUtilitiesGitScopeMixin
    from ._git.semantic import FlextInfraUtilitiesGitSemanticMixin
    from ._git.semantic_identity import FlextInfraUtilitiesGitSemanticIdentityMixin
    from ._git.semantic_index import FlextInfraUtilitiesGitSemanticIndexMixin
    from ._git.semantic_paths import FlextInfraUtilitiesGitSemanticPathsMixin
    from ._git.semantic_publish import FlextInfraUtilitiesGitSemanticPublishMixin
    from ._git.semantic_refs import FlextInfraUtilitiesGitSemanticRefsMixin
    from ._git.semantic_submodule import FlextInfraUtilitiesGitSemanticSubmoduleMixin
    from ._git.semantic_worktree import FlextInfraUtilitiesGitSemanticWorktreeMixin
    from ._git.worktree import FlextInfraUtilitiesGitWorktreeMixin
    from ._git.worktree_checkpoint import FlextInfraUtilitiesGitWorktreeCheckpointMixin
    from ._git.worktree_discovery import FlextInfraUtilitiesGitWorktreeDiscoveryMixin
    from ._git.worktree_io import git_stdin
    from ._git.worktree_materialization import (
        FlextInfraUtilitiesGitWorktreeMaterializationMixin,
    )
    from ._git.worktree_patch import FlextInfraUtilitiesGitWorktreePatchMixin
    from ._git.worktree_removal import FlextInfraUtilitiesGitWorktreeRemovalMixin
    from ._git.worktree_roots import FlextInfraUtilitiesGitWorktreeRootsMixin
    from ._git.worktree_status import FlextInfraUtilitiesGitWorktreeStatusMixin
    from ._github_sync import FlextInfraUtilitiesGithubSyncMixin
    from ._project_discovery_candidates import (
        FlextInfraUtilitiesProjectDiscoveryCandidatesMixin,
    )
    from ._project_discovery_shape import FlextInfraUtilitiesProjectDiscoveryShapeMixin
    from ._rope.pep695_patch import FlextInfraUtilitiesRopePep695Patch
    from ._rope_bracket_balance import FlextInfraUtilitiesRopeBracketBalanceMixin
    from ._rope_core_pymodule import FlextInfraUtilitiesRopeCorePyModuleMixin
    from ._rope_core_resources import FlextInfraUtilitiesRopeCoreResourcesMixin
    from ._rope_method_order import FlextInfraUtilitiesRopeMethodOrderMixin
    from .base import FlextInfraUtilitiesBase
    from .census import FlextInfraUtilitiesRefactorCensus
    from .codegen import FlextInfraUtilitiesCodegen
    from .deferred_self_reference_ast import (
        DeferredSelfReferenceFinding,
        collect_deferred_self_reference_findings,
    )
    from .dependencies import FlextInfraUtilitiesDependencies
    from .discovery import FlextInfraUtilitiesDiscovery
    from .docs import FlextInfraUtilitiesDocs
    from .docs_api import FlextInfraUtilitiesDocsApi
    from .docs_audit import FlextInfraUtilitiesDocsAudit
    from .docs_build import FlextInfraUtilitiesDocsBuild
    from .docs_contract import FlextInfraUtilitiesDocsContract
    from .docs_fix import FlextInfraUtilitiesDocsFix
    from .docs_generate import FlextInfraUtilitiesDocsGenerate
    from .docs_render import FlextInfraUtilitiesDocsRender
    from .docs_scope import FlextInfraUtilitiesDocsScope
    from .docs_validate import FlextInfraUtilitiesDocsValidate
    from .git import FlextInfraUtilitiesGit
    from .github import FlextInfraUtilitiesGithub
    from .log_parser import FlextInfraUtilitiesLogParser
    from .namespace import FlextInfraUtilitiesCodegenNamespace
    from .namespace_analysis import FlextInfraUtilitiesRefactorNamespaceFlext
    from .namespace_common import FlextInfraUtilitiesRefactorNamespaceCommon
    from .namespace_config import FlextInfraUtilitiesNamespaceConfig
    from .namespace_facades import FlextInfraUtilitiesRefactorNamespaceFacades
    from .namespace_moves import FlextInfraUtilitiesRefactorNamespaceMoves
    from .policy import FlextInfraUtilitiesRefactorPolicy
    from .process import FlextInfraUtilitiesProcess
    from .project_discovery import FlextInfraUtilitiesProjectDiscovery
    from .project_managed_artifacts import FlextInfraUtilitiesProjectManagedArtifacts
    from .protected_edit import FlextInfraUtilitiesProtectedEdit
    from .protected_edit_apply import FlextInfraUtilitiesProtectedEditApply
    from .protected_edit_linting import FlextInfraUtilitiesProtectedEditLinting
    from .protected_edit_preview import FlextInfraUtilitiesProtectedEditPreview
    from .protected_edit_writes import FlextInfraUtilitiesProtectedEditWrites
    from .pyproject import FlextInfraUtilitiesPyproject
    from .pyproject_conform import FlextInfraUtilitiesPyprojectConform
    from .refactor import FlextInfraUtilitiesRefactor
    from .refactor_discovery import FlextInfraUtilitiesRefactorDiscovery
    from .release import FlextInfraUtilitiesRelease
    from .repository import FlextInfraUtilitiesRepository
    from .resource_limits import FlextInfraUtilitiesResourceLimits
    from .rope_analysis import FlextInfraUtilitiesRopeAnalysis
    from .rope_analysis_introspection import (
        FlextInfraUtilitiesRopeAnalysisIntrospection,
    )
    from .rope_analysis_workspace import FlextInfraUtilitiesRopeAnalysisWorkspace
    from .rope_core import FlextInfraUtilitiesRopeCore
    from .rope_helpers import FlextInfraUtilitiesRopeHelpers
    from .rope_imports import FlextInfraUtilitiesRopeImports
    from .rope_inventory import FlextInfraUtilitiesRopeInventory
    from .rope_module_patch import FlextInfraUtilitiesRopeModulePatch
    from .rope_runtime import FlextInfraUtilitiesRopeRuntime
    from .rope_runtime_base import FlextInfraUtilitiesRopeRuntimeBase
    from .rope_runtime_modules import FlextInfraUtilitiesRopeRuntimeModules
    from .rope_runtime_refactors import FlextInfraUtilitiesRopeRuntimeRefactors
    from .rope_runtime_types import FlextInfraUtilitiesRopeRuntimeTypes
    from .rope_source import FlextInfraUtilitiesRopeSource
    from .rope_structure import FlextInfraUtilitiesRopeStructure
    from .safety import FlextInfraUtilitiesSafety
    from .silent_failure_ast import (
        collect_silent_failure_findings,
        collect_silent_failure_fixes,
    )
    from .snapshot import FlextInfraUtilitiesSnapshot
    from .versioning import FlextInfraUtilitiesVersioning
    from .workspace_fingerprint import FlextInfraUtilitiesWorkspaceFingerprint
    from .worktree_lifecycle import FlextInfraWorktreeLifecycle
    from .worktree_provisioning import FlextInfraWorktreeProvisioning
__all__: tuple[str, ...] = (
    "DeferredSelfReferenceFinding",
    "FlextInfraUtilitiesBase",
    "FlextInfraUtilitiesCodegen",
    "FlextInfraUtilitiesCodegenNamespace",
    "FlextInfraUtilitiesDependencies",
    "FlextInfraUtilitiesDiscovery",
    "FlextInfraUtilitiesDocs",
    "FlextInfraUtilitiesDocsApi",
    "FlextInfraUtilitiesDocsAudit",
    "FlextInfraUtilitiesDocsAuditDetectorsMixin",
    "FlextInfraUtilitiesDocsBuild",
    "FlextInfraUtilitiesDocsContract",
    "FlextInfraUtilitiesDocsFix",
    "FlextInfraUtilitiesDocsGenerate",
    "FlextInfraUtilitiesDocsGithubLinks",
    "FlextInfraUtilitiesDocsRender",
    "FlextInfraUtilitiesDocsScope",
    "FlextInfraUtilitiesDocsScopeBuildMixin",
    "FlextInfraUtilitiesDocsScopeSelectionMixin",
    "FlextInfraUtilitiesDocsValidate",
    "FlextInfraUtilitiesGit",
    "FlextInfraUtilitiesGitAttestationMixin",
    "FlextInfraUtilitiesGitRepo",
    "FlextInfraUtilitiesGitScopeMixin",
    "FlextInfraUtilitiesGitSemanticIdentityMixin",
    "FlextInfraUtilitiesGitSemanticIndexMixin",
    "FlextInfraUtilitiesGitSemanticMixin",
    "FlextInfraUtilitiesGitSemanticPathsMixin",
    "FlextInfraUtilitiesGitSemanticPublishMixin",
    "FlextInfraUtilitiesGitSemanticRefsMixin",
    "FlextInfraUtilitiesGitSemanticSubmoduleMixin",
    "FlextInfraUtilitiesGitSemanticWorktreeMixin",
    "FlextInfraUtilitiesGitWorktreeCheckpointMixin",
    "FlextInfraUtilitiesGitWorktreeDiscoveryMixin",
    "FlextInfraUtilitiesGitWorktreeMaterializationMixin",
    "FlextInfraUtilitiesGitWorktreeMixin",
    "FlextInfraUtilitiesGitWorktreePatchMixin",
    "FlextInfraUtilitiesGitWorktreeRemovalMixin",
    "FlextInfraUtilitiesGitWorktreeRootsMixin",
    "FlextInfraUtilitiesGitWorktreeStatusMixin",
    "FlextInfraUtilitiesGithub",
    "FlextInfraUtilitiesGithubSyncMixin",
    "FlextInfraUtilitiesLogParser",
    "FlextInfraUtilitiesNamespaceConfig",
    "FlextInfraUtilitiesProcess",
    "FlextInfraUtilitiesProjectDiscovery",
    "FlextInfraUtilitiesProjectDiscoveryCandidatesMixin",
    "FlextInfraUtilitiesProjectDiscoveryShapeMixin",
    "FlextInfraUtilitiesProjectManagedArtifacts",
    "FlextInfraUtilitiesProtectedEdit",
    "FlextInfraUtilitiesProtectedEditApply",
    "FlextInfraUtilitiesProtectedEditLinting",
    "FlextInfraUtilitiesProtectedEditPreview",
    "FlextInfraUtilitiesProtectedEditWrites",
    "FlextInfraUtilitiesPyproject",
    "FlextInfraUtilitiesPyprojectConform",
    "FlextInfraUtilitiesRefactor",
    "FlextInfraUtilitiesRefactorCensus",
    "FlextInfraUtilitiesRefactorDiscovery",
    "FlextInfraUtilitiesRefactorNamespaceCommon",
    "FlextInfraUtilitiesRefactorNamespaceFacades",
    "FlextInfraUtilitiesRefactorNamespaceFlext",
    "FlextInfraUtilitiesRefactorNamespaceMoves",
    "FlextInfraUtilitiesRefactorPolicy",
    "FlextInfraUtilitiesRelease",
    "FlextInfraUtilitiesRepository",
    "FlextInfraUtilitiesResourceLimits",
    "FlextInfraUtilitiesRopeAnalysis",
    "FlextInfraUtilitiesRopeAnalysisIntrospection",
    "FlextInfraUtilitiesRopeAnalysisWorkspace",
    "FlextInfraUtilitiesRopeBracketBalanceMixin",
    "FlextInfraUtilitiesRopeCore",
    "FlextInfraUtilitiesRopeCorePyModuleMixin",
    "FlextInfraUtilitiesRopeCoreResourcesMixin",
    "FlextInfraUtilitiesRopeHelpers",
    "FlextInfraUtilitiesRopeImports",
    "FlextInfraUtilitiesRopeInventory",
    "FlextInfraUtilitiesRopeMethodOrderMixin",
    "FlextInfraUtilitiesRopeModulePatch",
    "FlextInfraUtilitiesRopePep695Patch",
    "FlextInfraUtilitiesRopeRuntime",
    "FlextInfraUtilitiesRopeRuntimeBase",
    "FlextInfraUtilitiesRopeRuntimeModules",
    "FlextInfraUtilitiesRopeRuntimeRefactors",
    "FlextInfraUtilitiesRopeRuntimeTypes",
    "FlextInfraUtilitiesRopeSource",
    "FlextInfraUtilitiesRopeStructure",
    "FlextInfraUtilitiesSafety",
    "FlextInfraUtilitiesSnapshot",
    "FlextInfraUtilitiesVersioning",
    "FlextInfraUtilitiesWorkspaceFingerprint",
    "FlextInfraWorktreeLifecycle",
    "FlextInfraWorktreeProvisioning",
    "_git",
    "_rope",
    "canonical_origin_remote",
    "collect_deferred_self_reference_findings",
    "collect_silent_failure_findings",
    "collect_silent_failure_fixes",
    "git_stdin",
    "redact_origin_remote",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._docs_audit_detectors": ("FlextInfraUtilitiesDocsAuditDetectorsMixin",),
            "._docs_github_links": ("FlextInfraUtilitiesDocsGithubLinks",),
            "._docs_scope_build": ("FlextInfraUtilitiesDocsScopeBuildMixin",),
            "._docs_scope_selection": ("FlextInfraUtilitiesDocsScopeSelectionMixin",),
            "._git": ("_git",),
            "._git.attestation": ("FlextInfraUtilitiesGitAttestationMixin",),
            "._git.remote": ("canonical_origin_remote", "redact_origin_remote"),
            "._git.repo": ("FlextInfraUtilitiesGitRepo",),
            "._git.scope": ("FlextInfraUtilitiesGitScopeMixin",),
            "._git.semantic": ("FlextInfraUtilitiesGitSemanticMixin",),
            "._git.semantic_identity": ("FlextInfraUtilitiesGitSemanticIdentityMixin",),
            "._git.semantic_index": ("FlextInfraUtilitiesGitSemanticIndexMixin",),
            "._git.semantic_paths": ("FlextInfraUtilitiesGitSemanticPathsMixin",),
            "._git.semantic_publish": ("FlextInfraUtilitiesGitSemanticPublishMixin",),
            "._git.semantic_refs": ("FlextInfraUtilitiesGitSemanticRefsMixin",),
            "._git.semantic_submodule": (
                "FlextInfraUtilitiesGitSemanticSubmoduleMixin",
            ),
            "._git.semantic_worktree": ("FlextInfraUtilitiesGitSemanticWorktreeMixin",),
            "._git.worktree": ("FlextInfraUtilitiesGitWorktreeMixin",),
            "._git.worktree_checkpoint": (
                "FlextInfraUtilitiesGitWorktreeCheckpointMixin",
            ),
            "._git.worktree_discovery": (
                "FlextInfraUtilitiesGitWorktreeDiscoveryMixin",
            ),
            "._git.worktree_io": ("git_stdin",),
            "._git.worktree_materialization": (
                "FlextInfraUtilitiesGitWorktreeMaterializationMixin",
            ),
            "._git.worktree_patch": ("FlextInfraUtilitiesGitWorktreePatchMixin",),
            "._git.worktree_removal": ("FlextInfraUtilitiesGitWorktreeRemovalMixin",),
            "._git.worktree_roots": ("FlextInfraUtilitiesGitWorktreeRootsMixin",),
            "._git.worktree_status": ("FlextInfraUtilitiesGitWorktreeStatusMixin",),
            "._github_sync": ("FlextInfraUtilitiesGithubSyncMixin",),
            "._project_discovery_candidates": (
                "FlextInfraUtilitiesProjectDiscoveryCandidatesMixin",
            ),
            "._project_discovery_shape": (
                "FlextInfraUtilitiesProjectDiscoveryShapeMixin",
            ),
            "._rope": ("_rope",),
            "._rope.pep695_patch": ("FlextInfraUtilitiesRopePep695Patch",),
            "._rope_bracket_balance": ("FlextInfraUtilitiesRopeBracketBalanceMixin",),
            "._rope_core_pymodule": ("FlextInfraUtilitiesRopeCorePyModuleMixin",),
            "._rope_core_resources": ("FlextInfraUtilitiesRopeCoreResourcesMixin",),
            "._rope_method_order": ("FlextInfraUtilitiesRopeMethodOrderMixin",),
            ".base": ("FlextInfraUtilitiesBase",),
            ".census": ("FlextInfraUtilitiesRefactorCensus",),
            ".codegen": ("FlextInfraUtilitiesCodegen",),
            ".deferred_self_reference_ast": (
                "DeferredSelfReferenceFinding",
                "collect_deferred_self_reference_findings",
            ),
            ".dependencies": ("FlextInfraUtilitiesDependencies",),
            ".discovery": ("FlextInfraUtilitiesDiscovery",),
            ".docs": ("FlextInfraUtilitiesDocs",),
            ".docs_api": ("FlextInfraUtilitiesDocsApi",),
            ".docs_audit": ("FlextInfraUtilitiesDocsAudit",),
            ".docs_build": ("FlextInfraUtilitiesDocsBuild",),
            ".docs_contract": ("FlextInfraUtilitiesDocsContract",),
            ".docs_fix": ("FlextInfraUtilitiesDocsFix",),
            ".docs_generate": ("FlextInfraUtilitiesDocsGenerate",),
            ".docs_render": ("FlextInfraUtilitiesDocsRender",),
            ".docs_scope": ("FlextInfraUtilitiesDocsScope",),
            ".docs_validate": ("FlextInfraUtilitiesDocsValidate",),
            ".git": ("FlextInfraUtilitiesGit",),
            ".github": ("FlextInfraUtilitiesGithub",),
            ".log_parser": ("FlextInfraUtilitiesLogParser",),
            ".namespace": ("FlextInfraUtilitiesCodegenNamespace",),
            ".namespace_analysis": ("FlextInfraUtilitiesRefactorNamespaceFlext",),
            ".namespace_common": ("FlextInfraUtilitiesRefactorNamespaceCommon",),
            ".namespace_config": ("FlextInfraUtilitiesNamespaceConfig",),
            ".namespace_facades": ("FlextInfraUtilitiesRefactorNamespaceFacades",),
            ".namespace_moves": ("FlextInfraUtilitiesRefactorNamespaceMoves",),
            ".policy": ("FlextInfraUtilitiesRefactorPolicy",),
            ".process": ("FlextInfraUtilitiesProcess",),
            ".project_discovery": ("FlextInfraUtilitiesProjectDiscovery",),
            ".project_managed_artifacts": (
                "FlextInfraUtilitiesProjectManagedArtifacts",
            ),
            ".protected_edit": ("FlextInfraUtilitiesProtectedEdit",),
            ".protected_edit_apply": ("FlextInfraUtilitiesProtectedEditApply",),
            ".protected_edit_linting": ("FlextInfraUtilitiesProtectedEditLinting",),
            ".protected_edit_preview": ("FlextInfraUtilitiesProtectedEditPreview",),
            ".protected_edit_writes": ("FlextInfraUtilitiesProtectedEditWrites",),
            ".pyproject": ("FlextInfraUtilitiesPyproject",),
            ".pyproject_conform": ("FlextInfraUtilitiesPyprojectConform",),
            ".refactor": ("FlextInfraUtilitiesRefactor",),
            ".refactor_discovery": ("FlextInfraUtilitiesRefactorDiscovery",),
            ".release": ("FlextInfraUtilitiesRelease",),
            ".repository": ("FlextInfraUtilitiesRepository",),
            ".resource_limits": ("FlextInfraUtilitiesResourceLimits",),
            ".rope_analysis": ("FlextInfraUtilitiesRopeAnalysis",),
            ".rope_analysis_introspection": (
                "FlextInfraUtilitiesRopeAnalysisIntrospection",
            ),
            ".rope_analysis_workspace": ("FlextInfraUtilitiesRopeAnalysisWorkspace",),
            ".rope_core": ("FlextInfraUtilitiesRopeCore",),
            ".rope_helpers": ("FlextInfraUtilitiesRopeHelpers",),
            ".rope_imports": ("FlextInfraUtilitiesRopeImports",),
            ".rope_inventory": ("FlextInfraUtilitiesRopeInventory",),
            ".rope_module_patch": ("FlextInfraUtilitiesRopeModulePatch",),
            ".rope_runtime": ("FlextInfraUtilitiesRopeRuntime",),
            ".rope_runtime_base": ("FlextInfraUtilitiesRopeRuntimeBase",),
            ".rope_runtime_modules": ("FlextInfraUtilitiesRopeRuntimeModules",),
            ".rope_runtime_refactors": ("FlextInfraUtilitiesRopeRuntimeRefactors",),
            ".rope_runtime_types": ("FlextInfraUtilitiesRopeRuntimeTypes",),
            ".rope_source": ("FlextInfraUtilitiesRopeSource",),
            ".rope_structure": ("FlextInfraUtilitiesRopeStructure",),
            ".safety": ("FlextInfraUtilitiesSafety",),
            ".silent_failure_ast": (
                "collect_silent_failure_findings",
                "collect_silent_failure_fixes",
            ),
            ".snapshot": ("FlextInfraUtilitiesSnapshot",),
            ".versioning": ("FlextInfraUtilitiesVersioning",),
            ".workspace_fingerprint": ("FlextInfraUtilitiesWorkspaceFingerprint",),
            ".worktree_lifecycle": ("FlextInfraWorktreeLifecycle",),
            ".worktree_provisioning": ("FlextInfraWorktreeProvisioning",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
