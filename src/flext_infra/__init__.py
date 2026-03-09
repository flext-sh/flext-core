# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Public API for flext-infra.

Provides access to infrastructure services for workspace management, validation,
dependency handling, and build orchestration in the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_infra.__version__ import (
        __all__,
        __author__,
        __author_email__,
        __description__,
        __license__,
        __title__,
        __url__,
        __version__,
        __version_info__,
    )
    from flext_infra._utilities.discovery import FlextInfraUtilitiesDiscovery
    from flext_infra._utilities.git import FlextInfraUtilitiesGit
    from flext_infra._utilities.io import FlextInfraUtilitiesIo
    from flext_infra._utilities.output import FlextInfraUtilitiesOutput, output
    from flext_infra._utilities.paths import FlextInfraUtilitiesPaths
    from flext_infra._utilities.patterns import FlextInfraUtilitiesPatterns
    from flext_infra._utilities.reporting import FlextInfraUtilitiesReporting
    from flext_infra._utilities.selection import FlextInfraUtilitiesSelection
    from flext_infra._utilities.subprocess import FlextInfraUtilitiesSubprocess
    from flext_infra._utilities.templates import FlextInfraUtilitiesTemplates
    from flext_infra._utilities.terminal import FlextInfraUtilitiesTerminal
    from flext_infra._utilities.toml import (
        FlextInfraUtilitiesToml,
        array,
        as_container_list,
        as_string_list,
        as_toml_mapping,
        ensure_table,
        normalize_container_value,
        read_doc,
        table_string_keys,
        toml_get,
        unwrap_item,
    )
    from flext_infra._utilities.toml_parse import FlextInfraUtilitiesTomlParse
    from flext_infra._utilities.versioning import FlextInfraUtilitiesVersioning
    from flext_infra._utilities.yaml import FlextInfraUtilitiesYaml
    from flext_infra.basemk.engine import FlextInfraBaseMkTemplateEngine
    from flext_infra.basemk.generator import FlextInfraBaseMkGenerator
    from flext_infra.check.services import FlextInfraConfigFixer, _ProjectResult as r
    from flext_infra.check.workspace_check import (
        FlextInfraWorkspaceChecker,
        build_parser,
        run_cli,
    )
    from flext_infra.codegen.census import FlextInfraCodegenCensus
    from flext_infra.codegen.constants_quality_gate import (
        FlextInfraCodegenConstantsQualityGate,
    )
    from flext_infra.codegen.fixer import FlextInfraCodegenFixer
    from flext_infra.codegen.lazy_init import FlextInfraCodegenLazyInit
    from flext_infra.codegen.py_typed import FlextInfraCodegenPyTyped
    from flext_infra.codegen.scaffolder import FlextInfraCodegenScaffolder
    from flext_infra.codegen.transforms import FlextInfraCodegenTransforms
    from flext_infra.constants import FlextInfraConstants, c
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
    from flext_infra.deps.detection import (
        FlextInfraDependencyDetectionModels,
        FlextInfraDependencyDetectionService,
        build_project_report,
        classify_issues,
        discover_projects,
        dm,
        get_current_typings_from_pyproject,
        get_required_typings,
        load_dependency_limits,
        module_to_types_package,
        run_deptry,
        run_mypy_stub_hints,
        run_pip_check,
    )
    from flext_infra.deps.detector import (
        FlextInfraDependencyDetectorModels,
        FlextInfraRuntimeDevDependencyDetector,
        ddm,
    )
    from flext_infra.deps.extra_paths import (
        get_dep_paths,
        path_dep_paths,
        path_dep_paths_pep621,
        path_dep_paths_poetry,
        sync_extra_paths,
        sync_one,
    )
    from flext_infra.deps.internal_sync import FlextInfraInternalDependencySyncService
    from flext_infra.deps.modernizer import (
        ConsolidateGroupsPhase,
        EnsurePyreflyConfigPhase,
        EnsurePyrightConfigPhase,
        EnsurePytestConfigPhase,
        FlextInfraPyprojectModernizer,
        InjectCommentsPhase,
    )
    from flext_infra.deps.path_sync import (
        detect_mode,
        extract_dep_name,
        rewrite_dep_paths,
    )
    from flext_infra.deps.tool_config import (
        FlextInfraToolConfigDocument,
        load_tool_config,
    )
    from flext_infra.docs.auditor import FlextInfraDocAuditor
    from flext_infra.docs.builder import FlextInfraDocBuilder
    from flext_infra.docs.fixer import FlextInfraDocFixer
    from flext_infra.docs.generator import FlextInfraDocGenerator
    from flext_infra.docs.shared import FlextInfraDocsShared
    from flext_infra.docs.validator import FlextInfraDocValidator
    from flext_infra.github.linter import FlextInfraWorkflowLinter
    from flext_infra.github.pr import FlextInfraPrManager
    from flext_infra.github.pr_workspace import FlextInfraPrWorkspaceManager
    from flext_infra.github.workflows import FlextInfraWorkflowSyncer, SyncOperation
    from flext_infra.maintenance.python_version import (
        FlextInfraPythonVersionEnforcer,
        logger,
    )
    from flext_infra.models import FlextInfraModels, m
    from flext_infra.protocols import FlextInfraProtocols, p
    from flext_infra.refactor.analysis import (
        FlextInfraRefactorClassNestingAnalyzer,
        FlextInfraRefactorViolationAnalyzer,
    )
    from flext_infra.refactor.dependency_analyzer import (
        CompatibilityAliasDetector,
        CyclicImportDetector,
        DependencyAnalyzer,
        FlextInfraRefactorDependencyAnalyzerFacade,
        FutureAnnotationsDetector,
        ImportAliasDetector,
        InternalImportDetector,
        LooseObjectDetector,
        ManualProtocolDetector,
        ManualTypingAliasDetector,
        NamespaceFacadeScanner,
        RuntimeAliasDetector,
        load_python_module,
    )
    from flext_infra.refactor.engine import FlextInfraRefactorEngine
    from flext_infra.refactor.migrate_to_class_mro import (
        FlextInfraRefactorMigrateToClassMRO,
    )
    from flext_infra.refactor.mro_migrator import (
        FlextInfraRefactorMROMigrationTransformer,
    )
    from flext_infra.refactor.mro_resolver import (
        FlextInfraRefactorMROImportRewriter,
        FlextInfraRefactorMROMigrationScanner,
        FlextInfraRefactorMROResolver,
    )
    from flext_infra.refactor.namespace_enforcer import (
        FlextInfraNamespaceEnforcer,
        NamespaceEnforcementModels,
    )
    from flext_infra.refactor.namespace_rewriter import NamespaceEnforcementRewriter
    from flext_infra.refactor.output import render_namespace_enforcement_report
    from flext_infra.refactor.project_classifier import ProjectClassifier
    from flext_infra.refactor.pydantic_centralizer import (
        FlextInfraRefactorPydanticCentralizer,
    )
    from flext_infra.refactor.rule import (
        FlextInfraRefactorRule,
        FlextInfraRefactorRuleLoader,
    )
    from flext_infra.refactor.rules.class_nesting import ClassNestingRefactorRule
    from flext_infra.refactor.rules.class_reconstructor import (
        FlextInfraRefactorClassNestingReconstructor,
        FlextInfraRefactorClassReconstructorRule,
        PreCheckGate,
    )
    from flext_infra.refactor.rules.ensure_future_annotations import (
        FlextInfraRefactorEnsureFutureAnnotationsRule,
    )
    from flext_infra.refactor.rules.import_modernizer import (
        FlextInfraRefactorImportModernizerRule,
    )
    from flext_infra.refactor.rules.legacy_removal import (
        FlextInfraRefactorLegacyRemovalRule,
    )
    from flext_infra.refactor.rules.mro_class_migration import (
        FlextInfraRefactorMROClassMigrationRule,
    )
    from flext_infra.refactor.rules.mro_redundancy_checker import (
        FlextInfraRefactorMRORedundancyChecker,
    )
    from flext_infra.refactor.rules.pattern_corrections import (
        FlextInfraRefactorPatternCorrectionsRule,
    )
    from flext_infra.refactor.rules.symbol_propagation import (
        FlextInfraRefactorSignaturePropagationRule,
        FlextInfraRefactorSignaturePropagator,
        FlextInfraRefactorSymbolPropagationRule,
    )
    from flext_infra.refactor.safety import FlextInfraRefactorSafetyManager
    from flext_infra.refactor.scanner import FlextInfraRefactorLooseClassScanner
    from flext_infra.refactor.transformers.alias_remover import (
        FlextInfraRefactorAliasRemover,
    )
    from flext_infra.refactor.transformers.class_nesting import (
        FlextInfraRefactorClassNestingTransformer,
    )
    from flext_infra.refactor.transformers.class_reconstructor import (
        FlextInfraRefactorClassReconstructor,
    )
    from flext_infra.refactor.transformers.deprecated_remover import (
        FlextInfraRefactorDeprecatedRemover,
    )
    from flext_infra.refactor.transformers.helper_consolidation import (
        HelperConsolidationTransformer,
    )
    from flext_infra.refactor.transformers.import_bypass_remover import (
        FlextInfraRefactorImportBypassRemover,
    )
    from flext_infra.refactor.transformers.import_modernizer import (
        FlextInfraRefactorImportModernizer,
    )
    from flext_infra.refactor.transformers.lazy_import_fixer import (
        FlextInfraRefactorLazyImportFixer,
    )
    from flext_infra.refactor.transformers.mro_private_inline import (
        FlextInfraRefactorMROPrivateInlineTransformer,
    )
    from flext_infra.refactor.transformers.mro_reference_rewriter import (
        FlextInfraRefactorMROReferenceRewriter,
    )
    from flext_infra.refactor.transformers.mro_remover import (
        FlextInfraRefactorMRORemover,
    )
    from flext_infra.refactor.transformers.nested_class_propagation import (
        NestedClassPropagationTransformer,
    )
    from flext_infra.refactor.transformers.policy import (
        FlextInfraRefactorTransformerPolicyUtilities,
    )
    from flext_infra.refactor.transformers.symbol_propagator import (
        FlextInfraRefactorSymbolPropagator,
    )
    from flext_infra.refactor.validation import (
        FlextInfraRefactorCliSupport,
        FlextInfraRefactorMROMigrationValidator,
        FlextInfraRefactorRuleDefinitionValidator,
        PostCheckGate,
    )
    from flext_infra.release.orchestrator import FlextInfraReleaseOrchestrator
    from flext_infra.typings import FlextInfraTypes, t
    from flext_infra.utilities import FlextInfraUtilities, u
    from flext_infra.workspace.detector import (
        FlextInfraWorkspaceDetector,
        WorkspaceMode,
    )
    from flext_infra.workspace.migrator import FlextInfraProjectMigrator
    from flext_infra.workspace.orchestrator import FlextInfraOrchestratorService
    from flext_infra.workspace.sync import FlextInfraSyncService

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ClassNestingRefactorRule": (
        "flext_infra.refactor.rules.class_nesting",
        "ClassNestingRefactorRule",
    ),
    "CompatibilityAliasDetector": (
        "flext_infra.refactor.dependency_analyzer",
        "CompatibilityAliasDetector",
    ),
    "ConsolidateGroupsPhase": ("flext_infra.deps.modernizer", "ConsolidateGroupsPhase"),
    "CyclicImportDetector": (
        "flext_infra.refactor.dependency_analyzer",
        "CyclicImportDetector",
    ),
    "DependencyAnalyzer": (
        "flext_infra.refactor.dependency_analyzer",
        "DependencyAnalyzer",
    ),
    "EnsurePyreflyConfigPhase": (
        "flext_infra.deps.modernizer",
        "EnsurePyreflyConfigPhase",
    ),
    "EnsurePyrightConfigPhase": (
        "flext_infra.deps.modernizer",
        "EnsurePyrightConfigPhase",
    ),
    "EnsurePytestConfigPhase": (
        "flext_infra.deps.modernizer",
        "EnsurePytestConfigPhase",
    ),
    "FlextInfraBaseMkGenerator": (
        "flext_infra.basemk.generator",
        "FlextInfraBaseMkGenerator",
    ),
    "FlextInfraBaseMkTemplateEngine": (
        "flext_infra.basemk.engine",
        "FlextInfraBaseMkTemplateEngine",
    ),
    "FlextInfraBaseMkValidator": (
        "flext_infra.core.basemk_validator",
        "FlextInfraBaseMkValidator",
    ),
    "FlextInfraCodegenCensus": (
        "flext_infra.codegen.census",
        "FlextInfraCodegenCensus",
    ),
    "FlextInfraCodegenConstantsQualityGate": (
        "flext_infra.codegen.constants_quality_gate",
        "FlextInfraCodegenConstantsQualityGate",
    ),
    "FlextInfraCodegenFixer": ("flext_infra.codegen.fixer", "FlextInfraCodegenFixer"),
    "FlextInfraCodegenLazyInit": (
        "flext_infra.codegen.lazy_init",
        "FlextInfraCodegenLazyInit",
    ),
    "FlextInfraCodegenPyTyped": (
        "flext_infra.codegen.py_typed",
        "FlextInfraCodegenPyTyped",
    ),
    "FlextInfraCodegenScaffolder": (
        "flext_infra.codegen.scaffolder",
        "FlextInfraCodegenScaffolder",
    ),
    "FlextInfraCodegenTransforms": (
        "flext_infra.codegen.transforms",
        "FlextInfraCodegenTransforms",
    ),
    "FlextInfraConfigFixer": ("flext_infra.check.services", "FlextInfraConfigFixer"),
    "FlextInfraConstants": ("flext_infra.constants", "FlextInfraConstants"),
    "FlextInfraDependencyDetectionModels": (
        "flext_infra.deps.detection",
        "FlextInfraDependencyDetectionModels",
    ),
    "FlextInfraDependencyDetectionService": (
        "flext_infra.deps.detection",
        "FlextInfraDependencyDetectionService",
    ),
    "FlextInfraDependencyDetectorModels": (
        "flext_infra.deps.detector",
        "FlextInfraDependencyDetectorModels",
    ),
    "FlextInfraDocAuditor": ("flext_infra.docs.auditor", "FlextInfraDocAuditor"),
    "FlextInfraDocBuilder": ("flext_infra.docs.builder", "FlextInfraDocBuilder"),
    "FlextInfraDocFixer": ("flext_infra.docs.fixer", "FlextInfraDocFixer"),
    "FlextInfraDocGenerator": ("flext_infra.docs.generator", "FlextInfraDocGenerator"),
    "FlextInfraDocValidator": ("flext_infra.docs.validator", "FlextInfraDocValidator"),
    "FlextInfraDocsShared": ("flext_infra.docs.shared", "FlextInfraDocsShared"),
    "FlextInfraInternalDependencySyncService": (
        "flext_infra.deps.internal_sync",
        "FlextInfraInternalDependencySyncService",
    ),
    "FlextInfraInventoryService": (
        "flext_infra.core.inventory",
        "FlextInfraInventoryService",
    ),
    "FlextInfraModels": ("flext_infra.models", "FlextInfraModels"),
    "FlextInfraNamespaceEnforcer": (
        "flext_infra.refactor.namespace_enforcer",
        "FlextInfraNamespaceEnforcer",
    ),
    "FlextInfraNamespaceValidator": (
        "flext_infra.core.namespace_validator",
        "FlextInfraNamespaceValidator",
    ),
    "FlextInfraOrchestratorService": (
        "flext_infra.workspace.orchestrator",
        "FlextInfraOrchestratorService",
    ),
    "FlextInfraPrManager": ("flext_infra.github.pr", "FlextInfraPrManager"),
    "FlextInfraPrWorkspaceManager": (
        "flext_infra.github.pr_workspace",
        "FlextInfraPrWorkspaceManager",
    ),
    "FlextInfraProjectMigrator": (
        "flext_infra.workspace.migrator",
        "FlextInfraProjectMigrator",
    ),
    "FlextInfraProtocols": ("flext_infra.protocols", "FlextInfraProtocols"),
    "FlextInfraPyprojectModernizer": (
        "flext_infra.deps.modernizer",
        "FlextInfraPyprojectModernizer",
    ),
    "FlextInfraPytestDiagExtractor": (
        "flext_infra.core.pytest_diag",
        "FlextInfraPytestDiagExtractor",
    ),
    "FlextInfraPythonVersionEnforcer": (
        "flext_infra.maintenance.python_version",
        "FlextInfraPythonVersionEnforcer",
    ),
    "FlextInfraRefactorAliasRemover": (
        "flext_infra.refactor.transformers.alias_remover",
        "FlextInfraRefactorAliasRemover",
    ),
    "FlextInfraRefactorClassNestingAnalyzer": (
        "flext_infra.refactor.analysis",
        "FlextInfraRefactorClassNestingAnalyzer",
    ),
    "FlextInfraRefactorClassNestingReconstructor": (
        "flext_infra.refactor.rules.class_reconstructor",
        "FlextInfraRefactorClassNestingReconstructor",
    ),
    "FlextInfraRefactorClassNestingTransformer": (
        "flext_infra.refactor.transformers.class_nesting",
        "FlextInfraRefactorClassNestingTransformer",
    ),
    "FlextInfraRefactorClassReconstructor": (
        "flext_infra.refactor.transformers.class_reconstructor",
        "FlextInfraRefactorClassReconstructor",
    ),
    "FlextInfraRefactorClassReconstructorRule": (
        "flext_infra.refactor.rules.class_reconstructor",
        "FlextInfraRefactorClassReconstructorRule",
    ),
    "FlextInfraRefactorCliSupport": (
        "flext_infra.refactor.validation",
        "FlextInfraRefactorCliSupport",
    ),
    "FlextInfraRefactorDependencyAnalyzerFacade": (
        "flext_infra.refactor.dependency_analyzer",
        "FlextInfraRefactorDependencyAnalyzerFacade",
    ),
    "FlextInfraRefactorDeprecatedRemover": (
        "flext_infra.refactor.transformers.deprecated_remover",
        "FlextInfraRefactorDeprecatedRemover",
    ),
    "FlextInfraRefactorEngine": (
        "flext_infra.refactor.engine",
        "FlextInfraRefactorEngine",
    ),
    "FlextInfraRefactorEnsureFutureAnnotationsRule": (
        "flext_infra.refactor.rules.ensure_future_annotations",
        "FlextInfraRefactorEnsureFutureAnnotationsRule",
    ),
    "FlextInfraRefactorImportBypassRemover": (
        "flext_infra.refactor.transformers.import_bypass_remover",
        "FlextInfraRefactorImportBypassRemover",
    ),
    "FlextInfraRefactorImportModernizer": (
        "flext_infra.refactor.transformers.import_modernizer",
        "FlextInfraRefactorImportModernizer",
    ),
    "FlextInfraRefactorImportModernizerRule": (
        "flext_infra.refactor.rules.import_modernizer",
        "FlextInfraRefactorImportModernizerRule",
    ),
    "FlextInfraRefactorLazyImportFixer": (
        "flext_infra.refactor.transformers.lazy_import_fixer",
        "FlextInfraRefactorLazyImportFixer",
    ),
    "FlextInfraRefactorLegacyRemovalRule": (
        "flext_infra.refactor.rules.legacy_removal",
        "FlextInfraRefactorLegacyRemovalRule",
    ),
    "FlextInfraRefactorLooseClassScanner": (
        "flext_infra.refactor.scanner",
        "FlextInfraRefactorLooseClassScanner",
    ),
    "FlextInfraRefactorMROClassMigrationRule": (
        "flext_infra.refactor.rules.mro_class_migration",
        "FlextInfraRefactorMROClassMigrationRule",
    ),
    "FlextInfraRefactorMROImportRewriter": (
        "flext_infra.refactor.mro_resolver",
        "FlextInfraRefactorMROImportRewriter",
    ),
    "FlextInfraRefactorMROMigrationScanner": (
        "flext_infra.refactor.mro_resolver",
        "FlextInfraRefactorMROMigrationScanner",
    ),
    "FlextInfraRefactorMROMigrationTransformer": (
        "flext_infra.refactor.mro_migrator",
        "FlextInfraRefactorMROMigrationTransformer",
    ),
    "FlextInfraRefactorMROMigrationValidator": (
        "flext_infra.refactor.validation",
        "FlextInfraRefactorMROMigrationValidator",
    ),
    "FlextInfraRefactorMROPrivateInlineTransformer": (
        "flext_infra.refactor.transformers.mro_private_inline",
        "FlextInfraRefactorMROPrivateInlineTransformer",
    ),
    "FlextInfraRefactorMRORedundancyChecker": (
        "flext_infra.refactor.rules.mro_redundancy_checker",
        "FlextInfraRefactorMRORedundancyChecker",
    ),
    "FlextInfraRefactorMROReferenceRewriter": (
        "flext_infra.refactor.transformers.mro_reference_rewriter",
        "FlextInfraRefactorMROReferenceRewriter",
    ),
    "FlextInfraRefactorMRORemover": (
        "flext_infra.refactor.transformers.mro_remover",
        "FlextInfraRefactorMRORemover",
    ),
    "FlextInfraRefactorMROResolver": (
        "flext_infra.refactor.mro_resolver",
        "FlextInfraRefactorMROResolver",
    ),
    "FlextInfraRefactorMigrateToClassMRO": (
        "flext_infra.refactor.migrate_to_class_mro",
        "FlextInfraRefactorMigrateToClassMRO",
    ),
    "FlextInfraRefactorPatternCorrectionsRule": (
        "flext_infra.refactor.rules.pattern_corrections",
        "FlextInfraRefactorPatternCorrectionsRule",
    ),
    "FlextInfraRefactorPydanticCentralizer": (
        "flext_infra.refactor.pydantic_centralizer",
        "FlextInfraRefactorPydanticCentralizer",
    ),
    "FlextInfraRefactorRule": ("flext_infra.refactor.rule", "FlextInfraRefactorRule"),
    "FlextInfraRefactorRuleDefinitionValidator": (
        "flext_infra.refactor.validation",
        "FlextInfraRefactorRuleDefinitionValidator",
    ),
    "FlextInfraRefactorRuleLoader": (
        "flext_infra.refactor.rule",
        "FlextInfraRefactorRuleLoader",
    ),
    "FlextInfraRefactorSafetyManager": (
        "flext_infra.refactor.safety",
        "FlextInfraRefactorSafetyManager",
    ),
    "FlextInfraRefactorSignaturePropagationRule": (
        "flext_infra.refactor.rules.symbol_propagation",
        "FlextInfraRefactorSignaturePropagationRule",
    ),
    "FlextInfraRefactorSignaturePropagator": (
        "flext_infra.refactor.rules.symbol_propagation",
        "FlextInfraRefactorSignaturePropagator",
    ),
    "FlextInfraRefactorSymbolPropagationRule": (
        "flext_infra.refactor.rules.symbol_propagation",
        "FlextInfraRefactorSymbolPropagationRule",
    ),
    "FlextInfraRefactorSymbolPropagator": (
        "flext_infra.refactor.transformers.symbol_propagator",
        "FlextInfraRefactorSymbolPropagator",
    ),
    "FlextInfraRefactorTransformerPolicyUtilities": (
        "flext_infra.refactor.transformers.policy",
        "FlextInfraRefactorTransformerPolicyUtilities",
    ),
    "FlextInfraRefactorViolationAnalyzer": (
        "flext_infra.refactor.analysis",
        "FlextInfraRefactorViolationAnalyzer",
    ),
    "FlextInfraReleaseOrchestrator": (
        "flext_infra.release.orchestrator",
        "FlextInfraReleaseOrchestrator",
    ),
    "FlextInfraRuntimeDevDependencyDetector": (
        "flext_infra.deps.detector",
        "FlextInfraRuntimeDevDependencyDetector",
    ),
    "FlextInfraSkillValidator": (
        "flext_infra.core.skill_validator",
        "FlextInfraSkillValidator",
    ),
    "FlextInfraStubSupplyChain": (
        "flext_infra.core.stub_chain",
        "FlextInfraStubSupplyChain",
    ),
    "FlextInfraSyncService": ("flext_infra.workspace.sync", "FlextInfraSyncService"),
    "FlextInfraTextPatternScanner": (
        "flext_infra.core.scanner",
        "FlextInfraTextPatternScanner",
    ),
    "FlextInfraToolConfigDocument": (
        "flext_infra.deps.tool_config",
        "FlextInfraToolConfigDocument",
    ),
    "FlextInfraTypes": ("flext_infra.typings", "FlextInfraTypes"),
    "FlextInfraUtilities": ("flext_infra.utilities", "FlextInfraUtilities"),
    "FlextInfraUtilitiesDiscovery": (
        "flext_infra._utilities.discovery",
        "FlextInfraUtilitiesDiscovery",
    ),
    "FlextInfraUtilitiesGit": ("flext_infra._utilities.git", "FlextInfraUtilitiesGit"),
    "FlextInfraUtilitiesIo": ("flext_infra._utilities.io", "FlextInfraUtilitiesIo"),
    "FlextInfraUtilitiesOutput": (
        "flext_infra._utilities.output",
        "FlextInfraUtilitiesOutput",
    ),
    "FlextInfraUtilitiesPaths": (
        "flext_infra._utilities.paths",
        "FlextInfraUtilitiesPaths",
    ),
    "FlextInfraUtilitiesPatterns": (
        "flext_infra._utilities.patterns",
        "FlextInfraUtilitiesPatterns",
    ),
    "FlextInfraUtilitiesReporting": (
        "flext_infra._utilities.reporting",
        "FlextInfraUtilitiesReporting",
    ),
    "FlextInfraUtilitiesSelection": (
        "flext_infra._utilities.selection",
        "FlextInfraUtilitiesSelection",
    ),
    "FlextInfraUtilitiesSubprocess": (
        "flext_infra._utilities.subprocess",
        "FlextInfraUtilitiesSubprocess",
    ),
    "FlextInfraUtilitiesTemplates": (
        "flext_infra._utilities.templates",
        "FlextInfraUtilitiesTemplates",
    ),
    "FlextInfraUtilitiesTerminal": (
        "flext_infra._utilities.terminal",
        "FlextInfraUtilitiesTerminal",
    ),
    "FlextInfraUtilitiesToml": (
        "flext_infra._utilities.toml",
        "FlextInfraUtilitiesToml",
    ),
    "FlextInfraUtilitiesTomlParse": (
        "flext_infra._utilities.toml_parse",
        "FlextInfraUtilitiesTomlParse",
    ),
    "FlextInfraUtilitiesVersioning": (
        "flext_infra._utilities.versioning",
        "FlextInfraUtilitiesVersioning",
    ),
    "FlextInfraUtilitiesYaml": (
        "flext_infra._utilities.yaml",
        "FlextInfraUtilitiesYaml",
    ),
    "FlextInfraWorkflowLinter": (
        "flext_infra.github.linter",
        "FlextInfraWorkflowLinter",
    ),
    "FlextInfraWorkflowSyncer": (
        "flext_infra.github.workflows",
        "FlextInfraWorkflowSyncer",
    ),
    "FlextInfraWorkspaceChecker": (
        "flext_infra.check.workspace_check",
        "FlextInfraWorkspaceChecker",
    ),
    "FlextInfraWorkspaceDetector": (
        "flext_infra.workspace.detector",
        "FlextInfraWorkspaceDetector",
    ),
    "FutureAnnotationsDetector": (
        "flext_infra.refactor.dependency_analyzer",
        "FutureAnnotationsDetector",
    ),
    "HelperConsolidationTransformer": (
        "flext_infra.refactor.transformers.helper_consolidation",
        "HelperConsolidationTransformer",
    ),
    "ImportAliasDetector": (
        "flext_infra.refactor.dependency_analyzer",
        "ImportAliasDetector",
    ),
    "InjectCommentsPhase": ("flext_infra.deps.modernizer", "InjectCommentsPhase"),
    "InternalImportDetector": (
        "flext_infra.refactor.dependency_analyzer",
        "InternalImportDetector",
    ),
    "LooseObjectDetector": (
        "flext_infra.refactor.dependency_analyzer",
        "LooseObjectDetector",
    ),
    "ManualProtocolDetector": (
        "flext_infra.refactor.dependency_analyzer",
        "ManualProtocolDetector",
    ),
    "ManualTypingAliasDetector": (
        "flext_infra.refactor.dependency_analyzer",
        "ManualTypingAliasDetector",
    ),
    "NamespaceEnforcementModels": (
        "flext_infra.refactor.namespace_enforcer",
        "NamespaceEnforcementModels",
    ),
    "NamespaceEnforcementRewriter": (
        "flext_infra.refactor.namespace_rewriter",
        "NamespaceEnforcementRewriter",
    ),
    "NamespaceFacadeScanner": (
        "flext_infra.refactor.dependency_analyzer",
        "NamespaceFacadeScanner",
    ),
    "NestedClassPropagationTransformer": (
        "flext_infra.refactor.transformers.nested_class_propagation",
        "NestedClassPropagationTransformer",
    ),
    "PostCheckGate": ("flext_infra.refactor.validation", "PostCheckGate"),
    "PreCheckGate": ("flext_infra.refactor.rules.class_reconstructor", "PreCheckGate"),
    "ProjectClassifier": (
        "flext_infra.refactor.project_classifier",
        "ProjectClassifier",
    ),
    "RuntimeAliasDetector": (
        "flext_infra.refactor.dependency_analyzer",
        "RuntimeAliasDetector",
    ),
    "SyncOperation": ("flext_infra.github.workflows", "SyncOperation"),
    "WorkspaceMode": ("flext_infra.workspace.detector", "WorkspaceMode"),
    "__all__": ("flext_infra.__version__", "__all__"),
    "__author__": ("flext_infra.__version__", "__author__"),
    "__author_email__": ("flext_infra.__version__", "__author_email__"),
    "__description__": ("flext_infra.__version__", "__description__"),
    "__license__": ("flext_infra.__version__", "__license__"),
    "__title__": ("flext_infra.__version__", "__title__"),
    "__url__": ("flext_infra.__version__", "__url__"),
    "__version__": ("flext_infra.__version__", "__version__"),
    "__version_info__": ("flext_infra.__version__", "__version_info__"),
    "array": ("flext_infra._utilities.toml", "array"),
    "as_container_list": ("flext_infra._utilities.toml", "as_container_list"),
    "as_string_list": ("flext_infra._utilities.toml", "as_string_list"),
    "as_toml_mapping": ("flext_infra._utilities.toml", "as_toml_mapping"),
    "build_parser": ("flext_infra.check.workspace_check", "build_parser"),
    "build_project_report": ("flext_infra.deps.detection", "build_project_report"),
    "c": ("flext_infra.constants", "c"),
    "classify_issues": ("flext_infra.deps.detection", "classify_issues"),
    "ddm": ("flext_infra.deps.detector", "ddm"),
    "detect_mode": ("flext_infra.deps.path_sync", "detect_mode"),
    "discover_projects": ("flext_infra.deps.detection", "discover_projects"),
    "dm": ("flext_infra.deps.detection", "dm"),
    "ensure_table": ("flext_infra._utilities.toml", "ensure_table"),
    "extract_dep_name": ("flext_infra.deps.path_sync", "extract_dep_name"),
    "get_current_typings_from_pyproject": (
        "flext_infra.deps.detection",
        "get_current_typings_from_pyproject",
    ),
    "get_dep_paths": ("flext_infra.deps.extra_paths", "get_dep_paths"),
    "get_required_typings": ("flext_infra.deps.detection", "get_required_typings"),
    "load_dependency_limits": ("flext_infra.deps.detection", "load_dependency_limits"),
    "load_python_module": (
        "flext_infra.refactor.dependency_analyzer",
        "load_python_module",
    ),
    "load_tool_config": ("flext_infra.deps.tool_config", "load_tool_config"),
    "logger": ("flext_infra.maintenance.python_version", "logger"),
    "m": ("flext_infra.models", "m"),
    "module_to_types_package": (
        "flext_infra.deps.detection",
        "module_to_types_package",
    ),
    "normalize_container_value": (
        "flext_infra._utilities.toml",
        "normalize_container_value",
    ),
    "output": ("flext_infra._utilities.output", "output"),
    "p": ("flext_infra.protocols", "p"),
    "path_dep_paths": ("flext_infra.deps.extra_paths", "path_dep_paths"),
    "path_dep_paths_pep621": ("flext_infra.deps.extra_paths", "path_dep_paths_pep621"),
    "path_dep_paths_poetry": ("flext_infra.deps.extra_paths", "path_dep_paths_poetry"),
    "r": ("flext_infra.check.services", "_ProjectResult"),
    "read_doc": ("flext_infra._utilities.toml", "read_doc"),
    "render_namespace_enforcement_report": (
        "flext_infra.refactor.output",
        "render_namespace_enforcement_report",
    ),
    "rewrite_dep_paths": ("flext_infra.deps.path_sync", "rewrite_dep_paths"),
    "run_cli": ("flext_infra.check.workspace_check", "run_cli"),
    "run_deptry": ("flext_infra.deps.detection", "run_deptry"),
    "run_mypy_stub_hints": ("flext_infra.deps.detection", "run_mypy_stub_hints"),
    "run_pip_check": ("flext_infra.deps.detection", "run_pip_check"),
    "s": ("flext_infra.core.inventory", "FlextInfraInventoryService"),
    "sync_extra_paths": ("flext_infra.deps.extra_paths", "sync_extra_paths"),
    "sync_one": ("flext_infra.deps.extra_paths", "sync_one"),
    "t": ("flext_infra.typings", "t"),
    "table_string_keys": ("flext_infra._utilities.toml", "table_string_keys"),
    "toml_get": ("flext_infra._utilities.toml", "toml_get"),
    "u": ("flext_infra.utilities", "u"),
    "unwrap_item": ("flext_infra._utilities.toml", "unwrap_item"),
}

__all__ = [
    "ClassNestingRefactorRule",
    "CompatibilityAliasDetector",
    "ConsolidateGroupsPhase",
    "CyclicImportDetector",
    "DependencyAnalyzer",
    "EnsurePyreflyConfigPhase",
    "EnsurePyrightConfigPhase",
    "EnsurePytestConfigPhase",
    "FlextInfraBaseMkGenerator",
    "FlextInfraBaseMkTemplateEngine",
    "FlextInfraBaseMkValidator",
    "FlextInfraCodegenCensus",
    "FlextInfraCodegenConstantsQualityGate",
    "FlextInfraCodegenFixer",
    "FlextInfraCodegenLazyInit",
    "FlextInfraCodegenPyTyped",
    "FlextInfraCodegenScaffolder",
    "FlextInfraCodegenTransforms",
    "FlextInfraConfigFixer",
    "FlextInfraConstants",
    "FlextInfraDependencyDetectionModels",
    "FlextInfraDependencyDetectionService",
    "FlextInfraDependencyDetectorModels",
    "FlextInfraDocAuditor",
    "FlextInfraDocBuilder",
    "FlextInfraDocFixer",
    "FlextInfraDocGenerator",
    "FlextInfraDocValidator",
    "FlextInfraDocsShared",
    "FlextInfraInternalDependencySyncService",
    "FlextInfraInventoryService",
    "FlextInfraModels",
    "FlextInfraNamespaceEnforcer",
    "FlextInfraNamespaceValidator",
    "FlextInfraOrchestratorService",
    "FlextInfraPrManager",
    "FlextInfraPrWorkspaceManager",
    "FlextInfraProjectMigrator",
    "FlextInfraProtocols",
    "FlextInfraPyprojectModernizer",
    "FlextInfraPytestDiagExtractor",
    "FlextInfraPythonVersionEnforcer",
    "FlextInfraRefactorAliasRemover",
    "FlextInfraRefactorClassNestingAnalyzer",
    "FlextInfraRefactorClassNestingReconstructor",
    "FlextInfraRefactorClassNestingTransformer",
    "FlextInfraRefactorClassReconstructor",
    "FlextInfraRefactorClassReconstructorRule",
    "FlextInfraRefactorCliSupport",
    "FlextInfraRefactorDependencyAnalyzerFacade",
    "FlextInfraRefactorDeprecatedRemover",
    "FlextInfraRefactorEngine",
    "FlextInfraRefactorEnsureFutureAnnotationsRule",
    "FlextInfraRefactorImportBypassRemover",
    "FlextInfraRefactorImportModernizer",
    "FlextInfraRefactorImportModernizerRule",
    "FlextInfraRefactorLazyImportFixer",
    "FlextInfraRefactorLegacyRemovalRule",
    "FlextInfraRefactorLooseClassScanner",
    "FlextInfraRefactorMROClassMigrationRule",
    "FlextInfraRefactorMROImportRewriter",
    "FlextInfraRefactorMROMigrationScanner",
    "FlextInfraRefactorMROMigrationTransformer",
    "FlextInfraRefactorMROMigrationValidator",
    "FlextInfraRefactorMROPrivateInlineTransformer",
    "FlextInfraRefactorMRORedundancyChecker",
    "FlextInfraRefactorMROReferenceRewriter",
    "FlextInfraRefactorMRORemover",
    "FlextInfraRefactorMROResolver",
    "FlextInfraRefactorMigrateToClassMRO",
    "FlextInfraRefactorPatternCorrectionsRule",
    "FlextInfraRefactorPydanticCentralizer",
    "FlextInfraRefactorRule",
    "FlextInfraRefactorRuleDefinitionValidator",
    "FlextInfraRefactorRuleLoader",
    "FlextInfraRefactorSafetyManager",
    "FlextInfraRefactorSignaturePropagationRule",
    "FlextInfraRefactorSignaturePropagator",
    "FlextInfraRefactorSymbolPropagationRule",
    "FlextInfraRefactorSymbolPropagator",
    "FlextInfraRefactorTransformerPolicyUtilities",
    "FlextInfraRefactorViolationAnalyzer",
    "FlextInfraReleaseOrchestrator",
    "FlextInfraRuntimeDevDependencyDetector",
    "FlextInfraSkillValidator",
    "FlextInfraStubSupplyChain",
    "FlextInfraSyncService",
    "FlextInfraTextPatternScanner",
    "FlextInfraToolConfigDocument",
    "FlextInfraTypes",
    "FlextInfraUtilities",
    "FlextInfraUtilitiesDiscovery",
    "FlextInfraUtilitiesGit",
    "FlextInfraUtilitiesIo",
    "FlextInfraUtilitiesOutput",
    "FlextInfraUtilitiesPaths",
    "FlextInfraUtilitiesPatterns",
    "FlextInfraUtilitiesReporting",
    "FlextInfraUtilitiesSelection",
    "FlextInfraUtilitiesSubprocess",
    "FlextInfraUtilitiesTemplates",
    "FlextInfraUtilitiesTerminal",
    "FlextInfraUtilitiesToml",
    "FlextInfraUtilitiesTomlParse",
    "FlextInfraUtilitiesVersioning",
    "FlextInfraUtilitiesYaml",
    "FlextInfraWorkflowLinter",
    "FlextInfraWorkflowSyncer",
    "FlextInfraWorkspaceChecker",
    "FlextInfraWorkspaceDetector",
    "FutureAnnotationsDetector",
    "HelperConsolidationTransformer",
    "ImportAliasDetector",
    "InjectCommentsPhase",
    "InternalImportDetector",
    "LooseObjectDetector",
    "ManualProtocolDetector",
    "ManualTypingAliasDetector",
    "NamespaceEnforcementModels",
    "NamespaceEnforcementRewriter",
    "NamespaceFacadeScanner",
    "NestedClassPropagationTransformer",
    "PostCheckGate",
    "PreCheckGate",
    "ProjectClassifier",
    "RuntimeAliasDetector",
    "SyncOperation",
    "WorkspaceMode",
    "__all__",
    "__author__",
    "__author_email__",
    "__description__",
    "__license__",
    "__title__",
    "__url__",
    "__version__",
    "__version_info__",
    "array",
    "as_container_list",
    "as_string_list",
    "as_toml_mapping",
    "build_parser",
    "build_project_report",
    "c",
    "classify_issues",
    "ddm",
    "detect_mode",
    "discover_projects",
    "dm",
    "ensure_table",
    "extract_dep_name",
    "get_current_typings_from_pyproject",
    "get_dep_paths",
    "get_required_typings",
    "load_dependency_limits",
    "load_python_module",
    "load_tool_config",
    "logger",
    "m",
    "module_to_types_package",
    "normalize_container_value",
    "output",
    "p",
    "path_dep_paths",
    "path_dep_paths_pep621",
    "path_dep_paths_poetry",
    "r",
    "read_doc",
    "render_namespace_enforcement_report",
    "rewrite_dep_paths",
    "run_cli",
    "run_deptry",
    "run_mypy_stub_hints",
    "run_pip_check",
    "s",
    "sync_extra_paths",
    "sync_one",
    "t",
    "table_string_keys",
    "toml_get",
    "u",
    "unwrap_item",
]


def __getattr__(name: str) -> Any:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
