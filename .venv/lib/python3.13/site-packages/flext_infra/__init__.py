# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Infra package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

from .__version__ import __author__ as __author__
from .__version__ import __author_email__ as __author_email__
from .__version__ import __description__ as __description__
from .__version__ import __license__ as __license__
from .__version__ import __title__ as __title__
from .__version__ import __url__ as __url__
from .__version__ import __version__ as __version__
from .__version__ import __version_info__ as __version_info__

if TYPE_CHECKING:
    from . import check as check
    from . import codegen as codegen
    from . import codemod as codemod
    from . import deps as deps
    from . import detectors as detectors
    from . import docs as docs
    from . import fixers as fixers
    from . import gates as gates
    from . import maintenance as maintenance
    from . import refactor as refactor
    from . import release as release
    from . import services as services
    from . import transformers as transformers
    from . import validate as validate
    from . import workspace as workspace
    from flext_cli import d, e, h, r, x

    from ._config import config
    from ._settings import settings
    from .api import FlextInfra, infra
    from .base import FlextInfraServiceBase, FlextInfraServiceBase as s
    from .base_selection import FlextInfraProjectSelectionServiceBase
    from .check.workspace_check import FlextInfraWorkspaceChecker
    from .check.workspace_check_gates import (
        FlextInfraGateRegistry,
        FlextInfraWorkspaceCheckGatesMixin,
    )
    from .cli import FlextInfraCli, docs_main, main
    from .codegen.census import FlextInfraCodegenCensus
    from .codegen.codegen_generation import FlextInfraCodegenGeneration
    from .codegen.conform import FlextInfraCodegenConform
    from .codegen.consolidator import FlextInfraCodegenConsolidator
    from .codegen.constants_quality_gate import FlextInfraCodegenQualityGate
    from .codegen.fixer import FlextInfraCodegenFixer
    from .codegen.layout import FlextInfraCodegenLayout
    from .codegen.lazy_init import FlextInfraCodegenLazyInit
    from .codegen.lazy_init_planner import FlextInfraCodegenLazyInitPlanner
    from .codegen.make_bootstrap import FlextInfraCodegenMakeBootstrap
    from .codegen.managed_conflicts import FlextInfraCodegenManagedConflicts
    from .codegen.managed_conflicts_bootstrap import (
        ManagedConflictBootstrapError,
        prepare_managed_conflicts,
    )
    from .codegen.managed_conflicts_core import (
        ManagedConflictError,
        recover_managed_toml,
    )
    from .codegen.mise_artifacts import FlextInfraCodegenMiseArtifacts
    from .codegen.pipeline import FlextInfraCodegenPipeline
    from .codegen.project_new import FlextInfraCodegenProjectNew
    from .codegen.py_typed import FlextInfraCodegenPyTyped
    from .codegen.scaffolder import FlextInfraCodegenScaffolder
    from .codegen.version_file import FlextInfraCodegenVersionFile
    from .codemod.batch_apply import FlextInfraCodemodBatchApply
    from .codemod.batch_gates import (
        FlextInfraModGateEngine,
        FlextInfraModGateSnapshot,
        FlextInfraModScanReport,
    )
    from .codemod.discovery import discover_rule_ids, discover_rules
    from .constants import FlextInfraConstants, FlextInfraConstants as c
    from .deps.detection import FlextInfraDependencyDetectionService
    from .deps.detection_analysis import FlextInfraDependencyDetectionAnalysis
    from .deps.detector import FlextInfraRuntimeDevDependencyDetector
    from .deps.detector_runtime import FlextInfraDependencyDetectorRuntime
    from .deps.extra_paths import FlextInfraExtraPathsManager
    from .deps.fix_pyrefly_config import FlextInfraConfigFixer
    from .deps.modernizer import FlextInfraPyprojectModernizer
    from .deps.phases.consolidate_groups import FlextInfraConsolidateGroupsPhase
    from .deps.phases.ensure_coverage import FlextInfraEnsureCoverageConfigPhase
    from .deps.phases.ensure_formatting import FlextInfraEnsureFormattingToolingPhase
    from .deps.phases.ensure_mypy import FlextInfraEnsureMypyConfigPhase
    from .deps.phases.ensure_namespace import FlextInfraEnsureNamespaceToolingPhase
    from .deps.phases.ensure_packaging import FlextInfraEnsurePackagingPhase
    from .deps.phases.ensure_pydantic_mypy import (
        FlextInfraEnsurePydanticMypyConfigPhase,
    )
    from .deps.phases.ensure_pyrefly import FlextInfraEnsurePyreflyConfigPhase
    from .deps.phases.ensure_pyright import FlextInfraEnsurePyrightConfigPhase
    from .deps.phases.ensure_pytest import FlextInfraEnsurePytestConfigPhase
    from .deps.phases.ensure_ruff import FlextInfraEnsureRuffConfigPhase
    from .deps.phases.ensure_vulture import FlextInfraEnsureVultureConfigPhase
    from .deps.phases.inject_comments import FlextInfraInjectCommentsPhase
    from .deps.toml_phase import FlextInfraTomlPhaseService
    from .detectors.class_placement_detector import FlextInfraClassPlacementDetector
    from .detectors.compatibility_alias_detector import (
        FlextInfraCompatibilityAliasDetector,
    )
    from .detectors.cyclic_import_detector import FlextInfraCyclicImportDetector
    from .detectors.deferred_self_reference_detector import (
        FlextInfraDeferredSelfReferenceDetector,
    )
    from .detectors.future_annotations_detector import (
        FlextInfraFutureAnnotationsDetector,
    )
    from .detectors.import_alias_detector import FlextInfraImportAliasDetector
    from .detectors.inline_import_detector import FlextInfraInlineImportDetector
    from .detectors.internal_import_detector import FlextInfraInternalImportDetector
    from .detectors.loose_object_detector import FlextInfraLooseObjectDetector
    from .detectors.loose_test_function_detector import (
        FlextInfraLooseTestFunctionDetector,
    )
    from .detectors.manual_protocol_detector import FlextInfraManualProtocolDetector
    from .detectors.manual_typing_alias_detector import (
        FlextInfraManualTypingAliasDetector,
    )
    from .detectors.namespace_source_detector import FlextInfraNamespaceSourceDetector
    from .detectors.private_import_bypass_detector import (
        FlextInfraPrivateImportBypassDetector,
    )
    from .detectors.runtime_alias_detector import FlextInfraRuntimeAliasDetector
    from .detectors.silent_failure_detector import FlextInfraSilentFailureDetector
    from .docs.auditor import FlextInfraDocAuditor
    from .docs.auditor_mixin import FlextInfraDocAuditorMixin
    from .docs.base import FlextInfraDocServiceBase
    from .docs.builder import FlextInfraDocBuilder
    from .docs.fixer import FlextInfraDocFixer
    from .docs.generator import FlextInfraDocGenerator
    from .docs.server import FlextInfraDocServer
    from .docs.validator import FlextInfraDocValidator
    from .fixers.base import FlextInfraFixerAdapter
    from .fixers.gate_fixer import FlextInfraGateFixerAdapter
    from .fixers.orchestrator import FlextInfraEnforcementFixerOrchestrator
    from .fixers.transformer_fixer import FlextInfraTransformerFixerAdapter
    from .gates.abstraction_boundary import FlextInfraAbstractionBoundaryGate
    from .gates.bandit import FlextInfraBanditGate
    from .gates.base_gate import FlextInfraGate, FlextInfraScannerGateMixin
    from .gates.canonical_alias import FlextInfraCanonicalAliasGate
    from .gates.deferred_self_reference import FlextInfraDeferredSelfReferenceGate
    from .gates.direnv import FlextInfraDirenvGate
    from .gates.duplication import FlextInfraDuplicationGate
    from .gates.layout import FlextInfraLayoutGate
    from .gates.loc_cap import FlextInfraLocCapGate
    from .gates.markdown import FlextInfraMarkdownGate
    from .gates.mypy import FlextInfraMypyGate
    from .gates.namespace import FlextInfraNamespaceGate
    from .gates.pyrefly import FlextInfraPyreflyGate
    from .gates.pyright import FlextInfraPyrightGate
    from .gates.ruff_format import FlextInfraRuffFormatGate
    from .gates.ruff_lint import FlextInfraRuffLintGate
    from .gates.runtime_census import FlextInfraRuntimeCensusGate
    from .gates.silent_failure import FlextInfraSilentFailureGate
    from .gates.smells import FlextInfraSmellsGate
    from .gates.tier_whitelist import FlextInfraTierWhitelistGate
    from .git import FlextInfraGitService
    from .iteration import FlextInfraUtilitiesIteration
    from .maintenance.clean import FlextInfraCleanService
    from .maintenance.python_version import FlextInfraPythonVersionEnforcer
    from .models import FlextInfraModels, FlextInfraModels as m
    from .protocols import (
        FlextInfraProtocols,
        FlextInfraProtocols as p,
        FlextInfraProtocolsBase,
    )
    from .refactor.accessor_migration import FlextInfraAccessorMigrationOrchestrator
    from .refactor.census import FlextInfraRefactorCensus
    from .refactor.class_nesting_analyzer import FlextInfraRefactorClassNestingAnalyzer
    from .refactor.classvar_constant_autofix import (
        FlextInfraRefactorClassvarConstantAutofix,
    )
    from .refactor.file_executor import (
        FlextInfraClassNestingPostCheckGate,
        FlextInfraRefactorFileExecutor,
    )
    from .refactor.legacy_text_ops import FlextInfraRefactorLegacyTextOps
    from .refactor.loader import FlextInfraRefactorRuleLoader
    from .refactor.modernize_orchestrator import FlextInfraModernizeOrchestrator
    from .refactor.namespace_enforcer import FlextInfraNamespaceEnforcer
    from .refactor.namespace_enforcer_phases import (
        FlextInfraNamespaceEnforcerPhasesMixin,
    )
    from .refactor.orchestrator import FlextInfraRefactorOrchestrator
    from .refactor.project_classifier import FlextInfraProjectClassifier
    from .refactor.safety import FlextInfraRefactorSafetyManager
    from .refactor.scanner import FlextInfraRefactorLooseClassScanner
    from .refactor.service import FlextInfraRefactorService
    from .refactor.text_executor import FlextInfraRefactorTextExecutor
    from .refactor.violation_analyzer import FlextInfraRefactorViolationAnalyzer
    from .refactor.wrapper_root_namespace import FlextInfraWrapperRootNamespaceRefactor
    from .release.orchestrator import FlextInfraReleaseOrchestrator
    from .release.orchestrator_phases import FlextInfraReleaseOrchestratorPhases
    from .services.cli_dispatch import CliDispatchService
    from .services.cli_route_base import CliRouteBase
    from .services.cli_routes import CliRouteService
    from .services.cli_routes_codegen import CodegenRoutes
    from .services.cli_routes_refactor import RefactorRoutes
    from .services.cli_routes_validate import ValidationRoutes
    from .services.cli_routes_validate_commands import ValidationCommandRoutes
    from .services.cli_routes_workspace import WorkspaceRoutes
    from .services.codegen import FlextInfraCodegen
    from .transformers.base import (
        FlextInfraChangeTrackingTransformer,
        FlextInfraRopeTransformer,
    )
    from .transformers.cast_remover import FlextInfraRefactorCastRemover
    from .transformers.census_visitors import (
        FlextInfraCensusImportDiscoveryVisitor,
        FlextInfraCensusUsageCollector,
    )
    from .transformers.class_nesting import FlextInfraRefactorClassNestingTransformer
    from .transformers.class_reconstructor import FlextInfraRefactorClassReconstructor
    from .transformers.cli_modernizer import FlextInfraRefactorCliModernizer
    from .transformers.compatibility_alias import FlextInfraRefactorCompatibilityAlias
    from .transformers.deprecated_remover import FlextInfraRefactorDeprecatedRemover
    from .transformers.future_import import FlextInfraRefactorFutureImport
    from .transformers.hardcoded_version import FlextInfraRefactorHardcodedVersion
    from .transformers.helper_consolidation import (
        FlextInfraHelperConsolidationTransformer,
    )
    from .transformers.import_bypass_remover import (
        FlextInfraRefactorImportBypassRemover,
    )
    from .transformers.import_modernizer import FlextInfraRefactorImportModernizer
    from .transformers.lazy_import_fixer import FlextInfraRefactorLazyImportFixer
    from .transformers.logging_modernizer import FlextInfraRefactorLoggingModernizer
    from .transformers.nested_class_propagation import (
        FlextInfraNestedClassPropagationTransformer,
    )
    from .transformers.open_encoding import FlextInfraRefactorOpenEncoding
    from .transformers.pattern import FlextInfraRefactorPatternTransformer
    from .transformers.pattern_modernizer import FlextInfraRefactorPatternModernizer
    from .transformers.project_alias_migrator import (
        FlextInfraRefactorProjectAliasMigrator,
    )
    from .transformers.pydantic_modernizer import FlextInfraRefactorPydanticModernizer
    from .transformers.result_di_modernizer import FlextInfraRefactorResultDiModernizer
    from .transformers.signature_propagator import FlextInfraRefactorSignaturePropagator
    from .transformers.smells.base import (
        FlextInfraSmellFixer,
        auto_fixable_smell_tags,
        register_smell_fixer,
        smell_fixer_for,
    )
    from .transformers.smells.boolean_logic import FlextInfraBooleanLogicFixer
    from .transformers.symbol_propagator import FlextInfraRefactorSymbolPropagator
    from .transformers.tier0_import_fixer import FlextInfraTransformerTier0ImportFixer
    from .transformers.typing_dict_attr import FlextInfraRefactorTypingDictAttr
    from .transformers.typing_dict_import import FlextInfraRefactorTypingDictImport
    from .transformers.typing_unifier import FlextInfraRefactorTypingUnifier
    from .transformers.violation_census_visitor import FlextInfraViolationCensusVisitor
    from .typings import FlextInfraTypes, FlextInfraTypes as t
    from .utilities import FlextInfraUtilities, FlextInfraUtilities as u
    from .validate.cprofile_report import FlextInfraCProfileReport
    from .validate.fresh_import import FlextInfraValidateFreshImport
    from .validate.gate_contract import FlextInfraGateContractValidator
    from .validate.gate_contract_checks import FlextInfraGateContractChecksMixin
    from .validate.gate_contract_content import FlextInfraGateContractContentMixin
    from .validate.gate_contract_errors import (
        GateContractInfraError,
        GateContractUsageError,
    )
    from .validate.gate_contract_report import FlextInfraGateContractReportMixin
    from .validate.gate_contract_scan import FlextInfraGateContractScanMixin
    from .validate.import_cycles import FlextInfraValidateImportCycles
    from .validate.inventory import FlextInfraInventoryService
    from .validate.lazy_map_freshness import FlextInfraValidateLazyMapFreshness
    from .validate.loc_delta import FlextInfraLocDeltaValidator
    from .validate.manual_command import FlextInfraManualCommandValidator
    from .validate.metadata_discipline import FlextInfraValidateMetadataDiscipline
    from .validate.namespace_rules import FlextInfraNamespaceRules
    from .validate.namespace_validator import FlextInfraNamespaceValidator
    from .validate.pytest_diag import FlextInfraPytestDiagExtractor
    from .validate.pytest_runner import FlextInfraPytestRunner
    from .validate.pytest_selector import FlextInfraPytestSelectorValidator
    from .validate.runtime_census import FlextInfraRuntimeCensusValidator
    from .validate.scanner import FlextInfraTextPatternScanner
    from .validate.silent_failure import FlextInfraSilentFailureValidator
    from .validate.skill_validator import FlextInfraSkillValidator
    from .validate.stub_chain import FlextInfraStubSupplyChain
    from .validate.testmon_db import (
        FlextInfraTestmonCacheState,
        FlextInfraTestmonDbInspector,
    )
    from .validate.tier_whitelist import FlextInfraValidateTierWhitelist
    from .workspace.detector import FlextInfraWorkspaceDetector
    from .workspace.environment_contracts import envrc_contract_violations
    from .workspace.environment_provenance import (
        FlextInfraWorkspaceEnvironmentProvenance,
    )
    from .workspace.flext_binding import FlextInfraFlextBindingService
    from .workspace.orchestrator import FlextInfraOrchestratorService
    from .workspace.rope import FlextInfraRopeWorkspace
    from .workspace.sandbox_orchestrator import FlextInfraSandboxOrchestrator
    from .worktree import FlextInfraWorktreeService
__all__: tuple[str, ...] = (
    "CliDispatchService",
    "CliRouteBase",
    "CliRouteService",
    "CodegenRoutes",
    "FlextInfra",
    "FlextInfraAbstractionBoundaryGate",
    "FlextInfraAccessorMigrationOrchestrator",
    "FlextInfraBanditGate",
    "FlextInfraBooleanLogicFixer",
    "FlextInfraCProfileReport",
    "FlextInfraCanonicalAliasGate",
    "FlextInfraCensusImportDiscoveryVisitor",
    "FlextInfraCensusUsageCollector",
    "FlextInfraChangeTrackingTransformer",
    "FlextInfraClassNestingPostCheckGate",
    "FlextInfraClassPlacementDetector",
    "FlextInfraCleanService",
    "FlextInfraCli",
    "FlextInfraCodegen",
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
    "FlextInfraCodemodBatchApply",
    "FlextInfraCompatibilityAliasDetector",
    "FlextInfraConfigFixer",
    "FlextInfraConsolidateGroupsPhase",
    "FlextInfraConstants",
    "FlextInfraCyclicImportDetector",
    "FlextInfraDeferredSelfReferenceDetector",
    "FlextInfraDeferredSelfReferenceGate",
    "FlextInfraDependencyDetectionAnalysis",
    "FlextInfraDependencyDetectionService",
    "FlextInfraDependencyDetectorRuntime",
    "FlextInfraDirenvGate",
    "FlextInfraDocAuditor",
    "FlextInfraDocAuditorMixin",
    "FlextInfraDocBuilder",
    "FlextInfraDocFixer",
    "FlextInfraDocGenerator",
    "FlextInfraDocServer",
    "FlextInfraDocServiceBase",
    "FlextInfraDocValidator",
    "FlextInfraDuplicationGate",
    "FlextInfraEnforcementFixerOrchestrator",
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
    "FlextInfraFixerAdapter",
    "FlextInfraFlextBindingService",
    "FlextInfraFutureAnnotationsDetector",
    "FlextInfraGate",
    "FlextInfraGateContractChecksMixin",
    "FlextInfraGateContractContentMixin",
    "FlextInfraGateContractReportMixin",
    "FlextInfraGateContractScanMixin",
    "FlextInfraGateContractValidator",
    "FlextInfraGateFixerAdapter",
    "FlextInfraGateRegistry",
    "FlextInfraGitService",
    "FlextInfraHelperConsolidationTransformer",
    "FlextInfraImportAliasDetector",
    "FlextInfraInjectCommentsPhase",
    "FlextInfraInlineImportDetector",
    "FlextInfraInternalImportDetector",
    "FlextInfraInventoryService",
    "FlextInfraLayoutGate",
    "FlextInfraLocCapGate",
    "FlextInfraLocDeltaValidator",
    "FlextInfraLooseObjectDetector",
    "FlextInfraLooseTestFunctionDetector",
    "FlextInfraManualCommandValidator",
    "FlextInfraManualProtocolDetector",
    "FlextInfraManualTypingAliasDetector",
    "FlextInfraMarkdownGate",
    "FlextInfraModGateEngine",
    "FlextInfraModGateSnapshot",
    "FlextInfraModScanReport",
    "FlextInfraModels",
    "FlextInfraModernizeOrchestrator",
    "FlextInfraMypyGate",
    "FlextInfraNamespaceEnforcer",
    "FlextInfraNamespaceEnforcerPhasesMixin",
    "FlextInfraNamespaceGate",
    "FlextInfraNamespaceRules",
    "FlextInfraNamespaceSourceDetector",
    "FlextInfraNamespaceValidator",
    "FlextInfraNestedClassPropagationTransformer",
    "FlextInfraOrchestratorService",
    "FlextInfraPrivateImportBypassDetector",
    "FlextInfraProjectClassifier",
    "FlextInfraProjectSelectionServiceBase",
    "FlextInfraProtocols",
    "FlextInfraProtocolsBase",
    "FlextInfraPyprojectModernizer",
    "FlextInfraPyreflyGate",
    "FlextInfraPyrightGate",
    "FlextInfraPytestDiagExtractor",
    "FlextInfraPytestRunner",
    "FlextInfraPytestSelectorValidator",
    "FlextInfraPythonVersionEnforcer",
    "FlextInfraRefactorCastRemover",
    "FlextInfraRefactorCensus",
    "FlextInfraRefactorClassNestingAnalyzer",
    "FlextInfraRefactorClassNestingTransformer",
    "FlextInfraRefactorClassReconstructor",
    "FlextInfraRefactorClassvarConstantAutofix",
    "FlextInfraRefactorCliModernizer",
    "FlextInfraRefactorCompatibilityAlias",
    "FlextInfraRefactorDeprecatedRemover",
    "FlextInfraRefactorFileExecutor",
    "FlextInfraRefactorFutureImport",
    "FlextInfraRefactorHardcodedVersion",
    "FlextInfraRefactorImportBypassRemover",
    "FlextInfraRefactorImportModernizer",
    "FlextInfraRefactorLazyImportFixer",
    "FlextInfraRefactorLegacyTextOps",
    "FlextInfraRefactorLoggingModernizer",
    "FlextInfraRefactorLooseClassScanner",
    "FlextInfraRefactorOpenEncoding",
    "FlextInfraRefactorOrchestrator",
    "FlextInfraRefactorPatternModernizer",
    "FlextInfraRefactorPatternTransformer",
    "FlextInfraRefactorProjectAliasMigrator",
    "FlextInfraRefactorPydanticModernizer",
    "FlextInfraRefactorResultDiModernizer",
    "FlextInfraRefactorRuleLoader",
    "FlextInfraRefactorSafetyManager",
    "FlextInfraRefactorService",
    "FlextInfraRefactorSignaturePropagator",
    "FlextInfraRefactorSymbolPropagator",
    "FlextInfraRefactorTextExecutor",
    "FlextInfraRefactorTypingDictAttr",
    "FlextInfraRefactorTypingDictImport",
    "FlextInfraRefactorTypingUnifier",
    "FlextInfraRefactorViolationAnalyzer",
    "FlextInfraReleaseOrchestrator",
    "FlextInfraReleaseOrchestratorPhases",
    "FlextInfraRopeTransformer",
    "FlextInfraRopeWorkspace",
    "FlextInfraRuffFormatGate",
    "FlextInfraRuffLintGate",
    "FlextInfraRuntimeAliasDetector",
    "FlextInfraRuntimeCensusGate",
    "FlextInfraRuntimeCensusValidator",
    "FlextInfraRuntimeDevDependencyDetector",
    "FlextInfraSandboxOrchestrator",
    "FlextInfraScannerGateMixin",
    "FlextInfraServiceBase",
    "FlextInfraSilentFailureDetector",
    "FlextInfraSilentFailureGate",
    "FlextInfraSilentFailureValidator",
    "FlextInfraSkillValidator",
    "FlextInfraSmellFixer",
    "FlextInfraSmellsGate",
    "FlextInfraStubSupplyChain",
    "FlextInfraTestmonCacheState",
    "FlextInfraTestmonDbInspector",
    "FlextInfraTextPatternScanner",
    "FlextInfraTierWhitelistGate",
    "FlextInfraTomlPhaseService",
    "FlextInfraTransformerFixerAdapter",
    "FlextInfraTransformerTier0ImportFixer",
    "FlextInfraTypes",
    "FlextInfraUtilities",
    "FlextInfraUtilitiesIteration",
    "FlextInfraValidateFreshImport",
    "FlextInfraValidateImportCycles",
    "FlextInfraValidateLazyMapFreshness",
    "FlextInfraValidateMetadataDiscipline",
    "FlextInfraValidateTierWhitelist",
    "FlextInfraViolationCensusVisitor",
    "FlextInfraWorkspaceCheckGatesMixin",
    "FlextInfraWorkspaceChecker",
    "FlextInfraWorkspaceDetector",
    "FlextInfraWorkspaceEnvironmentProvenance",
    "FlextInfraWorktreeService",
    "FlextInfraWrapperRootNamespaceRefactor",
    "GateContractInfraError",
    "GateContractUsageError",
    "ManagedConflictBootstrapError",
    "ManagedConflictError",
    "RefactorRoutes",
    "ValidationCommandRoutes",
    "ValidationRoutes",
    "WorkspaceRoutes",
    "__author__",
    "__author_email__",
    "__description__",
    "__license__",
    "__title__",
    "__url__",
    "__version__",
    "__version_info__",
    "auto_fixable_smell_tags",
    "c",
    "check",
    "codegen",
    "codemod",
    "config",
    "d",
    "deps",
    "detectors",
    "discover_rule_ids",
    "discover_rules",
    "docs",
    "docs_main",
    "e",
    "envrc_contract_violations",
    "fixers",
    "gates",
    "h",
    "infra",
    "m",
    "main",
    "maintenance",
    "p",
    "prepare_managed_conflicts",
    "r",
    "recover_managed_toml",
    "refactor",
    "register_smell_fixer",
    "release",
    "s",
    "services",
    "settings",
    "smell_fixer_for",
    "t",
    "transformers",
    "u",
    "validate",
    "workspace",
    "x",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._config": ("config",),
            "._settings": ("settings",),
            ".api": ("FlextInfra", "infra"),
            ".base": ("FlextInfraServiceBase", "s"),
            ".base_selection": ("FlextInfraProjectSelectionServiceBase",),
            ".check": ("check",),
            ".check.workspace_check": ("FlextInfraWorkspaceChecker",),
            ".check.workspace_check_gates": (
                "FlextInfraGateRegistry",
                "FlextInfraWorkspaceCheckGatesMixin",
            ),
            ".cli": ("FlextInfraCli", "docs_main", "main"),
            ".codegen": ("codegen",),
            ".codegen.census": ("FlextInfraCodegenCensus",),
            ".codegen.codegen_generation": ("FlextInfraCodegenGeneration",),
            ".codegen.conform": ("FlextInfraCodegenConform",),
            ".codegen.consolidator": ("FlextInfraCodegenConsolidator",),
            ".codegen.constants_quality_gate": ("FlextInfraCodegenQualityGate",),
            ".codegen.fixer": ("FlextInfraCodegenFixer",),
            ".codegen.layout": ("FlextInfraCodegenLayout",),
            ".codegen.lazy_init": ("FlextInfraCodegenLazyInit",),
            ".codegen.lazy_init_planner": ("FlextInfraCodegenLazyInitPlanner",),
            ".codegen.make_bootstrap": ("FlextInfraCodegenMakeBootstrap",),
            ".codegen.managed_conflicts": ("FlextInfraCodegenManagedConflicts",),
            ".codegen.managed_conflicts_bootstrap": (
                "ManagedConflictBootstrapError",
                "prepare_managed_conflicts",
            ),
            ".codegen.managed_conflicts_core": (
                "ManagedConflictError",
                "recover_managed_toml",
            ),
            ".codegen.mise_artifacts": ("FlextInfraCodegenMiseArtifacts",),
            ".codegen.pipeline": ("FlextInfraCodegenPipeline",),
            ".codegen.project_new": ("FlextInfraCodegenProjectNew",),
            ".codegen.py_typed": ("FlextInfraCodegenPyTyped",),
            ".codegen.scaffolder": ("FlextInfraCodegenScaffolder",),
            ".codegen.version_file": ("FlextInfraCodegenVersionFile",),
            ".codemod": ("codemod",),
            ".codemod.batch_apply": ("FlextInfraCodemodBatchApply",),
            ".codemod.batch_gates": (
                "FlextInfraModGateEngine",
                "FlextInfraModGateSnapshot",
                "FlextInfraModScanReport",
            ),
            ".codemod.discovery": ("discover_rule_ids", "discover_rules"),
            ".constants": ("FlextInfraConstants", "c"),
            ".deps": ("deps",),
            ".deps.detection": ("FlextInfraDependencyDetectionService",),
            ".deps.detection_analysis": ("FlextInfraDependencyDetectionAnalysis",),
            ".deps.detector": ("FlextInfraRuntimeDevDependencyDetector",),
            ".deps.detector_runtime": ("FlextInfraDependencyDetectorRuntime",),
            ".deps.extra_paths": ("FlextInfraExtraPathsManager",),
            ".deps.fix_pyrefly_config": ("FlextInfraConfigFixer",),
            ".deps.modernizer": ("FlextInfraPyprojectModernizer",),
            ".deps.phases.consolidate_groups": ("FlextInfraConsolidateGroupsPhase",),
            ".deps.phases.ensure_coverage": ("FlextInfraEnsureCoverageConfigPhase",),
            ".deps.phases.ensure_formatting": (
                "FlextInfraEnsureFormattingToolingPhase",
            ),
            ".deps.phases.ensure_mypy": ("FlextInfraEnsureMypyConfigPhase",),
            ".deps.phases.ensure_namespace": ("FlextInfraEnsureNamespaceToolingPhase",),
            ".deps.phases.ensure_packaging": ("FlextInfraEnsurePackagingPhase",),
            ".deps.phases.ensure_pydantic_mypy": (
                "FlextInfraEnsurePydanticMypyConfigPhase",
            ),
            ".deps.phases.ensure_pyrefly": ("FlextInfraEnsurePyreflyConfigPhase",),
            ".deps.phases.ensure_pyright": ("FlextInfraEnsurePyrightConfigPhase",),
            ".deps.phases.ensure_pytest": ("FlextInfraEnsurePytestConfigPhase",),
            ".deps.phases.ensure_ruff": ("FlextInfraEnsureRuffConfigPhase",),
            ".deps.phases.ensure_vulture": ("FlextInfraEnsureVultureConfigPhase",),
            ".deps.phases.inject_comments": ("FlextInfraInjectCommentsPhase",),
            ".deps.toml_phase": ("FlextInfraTomlPhaseService",),
            ".detectors": ("detectors",),
            ".detectors.class_placement_detector": (
                "FlextInfraClassPlacementDetector",
            ),
            ".detectors.compatibility_alias_detector": (
                "FlextInfraCompatibilityAliasDetector",
            ),
            ".detectors.cyclic_import_detector": ("FlextInfraCyclicImportDetector",),
            ".detectors.deferred_self_reference_detector": (
                "FlextInfraDeferredSelfReferenceDetector",
            ),
            ".detectors.future_annotations_detector": (
                "FlextInfraFutureAnnotationsDetector",
            ),
            ".detectors.import_alias_detector": ("FlextInfraImportAliasDetector",),
            ".detectors.inline_import_detector": ("FlextInfraInlineImportDetector",),
            ".detectors.internal_import_detector": (
                "FlextInfraInternalImportDetector",
            ),
            ".detectors.loose_object_detector": ("FlextInfraLooseObjectDetector",),
            ".detectors.loose_test_function_detector": (
                "FlextInfraLooseTestFunctionDetector",
            ),
            ".detectors.manual_protocol_detector": (
                "FlextInfraManualProtocolDetector",
            ),
            ".detectors.manual_typing_alias_detector": (
                "FlextInfraManualTypingAliasDetector",
            ),
            ".detectors.namespace_source_detector": (
                "FlextInfraNamespaceSourceDetector",
            ),
            ".detectors.private_import_bypass_detector": (
                "FlextInfraPrivateImportBypassDetector",
            ),
            ".detectors.runtime_alias_detector": ("FlextInfraRuntimeAliasDetector",),
            ".detectors.silent_failure_detector": ("FlextInfraSilentFailureDetector",),
            ".docs": ("docs",),
            ".docs.auditor": ("FlextInfraDocAuditor",),
            ".docs.auditor_mixin": ("FlextInfraDocAuditorMixin",),
            ".docs.base": ("FlextInfraDocServiceBase",),
            ".docs.builder": ("FlextInfraDocBuilder",),
            ".docs.fixer": ("FlextInfraDocFixer",),
            ".docs.generator": ("FlextInfraDocGenerator",),
            ".docs.server": ("FlextInfraDocServer",),
            ".docs.validator": ("FlextInfraDocValidator",),
            ".fixers": ("fixers",),
            ".fixers.base": ("FlextInfraFixerAdapter",),
            ".fixers.gate_fixer": ("FlextInfraGateFixerAdapter",),
            ".fixers.orchestrator": ("FlextInfraEnforcementFixerOrchestrator",),
            ".fixers.transformer_fixer": ("FlextInfraTransformerFixerAdapter",),
            ".gates": ("gates",),
            ".gates.abstraction_boundary": ("FlextInfraAbstractionBoundaryGate",),
            ".gates.bandit": ("FlextInfraBanditGate",),
            ".gates.base_gate": ("FlextInfraGate", "FlextInfraScannerGateMixin"),
            ".gates.canonical_alias": ("FlextInfraCanonicalAliasGate",),
            ".gates.deferred_self_reference": ("FlextInfraDeferredSelfReferenceGate",),
            ".gates.direnv": ("FlextInfraDirenvGate",),
            ".gates.duplication": ("FlextInfraDuplicationGate",),
            ".gates.layout": ("FlextInfraLayoutGate",),
            ".gates.loc_cap": ("FlextInfraLocCapGate",),
            ".gates.markdown": ("FlextInfraMarkdownGate",),
            ".gates.mypy": ("FlextInfraMypyGate",),
            ".gates.namespace": ("FlextInfraNamespaceGate",),
            ".gates.pyrefly": ("FlextInfraPyreflyGate",),
            ".gates.pyright": ("FlextInfraPyrightGate",),
            ".gates.ruff_format": ("FlextInfraRuffFormatGate",),
            ".gates.ruff_lint": ("FlextInfraRuffLintGate",),
            ".gates.runtime_census": ("FlextInfraRuntimeCensusGate",),
            ".gates.silent_failure": ("FlextInfraSilentFailureGate",),
            ".gates.smells": ("FlextInfraSmellsGate",),
            ".gates.tier_whitelist": ("FlextInfraTierWhitelistGate",),
            ".git": ("FlextInfraGitService",),
            ".iteration": ("FlextInfraUtilitiesIteration",),
            ".maintenance": ("maintenance",),
            ".maintenance.clean": ("FlextInfraCleanService",),
            ".maintenance.python_version": ("FlextInfraPythonVersionEnforcer",),
            ".models": ("FlextInfraModels", "m"),
            ".protocols": ("FlextInfraProtocols", "FlextInfraProtocolsBase", "p"),
            ".refactor": ("refactor",),
            ".refactor.accessor_migration": (
                "FlextInfraAccessorMigrationOrchestrator",
            ),
            ".refactor.census": ("FlextInfraRefactorCensus",),
            ".refactor.class_nesting_analyzer": (
                "FlextInfraRefactorClassNestingAnalyzer",
            ),
            ".refactor.classvar_constant_autofix": (
                "FlextInfraRefactorClassvarConstantAutofix",
            ),
            ".refactor.file_executor": (
                "FlextInfraClassNestingPostCheckGate",
                "FlextInfraRefactorFileExecutor",
            ),
            ".refactor.legacy_text_ops": ("FlextInfraRefactorLegacyTextOps",),
            ".refactor.loader": ("FlextInfraRefactorRuleLoader",),
            ".refactor.modernize_orchestrator": ("FlextInfraModernizeOrchestrator",),
            ".refactor.namespace_enforcer": ("FlextInfraNamespaceEnforcer",),
            ".refactor.namespace_enforcer_phases": (
                "FlextInfraNamespaceEnforcerPhasesMixin",
            ),
            ".refactor.orchestrator": ("FlextInfraRefactorOrchestrator",),
            ".refactor.project_classifier": ("FlextInfraProjectClassifier",),
            ".refactor.safety": ("FlextInfraRefactorSafetyManager",),
            ".refactor.scanner": ("FlextInfraRefactorLooseClassScanner",),
            ".refactor.service": ("FlextInfraRefactorService",),
            ".refactor.text_executor": ("FlextInfraRefactorTextExecutor",),
            ".refactor.violation_analyzer": ("FlextInfraRefactorViolationAnalyzer",),
            ".refactor.wrapper_root_namespace": (
                "FlextInfraWrapperRootNamespaceRefactor",
            ),
            ".release": ("release",),
            ".release.orchestrator": ("FlextInfraReleaseOrchestrator",),
            ".release.orchestrator_phases": ("FlextInfraReleaseOrchestratorPhases",),
            ".services": ("services",),
            ".services.cli_dispatch": ("CliDispatchService",),
            ".services.cli_route_base": ("CliRouteBase",),
            ".services.cli_routes": ("CliRouteService",),
            ".services.cli_routes_codegen": ("CodegenRoutes",),
            ".services.cli_routes_refactor": ("RefactorRoutes",),
            ".services.cli_routes_validate": ("ValidationRoutes",),
            ".services.cli_routes_validate_commands": ("ValidationCommandRoutes",),
            ".services.cli_routes_workspace": ("WorkspaceRoutes",),
            ".services.codegen": ("FlextInfraCodegen",),
            ".transformers": ("transformers",),
            ".transformers.base": (
                "FlextInfraChangeTrackingTransformer",
                "FlextInfraRopeTransformer",
            ),
            ".transformers.cast_remover": ("FlextInfraRefactorCastRemover",),
            ".transformers.census_visitors": (
                "FlextInfraCensusImportDiscoveryVisitor",
                "FlextInfraCensusUsageCollector",
            ),
            ".transformers.class_nesting": (
                "FlextInfraRefactorClassNestingTransformer",
            ),
            ".transformers.class_reconstructor": (
                "FlextInfraRefactorClassReconstructor",
            ),
            ".transformers.cli_modernizer": ("FlextInfraRefactorCliModernizer",),
            ".transformers.compatibility_alias": (
                "FlextInfraRefactorCompatibilityAlias",
            ),
            ".transformers.deprecated_remover": (
                "FlextInfraRefactorDeprecatedRemover",
            ),
            ".transformers.future_import": ("FlextInfraRefactorFutureImport",),
            ".transformers.hardcoded_version": ("FlextInfraRefactorHardcodedVersion",),
            ".transformers.helper_consolidation": (
                "FlextInfraHelperConsolidationTransformer",
            ),
            ".transformers.import_bypass_remover": (
                "FlextInfraRefactorImportBypassRemover",
            ),
            ".transformers.import_modernizer": ("FlextInfraRefactorImportModernizer",),
            ".transformers.lazy_import_fixer": ("FlextInfraRefactorLazyImportFixer",),
            ".transformers.logging_modernizer": (
                "FlextInfraRefactorLoggingModernizer",
            ),
            ".transformers.nested_class_propagation": (
                "FlextInfraNestedClassPropagationTransformer",
            ),
            ".transformers.open_encoding": ("FlextInfraRefactorOpenEncoding",),
            ".transformers.pattern": ("FlextInfraRefactorPatternTransformer",),
            ".transformers.pattern_modernizer": (
                "FlextInfraRefactorPatternModernizer",
            ),
            ".transformers.project_alias_migrator": (
                "FlextInfraRefactorProjectAliasMigrator",
            ),
            ".transformers.pydantic_modernizer": (
                "FlextInfraRefactorPydanticModernizer",
            ),
            ".transformers.result_di_modernizer": (
                "FlextInfraRefactorResultDiModernizer",
            ),
            ".transformers.signature_propagator": (
                "FlextInfraRefactorSignaturePropagator",
            ),
            ".transformers.smells.base": (
                "FlextInfraSmellFixer",
                "auto_fixable_smell_tags",
                "register_smell_fixer",
                "smell_fixer_for",
            ),
            ".transformers.smells.boolean_logic": ("FlextInfraBooleanLogicFixer",),
            ".transformers.symbol_propagator": ("FlextInfraRefactorSymbolPropagator",),
            ".transformers.tier0_import_fixer": (
                "FlextInfraTransformerTier0ImportFixer",
            ),
            ".transformers.typing_dict_attr": ("FlextInfraRefactorTypingDictAttr",),
            ".transformers.typing_dict_import": ("FlextInfraRefactorTypingDictImport",),
            ".transformers.typing_unifier": ("FlextInfraRefactorTypingUnifier",),
            ".transformers.violation_census_visitor": (
                "FlextInfraViolationCensusVisitor",
            ),
            ".typings": ("FlextInfraTypes", "t"),
            ".utilities": ("FlextInfraUtilities", "u"),
            ".validate": ("validate",),
            ".validate.cprofile_report": ("FlextInfraCProfileReport",),
            ".validate.fresh_import": ("FlextInfraValidateFreshImport",),
            ".validate.gate_contract": ("FlextInfraGateContractValidator",),
            ".validate.gate_contract_checks": ("FlextInfraGateContractChecksMixin",),
            ".validate.gate_contract_content": ("FlextInfraGateContractContentMixin",),
            ".validate.gate_contract_errors": (
                "GateContractInfraError",
                "GateContractUsageError",
            ),
            ".validate.gate_contract_report": ("FlextInfraGateContractReportMixin",),
            ".validate.gate_contract_scan": ("FlextInfraGateContractScanMixin",),
            ".validate.import_cycles": ("FlextInfraValidateImportCycles",),
            ".validate.inventory": ("FlextInfraInventoryService",),
            ".validate.lazy_map_freshness": ("FlextInfraValidateLazyMapFreshness",),
            ".validate.loc_delta": ("FlextInfraLocDeltaValidator",),
            ".validate.manual_command": ("FlextInfraManualCommandValidator",),
            ".validate.metadata_discipline": ("FlextInfraValidateMetadataDiscipline",),
            ".validate.namespace_rules": ("FlextInfraNamespaceRules",),
            ".validate.namespace_validator": ("FlextInfraNamespaceValidator",),
            ".validate.pytest_diag": ("FlextInfraPytestDiagExtractor",),
            ".validate.pytest_runner": ("FlextInfraPytestRunner",),
            ".validate.pytest_selector": ("FlextInfraPytestSelectorValidator",),
            ".validate.runtime_census": ("FlextInfraRuntimeCensusValidator",),
            ".validate.scanner": ("FlextInfraTextPatternScanner",),
            ".validate.silent_failure": ("FlextInfraSilentFailureValidator",),
            ".validate.skill_validator": ("FlextInfraSkillValidator",),
            ".validate.stub_chain": ("FlextInfraStubSupplyChain",),
            ".validate.testmon_db": (
                "FlextInfraTestmonCacheState",
                "FlextInfraTestmonDbInspector",
            ),
            ".validate.tier_whitelist": ("FlextInfraValidateTierWhitelist",),
            ".workspace": ("workspace",),
            ".workspace.detector": ("FlextInfraWorkspaceDetector",),
            ".workspace.environment_contracts": ("envrc_contract_violations",),
            ".workspace.environment_provenance": (
                "FlextInfraWorkspaceEnvironmentProvenance",
            ),
            ".workspace.flext_binding": ("FlextInfraFlextBindingService",),
            ".workspace.orchestrator": ("FlextInfraOrchestratorService",),
            ".workspace.rope": ("FlextInfraRopeWorkspace",),
            ".workspace.sandbox_orchestrator": ("FlextInfraSandboxOrchestrator",),
            ".worktree": ("FlextInfraWorktreeService",),
            "flext_cli": ("d", "e", "h", "r", "x"),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
