"""Codegen, check, and dependency CLI route ownership."""

from __future__ import annotations

from typing import ClassVar

from flext_infra import c, m
from flext_infra.check.workspace_check import FlextInfraWorkspaceChecker
from flext_infra.codegen.census import FlextInfraCodegenCensus
from flext_infra.codegen.conform import FlextInfraCodegenConform
from flext_infra.codegen.consolidator import FlextInfraCodegenConsolidator
from flext_infra.codegen.constants_quality_gate import FlextInfraCodegenQualityGate
from flext_infra.codegen.fixer import FlextInfraCodegenFixer
from flext_infra.codegen.layout import FlextInfraCodegenLayout
from flext_infra.codegen.lazy_init import FlextInfraCodegenLazyInit
from flext_infra.codegen.make_bootstrap import FlextInfraCodegenMakeBootstrap
from flext_infra.codegen.mise_artifacts import FlextInfraCodegenMiseArtifacts
from flext_infra.codegen.pipeline import FlextInfraCodegenPipeline
from flext_infra.codegen.project_new import FlextInfraCodegenProjectNew
from flext_infra.codegen.py_typed import FlextInfraCodegenPyTyped
from flext_infra.codegen.scaffolder import FlextInfraCodegenScaffolder
from flext_infra.codegen.version_file import FlextInfraCodegenVersionFile
from flext_infra.deps.detector import FlextInfraRuntimeDevDependencyDetector
from flext_infra.deps.extra_paths import FlextInfraExtraPathsManager
from flext_infra.deps.fix_pyrefly_config import FlextInfraConfigFixer
from flext_infra.deps.modernizer import FlextInfraPyprojectModernizer
from flext_infra.fixers.orchestrator import FlextInfraEnforcementFixerOrchestrator
from flext_infra.services.cli_route_base import CliRouteBase


class CodegenRoutes(CliRouteBase):
    """Own check, codegen, and dependency command routes."""

    codegen_routes: ClassVar[dict[str, tuple[m.Cli.ResultCommandRoute, ...]]] = {
        c.Infra.CLI_GROUP_CHECK: (
            m.Cli.ResultCommandRoute(
                name=c.Infra.VERB_RUN,
                help_text="Run workspace quality gates",
                model_cls=m.Infra.RunCommand,
                handler=CliRouteBase.result_handler(
                    FlextInfraWorkspaceChecker.execute_payload
                ),
            ),
            m.Cli.ResultCommandRoute(
                name="fix-pyrefly-settings",
                help_text="Repair [tool.pyrefly] blocks",
                model_cls=m.Infra.FixPyreflyConfigCommand,
                handler=CliRouteBase.result_handler(
                    FlextInfraConfigFixer.execute_payload
                ),
            ),
            m.Cli.ResultCommandRoute(
                name="fix-enforcement",
                help_text="Auto-fix enforcement-catalog violations",
                model_cls=m.Infra.FixEnforcementCommand,
                handler=CliRouteBase.result_handler(
                    FlextInfraEnforcementFixerOrchestrator.execute_payload
                ),
            ),
        ),
        c.Infra.CLI_GROUP_CODEGEN: (
            m.Cli.ResultCommandRoute(
                name="conform",
                help_text="Conform generated project and workspace files",
                model_cls=m.Infra.CodegenConformRequest,
                handler=FlextInfraCodegenConform.execute_request,
                success_message="project conformance complete",
            ),
            *(
                m.Cli.ResultCommandRoute(
                    name=route_name,
                    help_text=help_text,
                    model_cls=model_cls,
                    handler=handler,
                    success_message=success_message,
                )
                for route_name, help_text, model_cls, handler, success_message in (
                    (
                        "new",
                        "Create a new FLEXT project from the canonical templates",
                        FlextInfraCodegenProjectNew,
                        CliRouteBase.result_handler(
                            FlextInfraCodegenProjectNew.execute_command
                        ),
                        "project created",
                    ),
                    (
                        "init",
                        "Bootstrap only the canonical generated Makefile",
                        FlextInfraCodegenMakeBootstrap,
                        CliRouteBase.result_handler(
                            FlextInfraCodegenMakeBootstrap.execute_command
                        ),
                        "Makefile bootstrap complete",
                    ),
                    (
                        "lazy-init",
                        (
                            "Regenerate PEP 562 lazy-import __init__.py files as "
                            "part of the continuous gen check/apply cycle (unlike "
                            "`init`, which only bootstraps a fresh, unprovisioned "
                            "repository)"
                        ),
                        FlextInfraCodegenLazyInit,
                        CliRouteBase.result_handler(
                            FlextInfraCodegenLazyInit.execute_command
                        ),
                        "lazy-init complete",
                    ),
                    (
                        "census",
                        "Count namespace violations across workspace projects",
                        FlextInfraCodegenCensus,
                        CliRouteBase.result_handler(
                            FlextInfraCodegenCensus.execute_command
                        ),
                        None,
                    ),
                    (
                        "scaffold",
                        "Generate missing base modules in src/ and tests/",
                        FlextInfraCodegenScaffolder,
                        CliRouteBase.result_handler(
                            FlextInfraCodegenScaffolder.execute_command
                        ),
                        None,
                    ),
                    (
                        "auto-fix",
                        "Auto-fix namespace violations (move Finals/TypeVars)",
                        FlextInfraCodegenFixer,
                        CliRouteBase.result_handler(
                            FlextInfraCodegenFixer.execute_command
                        ),
                        None,
                    ),
                    (
                        "py-typed",
                        "Create/remove PEP 561 py.typed markers",
                        FlextInfraCodegenPyTyped,
                        CliRouteBase.result_handler(
                            FlextInfraCodegenPyTyped.execute_command
                        ),
                        "py-typed markers updated",
                    ),
                    (
                        "pipeline",
                        "Run full codegen pipeline",
                        FlextInfraCodegenPipeline,
                        CliRouteBase.result_handler(
                            FlextInfraCodegenPipeline.execute_command
                        ),
                        None,
                    ),
                    (
                        "constants-quality-gate",
                        "Run constants migration quality gate",
                        FlextInfraCodegenQualityGate,
                        CliRouteBase.result_handler(
                            FlextInfraCodegenQualityGate.execute_command
                        ),
                        "constants quality gate passed",
                    ),
                    (
                        "consolidate",
                        "Consolidate inline constants into c.Infra.* references",
                        FlextInfraCodegenConsolidator,
                        CliRouteBase.result_handler(
                            FlextInfraCodegenConsolidator.execute_command
                        ),
                        None,
                    ),
                    (
                        "layout",
                        "Check/apply the canonical project layout (SSOT-driven)",
                        FlextInfraCodegenLayout,
                        CliRouteBase.result_handler(
                            FlextInfraCodegenLayout.execute_command
                        ),
                        "layout conformance complete",
                    ),
                    (
                        "mise-artifacts",
                        "Validate the generated Mise bundle read-only",
                        FlextInfraCodegenMiseArtifacts,
                        CliRouteBase.result_handler(
                            FlextInfraCodegenMiseArtifacts.execute_command
                        ),
                        "Mise artifact validation complete",
                    ),
                    (
                        "version-file",
                        "Generate __version__.py from project-metadata SSOT",
                        FlextInfraCodegenVersionFile,
                        CliRouteBase.result_handler(
                            FlextInfraCodegenVersionFile.execute_command
                        ),
                        "version-file generation complete",
                    ),
                )
            ),
        ),
        c.Infra.CLI_GROUP_DEPS: tuple(
            m.Cli.ResultCommandRoute(
                name=route_name,
                help_text=help_text,
                model_cls=model_cls,
                handler=CliRouteBase.result_handler(model_cls.execute_command),
            )
            for route_name, help_text, model_cls in (
                (
                    "detect",
                    "Detect runtime vs dev dependencies",
                    FlextInfraRuntimeDevDependencyDetector,
                ),
                (
                    "extra-paths",
                    "Synchronize pyright/mypy extraPaths",
                    FlextInfraExtraPathsManager,
                ),
                (
                    "modernize",
                    "Modernize workspace pyproject files",
                    FlextInfraPyprojectModernizer,
                ),
            )
        ),
    }


__all__: list[str] = ["CodegenRoutes"]
