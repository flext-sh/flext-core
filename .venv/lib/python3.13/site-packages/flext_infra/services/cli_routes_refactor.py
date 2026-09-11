"""Refactor CLI route ownership."""

from __future__ import annotations

import functools
from typing import ClassVar

from flext_infra import m
from flext_infra.codemod.batch_apply import FlextInfraCodemodBatchApply
from flext_infra.codemod.rules.refactor.apply_renames import FlextInfraApplyRenames
from flext_infra.refactor.accessor_migration import (
    FlextInfraAccessorMigrationOrchestrator,
)
from flext_infra.refactor.census import FlextInfraRefactorCensus
from flext_infra.refactor.modernize_orchestrator import FlextInfraModernizeOrchestrator
from flext_infra.refactor.namespace_enforcer import FlextInfraNamespaceEnforcer
from flext_infra.refactor.wrapper_root_namespace import (
    FlextInfraWrapperRootNamespaceRefactor,
)
from flext_infra.services.cli_route_base import CliRouteBase
from flext_infra.transformers.cli_modernizer import FlextInfraRefactorCliModernizer
from flext_infra.transformers.logging_modernizer import (
    FlextInfraRefactorLoggingModernizer,
)
from flext_infra.transformers.pattern_modernizer import (
    FlextInfraRefactorPatternModernizer,
)
from flext_infra.transformers.pydantic_modernizer import (
    FlextInfraRefactorPydanticModernizer,
)
from flext_infra.transformers.result_di_modernizer import (
    FlextInfraRefactorResultDiModernizer,
)


class RefactorRoutes(CliRouteBase):
    """Own the complete refactor command tuple."""

    refactor_routes: ClassVar[tuple[m.Cli.ResultCommandRoute, ...]] = (
        m.Cli.ResultCommandRoute(
            name="apply-renames",
            help_text="Check or apply an old,new CSV rename list",
            model_cls=m.Infra.ApplyRenamesInput,
            handler=FlextInfraApplyRenames.execute_command,
        ),
        m.Cli.ResultCommandRoute(
            name="namespace-enforce",
            help_text="Scan workspace for namespace governance violations",
            model_cls=m.Infra.RefactorNamespaceEnforceInput,
            handler=CliRouteBase.result_handler(
                FlextInfraNamespaceEnforcer.execute_command
            ),
        ),
        m.Cli.ResultCommandRoute(
            name="census",
            help_text="Run a Rope-only workspace census for Python objects",
            model_cls=FlextInfraRefactorCensus,
            handler=CliRouteBase.result_handler(
                FlextInfraRefactorCensus.execute_command
            ),
        ),
        m.Cli.ResultCommandRoute(
            name="accessor-migrate",
            help_text="Preview or apply automated get_/set_/is_ migration",
            model_cls=m.Infra.AccessorMigrationInput,
            handler=CliRouteBase.result_handler(
                FlextInfraAccessorMigrationOrchestrator.execute_payload
            ),
        ),
        m.Cli.ResultCommandRoute(
            name="wrapper-root-namespace",
            help_text=(
                "Normalize wrapper alias imports to wrapper root and "
                "flatten *.Core.Tests paths"
            ),
            model_cls=FlextInfraWrapperRootNamespaceRefactor,
            handler=FlextInfraWrapperRootNamespaceRefactor.execute,
        ),
        m.Cli.ResultCommandRoute(
            name="modernize-patterns",
            help_text=(
                "Fix u.Cli.print(), pdb, bare except and open() encoding in library code"
            ),
            model_cls=m.Infra.ModernizeInput,
            handler=functools.partial(
                FlextInfraModernizeOrchestrator.execute_command,
                transformer_factory=FlextInfraRefactorPatternModernizer,
                description="pattern modernizer",
            ),
        ),
        m.Cli.ResultCommandRoute(
            name="modernize-pydantic",
            help_text="Migrate Pydantic v1/legacy patterns to Pydantic v2",
            model_cls=m.Infra.ModernizeInput,
            handler=functools.partial(
                FlextInfraModernizeOrchestrator.execute_command,
                transformer_factory=FlextInfraRefactorPydanticModernizer,
                description="pydantic modernizer",
            ),
        ),
        m.Cli.ResultCommandRoute(
            name="modernize-logging",
            help_text="Migrate logging usage to u.fetch_logger",
            model_cls=m.Infra.ModernizeInput,
            handler=functools.partial(
                FlextInfraModernizeOrchestrator.execute_command,
                transformer_factory=FlextInfraRefactorLoggingModernizer,
                description="logging modernizer",
            ),
        ),
        m.Cli.ResultCommandRoute(
            name="modernize-result-di",
            help_text=(
                "Migrate result-flow and dependency-injector patterns "
                "to FLEXT canonical forms"
            ),
            model_cls=m.Infra.ModernizeInput,
            handler=functools.partial(
                FlextInfraModernizeOrchestrator.execute_command,
                transformer_factory=FlextInfraRefactorResultDiModernizer,
                description="result/DI modernizer",
            ),
        ),
        m.Cli.ResultCommandRoute(
            name="modernize-cli",
            help_text=(
                "Remove banned CLI helper imports and route u.Cli.print() "
                "to cli.display_text()"
            ),
            model_cls=m.Infra.ModernizeInput,
            handler=functools.partial(
                FlextInfraModernizeOrchestrator.execute_command,
                transformer_factory=FlextInfraRefactorCliModernizer,
                description="cli modernizer",
            ),
        ),
        m.Cli.ResultCommandRoute(
            name="mod",
            help_text=(
                "Batch-apply all ast-grep rules under the ruff/pyrefly rollback circuit"
            ),
            model_cls=FlextInfraCodemodBatchApply,
            handler=FlextInfraCodemodBatchApply.execute_command,
        ),
    )


__all__: list[str] = ["RefactorRoutes"]
