"""Workspace and release CLI route ownership."""

from __future__ import annotations

from typing import ClassVar

from flext_infra import c, m, p, t
from flext_infra.release.orchestrator import FlextInfraReleaseOrchestrator
from flext_infra.services._workspace.environment_beads import (
    FlextInfraWorkspaceEnvironmentSync,
)
from flext_infra.services.cli_route_base import CliRouteBase
from flext_infra.services.cli_routes_refactor import RefactorRoutes
from flext_infra.workspace.detector import FlextInfraWorkspaceDetector
from flext_infra.workspace.environment_provenance import (
    FlextInfraWorkspaceEnvironmentProvenance,
)
from flext_infra.workspace.flext_binding import FlextInfraFlextBindingService
from flext_infra.workspace.orchestrator import FlextInfraOrchestratorService


class WorkspaceRoutes(RefactorRoutes):
    """Own refactor, release, and workspace routes."""

    @staticmethod
    def _apply_flext_binding(
        params: m.Infra.FlextBindingRequest,
    ) -> p.Result[t.Cli.ResultValue]:
        """Apply the typed binding request through its service owner."""
        return FlextInfraFlextBindingService.apply(
            consumer_root=params.repository_root,
            flext_root=params.flext_root,
            python=params.python,
        ).map(CliRouteBase.as_route_value)

    @staticmethod
    def _sync_environment(
        params: m.Infra.WorkspaceEnvironmentCliRequest,
    ) -> p.Result[t.Cli.ResultValue]:
        """Keep the internal beads render context off the public CLI surface."""
        request = m.Infra.WorkspaceEnvironmentSyncRequest.model_validate(
            params.model_dump()
        )
        return FlextInfraWorkspaceEnvironmentSync.execute_request(request).map(
            CliRouteBase.as_route_value
        )

    workspace_routes: ClassVar[dict[str, tuple[m.Cli.ResultCommandRoute, ...]]] = {
        c.Infra.CLI_GROUP_REFACTOR: RefactorRoutes.refactor_routes,
        c.Infra.CLI_GROUP_RELEASE: (
            m.Cli.ResultCommandRoute(
                name=c.Infra.VERB_RUN,
                help_text="Run release orchestration CLI flow",
                model_cls=FlextInfraReleaseOrchestrator,
                handler=CliRouteBase.result_handler(
                    FlextInfraReleaseOrchestrator.execute_command
                ),
                success_message="Release completed successfully",
            ),
        ),
        c.Infra.CLI_GROUP_WORKSPACE: (
            m.Cli.ResultCommandRoute(
                name="verify-environment",
                help_text="Verify live workspace editable provenance",
                model_cls=m.Infra.WorkspaceEnvironmentRequest,
                handler=CliRouteBase.result_handler(
                    FlextInfraWorkspaceEnvironmentProvenance.execute_request
                ),
                success_message="workspace editable provenance verified",
            ),
            m.Cli.ResultCommandRoute(
                name="flext-binding",
                help_text="Bind this project onto a flext worktree for the session",
                model_cls=m.Infra.FlextBindingRequest,
                handler=_apply_flext_binding,
                success_message="flext worktree binding applied",
            ),
            *(
                m.Cli.ResultCommandRoute(
                    name=route_name,
                    help_text=help_text,
                    model_cls=model_cls,
                    handler=handler,
                )
                for route_name, help_text, model_cls, handler in (
                    (
                        "detect",
                        "Detect workspace or standalone mode",
                        FlextInfraWorkspaceDetector,
                        CliRouteBase.result_handler(
                            FlextInfraWorkspaceDetector.execute_command
                        ),
                    ),
                    (
                        "orchestrate",
                        "Run make verb across projects",
                        FlextInfraOrchestratorService,
                        CliRouteBase.result_handler(
                            FlextInfraOrchestratorService.execute_command
                        ),
                    ),
                    (
                        "sync-environment",
                        "Sync generated direnv/mise environment files",
                        m.Infra.WorkspaceEnvironmentCliRequest,
                        _sync_environment,
                    ),
                )
            ),
        ),
    }


__all__: list[str] = ["WorkspaceRoutes"]
