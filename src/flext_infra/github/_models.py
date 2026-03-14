"""Domain models for the github subpackage."""

from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict, Field

from flext_core import FlextModels


class FlextInfraGithubModels:
    """Models for GitHub PR orchestration and repository management."""

    class _PrExecutionResultModel(FlextModels.ArbitraryTypesModel):
        """Base model for PR execution result typing."""

        display: Annotated[
            str, Field(min_length=1, description="Repository display name")
        ]
        status: Annotated[str, Field(min_length=1, description="Execution status")]
        elapsed: Annotated[int, Field(ge=0, description="Elapsed time in seconds")]
        exit_code: Annotated[int, Field(description="Process exit code")]
        log_path: Annotated[
            str | None, Field(default=None, description="Log file path")
        ] = None

    class PrExecutionResult(_PrExecutionResultModel):
        """Result of a single PR operation on a repository."""

    class PrOrchestrationResult(FlextModels.ArbitraryTypesModel):
        """Aggregated result of workspace-wide PR orchestration."""

        total: Annotated[int, Field(ge=0, description="Total repositories processed")]
        success: Annotated[int, Field(ge=0, description="Successful executions")]
        fail: Annotated[int, Field(ge=0, description="Failed executions")]
        results: Annotated[
            tuple[FlextInfraGithubModels._PrExecutionResultModel, ...],
            Field(
                default_factory=tuple,
                description="Per-repository results",
            ),
        ] = Field(default_factory=tuple)

    class RepoUrls(FlextModels.ArbitraryTypesModel):
        """Repository URL pair with SSH and HTTPS variants."""

        ssh_url: Annotated[str, Field(default="", description="SSH clone URL")]
        https_url: Annotated[str, Field(default="", description="HTTPS clone URL")]

    class WorkflowLintResult(FlextModels.ArbitraryTypesModel):
        """Structured result payload for workflow lint execution."""

        status: Annotated[str, Field(min_length=1, description="Lint status")]
        reason: Annotated[
            str | None, Field(default=None, description="Skip reason")
        ] = None
        detail: Annotated[
            str | None, Field(default=None, description="Failure detail")
        ] = None
        exit_code: Annotated[
            int | None, Field(default=None, description="Process exit code")
        ] = None
        stdout: Annotated[
            str | None, Field(default=None, description="Captured stdout")
        ] = None
        stderr: Annotated[
            str | None, Field(default=None, description="Captured stderr")
        ] = None

    class SyncOperation(FlextModels.ArbitraryTypesModel):
        """Describe one workflow synchronization operation."""

        model_config = ConfigDict(frozen=True, extra="forbid")

        project: Annotated[str, Field(..., description="Project name.")]
        path: Annotated[
            str, Field(..., description="File path relative to project root.")
        ]
        action: Annotated[
            str,
            Field(
                ...,
                description="Sync action (create, update, noop, prune).",
            ),
        ]
        reason: Annotated[str, Field(..., description="Reason for the action.")]


__all__ = ["FlextInfraGithubModels"]
