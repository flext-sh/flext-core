"""Domain models for the github subpackage."""

from __future__ import annotations

from pydantic import Field

from flext_core import FlextModels


class GithubPrExecutionResultModel(FlextModels.ArbitraryTypesModel):
    """Base model for PR execution result typing."""

    display: str = Field(min_length=1, description="Repository display name")
    status: str = Field(min_length=1, description="Execution status")
    elapsed: int = Field(ge=0, description="Elapsed time in seconds")
    exit_code: int = Field(description="Process exit code")
    log_path: str | None = Field(default=None, description="Log file path")


class FlextInfraGithubModels:
    """Models for GitHub PR orchestration and repository management."""

    class PrExecutionResult(GithubPrExecutionResultModel):
        """Result of a single PR operation on a repository."""

    class PrOrchestrationResult(FlextModels.ArbitraryTypesModel):
        """Aggregated result of workspace-wide PR orchestration."""

        total: int = Field(ge=0, description="Total repositories processed")
        success: int = Field(ge=0, description="Successful executions")
        fail: int = Field(ge=0, description="Failed executions")
        results: tuple[GithubPrExecutionResultModel, ...] = Field(
            default_factory=tuple,
            description="Per-repository results",
        )

    class RepoUrls(FlextModels.ArbitraryTypesModel):
        """Repository URL pair with SSH and HTTPS variants."""

        ssh_url: str = Field(default="", description="SSH clone URL")
        https_url: str = Field(default="", description="HTTPS clone URL")

    class WorkflowLintResult(FlextModels.ArbitraryTypesModel):
        """Structured result payload for workflow lint execution."""

        status: str = Field(min_length=1, description="Lint status")
        reason: str | None = Field(default=None, description="Skip reason")
        detail: str | None = Field(default=None, description="Failure detail")
        exit_code: int | None = Field(default=None, description="Process exit code")
        stdout: str | None = Field(default=None, description="Captured stdout")
        stderr: str | None = Field(default=None, description="Captured stderr")

    class SyncOperation(FlextModels.ArbitraryTypesModel):
        """Describe one workflow synchronization operation."""

        model_config = {"frozen": True, "extra": "forbid"}

        project: str = Field(..., description="Project name.")
        path: str = Field(..., description="File path relative to project root.")
        action: str = Field(
            ..., description="Sync action (create, update, noop, prune).",
        )
        reason: str = Field(..., description="Reason for the action.")


__all__ = ["FlextInfraGithubModels"]
