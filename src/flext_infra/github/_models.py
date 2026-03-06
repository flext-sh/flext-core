"""Domain models for the github subpackage."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from flext_core import FlextModels


class _PrExecutionResult(FlextModels.ArbitraryTypesModel):
    """Result of a single PR operation on a repository."""

    display: str = Field(min_length=1, description="Repository display name")
    status: str = Field(min_length=1, description="Execution status")
    elapsed: int = Field(ge=0, description="Elapsed time in seconds")
    exit_code: int = Field(description="Process exit code")
    log_path: str | None = Field(default=None, description="Log file path")


class _WorkflowLintResult(FlextModels.ArbitraryTypesModel):
    """Structured result payload for workflow lint execution."""

    status: str = Field(min_length=1, description="Lint status")
    reason: str | None = Field(default=None, description="Skip reason")
    detail: str | None = Field(default=None, description="Failure detail")
    exit_code: int | None = Field(default=None, description="Process exit code")
    stdout: str | None = Field(default=None, description="Captured stdout")
    stderr: str | None = Field(default=None, description="Captured stderr")


class FlextInfraGithubModels:
    """Models for GitHub PR orchestration and repository management."""

    class PrExecutionResult(_PrExecutionResult):
        """Result of a single PR operation on a repository."""

    class PrOrchestrationResult(FlextModels.ArbitraryTypesModel):
        """Aggregated result of workspace-wide PR orchestration."""

        total: int = Field(ge=0, description="Total repositories processed")
        success: int = Field(ge=0, description="Successful executions")
        fail: int = Field(ge=0, description="Failed executions")
        results: Sequence[_PrExecutionResult] = Field(
            default_factory=list,
            description="Per-repository results",
        )

    class RepoUrls(FlextModels.ArbitraryTypesModel):
        """Repository URL pair with SSH and HTTPS variants."""

        ssh_url: str = Field(default="", description="SSH clone URL")
        https_url: str = Field(default="", description="HTTPS clone URL")

    class WorkflowLintResult(_WorkflowLintResult):
        """Structured result payload for workflow lint execution."""


__all__ = ["FlextInfraGithubModels"]
