"""Domain models for the docs subpackage."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextModels


class _DocsPhaseItemModel(BaseModel):
    """Unified item payload for docs phase reports."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    phase: str = Field(description="Docs phase: audit, fix, build, generate, validate")
    file: str = Field(default="", description="Relative file path")
    issue_type: str = Field(default="", description="Audit issue type")
    severity: str = Field(default="", description="Audit issue severity")
    message: str = Field(default="", description="Item detail message")
    links: int = Field(default=0, ge=0, description="Applied link fixes")
    toc: int = Field(default=0, ge=0, description="Applied TOC updates")
    path: str = Field(default="", description="Generated file path")
    written: bool = Field(default=False, description="Generated file write flag")


def _new_docs_phase_items() -> list[BaseModel]:
    return []


class FlextInfraDocsModels:
    """Models for documentation services."""

    class FlextInfraDocScope(FlextModels.ArbitraryTypesModel):
        """Documentation scope targeting a project or workspace root."""

        name: str = Field(min_length=1, description="Scope name")
        path: Path = Field(description="Absolute path to scope root")
        report_dir: Path = Field(description="Report output directory for scope")

    class AuditIssue(FlextModels.FrozenStrictModel):
        """Single documentation audit finding."""

        file: str = Field(description="File path relative to scope")
        issue_type: str = Field(description="Issue category")
        severity: str = Field(description="Issue severity")
        message: str = Field(description="Issue description")

    class GeneratedFile(FlextModels.FrozenStrictModel):
        """Record of a generated file operation."""

        path: str = Field(description="File path")
        written: bool = Field(default=False, description="Whether file was written")

    class DocsPhaseItem(_DocsPhaseItemModel):
        """Unified item payload for docs phase reports."""

    class DocsPhaseReport(FlextModels.FrozenStrictModel):
        """Unified report payload for docs phases."""

        phase: str = Field(
            description="Docs phase: audit, fix, build, generate, validate"
        )
        scope: str = Field(description="Scope name")
        result: str = Field(default="", description="Result status")
        reason: str = Field(default="", description="Result reason")
        message: str = Field(default="", description="Human-readable summary message")
        site_dir: str = Field(default="", description="Built site directory path")
        checks: list[str] = Field(default_factory=list, description="Executed checks")
        strict: bool = Field(default=False, description="Strict-mode flag")
        passed: bool = Field(default=False, description="Whether phase passed")
        changed_files: int = Field(default=0, ge=0, description="Changed files count")
        applied: bool = Field(default=False, description="Apply mode flag")
        generated: int = Field(default=0, ge=0, description="Generated files count")
        source: str = Field(
            default="", description="Source marker for generated content"
        )
        missing_adr_skills: list[str] = Field(
            default_factory=list,
            description="Missing ADR skill references",
        )
        todo_written: bool = Field(
            default=False, description="Whether TODOS.md was written"
        )
        items: Sequence[BaseModel] = Field(
            default_factory=_new_docs_phase_items,
            description="Phase-specific item payloads",
        )


__all__ = ["FlextInfraDocsModels"]
