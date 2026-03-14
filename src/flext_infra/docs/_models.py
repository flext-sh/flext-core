"""Domain models for the docs subpackage."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextModels


class FlextInfraDocsModels:
    """Models for documentation services."""

    class _DocsPhaseItemModel(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

        phase: Annotated[
            str,
            Field(description="Docs phase: audit, fix, build, generate, validate"),
        ]
        file: Annotated[str, Field(default="", description="Relative file path")] = ""
        issue_type: Annotated[
            str,
            Field(default="", description="Audit issue type"),
        ] = ""
        severity: Annotated[
            str,
            Field(default="", description="Audit issue severity"),
        ] = ""
        message: Annotated[
            str,
            Field(default="", description="Item detail message"),
        ] = ""
        links: Annotated[
            int,
            Field(default=0, ge=0, description="Applied link fixes"),
        ] = 0
        toc: Annotated[
            int,
            Field(default=0, ge=0, description="Applied TOC updates"),
        ] = 0
        path: Annotated[
            str,
            Field(default="", description="Generated file path"),
        ] = ""
        written: Annotated[
            bool,
            Field(default=False, description="Generated file write flag"),
        ] = False

    class FlextInfraDocScope(FlextModels.ArbitraryTypesModel):
        """Documentation scope targeting a project or workspace root."""

        name: Annotated[str, Field(min_length=1, description="Scope name")]
        path: Annotated[Path, Field(description="Absolute path to scope root")]
        report_dir: Annotated[
            Path, Field(description="Report output directory for scope")
        ]

    class AuditIssue(FlextModels.FrozenStrictModel):
        """Single documentation audit finding."""

        file: Annotated[str, Field(description="File path relative to scope")]
        issue_type: Annotated[str, Field(description="Issue category")]
        severity: Annotated[str, Field(description="Issue severity")]
        message: Annotated[str, Field(description="Issue description")]

    class GeneratedFile(FlextModels.FrozenStrictModel):
        """Record of a generated file operation."""

        path: Annotated[str, Field(description="File path")]
        written: Annotated[
            bool, Field(default=False, description="Whether file was written")
        ] = False

    class DocsPhaseItem(_DocsPhaseItemModel):
        """Unified item payload for docs phase reports."""

    class DocsPhaseReport(FlextModels.FrozenStrictModel):
        """Unified report payload for docs phases."""

        phase: Annotated[
            str,
            Field(
                description="Docs phase: audit, fix, build, generate, validate",
            ),
        ]
        scope: Annotated[str, Field(description="Scope name")]
        result: Annotated[str, Field(default="", description="Result status")] = ""
        reason: Annotated[str, Field(default="", description="Result reason")] = ""
        message: Annotated[
            str, Field(default="", description="Human-readable summary message")
        ] = ""
        site_dir: Annotated[
            str, Field(default="", description="Built site directory path")
        ] = ""
        checks: Annotated[
            list[str], Field(default_factory=list, description="Executed checks")
        ] = Field(default_factory=list)
        strict: Annotated[
            bool, Field(default=False, description="Strict-mode flag")
        ] = False
        passed: Annotated[
            bool, Field(default=False, description="Whether phase passed")
        ] = False
        changed_files: Annotated[
            int, Field(default=0, ge=0, description="Changed files count")
        ] = 0
        applied: Annotated[
            bool, Field(default=False, description="Apply mode flag")
        ] = False
        generated: Annotated[
            int, Field(default=0, ge=0, description="Generated files count")
        ] = 0
        source: Annotated[
            str,
            Field(
                default="",
                description="Source marker for generated content",
            ),
        ] = ""
        missing_adr_skills: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Missing ADR skill references",
            ),
        ] = Field(default_factory=list)
        todo_written: Annotated[
            bool,
            Field(
                default=False,
                description="Whether TODOS.md was written",
            ),
        ] = False
        items: Sequence[FlextInfraDocsModels.DocsPhaseItem] = Field(
            default_factory=lambda: list[FlextInfraDocsModels.DocsPhaseItem](),
            description="Phase-specific item payloads",
        )


__all__ = ["FlextInfraDocsModels"]
