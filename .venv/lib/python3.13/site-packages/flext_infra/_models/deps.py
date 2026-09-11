"""Domain models for the deps subpackage."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar

from flext_core import m
from flext_infra import t
from flext_infra._models._defaults import immutable_empty_mapping
from flext_infra._models.deps_toml import FlextInfraModelsDepsToml
from flext_infra._models.deps_tool_config import FlextInfraModelsDepsToolSettings
from flext_infra._models.mixins import FlextInfraModelsMixins as mm


class FlextInfraModelsDeps(FlextInfraModelsDepsToolSettings, FlextInfraModelsDepsToml):
    """Models for dependency detection and modernization reporting."""

    class DetectCommand(mm.WriteMixin, m.ContractModel):
        """Canonical CLI payload for ``flext-infra deps detect``.

        Inherits ``apply``/``dry_run``, ``workspace``, ``projects``,
        ``fail_fast``, ``verbose`` from ``WriteMixin``.
        """

        output_format: Annotated[
            str,
            m.Field(alias="format", description="Output format for dependency report"),
        ] = "text"
        output: Annotated[
            str | None, m.Field(None, description="Optional output report path")
        ] = None
        quiet: Annotated[bool, m.Field(False, description="Reduce command output")] = (
            False
        )
        no_fail: Annotated[
            bool,
            m.Field(
                alias="no-fail",
                description="Exit successfully even when issues are found",
            ),
        ] = False
        typings: Annotated[
            bool, m.Field(False, description="Detect required typing packages")
        ] = False
        apply_typings: Annotated[
            bool,
            m.Field(
                alias="apply-typings",
                description="Install missing typing packages into the typings group",
            ),
        ] = False
        no_pip_check: Annotated[
            bool, m.Field(alias="no-pip-check", description="Skip workspace pip check")
        ] = False
        limits: Annotated[
            str | None, m.Field(None, description="Path to dependency limits TOML")
        ] = None

        @property
        def output_path(self) -> Path | None:
            """Resolved explicit output path when provided."""
            if self.output is None:
                return None
            return Path(self.output).expanduser().resolve()

        @property
        def limits_path(self) -> Path | None:
            """Resolved dependency limits path when provided."""
            if self.limits is None:
                return None
            return Path(self.limits).expanduser().resolve()

    class ExtraPathsCommand(mm.WriteMixin, m.ContractModel):
        """Canonical CLI payload for ``flext-infra deps extra-paths``."""

    class ModernizeCommand(mm.WriteMixin, m.ContractModel):
        """Canonical CLI payload for ``flext-infra deps modernize``."""

        check: Annotated[bool, m.Field(False, description="Run in check mode")] = False
        audit: Annotated[
            bool, m.Field(description="Audit pyproject changes without writing")
        ] = False
        skip_check: Annotated[
            bool,
            m.Field(alias="skip-check", description="Skip post-write validation step"),
        ] = False
        skip_comments: Annotated[
            bool,
            m.Field(
                alias="skip-comments",
                description="Skip modernization of explanatory comments",
            ),
        ] = False
        rewrite_constraints: Annotated[
            bool,
            m.Field(
                alias="rewrite-constraints",
                description="Rewrite dependency constraints from uv.lock",
            ),
        ] = False

    # NOTE (multi-agent, flext-wkii.17.9): codegen consumes the pure pyproject
    # renderer directly, so no deps CLI payload remains for path/workspace modes.

    class PyprojectDocumentState(m.ArbitraryTypesModel):
        """Centralized normalized TOML state reused across deps workflows.

        Enforcement exemption: internal tooling model with intentional
        mutable state.
        """

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(validate_default=False)

        pyproject_path: Annotated[Path, m.Field(description="Resolved pyproject path")]
        original_rendered: Annotated[
            str, m.Field(description="Original TOML source text")
        ] = ""
        rendered: Annotated[
            str, m.Field(description="Canonical rendered TOML source text")
        ] = ""
        payload: Annotated[
            t.MutableJsonMapping, m.Field(description="Validated plain TOML payload")
        ] = m.Field(default_factory=dict)

    class DependencyLimitsInfo(m.ArbitraryTypesModel):
        """Dependency limits configuration metadata."""

        python_version: Annotated[
            str | None, m.Field(None, description="Python version")
        ] = None
        limits_path: Annotated[str, m.Field("", description="Path to limits file")] = ""

    class PipCheckReport(m.ArbitraryTypesModel):
        """Pip check execution report with status and output lines."""

        ok: Annotated[bool, m.Field(True, description="Whether pip check passed")] = (
            True
        )
        lines: Annotated[
            t.StrSequence, m.Field(description="Pip check output lines")
        ] = m.Field(default_factory=tuple)

    class DeptryIssueGroups(m.ArbitraryTypesModel):
        """Deptry issue grouping model by error code (DEP001-DEP004)."""

        dep001: Annotated[
            t.MutableSequenceOf[t.MappingKV[str, t.Primitives | None]],
            m.Field(description="DEP001 issues"),
        ] = m.Field(default_factory=list[t.MappingKV[str, t.Primitives | None]])
        dep002: Annotated[
            t.MutableSequenceOf[t.MappingKV[str, t.Primitives | None]],
            m.Field(description="DEP002 issues"),
        ] = m.Field(default_factory=list[t.MappingKV[str, t.Primitives | None]])
        dep003: Annotated[
            t.MutableSequenceOf[t.MappingKV[str, t.Primitives | None]],
            m.Field(description="DEP003 issues"),
        ] = m.Field(default_factory=list[t.MappingKV[str, t.Primitives | None]])
        dep004: Annotated[
            t.MutableSequenceOf[t.MappingKV[str, t.Primitives | None]],
            m.Field(description="DEP004 issues"),
        ] = m.Field(default_factory=list[t.MappingKV[str, t.Primitives | None]])

    class DeptryReport(m.ArbitraryTypesModel):
        """Deptry analysis report with categorized issue modules."""

        missing: Annotated[
            t.StrSequence, m.Field(description="Missing dependencies")
        ] = m.Field(default_factory=tuple)
        unused: Annotated[t.StrSequence, m.Field(description="Unused dependencies")] = (
            m.Field(default_factory=tuple)
        )
        transitive: Annotated[
            t.StrSequence, m.Field(description="Transitive dependencies")
        ] = m.Field(default_factory=tuple)
        dev_in_runtime: Annotated[
            t.StrSequence, m.Field(description="Dev dependencies in runtime")
        ] = m.Field(default_factory=tuple)
        raw_count: Annotated[
            t.NonNegativeInt, m.Field(0, description="Raw issue count")
        ] = 0

    class ProjectDependencyReport(mm.ProjectNameMixin, m.ArbitraryTypesModel):
        """Project-level dependency report combining deptry results."""

        deptry: FlextInfraModelsDeps.DeptryReport = m.Field(description="Deptry report")

    class TypingsReport(m.ArbitraryTypesModel):
        """Typing stubs analysis report with required/current/delta packages."""

        required_packages: Annotated[
            t.StrSequence, m.Field(description="Required packages")
        ] = m.Field(default_factory=tuple)
        hinted: Annotated[t.StrSequence, m.Field(description="Hinted packages")] = (
            m.Field(default_factory=tuple)
        )
        missing_modules: Annotated[
            t.StrSequence, m.Field(description="Missing modules")
        ] = m.Field(default_factory=tuple)
        current: Annotated[t.StrSequence, m.Field(description="Current typings")] = (
            m.Field(default_factory=tuple)
        )
        to_add: Annotated[t.StrSequence, m.Field(description="Typings to add")] = (
            m.Field(default_factory=tuple)
        )
        to_remove: Annotated[
            t.StrSequence, m.Field(description="Typings to remove")
        ] = m.Field(default_factory=tuple)
        limits_applied: Annotated[
            bool, m.Field(False, description="Whether limits were applied")
        ] = False
        python_version: Annotated[
            str | None, m.Field(None, description="Python version")
        ] = None

    class ProjectRuntimeReport(m.ArbitraryTypesModel):
        """Project runtime dependency and typings report."""

        deptry: FlextInfraModelsDeps.DeptryReport = m.Field(description="Deptry report")
        typings: FlextInfraModelsDeps.TypingsReport | None = m.Field(
            None, description="Typings report", validate_default=True
        )

    class WorkspaceDependencyReport(m.ArbitraryTypesModel):
        """Workspace-level dependency analysis report aggregating all projects.

        Enforcement exemption: internal tooling model with intentional
        mutable state.
        """

        workspace: Annotated[str, m.Field(description="Workspace name")]
        projects: t.MappingKV[str, FlextInfraModelsDeps.ProjectRuntimeReport] = m.Field(
            default_factory=immutable_empty_mapping, description="Per-project reports"
        )
        pip_check: FlextInfraModelsDeps.PipCheckReport | None = m.Field(
            None, description="Pip check report", validate_default=True
        )
        dependency_limits: FlextInfraModelsDeps.DependencyLimitsInfo | None = m.Field(
            None, description="Dependency limits", validate_default=True
        )


__all__: list[str] = ["FlextInfraModelsDeps"]
