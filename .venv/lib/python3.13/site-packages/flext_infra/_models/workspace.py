"""Domain models for the workspace subpackage."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar

from flext_cli import m
from flext_infra import c, t
from flext_infra._models._defaults import ImmutableEmptyMapping
from flext_infra._models.mixins import FlextInfraModelsMixins as mm


class FlextInfraModelsWorkspace:
    """Models for workspace discovery and orchestration.

    Canonical base policy:
    - ``ArbitraryTypesModel`` for mutable discovery payloads.
    - ``ContractModel`` reserved for immutable workspace settings contracts.
    """

    class WorkspaceEnvironmentRequest(m.ContractModel):
        """Read-only request for validating the active workspace environment."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(populate_by_name=True)

        repository_root: Annotated[
            Path, m.Field(alias="workspace", description="Repository root path")
        ]

    class FlextBindingRequest(m.ContractModel):
        """Session request binding one consumer onto a flext worktree."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(populate_by_name=True)

        repository_root: Annotated[
            Path, m.Field(alias="workspace", description="Consumer project root")
        ]
        flext_root: Annotated[
            Path, m.Field(description="Flext worktree supplying the packages")
        ]
        python: Annotated[
            Path, m.Field(description="Interpreter of the environment to rebind")
        ]

    class DirectUrlDirectoryInfo(m.ContractModel):
        """PEP 610 directory metadata for one installed distribution."""

        editable: Annotated[
            bool, m.Field(description="Distribution is installed as editable")
        ]

    class EditableDirectUrl(m.ContractModel):
        """Validated PEP 610 editable provenance payload."""

        url: Annotated[t.NonEmptyStr, m.Field(description="Editable source URL")]
        dir_info: Annotated[
            FlextInfraModelsWorkspace.DirectUrlDirectoryInfo,
            m.Field(description="Editable directory metadata"),
        ]

    class ProjectInfo(mm.ProjectEntryNameMixin, m.ArbitraryTypesModel):
        """Discovered project metadata for workspace operations."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            frozen=True, validate_default=False
        )

        path: Annotated[Path, m.Field(description="Absolute or relative project path")]
        stack: Annotated[t.NonEmptyStr, m.Field(description="Primary technology stack")]
        has_tests: Annotated[bool, m.Field(description="Project has test suite")] = (
            False
        )
        has_src: Annotated[
            bool, m.Field(description="Project has source directory")
        ] = True
        project_class: Annotated[
            t.NonEmptyStr, m.Field(description="Docs/governance project classification")
        ] = "platform"
        package_name: Annotated[
            str, m.Field(description="Primary Python package name")
        ] = ""
        make_profile: Annotated[
            c.Infra.MakeProfile,
            m.Field(description="Topology proven by this checkout's .gitmodules"),
        ] = c.Infra.MakeProfile.STANDALONE
        declared_subproject: Annotated[
            bool,
            m.Field(description="Whether the aggregate workspace declares this path"),
        ] = False

    class ProjectPyprojectState(m.ArbitraryTypesModel):
        """Centralized parsed pyproject state reused across discovery services.

        Enforcement exemption: internal tooling model with intentional
        mutable state.
        """

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            frozen=True, validate_default=False
        )

        project_root: Annotated[Path, m.Field(description="Project root path")]
        pyproject_path: Annotated[Path, m.Field(description="Resolved pyproject path")]
        payload: Annotated[
            t.JsonMapping, m.Field(description="Parsed pyproject payload")
        ] = m.Field(default_factory=ImmutableEmptyMapping)
        docs_meta: Annotated[
            t.JsonMapping, m.Field(description="Parsed tool.flext.docs payload")
        ] = m.Field(default_factory=ImmutableEmptyMapping)
        project_name: Annotated[str, m.Field(description="Declared project name")] = ""
        package_name: Annotated[str, m.Field(description="Primary package name")] = ""
        dependency_names: Annotated[
            t.StrSequence, m.Field(description="Declared dependency names")
        ] = m.Field(default_factory=tuple)


__all__: list[str] = ["FlextInfraModelsWorkspace"]
