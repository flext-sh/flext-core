"""Project-owned managed-artifact models for the deps subpackage.

Why this is a separate module: a ``default_factory`` runs while the class
body executes, so a nested model cannot name a sibling through the outer
class — that name is still unbound. The previous code hid this with a
``default_factory=lambda: Outer.Sibling()`` deferral, which turned a
structural defect into a runtime one.

The fix is the workspace's diamond-FLEXT composition: each layer declares the
model it owns and inherits the layer below, so every referenced model is a
resolved base-class attribute at definition time. Defaults stay direct
callables; no deferred-resolution lambda, no self-referential model.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Annotated

from flext_cli import m
from flext_infra import t
from flext_infra._models._defaults import ImmutableEmptyMapping


class FlextInfraModelsDepsToolConfigProjectRuff:
    """Innermost layer: the Ruff slice a project may own."""

    class ProjectRuffConfig(m.ArbitraryTypesModel):
        """Project-owned Ruff additions for generated managed artifacts."""

        per_file_ignores: Annotated[
            t.Infra.PerFileIgnores,
            m.Field(
                description="Project-local per-file rules merged with global policy."
            ),
        ] = m.Field(default_factory=ImmutableEmptyMapping)


class FlextInfraModelsDepsToolConfigProjectMise(
    FlextInfraModelsDepsToolConfigProjectRuff
):
    """Project-local Mise tools that extend, but never replace, fleet tools."""

    class ProjectMiseTool(m.ArbitraryTypesModel):
        """One project-owned Mise tool: exact version plus the platforms it ships."""

        version: Annotated[
            t.NonEmptyStr,
            m.Field(description="Exact version written to the generated .mise.toml."),
        ]
        platforms: Annotated[
            tuple[t.NonEmptyStr, ...] | None,
            m.Field(
                description=(
                    "Fleet lock platforms this tool publishes assets for. Absent "
                    "means every fleet platform; a subset records, in the project "
                    "that owns the tool, the platforms its backend cannot lock; an "
                    "explicit empty list declares a backend with no per-platform "
                    "assets (npm, pipx, cargo), so the lock carries none."
                )
            ),
        ] = None

    class ProjectMiseConfig(m.ArbitraryTypesModel):
        """Exact project-owned Mise selectors and their tool declarations."""

        tools: Annotated[
            t.MappingKV[
                t.NonEmptyStr, FlextInfraModelsDepsToolConfigProjectMise.ProjectMiseTool
            ],
            m.Field(description="Project-local Mise tools added to generated config."),
        ]


class FlextInfraModelsDepsToolConfigProjectGitignore(
    FlextInfraModelsDepsToolConfigProjectMise
):
    """Project-local ignore patterns appended to the generated ``.gitignore``."""

    class ProjectGitignoreConfig(m.ArbitraryTypesModel):
        """Repository-owned ignore patterns the fleet scaffold cannot know."""

        patterns: Annotated[
            tuple[t.NonEmptyStr, ...],
            m.Field(
                description=(
                    "Ignore patterns appended, in declaration order, as one "
                    "project-owned section of the generated .gitignore."
                )
            ),
        ] = ()


class FlextInfraModelsDepsToolConfigProjectArtifacts(
    FlextInfraModelsDepsToolConfigProjectGitignore
):
    """Managed-artifact layer; ``ProjectRuffConfig`` is an inherited attribute."""

    class ProjectManagedArtifactsConfig(m.ArbitraryTypesModel):
        """Project-owned configuration for generated artifacts."""

        Ruff: Annotated[
            FlextInfraModelsDepsToolConfigProjectRuff.ProjectRuffConfig,
            m.Field(description="Ruff additions owned by the current project."),
        ] = m.Field(
            default_factory=FlextInfraModelsDepsToolConfigProjectRuff.ProjectRuffConfig
        )
        Mise: Annotated[
            FlextInfraModelsDepsToolConfigProjectMise.ProjectMiseConfig,
            m.Field(description="Mise additions owned by the current project."),
        ] = m.Field(
            default_factory=lambda: (
                FlextInfraModelsDepsToolConfigProjectMise.ProjectMiseConfig(
                    tools=MappingProxyType({})
                )
            )
        )
        Gitignore: Annotated[
            FlextInfraModelsDepsToolConfigProjectGitignore.ProjectGitignoreConfig,
            m.Field(description="Ignore patterns owned by the current project."),
        ] = m.Field(
            default_factory=(
                FlextInfraModelsDepsToolConfigProjectGitignore.ProjectGitignoreConfig
            )
        )

    class ProjectManagedArtifactsResolution(m.ArbitraryTypesModel):
        """Composed project configuration plus selector provenance."""

        artifacts: Annotated[
            FlextInfraModelsDepsToolConfigProjectArtifacts.ProjectManagedArtifactsConfig,
            m.Field(description="Composed managed-artifact configuration."),
        ]
        mise_tool_sources: Annotated[
            t.MappingKV[t.NonEmptyStr, Path],
            m.Field(description="Source YAML path for every local Mise selector."),
        ]

    class ProjectManagedArtifactsSnapshot(m.ArbitraryTypesModel):
        """One immutable YAML snapshot and its single parsed resolution."""

        sources: Annotated[
            tuple[m.Cli.AtomicFileState, ...],
            m.Field(description="Ordered exact project configuration sources."),
        ]
        resolution: Annotated[
            FlextInfraModelsDepsToolConfigProjectArtifacts.ProjectManagedArtifactsResolution,
            m.Field(description="Managed artifacts parsed from those exact sources."),
        ]


class FlextInfraModelsDepsToolConfigProject(
    FlextInfraModelsDepsToolConfigProjectArtifacts
):
    """Document layer composing every project-owned managed-artifact model."""

    class ProjectConfigDocument(m.ArbitraryTypesModel):
        """Relevant managed-artifact slice loaded from project config files."""

        ManagedArtifacts: Annotated[
            FlextInfraModelsDepsToolConfigProjectArtifacts.ProjectManagedArtifactsConfig,
            m.Field(description="Project-local managed artifact configuration."),
        ] = m.Field(
            default_factory=(
                FlextInfraModelsDepsToolConfigProjectArtifacts.ProjectManagedArtifactsConfig
            )
        )


__all__: list[str] = [
    "FlextInfraModelsDepsToolConfigProject",
    "FlextInfraModelsDepsToolConfigProjectArtifacts",
    "FlextInfraModelsDepsToolConfigProjectMise",
    "FlextInfraModelsDepsToolConfigProjectRuff",
]
