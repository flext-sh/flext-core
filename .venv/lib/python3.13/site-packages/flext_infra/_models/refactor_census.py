"""Census models for the refactor subpackage."""

from __future__ import annotations

from typing import Annotated, ClassVar

from flext_core import m
from flext_infra import c, t
from flext_infra._models.mixins import FlextInfraModelsMixins as mm


class FlextInfraModelsRefactorCensus:
    """Census and FLEXT models for refactor workflows."""

    # -- FLEXT Generic Models ----------------------------------------------------

    class FLEXTFamilyTarget(m.ArbitraryTypesModel):
        """Parametrized target for an FLEXT family scan or operations."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        family: Annotated[
            t.NonEmptyStr, m.Field(description="Family alias letter (c/t/p/m/u)")
        ]
        class_suffix: Annotated[
            str, m.Field(description="Class name suffix (e.g. 'Utilities')")
        ]
        package_dir: Annotated[
            str,
            m.Field(
                description="Relative path to _xxx package dir (e.g. 'flext_core/_utilities')"
            ),
        ]
        facade_module: Annotated[
            str,
            m.Field(
                description="Relative path to facade (e.g. 'flext_core/utilities.py')"
            ),
        ]
        facade_class_prefix: Annotated[
            str, m.Field(description="Class name prefix for facade (e.g. 'Flext')")
        ] = "Flext"
        core_project: Annotated[
            str, m.Field(description="Core project directory name")
        ] = c.Infra.PKG_CORE

    # -- Census Models ---------------------------------------------------------

    class CensusMethodInfo(m.ArbitraryTypesModel):
        """A public method extracted from a _utilities class."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        name: Annotated[t.NonEmptyStr, m.Field(description="Method name")]
        method_type: Annotated[
            str, m.Field(description="Method kind: static, class, instance")
        ]
        source_file: Annotated[str, m.Field(description="Source filename")]

    class CensusUsageRecord(mm.ProjectNameMixin, m.ArbitraryTypesModel):
        """A single method usage found via CST analysis."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        class_name: Annotated[
            t.NonEmptyStr, m.Field(description="Utilities class name")
        ]
        method_name: Annotated[t.NonEmptyStr, m.Field(description="Method name")]
        access_mode: Annotated[
            str,
            m.Field(description="Access mode: alias_flat, alias_namespaced, direct"),
        ]
        file_path: Annotated[str, m.Field(description="Source file path")]

    class CensusMethodSummary(m.ArbitraryTypesModel):
        """Aggregated usage counts for a single method."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        name: Annotated[t.NonEmptyStr, m.Field(description="Method name")]
        method_type: Annotated[str, m.Field(description="Method kind")]
        alias_flat: Annotated[t.NonNegativeInt, m.Field(description="u.method count")]
        alias_namespaced: Annotated[
            t.NonNegativeInt, m.Field(description="u.Class.method count")
        ]
        direct: Annotated[
            t.NonNegativeInt, m.Field(description="Direct class.method count")
        ]
        total: Annotated[t.NonNegativeInt, m.Field(description="Total usages")]

    class CensusClassSummary(m.ArbitraryTypesModel):
        """Aggregated census for one _utilities class."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        class_name: Annotated[
            t.NonEmptyStr, m.Field(description="Utilities class name")
        ]
        source_file: Annotated[str, m.Field(description="Source filename")]
        methods: tuple[FlextInfraModelsRefactorCensus.CensusMethodSummary, ...] = (
            m.Field(default_factory=tuple, description="Method summaries")
        )

    class CensusProjectMethodUsage(m.ArbitraryTypesModel):
        """Usage of a method within a specific project."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        class_name: Annotated[
            t.NonEmptyStr, m.Field(description="Utilities class name")
        ]
        method_name: Annotated[t.NonEmptyStr, m.Field(description="Method name")]
        access_mode: Annotated[str, m.Field(description="Access mode")]
        count: Annotated[t.NonNegativeInt, m.Field(description="Usage count")]

    class CensusProjectSummary(mm.ProjectNameFieldMixin, m.ArbitraryTypesModel):
        """Usage breakdown for one project."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        usages: t.SequenceOf[
            FlextInfraModelsRefactorCensus.CensusProjectMethodUsage
        ] = m.Field(default_factory=tuple, description="Per-method usages")
        total: Annotated[
            t.NonNegativeInt, m.Field(description="Total usages in project")
        ]

    class UtilitiesCensusReport(m.ArbitraryTypesModel):
        """Full census report for _utilities method usage."""

        classes: tuple[FlextInfraModelsRefactorCensus.CensusClassSummary, ...] = (
            m.Field(default_factory=tuple, description="Per-class summaries")
        )
        projects: tuple[FlextInfraModelsRefactorCensus.CensusProjectSummary, ...] = (
            m.Field(default_factory=tuple, description="Per-project breakdowns")
        )
        total_classes: Annotated[
            t.NonNegativeInt, m.Field(description="Number of utility classes")
        ]
        total_methods: Annotated[
            t.NonNegativeInt, m.Field(description="Number of public methods")
        ]
        total_usages: Annotated[
            t.NonNegativeInt, m.Field(description="Total usage records")
        ]
        total_unused: Annotated[
            t.NonNegativeInt, m.Field(description="Methods with zero usages")
        ]
        files_scanned: Annotated[t.NonNegativeInt, m.Field(description="Files scanned")]
        parse_errors: Annotated[
            t.NonNegativeInt, m.Field(description="Files that failed to parse")
        ]


__all__: list[str] = ["FlextInfraModelsRefactorCensus"]
