"""CSV-driven rename models for refactor workflows."""

from __future__ import annotations

from typing import Annotated, ClassVar

from flext_core import m
from flext_infra import t
from flext_infra._models.mixins import FlextInfraModelsMixins as mm


class FlextInfraModelsRefactorRenames:
    """Contracts for the generic CSV-driven rename engine."""

    class ApplyRenamesInput(mm.WriteMixin, m.ContractModel):
        """Validated CLI request for CSV-driven symbol renames."""

        csv: Annotated[
            t.NonEmptyStr, m.Field(description="Path to the old,new rename-list CSV")
        ]
        roots: Annotated[
            t.StrSequence,
            m.Field(min_length=1, description="Directories to scan for rename targets"),
        ]

    class ApplyRenamesReport(m.ArbitraryTypesModel):
        """Summary of one CSV-driven rename pass."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        label: Annotated[t.NonEmptyStr, m.Field(description="Rename-list label")]
        files_scanned: Annotated[
            t.NonNegativeInt, m.Field(description="Text files scanned")
        ]
        occurrences: Annotated[
            t.NonNegativeInt, m.Field(description="Pending occurrences in check mode")
        ] = 0
        files_changed: Annotated[
            t.NonNegativeInt, m.Field(description="Files rewritten in apply mode")
        ] = 0
        applied: Annotated[bool, m.Field(description="Whether changes were applied")]


__all__: list[str] = ["FlextInfraModelsRefactorRenames"]
