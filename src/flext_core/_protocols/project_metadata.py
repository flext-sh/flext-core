"""Structural project metadata contracts exposed on ``p``.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_core._protocols._project_metadata import (
    FlextProtocolsProjectMetadataAggregates,
    FlextProtocolsProjectMetadataDocument,
    FlextProtocolsProjectMetadataFields,
    FlextProtocolsProjectMetadataPyprojectIngress,
)

if TYPE_CHECKING:
    from importlib.metadata import Distribution, DistributionFinder


class FlextProtocolsProjectMetadata(
    FlextProtocolsProjectMetadataDocument,
    FlextProtocolsProjectMetadataAggregates,
    FlextProtocolsProjectMetadataFields,
):
    """Protocols for project metadata consumed across FLEXT layers."""

    @runtime_checkable
    class DistributionSource(Protocol):
        """A ``sys.meta_path`` entry that reports installed distributions.

        Matches finder instances and finder classes alike (``PathFinder`` sits
        on ``sys.meta_path`` as a class), exactly as ``importlib.metadata``
        discovers them.
        """

        def find_distributions(
            self,
            context: DistributionFinder.Context = ...,
            /,
        ) -> Iterable[Distribution]: ...

    @runtime_checkable
    class PyprojectDocument(
        FlextProtocolsProjectMetadataPyprojectIngress.FlextModelsPyprojectIngressContract,
        Protocol,
    ):
        """Structural data contract projected from the owning model."""

        @property
        def project(self) -> FlextProtocolsProjectMetadataAggregates.Project | None:
            """Optional PEP 621 project table."""
            ...

        @property
        def tool(self) -> FlextProtocolsProjectMetadataDocument.PyprojectTool:
            """Owned tool tables."""
            ...


__all__: list[str] = ["FlextProtocolsProjectMetadata"]
