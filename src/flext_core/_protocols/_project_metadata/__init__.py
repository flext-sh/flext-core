# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Protocols. Project Metadata package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._protocols._project_metadata.aggregates import (
        FlextProtocolsProjectMetadataAggregates,
    )
    from flext_core._protocols._project_metadata.contract import (
        FlextProtocolsProjectMetadataContract,
    )
    from flext_core._protocols._project_metadata.document import (
        FlextProtocolsProjectMetadataDocument,
    )
    from flext_core._protocols._project_metadata.fields import (
        FlextProtocolsProjectMetadataFields,
    )
    from flext_core._protocols._project_metadata.pyproject_ingress import (
        FlextProtocolsProjectMetadataPyprojectIngress,
    )


__all__: tuple[str, ...] = (
    "FlextProtocolsProjectMetadataAggregates",
    "FlextProtocolsProjectMetadataContract",
    "FlextProtocolsProjectMetadataDocument",
    "FlextProtocolsProjectMetadataFields",
    "FlextProtocolsProjectMetadataPyprojectIngress",
)

install_lazy_exports(
    __name__,
    globals(),
    {
        "FlextProtocolsProjectMetadataAggregates": ".aggregates",
        "FlextProtocolsProjectMetadataContract": ".contract",
        "FlextProtocolsProjectMetadataDocument": ".document",
        "FlextProtocolsProjectMetadataFields": ".fields",
        "FlextProtocolsProjectMetadataPyprojectIngress": ".pyproject_ingress",
    },
    public_exports=__all__,
)
