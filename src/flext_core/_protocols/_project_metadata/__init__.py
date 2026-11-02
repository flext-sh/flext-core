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
    from flext_core._protocols._project_metadata.document import (
        FlextProtocolsProjectMetadataDocument,
    )
    from flext_core._protocols._project_metadata.fields import (
        FlextProtocolsProjectMetadataFields,
    )


__all__: tuple[str, ...] = (
    "FlextProtocolsProjectMetadataAggregates",
    "FlextProtocolsProjectMetadataDocument",
    "FlextProtocolsProjectMetadataFields",
)

install_lazy_exports(
    __name__,
    globals(),
    {
        "FlextProtocolsProjectMetadataAggregates": ".aggregates",
        "FlextProtocolsProjectMetadataDocument": ".document",
        "FlextProtocolsProjectMetadataFields": ".fields",
    },
    public_exports=__all__,
)
