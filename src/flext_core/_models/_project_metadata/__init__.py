# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models. Project Metadata package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import defer_lazy_bases, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._models._project_metadata.aggregates import (
        FlextModelsProjectMetadataAggregates,
    )
    from flext_core._models._project_metadata.contract import (
        FlextModelsProjectMetadataContract,
    )
    from flext_core._models._project_metadata.document import (
        FlextModelsProjectMetadataDocument,
    )
    from flext_core._models._project_metadata.fields import (
        FlextModelsProjectMetadataFields,
    )
    from flext_core._models._project_metadata.ingress import (
        FlextModelsPyprojectIngressContract,
    )


__all__: tuple[str, ...] = (
    "FlextModelsProjectMetadataAggregates",
    "FlextModelsProjectMetadataContract",
    "FlextModelsProjectMetadataDocument",
    "FlextModelsProjectMetadataFields",
    "FlextModelsPyprojectIngressContract",
)

install_lazy_exports(
    __name__,
    globals(),
    {
        "FlextModelsProjectMetadataAggregates": ".aggregates",
        "FlextModelsProjectMetadataContract": ".contract",
        "FlextModelsProjectMetadataDocument": ".document",
        "FlextModelsProjectMetadataFields": ".fields",
        "FlextModelsPyprojectIngressContract": ".ingress",
    },
    public_exports=__all__,
    deferred_members={
        "FlextModelsProjectMetadataAggregates": (
            "Project",
            "ProjectToolFlext",
        ),
        "FlextModelsProjectMetadataDocument": (
            "PyprojectTool",
            "ProjectMetadata",
        ),
        "FlextModelsProjectMetadataFields": (
            "ProjectAuthor",
            "ProjectUrls",
            "ProjectToolFlextProject",
            "ProjectToolFlextReadmeSection",
            "ProjectToolFlextDocs",
            "ProjectToolFlextWorkspace",
            "ProjectToolFlextNamespace",
        ),
    },
)

defer_lazy_bases(
    __name__,
    {
        "FlextModelsProjectMetadataAggregates": (
            (".", "FlextModelsProjectMetadataFields"),
        ),
        "FlextModelsProjectMetadataDocument": (
            (".", "FlextModelsProjectMetadataAggregates"),
        ),
    },
)
