"""FlextModelsExceptionParams - validated params for typed exception hierarchy.

Canonical home for exception parameter models. Used by:
- FlextExceptions (exceptions.py) for __init__ validation
- FlextModelsSettings ErrorConfig models (settings.py) via inheritance

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from flext_core import t
from flext_core._models.pydantic import FlextModelsPydantic as mp

from .flextmodelsexceptionparams_part_01 import (
    FlextModelsExceptionParams as FlextModelsExceptionParamsPart01,
)
from .flextmodelsexceptionparams_part_02 import (
    FlextModelsExceptionParams as FlextModelsExceptionParamsPart02,
)


class FlextModelsExceptionParams(FlextModelsExceptionParamsPart02):
    class OperationErrorParams(FlextModelsExceptionParamsPart01.ParamsModel):
        """Validated params for OperationError."""

        operation: Annotated[
            str | None,
            mp.Field(
                description="Operation name associated with the failure.",
            ),
        ] = None
        reason: Annotated[
            str | None,
            mp.Field(
                description="Short reason explaining the operation failure.",
            ),
        ] = None

    class ServiceLookupParams(
        FlextModelsExceptionParamsPart01.ExpectedActualTypeParams,
    ):
        """Validated params for service lookup and narrowing failures."""

        service_name: Annotated[
            str | None,
            mp.Field(
                default=None,
                description="Service identifier requested from the container/registry.",
                title="Service Name",
                examples=["command_bus"],
            ),
        ] = None

    class RegistryPluginParams(FlextModelsExceptionParamsPart01.ParamsModel):
        """Validated params for registry plugin registration lifecycle."""

        category: Annotated[
            str | None,
            mp.Field(
                default=None,
                description="Registry category for plugin registration.",
                title="Category",
                examples=["validators"],
            ),
        ] = None
        name: Annotated[
            str | None,
            mp.Field(
                default=None,
                description="Plugin name within a category.",
                title="Plugin Name",
                examples=["strict_typing"],
            ),
        ] = None
        scope: Annotated[
            str | None,
            mp.Field(
                default=None,
                description="Registration scope used by registry operations.",
                title="Scope",
                examples=["instance", "class"],
            ),
        ] = None

    class AttributeAccessErrorParams(FlextModelsExceptionParamsPart01.ParamsModel):
        """Validated params for AttributeAccessError."""

        attribute_name: Annotated[
            str | None,
            mp.Field(
                default=None,
                description="Attribute name that could not be accessed safely.",
                title="Attribute Name",
                examples=["token"],
            ),
        ] = None
        attribute_context: Annotated[
            t.RuntimeData | None,
            mp.Field(
                default=None,
                description="Context payload describing the state during access failure.",
                title="Attribute Context",
                examples=[{"owner": "session"}],
            ),
        ] = None


__all__: list[str] = ["FlextModelsExceptionParams"]
