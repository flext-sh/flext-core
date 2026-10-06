"""Typed outcomes of PEP 695 alias evaluation.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar, Literal

from flext_core._models._enforcement._base import FlextModelsEnforcementModelBase
from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._protocols.base import FlextProtocolsBase


class FlextModelsEnforcementResolution:
    """Value and source-proven deferral contracts available before consumers."""

    class ResolvedAlias(FlextModelsEnforcementModelBase):
        """A lazy alias evaluated successfully, retaining its runtime value."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(
                arbitrary_types_allowed=True,
            )
        )
        status: Literal["resolved"] = "resolved"
        value: FlextProtocolsBase.AttributeProbe

    class DeferredAlias(FlextModelsEnforcementModelBase):
        """Source-proven imports prevent runtime evaluation of a declared alias."""

        status: Literal["deferred"] = "deferred"
        module: str
        qualname: str
        file_path: str
        line_number: int
        unavailable_imports: tuple[str, ...]
