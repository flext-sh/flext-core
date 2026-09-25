"""Typed outcomes of PEP 695 alias evaluation."""

from __future__ import annotations

from typing import ClassVar, Literal

from ..._protocols.base import FlextProtocolsBase as p
from ..pydantic import FlextModelsPydantic as mp
from ._base import EnforcementModelBase


class FlextModelsEnforcementResolution:
    """Value and source-proven deferral contracts available before consumers."""

    class ResolvedAlias(EnforcementModelBase):
        """A lazy alias evaluated successfully, retaining its runtime value."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            arbitrary_types_allowed=True
        )
        status: Literal["resolved"] = "resolved"
        value: p.AttributeProbe

    class DeferredAlias(EnforcementModelBase):
        """Source-proven imports prevent runtime evaluation of a declared alias."""

        status: Literal["deferred"] = "deferred"
        module: str
        qualname: str
        file_path: str
        line_number: int
        unavailable_imports: tuple[str, ...]
