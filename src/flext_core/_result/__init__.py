# AUTO-GENERATED FILE — Regenerate with: make gen
"""Internal implementation hierarchy for the FlextResult railway container.

Defines the layered classes that compose the complete ``FlextResult`` MRO:
    - :class:`FlextResultBase`: Data schema and strict Pydantic validation.
    - :class:`FlextResultBehavior`: Context management, truthiness, and operators.
    - :class:`FlextResultConstruction`: Factories, error normalization, and guards.
    - :class:`FlextResultTransforms`: Monadic mappings, filtering, and chaining.
    - :class:`FlextResultComposition`: Sequence traversal and error accumulation.
    - :class:`FlextResultUnwrap`: Terminal value extraction helpers.
    - :class:`FlextResultComposed`: Unified internal base for the public facade.

See Also:
    - Workspace ADR-001 (Railway-Oriented Programming):
      ``flext/docs/architecture/adr/001-railway-oriented-programming.md``
    - Workspace ADR-018 (Generator Declarations):
      ``flext/docs/architecture/adr/018-generator-declarations.md``
    - :mod:`flext_core.result`: Public result facade and canonical ``r`` alias.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._result.base import FlextResultBase
    from flext_core._result.behavior import FlextResultBehavior
    from flext_core._result.composition import FlextResultComposition
    from flext_core._result.construction import FlextResultConstruction
    from flext_core._result.result import FlextResultComposed
    from flext_core._result.transforms import FlextResultTransforms
    from flext_core._result.unwrap import FlextResultUnwrap


__all__: tuple[str, ...] = (
    "FlextResultBase",
    "FlextResultBehavior",
    "FlextResultComposed",
    "FlextResultComposition",
    "FlextResultConstruction",
    "FlextResultTransforms",
    "FlextResultUnwrap",
)

install_lazy_exports(
    __name__,
    globals(),
    {
        "FlextResultBase": ".base",
        "FlextResultBehavior": ".behavior",
        "FlextResultComposed": ".result",
        "FlextResultComposition": ".composition",
        "FlextResultConstruction": ".construction",
        "FlextResultTransforms": ".transforms",
        "FlextResultUnwrap": ".unwrap",
    },
    public_exports=__all__,
)
