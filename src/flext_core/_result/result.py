"""Composed private generic result base over the _result pyramid.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, override

from flext_core._result import (
    FlextResultBehavior,
    FlextResultComposition,
    FlextResultConstruction,
    FlextResultTransforms,
    FlextResultUnwrap,
)

if TYPE_CHECKING:
    from flext_core import p


class FlextResultComposed[T](
    FlextResultUnwrap[T],
    FlextResultComposition[T],
    FlextResultTransforms[T],
    FlextResultConstruction[T],
    FlextResultBehavior[T],
    ABC,
):
    """Type-safe result with monadic railway-oriented operations.

    Composes the unwrap, composition, transform, construction and behavior
    layers into the single concrete base published as ``FlextResult``; every
    layer contributes one orthogonal slice of the result API and the MRO is
    the only composition mechanism.
    """

    @override
    def _as_result(self) -> p.Result[T]:
        """Return the complete public family by reference, without protocol scans.

        Returns:
            This result with its original identity, payload, and native failure.

        """
        return self


__all__: list[str] = ["FlextResultComposed"]
