"""Composed internal base unifying the FlextResult feature hierarchy.

Assembles the unwrapping, composition, transformation, construction, and
behavioral layers into a single abstract base class through C3 linearization.

See Also:
    - Workspace ADR-001 (Railway-Oriented Programming):
      ``flext/docs/architecture/adr/001-railway-oriented-programming.md``
    - :mod:`result`: Public concrete result facade.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from abc import ABC
from typing import override

from flext_core import p
from flext_core._result import (
    FlextResultBehavior,
    FlextResultComposition,
    FlextResultConstruction,
    FlextResultTransforms,
    FlextResultUnwrap,
)


class FlextResultComposed[T](
    FlextResultUnwrap[T],
    FlextResultComposition[T],
    FlextResultTransforms[T],
    FlextResultConstruction[T],
    FlextResultBehavior[T],
    ABC,
):
    """Monadic railway container uniting all result capabilities.

    Integrates unwrap helpers, sequence combinators, functional transforms,
    factory methods, and pythonic behavioral protocols into the primary base
    consumed by ``FlextResult``.

    Type Parameters:
        T: The type of payload carried on success.

    See Also:
        - Workspace ADR-001 (Railway-Oriented Programming):
          ``flext/docs/architecture/adr/001-railway-oriented-programming.md``
        - :class:`result.FlextResult`: Concrete public facade.

    """

    @override
    def _as_result(self) -> p.Result[T]:
        """Return the complete result view by direct self reference.

        Returns:
            p.Result[T]: This instance cast to its structural protocol contract.

        """
        return self


__all__: list[str] = ["FlextResultComposed"]
