"""Domain service base class for FLEXT applications.

FlextService[T] supplies validation, dependency injection, and railway-style
result handling for domain services. It relies on structural typing to satisfy
``p.Service`` and provides a clean service lifecycle.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import ConfigDict

from flext_core import FlextMixins as x, FlextProtocols as p, FlextTypes as t


class FlextService[TDomainResult: p.Base = p.Base](x):
    """Base class for domain services in FLEXT applications."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        strict=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        use_enum_values=True,
        validate_assignment=True,
    )

    def execute(self) -> p.Result[TDomainResult]:
        """Execute the service domain logic.

        Concrete services must override this method with their typed runtime result.
        """
        msg = f"{self.__class__.__name__}.execute() must be implemented"
        raise NotImplementedError(msg)


s = FlextService
__all__: t.MutableSequenceOf[str] = ["FlextService", "s"]
