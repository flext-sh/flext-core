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

from flext_core import p, t, x


class FlextService[TDomainResult: p.Base = p.Base](x):
    """Base class for domain services in FLEXT applications."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        strict=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        use_enum_values=True,
        validate_assignment=True,
    )


s = FlextService
__all__: t.MutableSequenceOf[str] = ["FlextService", "s"]
