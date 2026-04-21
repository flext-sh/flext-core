"""Domain service base class for FLEXT applications.

FlextService[T] supplies validation, dependency injection, and railway-style
result handling for domain services. It relies on structural typing to satisfy
``p.Service`` and provides a clean service lifecycle.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar

from flext_core import p, x
from pydantic import ConfigDict


class FlextService[TDomainResult: p.Base = p.Base](x):
    """Base class for domain services in FLEXT applications.

    DEPRECATED: This class depended on removed types (t.RuntimeData, t.RuntimeData, etc).
    Refactor to use explicit Pydantic models in m.* and protocols in p.*.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        strict=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        use_enum_values=True,
        validate_assignment=True,
    )

    # DEPRECATED: All methods removed - depended on t.RuntimeData, t.ServiceMap, t.RegisterableService
    # Refactor manually to use explicit m.* models and p.* protocols

    # Original methods commented for reference during manual refactoring:

    # @property
    # def settings(self) -> FlextSettings:
    #     """Resolve settings instance."""
    #     return FlextSettings.__new__()

    # @classmethod
    # def execute(cls, **kwargs: t.RuntimeData) -> p.Result[TDomainResult]:
    #     """Execute the service with given kwargs."""
    #     raise NotImplementedError

    # @staticmethod
    # def create_service_runtime(...) -> tuple[...]:
    #     """Create service runtime instances."""
    #     raise NotImplementedError

    # def validate_business_rules(self) -> p.Result[bool]:
    #     """Validate business rules."""
    #     return r[bool].ok(True)

    # def ok[T: t.RuntimeData | Sequence[t.RuntimeData]](self, value: T) -> p.Result[T]:
    #     """Wrap a successful value into a result."""
    #     return r[T].ok(value)

    # def fail_op(...) -> p.Result[TDomainResult]:
    #     """Return a failure result."""
    #     raise NotImplementedError


s = FlextService
__all__: list[str] = ["FlextService", "s"]
