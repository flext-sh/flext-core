"""Domain service base class for FLEXT applications.

`FlextService[TDomainResult]` supplies validation, dependency injection, and
railway-style result handling for domain services. It relies on structural
typing to satisfy `p.Service` and provides a clean service lifecycle.

Service contract (ADR-019):

- A dependency is a port: a field typed ``t.Port[p.X]``, where ``p.X`` is a
  plain ``@runtime_checkable`` Protocol, declared with
  ``m.Field(exclude=True, description=...)``. Pydantic validates the value with
  ``isinstance`` on construction and on assignment, and the port never enters
  the JSON Schema. A port whose type is not a plain Protocol class (a
  subscripted generic, a concrete class) is rejected when the subclass is
  created; Pydantic itself rejects a Protocol that is not runtime-checkable.
- The project's ``api.py`` is the single composition root: it constructs each
  service with its adapters. A service never reads a global to find a
  collaborator; settings reach it through the runtime its base declares.
- A failure leaves as ``p.Result`` carrying its cause; ``unwrap()`` chains it.

Singleton kernel (mirrors `FlextSettings`):

- per-class `_instance` ClassVar with thread-safe lock,
- `fetch_global()` — return the per-class shared singleton, built with no
  arguments, so it serves only services without ports; a service with a
  required port raises ``ValidationError`` there,
- `reset_for_testing()` — drop the singleton slot for test isolation.

Per-project `Flext<X>ServiceBase` MUST inherit `fetch_global` /
`reset_for_testing` from this root rather than redeclaring them
(ENFORCE-057 rejects per-project singleton hand-rolling).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
from types import NoneType, UnionType
from typing import ClassVar, Self, Unpack, get_args, get_origin, is_protocol, override

from pydantic import ConfigDict

from flext_core import c, p, t, x


class FlextService[TDomainResult = p.Base](x):
    """Base class for domain services in FLEXT applications."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        strict=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_by_name=True,
        validate_by_alias=True,
        use_enum_values=True,
        validate_assignment=True,
    )

    _lock: ClassVar[threading.RLock] = threading.RLock()
    _instance: ClassVar[Self | None] = None

    def __init_subclass__(cls, **kwargs: Unpack[ConfigDict]) -> None:
        """Inject a per-class singleton slot for every concrete subclass."""
        _ = kwargs
        super().__init_subclass__()
        cls._instance = None

    @classmethod
    @override
    def __pydantic_on_complete__(cls) -> None:
        """Reject a port whose type ``isinstance`` cannot validate."""
        super().__pydantic_on_complete__()
        for name, field in cls.model_fields.items():
            if get_origin(field.annotation) is not t.Port:
                continue
            (declared,) = get_args(field.annotation)
            members = (
                get_args(declared) if isinstance(declared, UnionType) else (declared,)
            )
            for member in members:
                if member is not NoneType and not is_protocol(member):
                    msg = c.ERR_SERVICE_PORT_TYPE.format(
                        service=cls.__name__, field=name, port_type=member
                    )
                    raise TypeError(msg)

    @classmethod
    def fetch_global(cls) -> Self:
        """Return the per-class shared singleton.

        Mirrors `FlextSettings.fetch_global` so consumers have a single
        canonical accessor across services and settings (§3.5).
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_for_testing(cls) -> None:
        """Drop the per-class singleton slot for test isolation."""
        with cls._lock:
            cls._instance = None

    @classmethod
    def with_settings(cls, settings: p.Settings) -> Self:
        """Return an isolated service snapshot with one runtime settings clone.

        Uses the structural `p.Settings.clone()` contract already consumed by the
        runtime bootstrap path so callers can inject a settings snapshot without
        coupling this service kernel to `FlextSettings`.
        """
        return cls(runtime_settings=settings.clone())

    def execute(self) -> p.Result[TDomainResult]:
        """Execute the service domain logic.

        Concrete services must override this method with their typed runtime result.
        """
        msg = f"{type(self).__name__}.execute() must be implemented"
        raise NotImplementedError(msg)


s = FlextService
__all__: t.MutableSequenceOf[str] = ["FlextService", "s"]
