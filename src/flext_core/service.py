"""Domain service base class for FLEXT applications.

`FlextService[TDomainResult]` supplies validation, dependency injection, and
railway-style result handling for domain services. It relies on structural
typing to satisfy `p.Service` and provides a clean service lifecycle.

Singleton kernel (mirrors `FlextSettingsBase`):

- per-class `_instance` ClassVar with thread-safe lock,
- `fetch_global()` — return the per-class shared singleton,
- `reset_for_testing()` — drop the singleton slot for test isolation.

Per-project `Flext<X>ServiceBase` MUST inherit `fetch_global` /
`reset_for_testing` from this root rather than redeclaring them
(ENFORCE-057 rejects per-project singleton hand-rolling).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
from typing import ClassVar, Self, Unpack

from pydantic import ConfigDict, PrivateAttr

from flext_core import p, t, x
from flext_core._settings.base import FlextSettingsBase


class FlextService[TDomainResult: p.Base = p.Base](x):
    """Base class for domain services in FLEXT applications."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        strict=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        use_enum_values=True,
        validate_assignment=True,
    )

    _lock: ClassVar[threading.RLock] = threading.RLock()
    _instance: ClassVar[Self | None] = None

    _injected_settings: FlextSettingsBase | None = PrivateAttr(default=None)

    def __init_subclass__(cls, **kwargs: Unpack[ConfigDict]) -> None:
        """Inject a per-class singleton slot for every concrete subclass."""
        super().__init_subclass__(**kwargs)
        cls._instance = None

    @classmethod
    def fetch_global(cls) -> Self:
        """Return the per-class shared singleton.

        Mirrors `FlextSettingsBase.fetch_global` so consumers have a single
        canonical accessor across services and settings (§3.5).
        """
        existing = getattr(cls, "_instance", None)
        if isinstance(existing, cls):
            return existing
        with cls._lock:
            existing = getattr(cls, "_instance", None)
            if isinstance(existing, cls):
                return existing
            instance = cls()
            cls._instance = instance
            return instance

    @classmethod
    def reset_for_testing(cls) -> None:
        """Drop the per-class singleton slot for test isolation."""
        with cls._lock:
            cls._instance = None

    @classmethod
    def with_settings(cls, settings: FlextSettingsBase) -> Self:
        """Return the per-class singleton with `settings` deep-cloned for injection.

        Reuses `FlextSettingsBase.clone_for_injection` so the caller's lifetime
        owns an isolated snapshot; the global settings singleton is unaffected.
        Per-project `Flext<X>ServiceBase` overrides may narrow `settings` parameter
        to their concrete leaf type via PEP 696 TypeVar default — see ENFORCE-058
        carve-out for infrastructure base classes.
        """
        instance = cls.fetch_global()
        instance._injected_settings = FlextSettingsBase.clone_for_injection(settings)
        return instance

    def execute(self) -> p.Result[TDomainResult]:
        """Execute the service domain logic.

        Concrete services must override this method with their typed runtime result.
        """
        msg = f"{type(self).__name__}.execute() must be implemented"
        raise NotImplementedError(msg)


s = FlextService
__all__: t.MutableSequenceOf[str] = ["FlextService", "s"]
