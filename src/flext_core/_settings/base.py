"""FlextSettingsBase — singleton-per-class mixin.

Replaces the centralised ``_instances`` dict with a per-class ``_instance``
ClassVar so every settings class manages its own singleton storage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
from typing import ClassVar, Self

from flext_core import FlextTypes as t


class FlextSettingsBase:
    """Mixin that gives every subclass its own singleton slot.

    ``__init_subclass__`` injects a fresh ``_instance = None`` into each
    concrete settings class so that ``FlextSettings`` and ``FlextCliSettings``
    never share the same slot.
    """

    _lock: ClassVar[threading.RLock] = threading.RLock()
    _singleton_enabled: ClassVar[bool] = True
    _instance: ClassVar[FlextSettingsBase | None] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls._instance = None

    def __new__(cls, **kwargs: t.SettingsInput) -> Self:
        """Create singleton instance.

        Unknown kwargs are filtered silently in ``__init__`` so consumer
        factories can pass arbitrary connection parameters without breaking
        when the target settings class does not declare them.
        """
        _ = kwargs
        if not cls._singleton_enabled:
            return super().__new__(cls)
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cls._instance = instance
        raw_instance = cls._instance
        if not isinstance(raw_instance, cls):
            cls_name = getattr(cls, "__name__", type(cls).__name__)
            msg = f"Singleton instance is not of expected type {cls_name}"
            raise TypeError(msg)
        return raw_instance

    @classmethod
    def _reset_instance(cls) -> None:
        """Reset singleton instance for testing purposes.

        This method is intended for use in tests only to allow
        clean state between test runs.
        """
        with cls._lock:
            cls._instance = None


__all__: list[str] = ["FlextSettingsBase"]
