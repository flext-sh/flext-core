"""FlextSettingsBase — singleton + Pydantic-2 facade for every settings class.

Houses the canonical helper API shared by `FlextSettings` (root) and
project-specific settings classes (`FlextLdifSettings`, `FlextApiSettings`, …):

- per-class singleton (``_instance`` ClassVar with thread-safe lock),
- ``fetch_global()`` — return shared singleton (rule 1; runtime mutations
  via ``update_global`` propagate),
- ``clone(**overrides)`` — `model_copy(deep=True)` + re-validation for
  custom-injection isolation (rule 2),
- ``update_global(**overrides)`` — replace ``cls._instance`` via
  `model_copy(update=…)` + re-validation; pure Pydantic-2 mutation
  (no ``setattr``, no ``__setattr__``),
- ``validate_overrides(**overrides)`` — typo guard before clone/update,
- ``clone_for_injection(settings)`` — single helper for service/container
  constructors that accept an explicit ``settings=…`` argument (rule 2),
- ``reset_for_testing()`` — drop singleton slot,
- ``resolve_env_file(namespace=None)`` — centralised .env discovery (rule 5).

Extends ``pydantic.BaseModel`` so Pydantic-only operations (``model_copy``,
``__pydantic_validator__``, ``model_fields``) are typed at this layer.
Project subclasses inherit ``(FlextSettingsBase, BaseSettings)`` directly,
giving them the helper API without inheriting root concrete fields (rule 3).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar, Self, Unpack

from pydantic import BaseModel, ConfigDict

from flext_core import (
    FlextConstants as c,
    FlextExceptions as e,
    FlextTypes as t,
)


class FlextSettingsBase(BaseModel):
    """Pydantic-2 base + per-class singleton + canonical helper API.

    Every settings class in the workspace inherits from this base, gaining the
    fetch/clone/update API plus an isolated singleton slot. Subclasses still
    layer in ``BaseSettings`` (or ``BaseModel``) for env-loading and field
    declarations.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    _lock: ClassVar[threading.RLock] = threading.RLock()
    _singleton_enabled: ClassVar[bool] = True
    _instance: ClassVar[FlextSettingsBase | None] = None

    def __init_subclass__(cls, **kwargs: Unpack[ConfigDict]) -> None:
        """Inject a per-class ``_instance`` slot for every concrete subclass."""
        super().__init_subclass__(**kwargs)
        cls._instance = None

    def __new__(cls, **kwargs: t.SettingsInput) -> Self:
        """Singleton factory.

        Unknown kwargs are ignored so consumer factories can pass arbitrary
        connection parameters without breaking when the target class does not
        declare them.
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
    def reset_instance(cls) -> None:
        """Reset the singleton slot for testing purposes."""
        with cls._lock:
            cls._instance = None

    @classmethod
    @contextmanager
    def singleton_disabled(cls) -> Generator[None]:
        """Temporarily disable singleton enforcement for clone operations."""
        with cls._lock:
            original = cls._singleton_enabled
            cls._singleton_enabled = False
            try:
                yield
            finally:
                cls._singleton_enabled = original

    @classmethod
    def fetch_global(cls, *, overrides: t.ScalarMapping | None = None) -> Self:
        """Return the shared per-class singleton (rule 1).

        When ``overrides`` is given, return an isolated deep clone instead —
        the global singleton is NOT mutated (use ``update_global`` for that).
        """
        existing = getattr(cls, "_instance", None)
        instance = existing if isinstance(existing, cls) else cls()
        if not overrides:
            return instance
        return instance.clone(**dict(overrides))

    def clone(self, **overrides: t.JsonPayload | None) -> Self:
        """Deep copy with optional field overrides + re-validation (rule 2).

        Used by service/container constructors that accept an explicit
        ``settings=`` argument so the caller's lifetime owns its snapshot
        independent of the global singleton. Nested submodels are deep-copied.
        """
        if not overrides:
            with self.__class__.singleton_disabled():
                return self.model_copy(deep=True)
        with self.__class__.singleton_disabled():
            copied = self.model_copy(update=dict(overrides), deep=True)
        copied.__pydantic_validator__.validate_python(
            copied.__dict__, self_instance=copied
        )
        return copied

    @classmethod
    def update_global(cls, **overrides: t.JsonPayload | None) -> Self:
        """Replace ``cls._instance`` via ``model_copy(update=…)`` + revalidate.

        Pure Pydantic-2 mutation: no ``setattr``, no ``__setattr__`` override,
        no ``apply_override`` ad-hoc method. Subsequent ``fetch_global()``
        calls return the new instance — propagates per rule 1.

        Raises:
            ValueError: if any override key is not a declared model field.

        """
        if not overrides:
            return cls.fetch_global()
        cls.validate_overrides(**overrides)
        current = cls.fetch_global()
        with cls.singleton_disabled():
            new_instance = current.model_copy(update=dict(overrides), deep=True)
        new_instance.__pydantic_validator__.validate_python(
            new_instance.__dict__, self_instance=new_instance
        )
        with cls._lock:
            cls._instance = new_instance
        return new_instance

    @classmethod
    def validate_overrides(cls, **overrides: t.JsonPayload | None) -> None:
        """Reject override keys that are not declared model fields.

        Typo guard at CLI/runtime override boundaries.
        """
        unknown = sorted(set(overrides) - set(cls.model_fields))
        if unknown:
            msg = e.render_template(
                "Unknown settings override(s) for {cls_name}: {unknown}",
                cls_name=cls.__name__,
                unknown=", ".join(unknown),
            )
            raise ValueError(msg)

    @classmethod
    def reset_for_testing(cls) -> None:
        """Reset the per-class singleton for test isolation."""
        cls.reset_instance()

    @classmethod
    def clone_for_injection(cls, settings: Self | None) -> Self | None:
        """Canonical helper for service/container constructors (rule 2).

        Returns:
            ``None`` when caller passed ``settings=None`` (caller resolves to
            ``cls.fetch_global()`` lazily so runtime mutations propagate);
            otherwise an isolated deep clone of the supplied instance.

        """
        if settings is None:
            return None
        if not isinstance(settings, cls):
            msg = e.render_template(
                "clone_for_injection expected {expected}, got {actual}",
                expected=cls.__name__,
                actual=type(settings).__name__,
            )
            raise TypeError(msg)
        return settings.clone()

    @classmethod
    def resolve_env_file(cls, namespace: str | None = None) -> str:
        """Centralised .env discovery (rule 5).

        Honours ``FLEXT_ENV_FILE`` env var; otherwise prefers a
        namespace-specific ``.env.flext-{namespace}`` (when ``namespace`` is
        given and the file exists) and falls back to the workspace ``.env``.
        """
        custom_env_file = os.environ.get(c.ENV_FILE_ENV_VAR)
        if custom_env_file:
            custom_path = Path(custom_env_file)
            if custom_path.exists():
                return str(custom_path.resolve())
            return custom_env_file
        if namespace:
            scoped = Path.cwd() / f".env.flext-{namespace}"
            if scoped.exists():
                return str(scoped.resolve())
        default_path = Path.cwd() / c.ENV_FILE_DEFAULT
        if default_path.exists():
            return str(default_path.resolve())
        return c.ENV_FILE_DEFAULT


__all__: list[str] = ["FlextSettingsBase"]
