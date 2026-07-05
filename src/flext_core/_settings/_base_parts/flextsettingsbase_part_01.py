"""FlextSettingsBase singleton and override helpers."""

from __future__ import annotations

import threading
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar, Self, Unpack

from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from flext_core._typings.base import FlextTypingBase as tb
    from flext_core._typings.services import FlextTypesServices as ts


class FlextSettingsBase(BaseSettings):
    """Pydantic-2 base + per-class singleton + canonical helper API.

    Every settings class in the workspace inherits from this base, gaining the
    fetch/clone/update API, env-loading support, and an isolated singleton
    slot.
    """

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    _lock: ClassVar[threading.RLock] = threading.RLock()

    _singleton_enabled: ClassVar[bool] = True

    _instance: ClassVar[FlextSettingsBase | None] = None

    def __init_subclass__(cls, **kwargs: Unpack[ConfigDict]) -> None:
        """Inject a per-class ``_instance`` slot for every concrete subclass."""
        _ = kwargs
        super().__init_subclass__()
        cls._instance = None

    def __new__(cls, **kwargs: ts.SettingsInput) -> Self:
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
    def _initialized_instance(cls) -> Self | None:
        """Return the cached singleton only after Pydantic finished initialization."""
        existing = getattr(cls, "_instance", None)
        if isinstance(existing, cls) and hasattr(existing, "__pydantic_fields_set__"):
            return existing
        if existing is not None:
            with cls._lock:
                if getattr(cls, "_instance", None) is existing:
                    cls._instance = None
        return None

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
    def fetch_global(cls, *, overrides: tb.ScalarMapping | None = None) -> Self:
        """Return the shared per-class singleton (rule 1).

        When ``overrides`` is given, return an isolated deep clone instead —
        the global singleton is NOT mutated (use ``update_global`` for that).
        """
        instance = cls._initialized_instance()
        if overrides:
            if instance is not None:
                return instance.clone(**overrides)
            with cls.singleton_disabled():
                return cls.model_validate(dict(overrides))
        if instance is not None:
            return instance
        with cls._lock:
            instance = cls._initialized_instance()
            if instance is not None:
                return instance
            with cls.singleton_disabled():
                created = cls()
            cls._instance = created
            return created

    @classmethod
    def merge_overrides(
        cls,
        settings: Self,
        **overrides: ts.SettingsOverride | None,
    ) -> dict[str, ts.SettingsOverride | None]:
        """Merge partial nested-model overrides onto the current settings state."""
        cls.validate_overrides(**overrides)
        merged_overrides: dict[str, ts.SettingsOverride | None] = {}
        for field_name, override_value in overrides.items():
            current_value = getattr(settings, field_name, None)
            if isinstance(current_value, BaseModel) and isinstance(
                override_value,
                Mapping,
            ):
                computed = set(type(current_value).model_computed_fields)
                merged_dict = {
                    **current_value.model_dump(mode="python", exclude=computed),
                    **{k: v for k, v in override_value.items() if k not in computed},
                }
                merged_overrides[field_name] = type(current_value).model_validate(
                    merged_dict,
                )
                continue
            merged_overrides[field_name] = override_value
        return merged_overrides

    def clone(self, **overrides: ts.SettingsOverride | None) -> Self:
        """Deep copy with optional field overrides + re-validation (rule 2).

        Used by service/container constructors that accept an explicit
        ``settings=`` argument so the caller's lifetime owns its snapshot
        independent of the global singleton. Nested submodels are deep-copied.
        """
        if not overrides:
            with self.__class__.singleton_disabled():
                return self.model_copy(deep=True)
        merged_overrides = self.__class__.merge_overrides(self, **overrides)
        with self.__class__.singleton_disabled():
            copied = self.model_copy(update=merged_overrides, deep=True)
            return type(copied).model_validate(copied, from_attributes=True)

    @classmethod
    def update_global(cls, **overrides: ts.SettingsOverride | None) -> Self:
        """Replace ``cls._instance`` via ``model_copy(update=…)`` + revalidate.

        Pure Pydantic-2 mutation: no ``setattr``, no ``__setattr__`` override,
        no ``apply_override`` ad-hoc method. Subsequent ``fetch_global()``
        calls return the new instance — propagates per rule 1.

        Raises:
            ValueError: if any override key is not a declared model field.

        """
        if not overrides:
            return cls.fetch_global()
        current = cls.fetch_global()
        merged_overrides = cls.merge_overrides(current, **overrides)
        with cls.singleton_disabled():
            new_instance = current.model_copy(update=merged_overrides, deep=True)
            validated = cls.model_validate(new_instance, from_attributes=True)
        with cls._lock:
            cls._instance = validated
        return validated

    @classmethod
    def validate_overrides(cls, **overrides: ts.SettingsOverride | None) -> None:
        """Reject override keys that are not declared model fields.

        Typo guard at CLI/runtime override boundaries.
        """
        unknown = sorted(set(overrides) - set(cls.model_fields))
        if unknown:
            msg = (
                f"Unknown settings override(s) for {cls.__name__}: {', '.join(unknown)}"
            )
            raise ValueError(msg)


__all__: list[str] = ["FlextSettingsBase"]
