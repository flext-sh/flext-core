"""Generic service surface for flext-tests."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Self

from flext_core import FlextContainer, s
from flext_tests import m, p, t
from flext_tests._settings import FlextTestsSettings


class FlextTestsServiceBase[TDomainResult: p.Base = p.Base](s[TDomainResult]):
    """Base service for test packages with the ``Tests`` settings namespace."""

    @classmethod
    def test_settings_type(cls) -> type[FlextTestsSettings]:
        """Return the concrete test settings type declared by the service MRO."""
        settings_type = cls._runtime_bootstrap_options().settings_type
        if settings_type is None:
            return FlextTestsSettings
        if not issubclass(settings_type, FlextTestsSettings):
            msg = (
                f"{cls.__name__} must bootstrap a FlextTestsSettings subclass, "
                f"got {settings_type.__name__}"
            )
            raise TypeError(msg)
        concrete_settings_type: type[FlextTestsSettings] = settings_type
        return concrete_settings_type

    @classmethod
    def fetch_logger(cls) -> p.Logger:
        """Return the canonical logger bound to the shared test service singleton."""
        return cls.fetch_global().logger

    @classmethod
    def fetch_settings(cls) -> FlextTestsSettings:
        """Return the most specific test settings type declared by the service."""
        settings_type = cls.test_settings_type()
        resolved_settings = cls.fetch_global().settings
        if isinstance(resolved_settings, settings_type):
            return resolved_settings
        return settings_type.model_validate(resolved_settings)

    @classmethod
    def with_test_settings(cls, **overrides: t.SettingsOverride | None) -> Self:
        """Bind an isolated snapshot of the most specific test settings tree."""
        return cls.with_settings(cls.fetch_settings().clone(**overrides))

    @classmethod
    @contextmanager
    def isolated_test_runtime(
        cls, **overrides: t.SettingsOverride | None
    ) -> Generator[Self]:
        """Yield one isolated service/settings/container cycle for a test body."""
        settings_type = cls.test_settings_type()
        FlextContainer.reset_for_testing()
        settings_type.reset_for_testing()
        cls.reset_for_testing()
        try:
            yield (
                cls.with_test_settings(**overrides) if overrides else cls.fetch_global()
            )
        finally:
            cls.reset_for_testing()
            settings_type.reset_for_testing()
            FlextContainer.reset_for_testing()

    @classmethod
    def _runtime_bootstrap_options(cls) -> m.RuntimeBootstrapOptions:
        return m.RuntimeBootstrapOptions(settings_type=FlextTestsSettings)


class FlextTestsCase:
    """Pytest class MRO surface populated by the shared test runtime fixture."""

    service: FlextTestsServiceBase[p.Base]
    settings: FlextTestsSettings
    logger: p.Logger
    c: type
    e: type
    m: type
    p: type
    r: type
    t: type
    u: type


s = FlextTestsServiceBase

__all__: list[str] = ["FlextTestsCase", "FlextTestsServiceBase", "s"]
