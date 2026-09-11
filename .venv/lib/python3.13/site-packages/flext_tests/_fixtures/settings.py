"""Centralized FlextSettings fixtures for all project test suites.

Provides:
- reset_settings: Fixture that resets FlextSettings singleton between tests
- settings: Base FlextSettings fixture with test defaults
- settings_factory: Factory fixture for creating project-specific settings

Eliminates duplicated reset_settings_singleton / _reset_instance patterns
found across flext-core, flext-oracle-wms, flext-target-oracle, flext-auth,
flext-cli, and other project conftest.py files.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from flext_core import FlextContainer, FlextContext, FlextSettings
from flext_tests import c, e, m, p, r, s, t, u
from flext_tests._settings import FlextTestsSettings
from flext_tests.base import FlextTestsCase

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path


def _reset_runtime_state() -> None:
    """Reset root/test settings singletons and the DI container."""
    FlextSettings.reset_for_testing()
    FlextTestsSettings.reset_for_testing()
    FlextContainer.reset_for_testing()


def _bind_runtime_aliases(
    *, module: ModuleType, instance: FlextTestsCase | None
) -> None:
    """Bind canonical FLEXT runtime aliases onto pytest class instances."""
    package_root = module.__package__ or module.__name__
    package_name = package_root.split(".", maxsplit=1)[0]
    tests_package = importlib.import_module(package_name)
    service_type = getattr(tests_package, "s", s)
    if not isinstance(service_type, type) or not issubclass(service_type, s):
        service_type = s
    service = service_type.fetch_global()
    if instance is None:
        return
    instance.service = service
    instance.settings = service.fetch_settings()
    instance.logger = service.logger
    for alias_name, fallback in (
        ("c", c),
        ("e", e),
        ("m", m),
        ("p", p),
        ("r", r),
        ("t", t),
        ("u", u),
    ):
        setattr(instance, alias_name, getattr(tests_package, alias_name, fallback))


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Reset and bind the canonical runtime before each pytest function."""
    _reset_runtime_state()
    if not isinstance(item, pytest.Function):
        return
    module = item.module
    if not isinstance(module, ModuleType):
        return
    instance = item.instance
    _bind_runtime_aliases(
        module=module,
        instance=instance if isinstance(instance, FlextTestsCase) else None,
    )


def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> None:
    """Reset the canonical runtime after each pytest function."""
    _ = item, nextitem
    _reset_runtime_state()


@pytest.fixture
def reset_settings() -> Iterator[None]:
    """Reset root + test settings singletons and the DI container.

    Prevents state leakage across tests by clearing both the root settings
    singleton, the test-settings singleton, and the container singleton.
    Runs as autouse so every test gets clean state automatically.
    """
    _reset_runtime_state()
    try:
        yield
    finally:
        _reset_runtime_state()


@pytest.fixture
def test_runtime(request: pytest.FixtureRequest, reset_settings: None) -> None:
    """Bind the canonical FLEXT test runtime aliases onto pytest class instances."""
    _ = reset_settings
    if not isinstance(request.node, pytest.Function):
        return
    instance = request.instance
    _bind_runtime_aliases(
        module=request.module,
        instance=instance if isinstance(instance, FlextTestsCase) else None,
    )


@pytest.fixture
def settings() -> FlextTestsSettings:
    """Provide a clean FlextTestsSettings instance with test defaults.

    Resets the root/test singletons first, then creates a fresh typed
    tests settings instance with verbose core defaults.
    """
    FlextSettings.reset_for_testing()
    FlextTestsSettings.reset_for_testing()
    return FlextTestsSettings(debug=True, trace=False)


@pytest.fixture
def settings_factory() -> Callable[..., FlextSettings]:
    """Create project-specific settings instances in tests.

    Usage in test::

        def test_something(settings_factory):
            settings = settings_factory(FlextApiSettings, base_url="http://test")
            assert settings.base_url == "http://test"

    The factory resets the singleton for the given class before creating
    a new instance, ensuring test isolation.
    """

    def _create[TSettings: FlextSettings](
        settings_cls: type[TSettings], **overrides: t.Scalar | None
    ) -> TSettings:
        settings_cls.reset_for_testing()
        filtered_overrides: t.ScalarMapping = {
            key: value for key, value in overrides.items() if value is not None
        }
        return settings_cls.fetch_global(overrides=filtered_overrides)

    return _create


@pytest.fixture
def test_context() -> FlextContext:
    """Provide the canonical FlextContext fixture for test runtimes."""
    return FlextContext()


@pytest.fixture
def clean_container() -> Iterator[p.Container]:
    """Provide an isolated DI container for tests that mutate registrations."""
    FlextContainer.reset_for_testing()
    try:
        yield FlextContainer()
    finally:
        FlextContainer.reset_for_testing()


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provide the canonical temporary directory fixture for tests."""
    return tmp_path


@pytest.fixture
def temp_file(temp_dir: Path) -> Path:
    """Provide the canonical temporary file path rooted at ``temp_dir``."""
    return temp_dir / "test_file.txt"


@pytest.fixture
def sample_data() -> t.JsonMapping:
    """Provide a small canonical JSON payload for generic test cases."""
    return {"string": "test_value", "number": 42, "enabled": True}
