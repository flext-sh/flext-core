"""Pytest plugin wiring the canonical flext-tests fixture modules.

Usage in any project's conftest.py::

    pytest_plugins = ["flext_tests.conftest_plugin"]

This plugin delegates to the shared markdown-validation and test-runtime
fixture modules so projects get one canonical owner for CLI options,
autouse runtime setup, and shared helper fixtures.
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the local markdown fallback without eager product imports."""
    if find_spec("pytest_markdown_docs") is not None:
        return
    from flext_tests._fixtures.markdown_validation import pytest_addoption as register

    register(parser)


def pytest_configure(config: pytest.Config) -> None:
    """Register fixture plugins after startup instrumentation is active."""
    # `_fixtures/__init__` exports a fixture function named `settings`, which
    # shadows the submodule of the same name: `from ... import settings` binds
    # the fixture, and registering a function as a plugin silently registers no
    # fixtures at all. `import_module` names the module unambiguously.
    settings = import_module("flext_tests._fixtures.settings")
    connectivity = import_module("flext_tests._fixtures.connectivity")

    if settings not in config.pluginmanager.get_plugins():
        config.pluginmanager.register(settings, settings.__name__)
    # Connectivity-bound tests skip - never fail - when their external service
    # is unreachable (AGENTS.md external/docker skip rule).
    if connectivity not in config.pluginmanager.get_plugins():
        config.pluginmanager.register(connectivity, connectivity.__name__)
    if find_spec("pytest_markdown_docs") is None:
        from flext_tests._fixtures import markdown_validation

        if markdown_validation not in config.pluginmanager.get_plugins():
            config.pluginmanager.register(
                markdown_validation, markdown_validation.__name__
            )


# Enforcement dispatcher (flext_tests._fixtures.enforcement) is loaded via
# the ``flext_tests_enforcement`` pytest11 entry point in pyproject.toml —
# re-exporting its hooks here would double-register CLI options when both
# paths are active. The plugin module registers the canonical fixture modules
# during ``pytest_configure``. It must not expose ``pytest_plugins``: pytest
# processes that declaration while loading the ``pytest11`` entry point,
# before pytest-cov starts measuring the product package.


__all__: list[str] = ["pytest_addoption", "pytest_configure"]
