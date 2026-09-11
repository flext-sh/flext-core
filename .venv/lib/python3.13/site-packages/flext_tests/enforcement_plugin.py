"""Lightweight ``pytest11`` adapter for the enforcement dispatcher.

Pytest imports installed entry points before pytest-cov starts measurement.
This module therefore owns only the external hook boundary and defers product
imports until each lifecycle hook is actually called.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

    from flext_tests import p

SLOW_TIMEOUT_INI_OPTION = "flext_slow_timeout_seconds"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the enforcement dispatcher's stable command-line contract."""
    parser.addini(
        SLOW_TIMEOUT_INI_OPTION,
        "Config-owned timeout in seconds for items explicitly marked slow.",
        default="",
    )
    group = parser.getgroup("flext-enforce", "FLEXT cross-layer enforcement catalog")
    group.addoption(
        "--flext-enforce",
        action="store_true",
        default=False,
        help=(
            "Force-enable the FLEXT enforcement dispatcher (default: auto - "
            "enabled when the pytest rootdir is the workspace root)."
        ),
    )
    group.addoption(
        "--no-flext-enforce",
        action="store_true",
        default=False,
        help="Disable the FLEXT enforcement dispatcher.",
    )
    group.addoption(
        "--flext-enforce-strict",
        action="store_true",
        default=False,
        help="Promote runtime enforcement warnings to pytest failures.",
    )
    group.addoption(
        "--flext-enforce-rules",
        action="store",
        default="",
        help="Comma-separated ENFORCE-NNN allow list.",
    )
    group.addoption(
        "--flext-enforce-exclude-rules",
        action="store",
        default="",
        help="Comma-separated ENFORCE-NNN block list.",
    )
    group.addoption(
        "--flext-enforce-workspace-root",
        action="store",
        default="",
        help="Override workspace root auto-detection with an explicit path.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Resolve enforcement only after startup instrumentation is active."""
    from flext_tests._fixtures._enforcement_parts.config import pytest_configure

    pytest_configure(config)


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Delegate collection-time enforcement."""
    from flext_tests._fixtures._enforcement_parts.hooks import (
        pytest_collection_modifyitems,
    )

    pytest_collection_modifyitems(session, config, items)


def pytest_warning_recorded(
    warning_message: p.AttributeProbe,
    when: str,
    nodeid: str,
    location: tuple[str, int, str] | None,
) -> None:
    """Delegate runtime-warning capture."""
    from flext_tests._fixtures._enforcement_parts.hooks import pytest_warning_recorded

    pytest_warning_recorded(warning_message, when, nodeid, location)


def pytest_sessionstart(session: pytest.Session) -> None:
    """Delegate session initialization."""
    from flext_tests._fixtures._enforcement_parts.hooks import pytest_sessionstart

    pytest_sessionstart(session)


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter, exitstatus: int, config: pytest.Config
) -> None:
    """Delegate the enforcement summary."""
    from flext_tests._fixtures._enforcement_parts.hooks import pytest_terminal_summary

    pytest_terminal_summary(terminalreporter, exitstatus, config)


__all__: list[str] = [
    "SLOW_TIMEOUT_INI_OPTION",
    "pytest_addoption",
    "pytest_collection_modifyitems",
    "pytest_configure",
    "pytest_sessionstart",
    "pytest_terminal_summary",
    "pytest_warning_recorded",
]
