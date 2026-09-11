"""Pytest hooks for enforcement dispatch."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from flext_tests._fixtures._enforcement_parts.build import build_items
from flext_tests._fixtures._enforcement_parts.config import (
    SessionConfig,
    active_rules,
    resolve_config,
)
from flext_tests.enforcement_plugin import SLOW_TIMEOUT_INI_OPTION

if TYPE_CHECKING:
    from flext_tests import p


def _apply_slow_timeout_policy(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply the config-owned item budget extension to explicit slow tests."""
    raw_timeout = str(config.getini(SLOW_TIMEOUT_INI_OPTION)).strip()
    if not raw_timeout:
        return
    try:
        slow_timeout = float(raw_timeout)
    except ValueError as err:
        msg = (
            "FLEXT slow timeout policy: "
            f"{SLOW_TIMEOUT_INI_OPTION} must be a positive finite number"
        )
        raise pytest.UsageError(msg) from err
    if not math.isfinite(slow_timeout) or slow_timeout <= 0:
        msg = (
            "FLEXT slow timeout policy: "
            f"{SLOW_TIMEOUT_INI_OPTION} must be a positive finite number"
        )
        raise pytest.UsageError(msg)
    if not any(
        config.pluginmanager.hasplugin(plugin_name)
        for plugin_name in ("timeout", "pytest_timeout")
    ):
        msg = "FLEXT slow timeout policy requires the pytest-timeout plugin"
        raise pytest.UsageError(msg)
    for item in items:
        if item.get_closest_marker("timeout") is not None:
            msg = (
                "FLEXT slow timeout policy: explicit pytest.mark.timeout is "
                f"forbidden; {item.nodeid} must use the config-owned item budget"
            )
            raise pytest.UsageError(msg)
        if item.get_closest_marker("slow") is not None:
            item.add_marker(pytest.mark.timeout(slow_timeout), append=False)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Append dispatcher items to the collection when active."""
    _apply_slow_timeout_policy(config, items)
    cfg = resolve_config(config)
    if not cfg.active:
        return
    if hasattr(config, "workerinput"):
        return
    generated = build_items(session, cfg, collected_items=items)
    if not generated:
        return
    items.extend(generated)


def pytest_warning_recorded(
    warning_message: p.AttributeProbe,
    when: str,
    nodeid: str,
    location: tuple[str, int, str] | None,
) -> None:
    """Track runtime enforcement warnings."""
    _ = when, nodeid, location
    if SessionConfig.value is None:
        return
    cfg = resolve_config(SessionConfig.value)
    if not cfg.active:
        return
    category = getattr(warning_message, "category", None)
    if category is None:
        return
    dotted = f"{category.__module__}.{category.__qualname__}"
    counter = cfg.warning_counter
    counter[dotted] = counter.get(dotted, 0) + 1


def pytest_sessionstart(session: pytest.Session) -> None:
    """Expose the session config for warning-capture plumbing."""
    SessionConfig.value = session.config


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter, exitstatus: int, config: pytest.Config
) -> None:
    """Print the per-kind breakdown at the end of the session."""
    _ = exitstatus
    cfg = resolve_config(config)
    if not cfg.active:
        return
    active = active_rules(cfg)
    kinds: dict[str, int] = {}
    for rule in active:
        kinds[rule.source.kind] = kinds.get(rule.source.kind, 0) + 1
    terminalreporter.write_sep("-", "flext-enforce", yellow=True)
    terminalreporter.write_line(
        f"catalog active: {len(active)} rules across {len(kinds)} source kinds"
    )
    for kind in sorted(kinds):
        terminalreporter.write_line(f"  {kind}: {kinds[kind]}")
    terminalreporter.write_line(
        f"runtime warnings captured: {sum(cfg.warning_counter.values())}"
    )


__all__: list[str] = [
    "pytest_collection_modifyitems",
    "pytest_sessionstart",
    "pytest_terminal_summary",
    "pytest_warning_recorded",
]
