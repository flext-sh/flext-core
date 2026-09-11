"""DSL service for external process runtime helpers."""

from __future__ import annotations

from flext_cli import s, t
from flext_cli._utilities.runtime import FlextCliUtilitiesRuntime


class FlextCliRuntime(s, FlextCliUtilitiesRuntime):
    """Expose process execution helpers through ``cli`` and ``FlextCli``."""


__all__: t.MutableSequenceOf[str] = ["FlextCliRuntime"]
