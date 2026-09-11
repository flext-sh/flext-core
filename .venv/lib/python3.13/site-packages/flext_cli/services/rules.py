"""DSL service for declarative local rule loading."""

from __future__ import annotations

from flext_cli import s, t
from flext_cli._utilities.rules import FlextCliUtilitiesRules


class FlextCliRules(s, FlextCliUtilitiesRules):
    """Expose the generic rule-loading DSL through ``cli`` and ``u.Cli``."""


__all__: t.MutableSequenceOf[str] = ["FlextCliRules"]
