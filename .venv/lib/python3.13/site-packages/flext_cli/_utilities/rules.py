"""Generic local-rule loading helpers shared through ``u.Cli.rules_*``."""

from __future__ import annotations

from flext_cli._utilities._rules._loaders import FlextCliUtilitiesRulesLoadersMixin

# NOTE (multi-agent): mro-i6nq.13 — composed from the _rules/{_loaders,_matchers}
# mixin chain (replacing the numbered _rules_parts).


class FlextCliUtilitiesRules(FlextCliUtilitiesRulesLoadersMixin):
    """Public facade for the generic local-rule helpers behind ``u.Cli.rules_*``."""


__all__: list[str] = ["FlextCliUtilitiesRules"]
