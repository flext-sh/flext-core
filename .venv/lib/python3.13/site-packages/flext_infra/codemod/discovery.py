"""Discover ast-grep rules from installed FLEXT packages in cascade order."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Final

# Cascade order: last wins on rule ID conflict. The local project is appended
# by the caller through ``extra_packages`` so it always overrides the library.
_CASCADE_PACKAGES: Final[tuple[str, ...]] = ("flext_core", "flext_cli", "flext_infra")


def discover_rules(*extra_packages: str) -> list[Path]:
    """Discover ast-grep rule files from installed packages.

    Rules are read from each package's ``codemod/rules/`` directory via
    importlib.resources, so they travel inside the wheel instead of being
    projected into every repository. Packages are searched in cascade order;
    later packages override earlier ones on rule ID (filename stem) conflict.

    Args:
        extra_packages: Additional package names to search after the cascade
            (e.g. the local project package).

    Returns:
        Sorted list of rule file paths, deduplicated by rule ID.

    """
    rules: dict[str, Path] = {}
    for pkg_name in (*_CASCADE_PACKAGES, *extra_packages):
        try:
            rules_dir = Path(str(files(pkg_name) / "codemod" / "rules"))
        except (ModuleNotFoundError, FileNotFoundError, TypeError):
            continue
        if not rules_dir.is_dir():
            continue
        for rule_file in sorted(rules_dir.rglob("*.yml")):
            rules[rule_file.stem] = rule_file
    return sorted(rules.values())


def discover_rule_ids(*extra_packages: str) -> list[str]:
    """Return rule IDs (filename stems) for every discovered rule."""
    return [rule.stem for rule in discover_rules(*extra_packages)]


__all__: list[str] = ["discover_rule_ids", "discover_rules"]
