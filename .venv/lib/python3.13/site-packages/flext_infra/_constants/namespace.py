"""Bootstrap-safe namespace constants."""

from __future__ import annotations

from typing import Final


class FlextInfraConstantsNamespace:
    """Namespace constants shared by bootstrap-sensitive utilities."""

    NAMESPACE_SETTINGS_FILE_NAMES: Final[frozenset[str]] = frozenset({
        "settings.py",
        "_settings.py",
    })
    NAMESPACE_PROTECTED_FILES: Final[frozenset[str]] = frozenset({
        "settings.py",
        "_settings.py",
        "typings.py",
        "_typings.py",
        "__init__.py",
        "__main__.py",
        "__version__.py",
        "conftest.py",
        "py.typed",
    })
    NAMESPACE_CANONICAL_ALIAS_MODULE_STEMS: Final[frozenset[str]] = frozenset({
        "ldif",
        "cli",
        "main",
    })


__all__: list[str] = ["FlextInfraConstantsNamespace"]
