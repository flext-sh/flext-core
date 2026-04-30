"""FlextConstantsProjectMetadata - canonical project-metadata SSOT."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

from flext_core._typings.base import FlextTypingBase as t


class FlextConstantsProjectMetadata:
    """Canonical project-metadata constants exposed flat on ``c.*``."""

    ALIAS_TO_SUFFIX: Final[t.StrMapping] = MappingProxyType({
        "c": "Constants",
        "d": "Decorators",
        "e": "Exceptions",
        "h": "Handlers",
        "m": "Models",
        "p": "Protocols",
        "r": "Result",
        "s": "Service",
        "t": "Types",
        "u": "Utilities",
        "x": "Mixins",
    })

    RUNTIME_ALIAS_NAMES: Final[frozenset[str]] = frozenset(ALIAS_TO_SUFFIX)
    FACADE_ALIAS_NAMES: Final[frozenset[str]] = frozenset({"c", "m", "p", "t", "u"})
    FACADE_MODULE_NAMES: Final[frozenset[str]] = frozenset({
        "constants",
        "models",
        "protocols",
        "typings",
        "utilities",
    })

    TIER_FACADE_PREFIX: Final[t.StrMapping] = MappingProxyType({
        "src": "Flext",
        "tests": "TestsFlext",
        "examples": "ExamplesFlext",
        "scripts": "ScriptsFlext",
        "docs": "DocsFlext",
    })
    SCAN_DIRECTORIES: Final[t.VariadicTuple[str]] = tuple(TIER_FACADE_PREFIX)
    TIER_SUB_NAMESPACE: Final[t.StrMapping] = MappingProxyType({
        "src": "",
        "tests": "Tests",
        "examples": "Examples",
        "scripts": "Scripts",
        "docs": "Docs",
    })
    UNIVERSAL_ALIAS_PARENT_SOURCES: Final[t.StrMapping] = MappingProxyType({
        "r": "flext_core",
        "e": "flext_core",
        "d": "flext_core",
        "x": "flext_core",
    })
    SPECIAL_NAME_OVERRIDES: Final[t.StrMapping] = MappingProxyType({
        "flext": "FlextRoot",
        "flext-core": "Flext",
    })

    PYPROJECT_FILENAME: Final[str] = "pyproject.toml"
