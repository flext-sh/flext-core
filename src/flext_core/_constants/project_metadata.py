"""FlextConstantsProjectMetadata — canonical project-metadata SSOT.

Tier 0 module: imports nothing from flext-core. Owned solely by
flext-core; every other reader/writer in the monorepo references this
class via ``c.Project.*``. Duplication of these values in flext-infra,
flext-tests, or any other subproject is forbidden by the
metadata-discipline enforcement rule.

Architecture: Tier 0 — zero internal dependencies (safe from cycles).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar, Final


class FlextConstantsProjectMetadata:
    """Namespace holder for project-metadata constants SSOT.

    The actual constants live under the nested ``Project`` class so that
    ``FlextConstants`` can inherit this class via MRO and expose
    ``c.Project.ALIAS_TO_SUFFIX`` (sub-namespace access) rather than
    flat ``c.ALIAS_TO_SUFFIX``.
    """

    _flext_enforcement_exempt: ClassVar[bool] = True

    class Project:
        """Project-metadata constants — SSOT (accessible as ``c.Project.*``)."""

        _flext_enforcement_exempt: ClassVar[bool] = True

        ALIAS_TO_SUFFIX: Final[Mapping[str, str]] = MappingProxyType(
            {
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
            }
        )

        RUNTIME_ALIAS_NAMES: Final[frozenset[str]] = frozenset(ALIAS_TO_SUFFIX)

        TIER_FACADE_PREFIX: Final[Mapping[str, str]] = MappingProxyType(
            {
                "src": "Flext",
                "tests": "TestsFlext",
                "examples": "ExamplesFlext",
                "scripts": "ScriptsFlext",
                "docs": "DocsFlext",
            }
        )

        SCAN_DIRECTORIES: Final[tuple[str, ...]] = tuple(TIER_FACADE_PREFIX)

        TIER_SUB_NAMESPACE: Final[Mapping[str, str]] = MappingProxyType(
            {
                "src": "",
                "tests": "Tests",
                "examples": "Examples",
                "scripts": "Scripts",
                "docs": "Docs",
            }
        )

        UNIVERSAL_ALIAS_PARENT_SOURCES: Final[Mapping[str, str]] = MappingProxyType(
            {
                "r": "flext_core",
                "e": "flext_core",
                "d": "flext_core",
                "x": "flext_core",
            }
        )

        SPECIAL_NAME_OVERRIDES: Final[Mapping[str, str]] = MappingProxyType(
            {
                "flext": "FlextRoot",
                "flext-core": "Flext",
            }
        )

        MANAGED_PYPROJECT_KEYS: Final[tuple[str, ...]] = (
            "tool.flext.project",
            "tool.flext.namespace",
            "tool.flext.docs",
            "tool.flext.aliases",
        )
