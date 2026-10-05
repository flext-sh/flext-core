"""Namespace-target enforcement constants for FlextConstantsEnforcement.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Mapping


class FlextConstantsEnforcementTargets:
    """Target sets and external library ownership constants."""

    ENFORCEMENT_CANONICAL_FILES: ClassVar[frozenset[str]] = frozenset({
        "constants.py",
        "models.py",
        "protocols.py",
        "typings.py",
        "utilities.py",
    })
    """The five canonical facade files per project (AGENTS.md §2.2)."""

    ENFORCEMENT_PRIVATE_FAMILY_PACKAGES: ClassVar[frozenset[str]] = frozenset(
        f"_{name.removesuffix('.py')}" for name in ENFORCEMENT_CANONICAL_FILES
    )
    """Private-family sub-package suffixes derived from canonical files.

    When a canonical file (``constants.py``) imports a class whose runtime
    origin lives under the matching private sub-package (``_constants``),
    the ``no_concrete_namespace_import`` rule (ENFORCE-046) treats it as a
    legitimate local family import — not a forbidden bare ``Flext*`` class
    import.  Facade classes must compose their private family
    (``FlextApi[Tipo]*``) classes through MRO (R1), so this exemption is
    scoped to same-package, private-sub-package origins only.

    Evaluation is automatic: derived from :attr:`ENFORCEMENT_CANONICAL_FILES`,
    so adding a new canonical file automatically extends the allowed family
    set.  No parallel hardcoded list exists.
    """

    ENFORCEMENT_LIBRARY_OWNERS: ClassVar[Mapping[str, str]] = MappingProxyType({
        "pydantic": "flext-core",
        "pydantic_settings": "flext-core",
        "pydantic_core": "flext-core",
        "dependency_injector": "flext-core",
        "returns": "flext-core",
        "structlog": "flext-core",
        "rich": "flext-cli",
        "rope": "flext-infra",
        "orjson": "flext-cli",
        # Why: flext-core declares pyyaml as its own direct runtime dependency
        # and uses it in _config.py (foundational config loading); flext-cli
        # depends on flext-core, so the owner must be the lower layer.
        "yaml": "flext-core",
        "pyyaml": "flext-core",
        "click": "flext-cli",
        "ldap3": "flext-ldap",
        "singer_sdk": "flext-meltano",
        "sqlalchemy": "flext-db-oracle",
        "oracledb": "flext-db-oracle",
        "grpc": "flext-grpc",
        "fastapi": "flext-web",
        "httpx": "flext-api",
    })
    """SSOT mapping: external library → owning FLEXT abstraction project (§2.7).

    Every consumer accesses these libraries via the owning project's facades
    (``c/m/p/t/u``), never via a bare top-level import. The runtime
    LIBRARY_IMPORT predicate (``m.Enforcement.LibraryImportParams``) and the
    rope-based source-level tier-whitelist validator both source their data
    from this mapping. Adding a new abstracted library = one entry here, no
    parallel list elsewhere.
    """


__all__: list[str] = ["FlextConstantsEnforcementTargets"]
