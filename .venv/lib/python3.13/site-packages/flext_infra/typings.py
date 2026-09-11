"""Type aliases for flext-infra.

Re-exports and extends flext-cli typings for infrastructure services.
Infra-specific type aliases live inside ``FlextInfraTypes`` so they are
accessed via ``t.Infra.Payload``, ``t.JsonMapping``, etc.

Non-recursive aliases use ``type X = ...`` (PEP 695 Python 3.13+ syntax).
See AGENTS.md §3 AXIOMATIC rule.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_cli import t
from flext_infra._typings.adapters import FlextInfraTypesAdapters
from flext_infra._typings.base import FlextInfraTypesBase
from flext_infra._typings.rope import FlextInfraTypesRope


class FlextInfraTypes(t):
    """Type namespace for flext-infra; extends FlextTypes via FLEXT.

    Infra-specific types are nested under the ``Infra`` inner class to
    keep the namespace explicit (``t.Infra.Payload``, ``t.JsonMapping``).
    Parent types (``t.Scalar``, ``t.StrMapping``, etc.) are inherited
    transparently from ``FlextTypes`` via FLEXT.
    """

    class Infra(FlextInfraTypesAdapters, FlextInfraTypesRope, FlextInfraTypesBase):
        """Infrastructure-domain type aliases.

        These aliases compose CLI JSON contracts and infrastructure-specific
        collection generics for common tooling payloads.
        """


t = FlextInfraTypes

__all__: list[str] = ["FlextInfraTypes", "t"]
