"""Centralized TypeAdapter instances for flext-infra.

Provides SSOT TypeAdapter singletons for common validation patterns.
All modules should import these via ``t.Infra.<ADAPTER_NAME>`` instead
of creating local TypeAdapter instances.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_infra._constants.adapters import FlextInfraConstantsAdapters


class FlextInfraTypesAdapters(FlextInfraConstantsAdapters):
    """Centralized TypeAdapter instances for infrastructure validation.

    Usage::

        from flext_infra import t

        validated = t.Infra.INFRA_MAPPING_ADAPTER.validate_python(raw)
    """


__all__: list[str] = ["FlextInfraTypesAdapters"]
