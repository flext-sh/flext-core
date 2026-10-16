"""Type-safe result type for operations.

The composed private base lives in the ``_result`` family package
(one top-level class per module, NS-000); this namespace module publishes
the public concrete facade and the canonical ``r`` letter alias.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._result import FlextResultComposed


class FlextResult[T](FlextResultComposed[T]):
    """Public concrete result facade; runtime and typing share one MRO.

    Instances carry either a success payload or a failure message plus the
    exception that produced it. Every operation returns a new ``FlextResult``
    so failures propagate explicitly instead of raising at the call site;
    use the ``unwrap``/``behavior`` helpers to extract payloads or bind the
    error. ``r`` is the canonical letter alias for this class.
    """


r = FlextResult


__all__: list[str] = ["FlextResult", "r"]
