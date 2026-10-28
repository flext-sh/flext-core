"""FlextModelsNamespace — common root of every facade namespace class.

Namespace governance is static analysis owned by flext-infra's declarative
rule catalog; class creation runs no enforcement.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from collections.abc import MutableSequence


class FlextModelsNamespace:
    """Common root shared by the facade namespace classes (c, p, t, m, u)."""


__all__: MutableSequence[str] = ["FlextModelsNamespace"]
