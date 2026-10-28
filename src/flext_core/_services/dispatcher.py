"""FlextServiceDispatcher - dispatcher access via MRO.

Copyright (c) 2026 FLEXT Team. All rights reserved.
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

from flext_core import p, s


class FlextServiceDispatcher(s[p.Base]):
    """Provide dispatcher access through the public facade."""

    @property
    def dispatcher(self) -> p.Dispatcher:
        """Dispatcher from this component's container."""
        return self.container.dispatcher().unwrap()


__all__: list[str] = ["FlextServiceDispatcher"]
