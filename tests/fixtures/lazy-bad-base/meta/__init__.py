"""Package whose export has a metaclass other than type.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC


class Abstract(ABC):
    """Real class export whose metaclass is ``ABCMeta``."""

    KIND = "abstract"


__all__: list[str] = ["Abstract"]
