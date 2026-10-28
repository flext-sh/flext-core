"""Constants for flext.

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

from collections.abc import MutableSequence

from flext_core import FlextConstants


class ScriptsFlextConstants(FlextConstants):
    """Constants for flext."""


c = ScriptsFlextConstants

__all__: MutableSequence[str] = ["ScriptsFlextConstants", "c"]
