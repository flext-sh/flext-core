"""Facade for FlextUtilitiesMapperExtract.

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

from flext_core._utilities._mapper_extract.mapper_extract_part_02 import (
    FlextUtilitiesMapperExtractPart02,
)


class FlextUtilitiesMapperExtract(FlextUtilitiesMapperExtractPart02):
    """Compose mapper extraction through its internal part hierarchy."""


__all__: list[str] = ["FlextUtilitiesMapperExtract"]
