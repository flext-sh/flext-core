"""Reusable service mixins facade.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import flext_core._models.flext_mixins

# NOTE (multi-agent): mro-i6nq.12 — consolidated _mixins_parts/part_01+part_02 into
# this single domain module and refactored the runtime-bootstrap tower
# (manual __init__/_coerce_*/_apply/property+setter pairs) into 4 native Pydantic
# fields mirroring m.RuntimeBootstrapOptions; dead track/_init_service/
# _register_in_container removed (zero callers).

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, MutableMapping


x = flext_core._models.flext_mixins.FlextMixins

__all__: list[str] = ["FlextMixins", "x"]
