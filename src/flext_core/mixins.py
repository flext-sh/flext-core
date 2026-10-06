"""Reusable service mixins facade.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import t
from flext_core._models.flext_mixins import FlextMixins

# NOTE (multi-agent): mro-i6nq.12 — consolidated _mixins_parts/part_01+part_02 into
# this single domain module and refactored the runtime-bootstrap tower
# (manual __init__/_coerce_*/_apply/property+setter pairs) into 4 native Pydantic
# fields mirroring m.RuntimeBootstrapOptions; dead track/_init_service/
# _register_in_container removed (zero callers).


# Inheritance base for FlextService/FlextHandlers (x namespace, consumed by
# service.py, _handlers_parts and flext-tests support bases).
x = FlextMixins

__all__: t.StrSequence = ("FlextMixins", "x")
