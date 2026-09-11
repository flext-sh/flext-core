"""Types extraction for flext_tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TypeIs

from flext_core import m
from flext_tests._typings.base import FlextTestsBaseTypesMixin as tb


class FlextTestsGuardsTypesMixin:
    @staticmethod
    def general_value(value: tb.Testobject) -> TypeIs[tb.Testobject]:
        if value is None:
            return True
        if isinstance(value, (str, int, float, bool, bytes)):
            return True
        if isinstance(value, m.BaseModel):
            return True
        return isinstance(value, (list, dict))
