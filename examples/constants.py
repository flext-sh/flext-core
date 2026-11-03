"""Public examples constants facade for flext-core.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum, unique
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

from flext_core import c as flext_core_c

if TYPE_CHECKING:
    from flext_core import t


class ExamplesFlextConstants(flext_core_c):
    """Public examples constants facade extending flext-core constants."""

    class Examples:
        """Canonical namespace for example constants."""

        @unique
        class StatusEnum(StrEnum):
            """User account status of the mixins example."""

            ACTIVE = "active"
            PENDING = "pending"
            INACTIVE = "inactive"

        class OutputKind(StrEnum):
            """Supported output result kinds."""

            SUCCESS = "PASS"
            FAIL = "FAIL"
            GENERATED = "GENERATED"

        class OutputTemplate(StrEnum):
            """Canonical output templates for example verification."""

            SUCCESS = "{kind}: {stem} ({checks} checks)\\n"
            FAIL = "{kind}: {stem} — diff {expected_name} {actual_name}\\n"
            GENERATED = "{kind}: {expected_name} ({checks} checks)\\n"

        LABEL_VALUE_SEPARATOR: ClassVar[str] = ": "
        RESULT_LINE_PATTERN: ClassVar[t.RegexPattern] = (
            flext_core_c.PATTERN_EXAMPLE_RESULT_LINE_RE
        )
        TEMPLATE_BY_KIND: ClassVar[Mapping[OutputKind, OutputTemplate]] = (
            MappingProxyType({
                OutputKind.SUCCESS: OutputTemplate.SUCCESS,
                OutputKind.FAIL: OutputTemplate.FAIL,
                OutputKind.GENERATED: OutputTemplate.GENERATED,
            })
        )


c = ExamplesFlextConstants
__all__: list[str] = ["ExamplesFlextConstants", "c"]
