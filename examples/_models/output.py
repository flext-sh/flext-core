"""Centralized output strings and patterns for public examples."""

from __future__ import annotations

from collections.abc import (
    Mapping,
)
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

from examples import c

if TYPE_CHECKING:
    from examples import t


class ExamplesFlextModelsOutput:
    """Output namespace used by example scripts."""

    class Examples:
        """Examples output namespace."""

        class OutputKind(StrEnum):
            """Supported output result kinds."""

            PASS = "PASS"
            FAIL = "FAIL"
            GENERATED = "GENERATED"

        class OutputTemplate(StrEnum):
            """Canonical output templates for example verification."""

            PASS = "{kind}: {stem} ({checks} checks)\\n"
            FAIL = "{kind}: {stem} — diff {expected_name} {actual_name}\\n"
            GENERATED = "{kind}: {expected_name} ({checks} checks)\\n"

        LABEL_VALUE_SEPARATOR: ClassVar[str] = ": "
        RESULT_LINE_PATTERN: ClassVar[t.RegexPattern] = c.PATTERN_EXAMPLE_RESULT_LINE_RE
        TEMPLATE_BY_KIND: ClassVar[Mapping[OutputKind, OutputTemplate]] = (
            MappingProxyType(
                {
                    OutputKind.PASS: OutputTemplate.PASS,
                    OutputKind.FAIL: OutputTemplate.FAIL,
                    OutputKind.GENERATED: OutputTemplate.GENERATED,
                },
            )
        )
