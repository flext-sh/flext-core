"""Validation and failing service factory helpers for flext-core tests.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests import p
from tests._utilities import TestsFlextUtilitiesServicesMixin

if TYPE_CHECKING:
    from collections.abc import Callable


class TestsFlextUtilitiesValidationFactoriesMixin(
    TestsFlextUtilitiesServicesMixin,
):
    """Validation and failing service factory helpers."""

    @staticmethod
    def input_reader() -> Callable[[str], str]:
        """Return the native input reader for a private-attribute factory.

        Returns:
            The builtin input callable.

        """
        return input

    @staticmethod
    def validated_label(values: Mapping[str, p.AttributeProbe]) -> str:
        """Derive a field default from Pydantic's already validated input.

        Returns:
            The validated label passed to the field factory.

        Raises:
            TypeError: If the validated label is not a string.

        """
        label = values["label"]
        if not isinstance(label, str):
            msg = "Validated label must be a string"
            raise TypeError(msg)
        return label


__all__: list[str] = ["TestsFlextUtilitiesValidationFactoriesMixin"]
