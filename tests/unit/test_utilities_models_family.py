"""Utilities never carry the models family into the classes that compose them.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TypeAlias

from flext_core import FlextContext, FlextUtilities, m
from tests import tm


class TestsFlextCoreUtilitiesModelsFamily:
    """A Pydantic model composing ``u`` gains no field from the models family."""

    @staticmethod
    def test_models_facade_is_not_in_the_utilities_mro() -> None:
        """``u`` composes utility parts only, never the ``m`` facade."""
        tm.that(m in FlextUtilities.__mro__, eq=False)

    @staticmethod
    def test_model_composing_utilities_has_no_type_alias_fields() -> None:
        """``FlextContext`` builds and none of its fields is a ``TypeAlias``."""
        annotations = [field.annotation for field in FlextContext.model_fields.values()]
        tm.that(
            any(id(annotation) == id(TypeAlias) for annotation in annotations),
            eq=False,
        )
        tm.that("PyprojectDocument" in FlextContext.model_fields, eq=False)


__all__: list[str] = ["TestsFlextCoreUtilitiesModelsFamily"]
