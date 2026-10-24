# AUTO-GENERATED FILE — Regenerate with: make gen
"""Public examples model facade for flext-core.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from examples._models import (
    ExamplesFlextModelsErrors,
    ExamplesFlextModelsEx00,
    ExamplesFlextModelsEx01,
    ExamplesFlextModelsEx02,
    ExamplesFlextModelsEx03,
    ExamplesFlextModelsEx04,
    ExamplesFlextModelsEx05,
    ExamplesFlextModelsEx07,
    ExamplesFlextModelsEx08,
    ExamplesFlextModelsEx10,
    ExamplesFlextModelsEx11,
    ExamplesFlextModelsEx12,
    ExamplesFlextModelsEx14,
    ExamplesFlextModelsOutput,
    ExamplesFlextSharedHandle,
    ExamplesFlextSharedPerson,
)
from flext_core import FlextModels

if TYPE_CHECKING:
    from examples import t


class ExamplesFlextModels(FlextModels):
    """Public examples model facade — composes all _models/* via MRO."""

    class Examples(
        ExamplesFlextModelsEx00,
        ExamplesFlextModelsEx01,
        ExamplesFlextModelsEx02,
        ExamplesFlextModelsEx03,
        ExamplesFlextModelsEx04,
        ExamplesFlextModelsEx05,
        ExamplesFlextModelsEx07,
        ExamplesFlextModelsEx08,
        ExamplesFlextModelsEx10,
        ExamplesFlextModelsEx11,
        ExamplesFlextModelsEx12,
        ExamplesFlextModelsEx14,
        ExamplesFlextSharedHandle,
        ExamplesFlextSharedPerson,
        ExamplesFlextModelsErrors.Examples,
        ExamplesFlextModelsOutput.Examples,
    ):
        """Canonical namespace for all example domain models.

        Access via: from examples import m; m.Examples.<ClassName>
        """

        class Person(ExamplesFlextSharedPerson):
            """Shared person model used by public examples."""

        class Handle(ExamplesFlextSharedHandle):
            """Shared resource-handle model used by public examples."""


m = ExamplesFlextModels

__all__: t.MutableSequenceOf[str] = ["ExamplesFlextModels", "m"]
