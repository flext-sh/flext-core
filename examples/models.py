# AUTO-GENERATED FILE — Regenerate with: make gen
"""Public examples model facade for flext-core.

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
    ExamplesFlextModelsHandle,
    ExamplesFlextModelsOutput,
    ExamplesFlextModelsPerson,
)
from flext_core import m as flext_core_m


class ExamplesFlextModels(flext_core_m):
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
        ExamplesFlextModelsHandle,
        ExamplesFlextModelsPerson,
        ExamplesFlextModelsErrors.Examples,
        ExamplesFlextModelsOutput.Examples,
    ):
        """Canonical namespace for all example domain models.

        Access via: from examples import m; m.Examples.<ClassName>
        """


m = ExamplesFlextModels
__all__: list[str] = ["ExamplesFlextModels", "m"]
