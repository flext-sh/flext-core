"""Public examples model facade for flext-core."""

from __future__ import annotations

from examples import t
from examples._models.errors import ExamplesFlextModelsErrors
from examples._models.ex00 import ExamplesFlextModelsEx00
from examples._models.ex01 import ExamplesFlextModelsEx01
from examples._models.ex02 import ExamplesFlextModelsEx02
from examples._models.ex03 import ExamplesFlextModelsEx03
from examples._models.ex04 import ExamplesFlextModelsEx04
from examples._models.ex05 import ExamplesFlextModelsEx05
from examples._models.ex07 import ExamplesFlextModelsEx07
from examples._models.ex08 import ExamplesFlextModelsEx08
from examples._models.ex10 import ExamplesFlextModelsEx10
from examples._models.ex11 import ExamplesFlextModelsEx11
from examples._models.ex12 import ExamplesFlextModelsEx12
from examples._models.ex14 import ExamplesFlextModelsEx14
from examples._models.output import ExamplesFlextModelsOutput
from examples._models.shared import (
    ExamplesFlextSharedHandle,
    ExamplesFlextSharedPerson,
)
from flext_core import m


class ExamplesFlextModels(
    m,
):
    """Public examples model facade — composes all _models/* via MRO."""

    class Examples(
        ExamplesFlextModelsErrors.Examples,
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
