"""Public examples model facade for flext-core."""

from __future__ import annotations

from examples import t
from examples._models.ex00 import ExamplesFlextCoreModelsEx00
from examples._models.ex01 import ExamplesFlextCoreModelsEx01
from examples._models.ex02 import ExamplesFlextCoreModelsEx02
from examples._models.ex03 import ExamplesFlextCoreModelsEx03
from examples._models.ex04 import ExamplesFlextCoreModelsEx04
from examples._models.ex05 import ExamplesFlextCoreModelsEx05
from examples._models.ex07 import ExamplesFlextCoreModelsEx07
from examples._models.ex08 import ExamplesFlextCoreModelsEx08
from examples._models.ex10 import ExamplesFlextCoreModelsEx10
from examples._models.ex11 import ExamplesFlextCoreModelsEx11
from examples._models.ex12 import ExamplesFlextCoreModelsEx12
from examples._models.ex14 import ExamplesFlextCoreModelsEx14
from flext_core import m


class ExamplesFlextCoreModels(
    m,
):
    """Public examples model facade — composes all _models/* via MRO."""

    class Examples(
        ExamplesFlextCoreModelsEx00,
        ExamplesFlextCoreModelsEx01,
        ExamplesFlextCoreModelsEx02,
        ExamplesFlextCoreModelsEx03,
        ExamplesFlextCoreModelsEx04,
        ExamplesFlextCoreModelsEx05,
        ExamplesFlextCoreModelsEx07,
        ExamplesFlextCoreModelsEx08,
        ExamplesFlextCoreModelsEx10,
        ExamplesFlextCoreModelsEx11,
        ExamplesFlextCoreModelsEx12,
        ExamplesFlextCoreModelsEx14,
    ):
        """Canonical namespace for all example domain models.

        Access via: from examples import m; m.Examples.<ClassName>
        """


m = ExamplesFlextCoreModels

__all__: t.MutableSequenceOf[str] = ["ExamplesFlextCoreModels", "m"]
