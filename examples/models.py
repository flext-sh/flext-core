"""Public examples model facade for flext-core."""

from __future__ import annotations

from examples._models.errors import ExamplesFlextCoreModelsErrors
from examples._models.output import ExamplesFlextCoreModelsOutput
from examples._models.shared import (
    ExamplesFlextCoreSharedHandle,
    ExamplesFlextCoreSharedPerson,
)
from flext_core import m


class ExamplesFlextCoreModels(
    ExamplesFlextCoreModelsErrors,
    ExamplesFlextCoreModelsOutput,
    m,
):
    """Public examples model facade extending flext-core models."""

    class Examples(
        ExamplesFlextCoreModelsErrors.Examples,
        ExamplesFlextCoreModelsOutput.Examples,
    ):
        """Examples namespace for shared example-domain models."""

        class Person(ExamplesFlextCoreSharedPerson):
            """Shared person model used by public examples."""

        class Handle(ExamplesFlextCoreSharedHandle):
            """Shared resource-handle model used by public examples."""


m = ExamplesFlextCoreModels

__all__: list[str] = ["ExamplesFlextCoreModels", "m"]
