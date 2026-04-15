"""Public examples model facade for flext-core."""

from __future__ import annotations

from examples import (
    ExamplesFlextCoreModelsErrors,
    ExamplesFlextCoreModelsOutput,
    SharedHandle,
    SharedPerson,
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

        class Person(SharedPerson):
            """Shared person model used by public examples."""

        class Handle(SharedHandle):
            """Shared resource-handle model used by public examples."""


m = ExamplesFlextCoreModels

__all__: list[str] = ["ExamplesFlextCoreModels", "m"]
