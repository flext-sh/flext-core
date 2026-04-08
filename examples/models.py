"""Public examples model facade for flext-core."""

from __future__ import annotations

from examples._models.shared import SharedHandle, SharedPerson
from flext_core import FlextModels


class ExamplesFlextCoreModels(FlextModels):
    """Public examples model facade extending flext-core models."""

    class Examples:
        """Examples namespace for shared example-domain models."""

        class Person(SharedPerson):
            """Shared person model used by public examples."""

        class Handle(SharedHandle):
            """Shared resource-handle model used by public examples."""


m = ExamplesFlextCoreModels

__all__ = ["ExamplesFlextCoreModels", "m"]
