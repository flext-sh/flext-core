"""Public examples model facade for flext-core."""

from __future__ import annotations

from flext_core import FlextModels

from ._models import SharedHandle, SharedPerson


class FlextCoreExamplesModels(FlextModels):
    """Public examples model facade extending flext-core models."""

    class Examples:
        """Examples namespace for shared example-domain models."""

        class Person(SharedPerson):
            """Shared person model used by public examples."""

        class Handle(SharedHandle):
            """Shared resource-handle model used by public examples."""


m = FlextCoreExamplesModels
Person = FlextCoreExamplesModels.Examples.Person
Handle = FlextCoreExamplesModels.Examples.Handle

__all__ = ["FlextCoreExamplesModels", "Handle", "Person", "m"]
